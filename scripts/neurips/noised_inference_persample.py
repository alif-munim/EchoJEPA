"""
LVEF noised inference with per-sample output for bootstrap CIs.

Saves per-sample (prediction, label) pairs for each condition, enabling
bootstrap resampling over 1,277 test videos to compute R² CIs.

Output: CSV with columns [sample_idx, condition, prediction, label]
"""

import argparse
import csv
import hashlib
import sys

import decord
import numpy as np
import torch
import torchvision.transforms.functional as TF

sys.path.insert(0, ".")
import src.models.vision_transformer as vit
from scripts.neurips.echo_perturbations import (
    PERTURBATIONS,
    SEVERITY_LEVELS,
    TRANSDUCER_PRESETS,
    apply_perturbation,
    create_scan_mask,
)
from src.models.attentive_pooler import AttentiveRegressor
from src.utils.checkpoint_loader import robust_checkpoint_loader

decord.bridge.set_bridge("torch")

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)


def load_clip(video_path, frames=16, frame_step=2, resolution=224):
    vr = decord.VideoReader(video_path, num_threads=1)
    total = len(vr)
    needed = frames * frame_step
    start = max(0, (total - needed) // 2)
    indices = list(range(start, min(start + needed, total), frame_step))
    while len(indices) < frames:
        indices.append(indices[-1])
    indices = indices[:frames]
    clip = vr.get_batch(indices).permute(3, 0, 1, 2).float() / 255.0
    clip = TF.resize(clip, [resolution, resolution], antialias=True)
    return clip


def read_csv(csv_path):
    paths, labels = [], []
    with open(csv_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                paths.append(parts[0])
                labels.append(float(parts[1]))
    return paths, labels


def load_vjepa_encoder(checkpoint, model_name, checkpoint_key, device, resolution=224, frames=16):
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = vit.__dict__[model_name](
        img_size=resolution, num_frames=frames, patch_size=16, tubelet_size=2,
        uniform_power=True, use_rope=True,
    )
    if checkpoint_key in ckpt:
        state = ckpt[checkpoint_key]
    else:
        for k in ["encoder", "model", "state_dict"]:
            if k in ckpt:
                state = ckpt[k]
                break
        else:
            state = ckpt
    state = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state.items()}
    model_sd = model.state_dict()
    for k in list(state.keys()):
        if k in model_sd and state[k].shape != model_sd[k].shape:
            del state[k]
    model.load_state_dict(state, strict=False)
    model.eval().to(device)
    return model


def load_videomae_encoder(checkpoint, device, resolution=224, frames=16):
    from evals.video_classification_frozen.modelcustom.videomae_encoder import (
        _convert_pretrain_to_finetune_state_dict,
        _import_modeling_finetune,
    )
    mf = _import_modeling_finetune()
    model = mf.vit_large_patch16_224(
        img_size=resolution, all_frames=frames, tubelet_size=2, num_classes=1000,
    )
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)
    state = _convert_pretrain_to_finetune_state_dict(state, model.state_dict())
    model.load_state_dict(state, strict=False)
    model.eval().to(device)
    return model


def load_probe(probe_checkpoint, embed_dim, device):
    ckpt = robust_checkpoint_loader(probe_checkpoint, map_location="cpu")
    best_vals = ckpt["best_val_acc_per_head"]
    best_idx = best_vals.index(min(best_vals))
    state_dict = ckpt["classifiers"][best_idx]
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    sa_block_indices = set()
    for k in state_dict:
        if k.startswith("pooler.blocks."):
            sa_block_indices.add(int(k.split(".")[2]))
    depth = len(sa_block_indices) + 1
    num_heads = embed_dim // 64
    linear_key = "regressor.weight"
    num_targets = state_dict[linear_key].shape[0]

    probe = AttentiveRegressor(
        embed_dim=embed_dim, num_heads=num_heads, depth=depth, num_targets=num_targets,
    )
    probe.load_state_dict(state_dict)
    probe.eval().to(device)

    metadata = {
        "target_mean": ckpt.get("target_mean"),
        "target_std": ckpt.get("target_std"),
        "best_head_idx": best_idx,
        "best_val": best_vals[best_idx],
    }
    print(f"  Probe: depth={depth}, heads={num_heads}, best_head={best_idx} (val={best_vals[best_idx]:.4f})")
    return probe, metadata


@torch.no_grad()
def run_condition_persample(encoder, probe, paths, labels, device, is_videomae,
                            target_mean, target_std,
                            perturbation_type=None, severity=None,
                            resolution=224, frames=16, frame_step=2,
                            transducer_pos=(0.5, 0.0)):
    """Run inference, returning per-sample (prediction, label) pairs."""
    results = []
    n_skipped = 0

    for i, (video_path, label) in enumerate(zip(paths, labels)):
        try:
            clip = load_clip(video_path, frames, frame_step, resolution)
        except Exception:
            n_skipped += 1
            continue

        if perturbation_type is not None:
            seed = int(hashlib.md5(video_path.encode()).hexdigest()[:8], 16)
            mask = create_scan_mask(clip[:, 0, :, :])
            clip = apply_perturbation(
                clip, perturbation_type, severity, scan_mask=mask, seed=seed,
                transducer_pos=transducer_pos,
            )

        clip = (clip - IMAGENET_MEAN) / IMAGENET_STD
        clip = clip.unsqueeze(0).to(device)

        if is_videomae:
            features = encoder.forward_features(clip)
            if features.dim() == 2:
                features = features.unsqueeze(0)
        else:
            features = encoder(clip)

        pred = probe(features).cpu().squeeze().item()

        # Un-normalize prediction
        if target_mean is not None and target_std is not None:
            pred = pred * target_std + target_mean

        results.append({"sample_idx": i, "prediction": pred, "label": label})

        if (i + 1) % 200 == 0:
            print(f"    {i + 1}/{len(paths)}", flush=True)

    if n_skipped:
        print(f"    (skipped {n_skipped} videos)")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder_type", required=True, choices=["vjepa", "videomae"])
    parser.add_argument("--encoder_checkpoint", required=True)
    parser.add_argument("--encoder_model_name", default="vit_large")
    parser.add_argument("--encoder_key", default="target_encoder")
    parser.add_argument("--probe_checkpoint", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--frame_step", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--label", default="model")
    args = parser.parse_args()

    device = torch.device(args.device)
    is_videomae = args.encoder_type == "videomae"

    print(f"Loading encoder: {args.encoder_type} from {args.encoder_checkpoint}")
    if is_videomae:
        encoder = load_videomae_encoder(args.encoder_checkpoint, device, args.resolution, args.frames)
        embed_dim = 1024
    else:
        encoder = load_vjepa_encoder(
            args.encoder_checkpoint, args.encoder_model_name, args.encoder_key,
            device, args.resolution, args.frames,
        )
        embed_dims = {"vit_large": 1024, "vit_giant_xformers": 1408, "vit_huge": 1280}
        embed_dim = embed_dims.get(args.encoder_model_name, 1024)

    print(f"Loading probe: {args.probe_checkpoint}")
    probe, metadata = load_probe(args.probe_checkpoint, embed_dim, device)
    target_mean = metadata.get("target_mean")
    target_std = metadata.get("target_std")

    print(f"Loading test data: {args.test_csv}")
    paths, labels = read_csv(args.test_csv)
    print(f"  {len(paths)} videos")

    transducer_pos = TRANSDUCER_PRESETS["standard"]

    conditions = [("clean", None, None)]
    for ptype in PERTURBATIONS:
        for sev in SEVERITY_LEVELS:
            conditions.append((f"{ptype}/{sev}", ptype, sev))

    out_path = f"scripts/neurips/samples/{args.label}_noised_lvef_persample.csv"
    fieldnames = ["sample_idx", "condition", "prediction", "label"]

    all_rows = []
    for cond_name, ptype, sev in conditions:
        print(f"  {cond_name} ...", flush=True)
        rows = run_condition_persample(
            encoder, probe, paths, labels, device, is_videomae,
            target_mean, target_std, ptype, sev,
            args.resolution, args.frames, args.frame_step, transducer_pos,
        )
        for r in rows:
            r["condition"] = cond_name
        all_rows.extend(rows)

        # Quick aggregate
        preds = np.array([r["prediction"] for r in rows])
        labs = np.array([r["label"] for r in rows])
        ss_res = np.sum((labs - preds) ** 2)
        ss_tot = np.sum((labs - labs.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        print(f"    R²={r2:.4f} (n={len(rows)})")

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nSaved {len(all_rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
