"""
6-condition frame shuffling temporal ablation on arbitrary encoder+probe.

Extends frame_shuffle_task.py to support all 6 conditions from the ICML rebuttal:
  1. clean — original frame order
  2. tubelet — permute at 2-frame tubelet granularity
  3. reverse — play video backwards
  4. shuffle — full frame permutation (N seeds)
  5. matched — tubelet shuffle with FIXED perm (same for all videos)
  6. matched_frame — frame shuffle with FIXED perm (most rigorous)

Note: matched/matched_frame ideally involve RoPE remapping, which requires
modifying the encoder. In this standalone script, we apply the fixed permutation
WITHOUT RoPE remapping. This means "matched" here is equivalent to "tubelet
with fixed perm" and "matched_frame" is "frame with fixed perm." The RoPE
mismatch is a known limitation — see frame-shuffling.md for the full analysis.

Usage:
    python scripts/rebuttal/frame_shuffle_6cond.py \
        --test_csv data/csv/echonet_dynamic_test_local.csv \
        --models JEPA-IN21K-e100 \
        --device cuda:0 \
        --output scripts/rebuttal/samples/6cond_JEPA_IN21K_e100.csv
"""

import argparse
import hashlib
import random
import sys

import decord
import numpy as np
import torch
import torchvision.transforms.functional as TF

sys.path.insert(0, ".")

import src.models.vision_transformer as vit
from src.models.attentive_pooler import AttentiveRegressor
from src.utils.checkpoint_loader import robust_checkpoint_loader

decord.bridge.set_bridge("torch")

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)

# Import model configs from severity script
from scripts.rebuttal.frame_shuffle_severity import ALL_CONFIGS, PT50_END_CONFIGS


CONDITIONS = ["clean", "tubelet", "reverse", "matched", "shuffle", "matched_frame"]


def load_clip(video_path, frames=16, frame_step=2, resolution=224):
    vr = decord.VideoReader(video_path, num_threads=1)
    total = len(vr)
    needed = frames * frame_step
    start = max(0, (total - needed) // 2)
    indices = list(range(start, min(start + needed, total), frame_step))
    while len(indices) < frames:
        indices.append(indices[-1])
    indices = indices[:frames]
    clip = vr.get_batch(indices)
    clip = clip.permute(3, 0, 1, 2).float() / 255.0
    clip = TF.resize(clip, [resolution, resolution], antialias=True)
    return clip


def apply_temporal_ablation(clip, condition, seed, video_path):
    """Apply one of 6 temporal ablation conditions to a clip [C, T, H, W]."""
    T = clip.shape[1]
    video_hash = int(hashlib.md5(str(video_path).encode()).hexdigest()[:8], 16)
    rng = np.random.RandomState(video_hash + seed)

    if condition == "clean":
        return clip
    elif condition == "reverse":
        return clip[:, torch.arange(T - 1, -1, -1), :, :]
    elif condition == "tubelet":
        n_tubelets = T // 2
        tubelet_perm = rng.permutation(n_tubelets)
        indices = []
        for ti in tubelet_perm:
            indices.extend([ti * 2, ti * 2 + 1])
        if T % 2 == 1:
            indices.append(T - 1)
        return clip[:, indices, :, :]
    elif condition == "matched":
        # Fixed tubelet-level perm (same for all videos in this seed)
        n_tubelets = T // 2
        fixed_rng = np.random.RandomState(seed)
        tubelet_perm = fixed_rng.permutation(n_tubelets)
        indices = []
        for ti in tubelet_perm:
            indices.extend([ti * 2, ti * 2 + 1])
        if T % 2 == 1:
            indices.append(T - 1)
        return clip[:, indices, :, :]
    elif condition == "shuffle":
        perm = rng.permutation(T)
        return clip[:, perm, :, :]
    elif condition == "matched_frame":
        # Fixed frame-level perm (same for all videos in this seed)
        fixed_rng = np.random.RandomState(seed)
        frame_perm = fixed_rng.permutation(T)
        return clip[:, frame_perm, :, :]
    else:
        raise ValueError(f"Unknown condition: {condition}")


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
    state = ckpt[checkpoint_key]
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
    num_targets = state_dict["regressor.weight"].shape[0]
    probe = AttentiveRegressor(
        embed_dim=embed_dim, num_heads=num_heads, depth=depth, num_targets=num_targets,
    )
    probe.load_state_dict(state_dict)
    probe.eval().to(device)
    target_mean = ckpt.get("target_mean")
    target_std = ckpt.get("target_std")
    return probe, target_mean, target_std


@torch.no_grad()
def run_condition(encoder, probe, paths, labels, device, is_videomae,
                  target_mean, target_std, condition, seed,
                  resolution=224, frames=16, frame_step=2):
    all_preds, all_labels = [], []
    n_skipped = 0
    for i, (path, label) in enumerate(zip(paths, labels)):
        try:
            clip = load_clip(path, frames, frame_step, resolution)
        except Exception as e:
            n_skipped += 1
            if n_skipped <= 3:
                print(f"  Skip {path}: {e}")
            continue

        clip = apply_temporal_ablation(clip, condition, seed, path)
        clip = (clip - IMAGENET_MEAN) / IMAGENET_STD
        clip = clip.unsqueeze(0).to(device)

        if is_videomae:
            features = encoder.forward_features(clip)
            if features.dim() == 2:
                features = features.unsqueeze(0)
        else:
            features = encoder(clip)

        pred = probe(features).cpu().squeeze().item()
        if target_mean is not None and target_std is not None:
            pred = pred * target_std + target_mean
        all_preds.append(pred)
        all_labels.append(label)

        if (i + 1) % 400 == 0:
            print(f"    {i + 1}/{len(paths)}")

    return np.array(all_preds), np.array(all_labels), n_skipped


def compute_metrics(preds, labels):
    from scipy.stats import pearsonr
    residuals = labels - preds
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((labels - labels.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    mae = np.mean(np.abs(residuals))
    r, _ = pearsonr(labels, preds) if len(labels) > 2 else (0.0, 1.0)
    return {"R2": r2, "MAE": mae, "Pearson": r}


def main():
    parser = argparse.ArgumentParser(description="6-condition frame shuffling")
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--conditions", nargs="*", default=None,
                        help=f"Conditions to run (default: all). Options: {CONDITIONS}")
    parser.add_argument("--n_seeds", type=int, default=3,
                        help="Seeds for stochastic conditions (shuffle, matched_frame)")
    parser.add_argument("--n_videos", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--frame_step", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    device = torch.device(args.device)
    conditions = args.conditions if args.conditions else CONDITIONS

    combined = {**PT50_END_CONFIGS, **ALL_CONFIGS}
    if args.models:
        configs = {k: combined[k] for k in args.models if k in combined}
    else:
        configs = ALL_CONFIGS

    paths, labels = read_csv(args.test_csv)
    if args.n_videos and args.n_videos < len(paths):
        indices = random.sample(range(len(paths)), args.n_videos)
        paths = [paths[i] for i in indices]
        labels = [labels[i] for i in indices]
    print(f"Test set: {len(paths)} videos")
    print(f"Conditions: {conditions}")

    all_results = []

    for model_name, cfg in configs.items():
        print(f"\n{'=' * 70}")
        print(f"Model: {model_name}")
        print(f"{'=' * 70}")

        is_videomae = cfg["encoder_type"] == "videomae"
        if is_videomae:
            encoder = load_videomae_encoder(
                cfg["encoder_checkpoint"], device, args.resolution, args.frames)
        else:
            encoder = load_vjepa_encoder(
                cfg["encoder_checkpoint"], cfg["encoder_model_name"],
                cfg["encoder_key"], device, args.resolution, args.frames)
        embed_dim = 1024

        probe, target_mean, target_std = load_probe(cfg["probe_checkpoint"], embed_dim, device)

        for cond in conditions:
            # Deterministic conditions: single run
            if cond in ("clean", "reverse"):
                print(f"\n  [{cond}]")
                preds, labs, skipped = run_condition(
                    encoder, probe, paths, labels, device, is_videomae,
                    target_mean, target_std, cond, seed=100,
                    resolution=args.resolution, frames=args.frames, frame_step=args.frame_step,
                )
                metrics = compute_metrics(preds, labs)
                print(f"    R²={metrics['R2']:.4f}  Pearson={metrics['Pearson']:.4f}  MAE={metrics['MAE']:.3f}")
                all_results.append({
                    "model": model_name, "condition": cond, "seed": 0,
                    **{k: f"{v:.6f}" for k, v in metrics.items()},
                })

            # Stochastic conditions: multiple seeds
            else:
                for seed_idx in range(args.n_seeds):
                    seed = seed_idx + 100
                    print(f"\n  [{cond}, seed={seed}]")
                    preds, labs, skipped = run_condition(
                        encoder, probe, paths, labels, device, is_videomae,
                        target_mean, target_std, cond, seed=seed,
                        resolution=args.resolution, frames=args.frames, frame_step=args.frame_step,
                    )
                    metrics = compute_metrics(preds, labs)
                    print(f"    R²={metrics['R2']:.4f}  Pearson={metrics['Pearson']:.4f}  MAE={metrics['MAE']:.3f}")
                    all_results.append({
                        "model": model_name, "condition": cond, "seed": seed,
                        **{k: f"{v:.6f}" for k, v in metrics.items()},
                    })

        del encoder, probe
        torch.cuda.empty_cache()

    # Save CSV
    if args.output is None:
        args.output = "scripts/rebuttal/samples/frame_shuffle_6cond_results.csv"
    with open(args.output, "w") as f:
        f.write("model,condition,seed,R2,Pearson,MAE\n")
        for r in all_results:
            f.write(f"{r['model']},{r['condition']},{r['seed']},{r['R2']},{r['Pearson']},{r['MAE']}\n")
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
