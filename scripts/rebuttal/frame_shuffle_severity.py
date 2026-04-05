"""
P1.5a: Frame shuffle severity gradient — task-level evaluation.

Tests whether temporal integration is global or local by shuffling only a
FRACTION of frames (0%, 25%, 50%, 75%, 100%) and measuring R² degradation.

Predictions:
  - JEPA: linear degradation → global temporal integration
  - MAE: sublinear (most damage from first 25%) → local temporal structure
  - BYOL: concave (rapid collapse then plateau) → fragile temporal coherence

Uses the same frozen encoder + probe pipeline as frame_shuffle_task.py.
Runs "shuffle" variant (frame-level permutation, RoPE positions NOT remapped).

Usage:
    # All 3 models, all severity levels (run on 3 GPUs in parallel)
    python scripts/rebuttal/frame_shuffle_severity.py \
        --test_csv data/csv/echonet_dynamic_test_local.csv \
        --models JEPA-L-pt50 \
        --device cuda:0

    # Quick test
    python scripts/rebuttal/frame_shuffle_severity.py \
        --test_csv data/csv/echonet_dynamic_test_local.csv \
        --models JEPA-L-pt50 \
        --n_videos 50 \
        --device cuda:0
"""

import argparse
import json
import random
import sys
import time

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

PT50_END_CONFIGS = {
    "JEPA-L-pt50": {
        "encoder_type": "vjepa",
        "encoder_checkpoint": "checkpoints/echojepa-l-pt50.pt",
        "encoder_model_name": "vit_large",
        "encoder_key": "target_encoder",
        "probe_checkpoint": "evals/vitl/icml/echojepa_pt50_end_lvef_224/"
                            "video_classification_frozen/icml-echojepa-l-pt50-end-lvef-d4/best.pt",
    },
    "BYOL-L-pt50": {
        "encoder_type": "vjepa",
        "encoder_checkpoint": "checkpoints/byol_vitl_imagenet_v2_e50.pt",
        "encoder_model_name": "vit_large",
        "encoder_key": "target_encoder",
        "probe_checkpoint": "evals/vitl/icml/echobyol_pt50_end_lvef_224/"
                            "video_classification_frozen/icml-echobyol-l-pt50-end-lvef-d4-224/best.pt",
    },
    "MAE-L-pt50": {
        "encoder_type": "videomae",
        "encoder_checkpoint": "checkpoints/videomae_l_mimic_ep50.pth",
        "encoder_model_name": None,
        "encoder_key": None,
        "probe_checkpoint": "evals/vitl/icml/echomae_pt50_end_lvef_224/"
                            "video_classification_frozen/icml-echomae-l-pt50-end-lvef-d4/best.pt",
    },
}

# Training dynamics configs: BYOL and MAE at e24, e50, e75, e100
ALL_CONFIGS = {
    # JEPA (only pt50 available)
    "JEPA-L-e50": PT50_END_CONFIGS["JEPA-L-pt50"],
    # BYOL v2 (ImageNet init)
    "BYOL-L-e24": {
        "encoder_type": "vjepa",
        "encoder_checkpoint": "checkpoints/byol_vitl_imagenet_v2_e24.pt",
        "encoder_model_name": "vit_large",
        "encoder_key": "target_encoder",
        "probe_checkpoint": "evals/vitl/icml/echobyol_e24_end_lvef_224/"
                            "video_classification_frozen/icml-echobyol-l-e24-end-lvef-d4/best.pt",
    },
    "BYOL-L-e50": PT50_END_CONFIGS["BYOL-L-pt50"],
    "BYOL-L-e75": {
        "encoder_type": "vjepa",
        "encoder_checkpoint": "checkpoints/byol_vitl_imagenet_v2_e75.pt",
        "encoder_model_name": "vit_large",
        "encoder_key": "target_encoder",
        "probe_checkpoint": "evals/vitl/icml/echobyol_e75_end_lvef_224/"
                            "video_classification_frozen/icml-echobyol-l-e75-end-lvef-d4/best.pt",
    },
    "BYOL-L-e100": {
        "encoder_type": "vjepa",
        "encoder_checkpoint": "checkpoints/byol_vitl_imagenet_v2_e100.pt",
        "encoder_model_name": "vit_large",
        "encoder_key": "target_encoder",
        "probe_checkpoint": "evals/vitl/icml/echobyol_e100_end_lvef_224/"
                            "video_classification_frozen/icml-echobyol-l-e100-end-lvef-d4/best.pt",
    },
    # VideoMAE (random init)
    "MAE-L-e24": {
        "encoder_type": "videomae",
        "encoder_checkpoint": "checkpoints/videomae_l_mimic_ep24.pth",
        "encoder_model_name": None,
        "encoder_key": None,
        "probe_checkpoint": "evals/vitl/icml/echomae_e24_end_lvef_224/"
                            "video_classification_frozen/icml-echomae-l-e24-end-lvef-d4/best.pt",
    },
    "MAE-L-e50": PT50_END_CONFIGS["MAE-L-pt50"],
    "MAE-L-e74": {
        "encoder_type": "videomae",
        "encoder_checkpoint": "checkpoints/videomae_l_mimic_ep74.pth",
        "encoder_model_name": None,
        "encoder_key": None,
        "probe_checkpoint": "evals/vitl/icml/echomae_e74_end_lvef_224/"
                            "video_classification_frozen/icml-echomae-l-e74-end-lvef-d4/best.pt",
    },
    "MAE-L-e99": {
        "encoder_type": "videomae",
        "encoder_checkpoint": "checkpoints/videomae_l_mimic_ep99.pth",
        "encoder_model_name": None,
        "encoder_key": None,
        "probe_checkpoint": "evals/vitl/icml/echomae_e99_end_lvef_224/"
                            "video_classification_frozen/icml-echomae-l-e99-end-lvef-d4/best.pt",
    },
}

SEVERITY_FRACTIONS = [0.0, 0.25, 0.50, 0.75, 1.0]


# --- Data ---


def load_clip(video_path, frames=16, frame_step=2, resolution=224):
    """Load clip as [C, T, H, W] float32 in [0, 1]."""
    vr = decord.VideoReader(video_path, num_threads=1)
    total = len(vr)
    needed = frames * frame_step
    start = max(0, (total - needed) // 2)
    indices = list(range(start, min(start + needed, total), frame_step))
    while len(indices) < frames:
        indices.append(indices[-1])
    indices = indices[:frames]
    clip = vr.get_batch(indices)  # [T, H, W, C]
    clip = clip.permute(3, 0, 1, 2).float() / 255.0  # [C, T, H, W]
    clip = TF.resize(clip, [resolution, resolution], antialias=True)
    return clip


def partial_shuffle(clip, fraction, rng):
    """
    Shuffle a fraction of frames, keeping the rest in original order.

    Args:
        clip: [C, T, H, W] tensor
        fraction: float in [0, 1]. 0 = clean, 1 = full shuffle.
        rng: np.random.RandomState for reproducibility.

    Returns:
        [C, T, H, W] tensor with `fraction` of frames permuted.
    """
    T = clip.shape[1]
    k = int(round(T * fraction))
    if k < 2:
        return clip

    # Select which frame positions to shuffle
    positions = sorted(rng.choice(T, size=k, replace=False).tolist())

    # Generate a random permutation of those positions
    shuffled_values = rng.permutation(positions).tolist()

    # Build the full index: unchanged positions keep identity mapping
    indices = list(range(T))
    for orig_pos, new_val in zip(positions, shuffled_values):
        indices[orig_pos] = new_val

    return clip[:, indices, :, :]


def read_csv(csv_path):
    paths, labels = [], []
    with open(csv_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                paths.append(parts[0])
                labels.append(float(parts[1]))
    return paths, labels


# --- Model Loading ---


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
    print(f"  Probe: depth={depth}, heads={num_heads}, best_head={best_idx}, "
          f"z-score={target_mean}/{target_std}")
    return probe, target_mean, target_std


# --- Inference ---


@torch.no_grad()
def run_inference(encoder, probe, paths, labels, device, is_videomae,
                  target_mean, target_std,
                  resolution=224, frames=16, frame_step=2,
                  shuffle_fraction=0.0, shuffle_rng=None):
    """Run inference with optional partial frame shuffling."""
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

        if shuffle_rng is not None and shuffle_fraction > 0:
            clip = partial_shuffle(clip, shuffle_fraction, shuffle_rng)

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
    parser = argparse.ArgumentParser(description="P1.5a: Frame shuffle severity gradient")
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--models", nargs="*", default=None,
                        help="Model names from PT50_END_CONFIGS or ALL_CONFIGS")
    parser.add_argument("--all", action="store_true",
                        help="Run all 9 models (training dynamics sweep)")
    parser.add_argument("--fractions", nargs="*", type=float, default=None,
                        help=f"Shuffle fractions (default: {SEVERITY_FRACTIONS})")
    parser.add_argument("--n_videos", type=int, default=None)
    parser.add_argument("--n_seeds", type=int, default=3)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--frame_step", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    device = torch.device(args.device)
    fractions = args.fractions if args.fractions else SEVERITY_FRACTIONS

    # Model selection: --all uses all 9, --models picks from either dict
    combined = {**PT50_END_CONFIGS, **ALL_CONFIGS}
    if args.all:
        configs = ALL_CONFIGS
    elif args.models:
        configs = {k: combined[k] for k in args.models if k in combined}
    else:
        configs = PT50_END_CONFIGS

    paths, labels = read_csv(args.test_csv)
    if args.n_videos and args.n_videos < len(paths):
        indices = random.sample(range(len(paths)), args.n_videos)
        paths = [paths[i] for i in indices]
        labels = [labels[i] for i in indices]
    print(f"Test set: {len(paths)} videos")
    print(f"Fractions: {fractions}")
    print(f"Seeds per fraction: {args.n_seeds}")

    all_results = {}

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

        model_results = {}
        t0 = time.time()

        for frac in fractions:
            if frac == 0.0:
                # Clean baseline — single run, no RNG needed
                print(f"\n  [fraction=0.00 — clean]")
                preds, labs, skipped = run_inference(
                    encoder, probe, paths, labels, device, is_videomae,
                    target_mean, target_std, args.resolution, args.frames, args.frame_step,
                    shuffle_fraction=0.0, shuffle_rng=None,
                )
                metrics = compute_metrics(preds, labs)
                model_results[0.0] = {
                    "mean": metrics,
                    "std": {"R2": 0.0, "Pearson": 0.0, "MAE": 0.0},
                    "per_seed": [metrics],
                }
                print(f"    R²={metrics['R2']:.4f}  Pearson={metrics['Pearson']:.4f}  "
                      f"MAE={metrics['MAE']:.3f}  (skipped {skipped})")
            else:
                # Multiple seeds
                seed_metrics = []
                for seed_idx in range(args.n_seeds):
                    shuffle_seed = seed_idx + 100
                    rng = np.random.RandomState(shuffle_seed)
                    print(f"\n  [fraction={frac:.2f}, seed={shuffle_seed}]")

                    preds, labs, skipped = run_inference(
                        encoder, probe, paths, labels, device, is_videomae,
                        target_mean, target_std, args.resolution, args.frames, args.frame_step,
                        shuffle_fraction=frac, shuffle_rng=rng,
                    )
                    metrics = compute_metrics(preds, labs)
                    seed_metrics.append(metrics)
                    print(f"    R²={metrics['R2']:.4f}  Pearson={metrics['Pearson']:.4f}  "
                          f"MAE={metrics['MAE']:.3f}")

                mean_metrics = {
                    k: float(np.mean([m[k] for m in seed_metrics]))
                    for k in ("R2", "Pearson", "MAE")
                }
                std_metrics = {
                    k: float(np.std([m[k] for m in seed_metrics]))
                    for k in ("R2", "Pearson", "MAE")
                }
                model_results[frac] = {
                    "mean": mean_metrics,
                    "std": std_metrics,
                    "per_seed": seed_metrics,
                }

        elapsed = time.time() - t0
        all_results[model_name] = model_results

        # Per-model summary
        print(f"\n  --- {model_name} summary ({elapsed:.0f}s) ---")
        clean_r2 = model_results[0.0]["mean"]["R2"]
        for frac in fractions:
            r = model_results[frac]
            r2 = r["mean"]["R2"]
            r2_std = r["std"]["R2"]
            drop = (clean_r2 - r2) / abs(clean_r2) * 100 if clean_r2 != 0 else 0
            print(f"    frac={frac:.2f}: R²={r2:.4f} ± {r2_std:.4f}  ({drop:+.1f}%)")

        del encoder, probe
        torch.cuda.empty_cache()

    # --- Final summary ---
    print(f"\n{'=' * 100}")
    print("P1.5a: FRAME SHUFFLE SEVERITY GRADIENT")
    print(f"Dataset: {args.test_csv} ({len(paths)} videos), Seeds: {args.n_seeds}")
    print(f"{'=' * 100}")

    header = f"{'Fraction':<10}"
    for name in all_results:
        header += f"  {name:<25}"
    print(header)
    print("-" * 100)

    for frac in fractions:
        row = f"{frac:<10.2f}"
        for name in all_results:
            r = all_results[name][frac]
            r2 = r["mean"]["R2"]
            r2_std = r["std"]["R2"]
            clean_r2 = all_results[name][0.0]["mean"]["R2"]
            drop = (clean_r2 - r2) / abs(clean_r2) * 100 if clean_r2 != 0 else 0
            if frac == 0.0:
                row += f"  {r2:.4f} (baseline)          "
            else:
                row += f"  {r2:.4f}±{r2_std:.4f} ({drop:+.1f}%)  "
        print(row)

    # --- Save CSV ---
    if args.output is None:
        args.output = "scripts/rebuttal/samples/frame_shuffle_severity_results.csv"
    with open(args.output, "w") as f:
        f.write("model,fraction,seed,R2,Pearson,MAE\n")
        for name, model_results in all_results.items():
            for frac in fractions:
                r = model_results[frac]
                for seed_idx, m in enumerate(r["per_seed"]):
                    seed_val = 0 if frac == 0.0 else seed_idx + 100
                    f.write(f"{name},{frac:.2f},{seed_val},"
                            f"{m['R2']:.6f},{m['Pearson']:.6f},{m['MAE']:.6f}\n")
    print(f"\nSaved: {args.output}")

    # --- Save JSON for plotting ---
    json_output = args.output.replace(".csv", ".json")
    json_data = {}
    for name, model_results in all_results.items():
        json_data[name] = {}
        for frac in fractions:
            r = model_results[frac]
            json_data[name][str(frac)] = {
                "mean": r["mean"],
                "std": r["std"],
            }
    with open(json_output, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"Saved: {json_output}")


if __name__ == "__main__":
    main()
