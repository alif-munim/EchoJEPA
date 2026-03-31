"""
Frame shuffling temporal ablation — TASK-LEVEL evaluation.

Measures how much downstream task performance (R², Pearson) degrades when
frame order is destroyed. Complements the representation-level cosine
similarity analysis in frame_shuffling.py.

Hypothesis: EchoJEPA encodes temporal dynamics (cardiac cycle phase, wall
motion) so shuffling destroys task-relevant information. EchoMAE learns
static frame appearance, so shuffling has minimal effect.

Runs frozen encoder + trained probe on:
  1. Clean (original frame order) — baseline
  2. Shuffled × N seeds — permuted frame order

Reports R², Pearson, MAE for each condition with mean +/- std across seeds.

Usage:
    # All 3 pt50 models on EchoNet-Dynamic test (1,277 videos)
    python scripts/rebuttal/frame_shuffle_task.py \
        --test_csv data/csv/echonet_dynamic_test_local.csv \
        --device cuda:0

    # Single model, quick test
    python scripts/rebuttal/frame_shuffle_task.py \
        --test_csv data/csv/echonet_dynamic_test_local.csv \
        --models JEPA-L-pt50 \
        --n_videos 50 \
        --device cuda:0
"""

import argparse
import random
import sys

import decord
import numpy as np
import torch
import torchvision.transforms.functional as TF

sys.path.insert(0, ".")

import src.models.vision_transformer as vit
from src.models.attentive_pooler import AttentiveClassifier, AttentiveRegressor
from src.utils.checkpoint_loader import robust_checkpoint_loader

decord.bridge.set_bridge("torch")

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)

# --- Model + probe configs for pt50 3-way on EchoNet-Dynamic LVEF ---
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


def read_csv(csv_path):
    """Read space-delimited CSV: path label."""
    paths, labels = [], []
    with open(csv_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                paths.append(parts[0])
                labels.append(float(parts[1]))
    return paths, labels


# --- Model Loading (from noised_inference.py) ---


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
            print(f"  Shape mismatch, skipping: {k}")
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
    """Load best probe head from multi-head checkpoint."""
    ckpt = robust_checkpoint_loader(probe_checkpoint, map_location="cpu")

    best_vals = ckpt["best_val_acc_per_head"]
    best_idx = best_vals.index(min(best_vals))  # regression: lower MAE = better

    state_dict = ckpt["classifiers"][best_idx]
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    # Infer depth from state dict
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

    target_mean = ckpt.get("target_mean")
    target_std = ckpt.get("target_std")
    print(f"  Probe: depth={depth}, heads={num_heads}, best_head={best_idx} "
          f"(val={best_vals[best_idx]:.4f}), z-score={target_mean}/{target_std}")
    return probe, target_mean, target_std


# --- Inference ---


@torch.no_grad()
def run_inference(encoder, probe, paths, labels, device, is_videomae,
                  target_mean, target_std,
                  resolution=224, frames=16, frame_step=2,
                  shuffle_rng=None):
    """
    Run inference on all videos, optionally with frame shuffling.

    Args:
        shuffle_rng: If provided, shuffle frame order using this RNG.
                     If None, use original frame order (clean baseline).
    """
    all_preds = []
    all_labels = []
    n_skipped = 0

    for i, (path, label) in enumerate(zip(paths, labels)):
        try:
            clip = load_clip(path, frames, frame_step, resolution)
        except Exception as e:
            n_skipped += 1
            if n_skipped <= 3:
                print(f"  Skip {path}: {e}")
            continue

        # Frame shuffling: permute temporal dimension
        if shuffle_rng is not None:
            T = clip.shape[1]
            perm = shuffle_rng.permutation(T)
            clip = clip[:, perm, :, :]

        # Normalize and encode
        clip = (clip - IMAGENET_MEAN) / IMAGENET_STD
        clip = clip.unsqueeze(0).to(device)  # [1, C, T, H, W]

        if is_videomae:
            features = encoder.forward_features(clip)
            if features.dim() == 2:
                features = features.unsqueeze(0)
        else:
            features = encoder(clip)  # [1, N, D]

        pred = probe(features).cpu().squeeze().item()

        # Un-normalize prediction
        if target_mean is not None and target_std is not None:
            pred = pred * target_std + target_mean

        all_preds.append(pred)
        all_labels.append(label)

        if (i + 1) % 200 == 0:
            print(f"    {i + 1}/{len(paths)}")

    return np.array(all_preds), np.array(all_labels), n_skipped


def compute_metrics(preds, labels):
    """Compute R², Pearson, MAE."""
    from scipy.stats import pearsonr

    residuals = labels - preds
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((labels - labels.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    mae = np.mean(np.abs(residuals))
    r, _ = pearsonr(labels, preds) if len(labels) > 2 else (0.0, 1.0)
    return {"R2": r2, "MAE": mae, "Pearson": r}


def main():
    parser = argparse.ArgumentParser(description="Frame shuffling — task-level temporal ablation")
    parser.add_argument("--test_csv", required=True, help="Space-delimited CSV: path raw_label")
    parser.add_argument("--models", nargs="*", default=None,
                        help=f"Models to run (default: all). Options: {list(PT50_END_CONFIGS.keys())}")
    parser.add_argument("--n_videos", type=int, default=None,
                        help="Subsample N videos (default: all)")
    parser.add_argument("--n_shuffle_seeds", type=int, default=3,
                        help="Number of shuffle seeds (default: 3)")
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--frame_step", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="scripts/rebuttal/samples/frame_shuffle_task_results.csv")
    args = parser.parse_args()

    random.seed(args.seed)
    device = torch.device(args.device)

    # Select models
    if args.models:
        configs = {k: PT50_END_CONFIGS[k] for k in args.models if k in PT50_END_CONFIGS}
    else:
        configs = PT50_END_CONFIGS

    # Load test data
    paths, labels = read_csv(args.test_csv)
    if args.n_videos and args.n_videos < len(paths):
        indices = random.sample(range(len(paths)), args.n_videos)
        paths = [paths[i] for i in indices]
        labels = [labels[i] for i in indices]
    print(f"Test set: {len(paths)} videos")

    all_results = {}

    for model_name, cfg in configs.items():
        print(f"\n{'=' * 70}")
        print(f"Model: {model_name}")
        print(f"{'=' * 70}")

        # Load encoder
        is_videomae = cfg["encoder_type"] == "videomae"
        if is_videomae:
            encoder = load_videomae_encoder(
                cfg["encoder_checkpoint"], device, args.resolution, args.frames)
            embed_dim = 1024
        else:
            encoder = load_vjepa_encoder(
                cfg["encoder_checkpoint"], cfg["encoder_model_name"],
                cfg["encoder_key"], device, args.resolution, args.frames)
            embed_dim = 1024  # ViT-L

        # Load probe
        probe, target_mean, target_std = load_probe(cfg["probe_checkpoint"], embed_dim, device)

        # --- Clean baseline ---
        print("\n  [Clean — original frame order]")
        preds_clean, labs, skipped = run_inference(
            encoder, probe, paths, labels, device, is_videomae,
            target_mean, target_std, args.resolution, args.frames, args.frame_step,
            shuffle_rng=None,
        )
        clean_metrics = compute_metrics(preds_clean, labs)
        print(f"    R²={clean_metrics['R2']:.4f}  Pearson={clean_metrics['Pearson']:.4f}  "
              f"MAE={clean_metrics['MAE']:.3f}  (skipped {skipped})")

        # --- Shuffled × N seeds ---
        shuffle_metrics = []
        for seed_idx in range(args.n_shuffle_seeds):
            shuffle_seed = seed_idx + 100
            rng = np.random.RandomState(shuffle_seed)
            print(f"\n  [Shuffled — seed {shuffle_seed}]")

            preds_shuf, labs_shuf, skipped_shuf = run_inference(
                encoder, probe, paths, labels, device, is_videomae,
                target_mean, target_std, args.resolution, args.frames, args.frame_step,
                shuffle_rng=rng,
            )
            metrics = compute_metrics(preds_shuf, labs_shuf)
            shuffle_metrics.append(metrics)
            print(f"    R²={metrics['R2']:.4f}  Pearson={metrics['Pearson']:.4f}  "
                  f"MAE={metrics['MAE']:.3f}")

        # Aggregate shuffled results
        mean_r2 = np.mean([m["R2"] for m in shuffle_metrics])
        std_r2 = np.std([m["R2"] for m in shuffle_metrics])
        mean_pearson = np.mean([m["Pearson"] for m in shuffle_metrics])
        std_pearson = np.std([m["Pearson"] for m in shuffle_metrics])
        mean_mae = np.mean([m["MAE"] for m in shuffle_metrics])
        std_mae = np.std([m["MAE"] for m in shuffle_metrics])

        r2_drop = (clean_metrics["R2"] - mean_r2) / abs(clean_metrics["R2"]) * 100 if clean_metrics["R2"] != 0 else 0
        pearson_drop = (clean_metrics["Pearson"] - mean_pearson) / abs(clean_metrics["Pearson"]) * 100 if clean_metrics["Pearson"] != 0 else 0

        all_results[model_name] = {
            "clean": clean_metrics,
            "shuffled_mean": {"R2": mean_r2, "Pearson": mean_pearson, "MAE": mean_mae},
            "shuffled_std": {"R2": std_r2, "Pearson": std_pearson, "MAE": std_mae},
            "R2_drop_pct": r2_drop,
            "Pearson_drop_pct": pearson_drop,
        }

        # Free GPU memory
        del encoder, probe
        torch.cuda.empty_cache()

    # --- Summary ---
    print(f"\n{'=' * 90}")
    print("FRAME SHUFFLING — TASK-LEVEL TEMPORAL ABLATION")
    print(f"Dataset: {args.test_csv} ({len(paths)} videos)")
    print(f"Shuffle seeds: {args.n_shuffle_seeds}")
    print(f"{'=' * 90}")

    header = f"{'Model':<18} {'Clean R²':<12} {'Shuffled R²':<20} {'R² Drop':<12} {'Clean Pearson':<15} {'Shuf Pearson':<20} {'Pear Drop':<12}"
    print(header)
    print("-" * 109)
    for name, r in all_results.items():
        c = r["clean"]
        sm = r["shuffled_mean"]
        ss = r["shuffled_std"]
        print(f"{name:<18} {c['R2']:<12.4f} {sm['R2']:.4f} +/- {ss['R2']:.4f}   "
              f"{r['R2_drop_pct']:>+7.1f}%    {c['Pearson']:<15.4f} "
              f"{sm['Pearson']:.4f} +/- {ss['Pearson']:.4f}   {r['Pearson_drop_pct']:>+7.1f}%")

    # Rebuttal-ready summary
    print(f"\n{'=' * 90}")
    print("REBUTTAL SUMMARY (copy-paste)")
    print(f"{'=' * 90}")
    for name, r in all_results.items():
        c = r["clean"]
        sm = r["shuffled_mean"]
        print(f"  {name}: R² {c['R2']:.3f} -> {sm['R2']:.3f} ({r['R2_drop_pct']:+.1f}%), "
              f"Pearson {c['Pearson']:.3f} -> {sm['Pearson']:.3f} ({r['Pearson_drop_pct']:+.1f}%)")

    # Save CSV
    with open(args.output, "w") as f:
        f.write("model,condition,R2,Pearson,MAE\n")
        for name, r in all_results.items():
            c = r["clean"]
            sm = r["shuffled_mean"]
            f.write(f"{name},clean,{c['R2']:.6f},{c['Pearson']:.6f},{c['MAE']:.6f}\n")
            f.write(f"{name},shuffled_mean,{sm['R2']:.6f},{sm['Pearson']:.6f},{sm['MAE']:.6f}\n")
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
