"""
Noise autocorrelation sweep — causal test of the frame-varying noise mechanism.

Adds multiplicative speckle-like noise with controllable temporal correlation
to echo videos. Sweeps correlation time τ from ∞ (static, same noise every
frame) to 0 (iid, independent noise each frame).

Hypothesis: MAE degrades as noise becomes more frame-varying (must reconstruct
different noise each frame). JEPA is robust because EMA target averages over
frame-varying noise. If the ranking inverts as a function of τ, that's causal
proof.

Noise model: AR(1) multiplicative Rayleigh speckle.
  - Generate iid Rayleigh noise N_t for each frame
  - Temporal smoothing: S_0 = N_0, S_t = α * S_{t-1} + (1-α) * N_t
  - α = exp(-1/τ) where τ is correlation time in frames
  - τ=∞ → α=1 → fully static (same noise every frame)
  - τ=0 → α=0 → fully iid (independent each frame)
  - clip_noised = clip * S_t (multiplicative)

Usage:
    python scripts/neurips/noise_autocorrelation_sweep.py \
        --test_csv data/csv/echonet_dynamic_test_local.csv \
        --models JEPA-IN21K-e100 \
        --device cuda:0 \
        --output scripts/neurips/samples/autocorr_JEPA_IN21K_e100.csv

    # All 3 primary models in parallel
    python scripts/neurips/noise_autocorrelation_sweep.py \
        --test_csv data/csv/echonet_dynamic_test_local.csv \
        --models JEPA-IN21K-e100 --device cuda:0 \
        --output scripts/neurips/samples/autocorr_JEPA_IN21K_e100.csv &
    python scripts/neurips/noise_autocorrelation_sweep.py \
        --test_csv data/csv/echonet_dynamic_test_local.csv \
        --models BYOL-L-e100 --device cuda:1 \
        --output scripts/neurips/samples/autocorr_BYOL_e100.csv &
    python scripts/neurips/noise_autocorrelation_sweep.py \
        --test_csv data/csv/echonet_dynamic_test_local.csv \
        --models MAE-L-e99 --device cuda:2 \
        --output scripts/neurips/samples/autocorr_MAE_e99.csv &
"""

import argparse
import math
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

# Import model configs
from scripts.neurips.frame_shuffle_severity import ALL_CONFIGS, PT50_END_CONFIGS

# Correlation times to sweep (in frames)
# inf = static (same noise every frame), 0 = iid (independent each frame)
TAU_VALUES = [float("inf"), 8.0, 4.0, 2.0, 1.0, 0.5, 0.0]

# Noise severity levels
NOISE_SEVERITIES = {
    "mild": 0.3,       # σ for Rayleigh scale parameter
    "moderate": 0.5,
    "severe": 0.7,
}


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


def generate_speckle_noise(T, H, W, tau, severity, rng):
    """
    Generate temporally-correlated multiplicative speckle noise.

    Args:
        T: number of frames
        H, W: spatial dimensions
        tau: temporal correlation time (frames). inf=static, 0=iid.
        severity: Rayleigh scale parameter (higher = more noise)
        rng: np.random.RandomState

    Returns:
        [T, H, W] noise tensor, multiplicative (centered around 1.0)
    """
    norm_factor = np.float32(severity * np.sqrt(np.pi / 2))  # keep everything float32

    if tau == float("inf"):
        # Static: same noise for all frames
        noise_frame = rng.rayleigh(scale=severity, size=(H, W)).astype(np.float32)
        noise_frame = noise_frame / norm_factor  # normalize mean to ~1.0
        noise = np.stack([noise_frame] * T, axis=0)
    elif tau == 0.0:
        # Fully iid: independent noise each frame
        noise = rng.rayleigh(scale=severity, size=(T, H, W)).astype(np.float32)
        noise = noise / norm_factor
    else:
        # AR(1) process: S_t = alpha * S_{t-1} + (1-alpha) * N_t
        alpha = np.float32(math.exp(-1.0 / tau))
        one_minus_alpha = np.float32(1.0) - alpha
        noise = np.zeros((T, H, W), dtype=np.float32)
        # First frame
        n0 = rng.rayleigh(scale=severity, size=(H, W)).astype(np.float32) / norm_factor
        noise[0] = n0
        # Subsequent frames with temporal smoothing
        for t in range(1, T):
            nt = rng.rayleigh(scale=severity, size=(H, W)).astype(np.float32) / norm_factor
            noise[t] = alpha * noise[t - 1] + one_minus_alpha * nt
    return torch.from_numpy(noise)


def apply_speckle_noise(clip, tau, severity, rng):
    """
    Apply temporally-correlated multiplicative speckle to a clip.

    Args:
        clip: [C, T, H, W] float32 in [0, 1]
        tau: correlation time in frames
        severity: noise strength
        rng: np.random.RandomState

    Returns:
        [C, T, H, W] noised clip, clamped to [0, 1]
    """
    C, T, H, W = clip.shape
    noise = generate_speckle_noise(T, H, W, tau, severity, rng)  # [T, H, W]
    noise = noise.unsqueeze(0)  # [1, T, H, W] — broadcast over channels
    noised = clip * noise
    return torch.clamp(noised, 0.0, 1.0)


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
def run_inference(encoder, probe, paths, labels, device, is_videomae,
                  target_mean, target_std,
                  resolution=224, frames=16, frame_step=2,
                  tau=None, severity=0.0, noise_rng=None):
    """Run inference with optional temporally-correlated noise."""
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

        # Apply noise if specified
        if tau is not None and severity > 0:
            clip = apply_speckle_noise(clip, tau, severity, noise_rng)

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
    parser = argparse.ArgumentParser(description="Noise autocorrelation sweep")
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--taus", nargs="*", type=float, default=None,
                        help=f"Correlation times (default: {TAU_VALUES})")
    parser.add_argument("--severity", default="moderate",
                        help=f"Noise severity (default: moderate). Options: {list(NOISE_SEVERITIES.keys())} or float")
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
    taus = args.taus if args.taus else TAU_VALUES

    # Parse severity
    try:
        severity = float(args.severity)
    except ValueError:
        severity = NOISE_SEVERITIES[args.severity]

    combined = {**PT50_END_CONFIGS, **ALL_CONFIGS}
    if args.models:
        configs = {k: combined[k] for k in args.models if k in combined}
    else:
        # Default to primary comparison
        configs = {k: ALL_CONFIGS[k] for k in ["JEPA-IN21K-e100", "BYOL-L-e100", "MAE-L-e99"]
                   if k in ALL_CONFIGS}

    paths, labels = read_csv(args.test_csv)
    if args.n_videos and args.n_videos < len(paths):
        indices = random.sample(range(len(paths)), args.n_videos)
        paths = [paths[i] for i in indices]
        labels = [labels[i] for i in indices]
    print(f"Test set: {len(paths)} videos")
    print(f"Tau values: {taus}")
    print(f"Severity: {severity}")
    print(f"Seeds: {args.n_seeds}")

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

        # --- Clean baseline (no noise) ---
        print(f"\n  [clean — no noise]")
        preds, labs, skipped = run_inference(
            encoder, probe, paths, labels, device, is_videomae,
            target_mean, target_std, args.resolution, args.frames, args.frame_step,
        )
        clean_metrics = compute_metrics(preds, labs)
        print(f"    R²={clean_metrics['R2']:.4f}  Pearson={clean_metrics['Pearson']:.4f}  "
              f"MAE={clean_metrics['MAE']:.3f}")
        all_results.append({
            "model": model_name, "tau": "clean", "seed": 0, "severity": 0,
            **{k: f"{v:.6f}" for k, v in clean_metrics.items()},
        })

        # --- Sweep tau values ---
        for tau in taus:
            tau_label = "inf" if tau == float("inf") else f"{tau:.1f}"
            seed_metrics = []
            for seed_idx in range(args.n_seeds):
                noise_seed = seed_idx + 200
                rng = np.random.RandomState(noise_seed)
                print(f"\n  [tau={tau_label}, severity={severity}, seed={noise_seed}]")

                preds, labs, skipped = run_inference(
                    encoder, probe, paths, labels, device, is_videomae,
                    target_mean, target_std, args.resolution, args.frames, args.frame_step,
                    tau=tau, severity=severity, noise_rng=rng,
                )
                metrics = compute_metrics(preds, labs)
                seed_metrics.append(metrics)
                print(f"    R²={metrics['R2']:.4f}  Pearson={metrics['Pearson']:.4f}  "
                      f"MAE={metrics['MAE']:.3f}")
                all_results.append({
                    "model": model_name, "tau": tau_label, "seed": noise_seed,
                    "severity": severity,
                    **{k: f"{v:.6f}" for k, v in metrics.items()},
                })

            # Summary for this tau
            mean_r2 = np.mean([m["R2"] for m in seed_metrics])
            std_r2 = np.std([m["R2"] for m in seed_metrics])
            drop = (clean_metrics["R2"] - mean_r2) / abs(clean_metrics["R2"]) * 100 if clean_metrics["R2"] != 0 else 0
            print(f"  → tau={tau_label}: R²={mean_r2:.4f}±{std_r2:.4f} ({drop:+.1f}% from clean)")

        del encoder, probe
        torch.cuda.empty_cache()

    # --- Save CSV ---
    if args.output is None:
        args.output = "scripts/neurips/samples/noise_autocorrelation_results.csv"
    with open(args.output, "w") as f:
        f.write("model,tau,seed,severity,R2,Pearson,MAE\n")
        for r in all_results:
            f.write(f"{r['model']},{r['tau']},{r['seed']},{r['severity']},"
                    f"{r['R2']},{r['Pearson']},{r['MAE']}\n")
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
