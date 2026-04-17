"""
Generate perturbed video tensors with echo-specific perturbations (USAugment).

Loads N validation videos, applies three perturbation types (depth attenuation,
acoustic shadow, haze) at three severity levels each, and saves as a tensor
cache (.pt) for reuse by CKA and noise-level probe scripts.

Perturbations model real clinical degradation modes:
  - Depth attenuation: signal falls off with depth (obese patients, poor windows)
  - Acoustic shadow: localized signal dropout (ribs, calcification)
  - Haze artifact: reverberation haze (poor acoustic windows)

Usage:
    python scripts/neurips/generate_perturbed_videos.py \
        --csv data/csv/uhn_views_22k_val.csv \
        --n_videos 200 \
        --output scripts/neurips/perturbed_cache.pt \
        --resolution 224 --frames 16 --frame_step 2
"""

import argparse
import random
import sys

import decord
import torch
import torchvision.transforms.functional as TF

# Add repo root to path for imports
sys.path.insert(0, ".")
from scripts.neurips.echo_perturbations import (
    PERTURBATIONS,
    SEVERITY_LEVELS,
    apply_perturbation,
    create_scan_mask,
)

decord.bridge.set_bridge("torch")


def load_clip(video_path, frames=16, frame_step=2, resolution=224):
    """Load a single clip from a video file."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Validation CSV (space-delimited: path label)")
    parser.add_argument("--n_videos", type=int, default=200)
    parser.add_argument("--output", default="scripts/neurips/perturbed_cache.pt")
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--frame_step", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # Read CSV
    paths = []
    with open(args.csv) as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                paths.append(parts[0])

    # Sample N videos
    if len(paths) > args.n_videos:
        paths = random.sample(paths, args.n_videos)

    print(f"Loading {len(paths)} videos...")
    clean_clips = []
    valid_paths = []
    for i, p in enumerate(paths):
        try:
            clip = load_clip(p, args.frames, args.frame_step, args.resolution)
            clean_clips.append(clip)
            valid_paths.append(p)
        except Exception as e:
            print(f"  Skip {p}: {e}")
        if (i + 1) % 50 == 0:
            print(f"  Loaded {i + 1}/{len(paths)}")

    clean_tensor = torch.stack(clean_clips)  # [N, C, T, H, W]
    print(f"Clean tensor: {clean_tensor.shape}")

    # Pre-compute scan masks (one per video, from first frame)
    scan_masks = [create_scan_mask(c[:, 0, :, :]) for c in clean_clips]

    # Generate perturbed versions for each perturbation type × severity
    perturbed = {}
    perturbation_types = list(PERTURBATIONS.keys())

    for ptype in perturbation_types:
        perturbed[ptype] = {}
        for severity in SEVERITY_LEVELS:
            print(f"Generating {ptype} / {severity}...")
            noisy_clips = []
            for idx, (clip, mask) in enumerate(zip(clean_clips, scan_masks)):
                # Use video index as seed for reproducible shadow/haze position
                p_clip = apply_perturbation(clip, ptype, severity, scan_mask=mask, seed=idx)
                noisy_clips.append(p_clip)
            perturbed[ptype][severity] = torch.stack(noisy_clips)
            print(f"  {perturbed[ptype][severity].shape}")

    cache = {
        "clean": clean_tensor,
        "perturbed": perturbed,
        "perturbation_types": perturbation_types,
        "severity_levels": SEVERITY_LEVELS,
        "paths": valid_paths,
    }
    torch.save(cache, args.output)

    n_conditions = len(perturbation_types) * len(SEVERITY_LEVELS)
    print(
        f"Saved cache to {args.output} "
        f"({len(valid_paths)} videos, {len(perturbation_types)} perturbation types, "
        f"{len(SEVERITY_LEVELS)} severity levels, {n_conditions} conditions total)"
    )


if __name__ == "__main__":
    main()
