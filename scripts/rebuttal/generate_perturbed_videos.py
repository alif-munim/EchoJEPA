"""
Generate perturbed video tensors with synthetic Rayleigh speckle noise.

Loads N validation videos, applies speckle at 5 intensity levels, and saves
as a tensor cache (.pt) for reuse by CKA and noise-level probe scripts.

Usage:
    python scripts/rebuttal/generate_perturbed_videos.py \
        --csv data/csv/uhn_views_22k_val.csv \
        --n_videos 200 \
        --output scripts/rebuttal/perturbed_cache.pt \
        --resolution 224 --frames 16 --frame_step 2
"""

import argparse
import random

import decord
import torch
import torchvision.transforms.functional as TF

decord.bridge.set_bridge("torch")

SIGMA_LEVELS = [0.05, 0.1, 0.2, 0.4, 0.8]


def load_clip(video_path, frames=16, frame_step=2, resolution=224):
    """Load a single clip from a video file."""
    vr = decord.VideoReader(video_path, num_threads=1)
    total = len(vr)
    needed = frames * frame_step
    start = max(0, (total - needed) // 2)
    indices = list(range(start, min(start + needed, total), frame_step))
    # Pad if too short
    while len(indices) < frames:
        indices.append(indices[-1])
    indices = indices[:frames]
    clip = vr.get_batch(indices)  # [T, H, W, C]
    clip = clip.permute(3, 0, 1, 2).float() / 255.0  # [C, T, H, W]
    clip = TF.resize(clip, [resolution, resolution], antialias=True)
    return clip


def apply_rayleigh_speckle(clip, sigma):
    """Multiply clip by Rayleigh noise: x_noisy = x * noise, noise ~ Rayleigh(sigma)."""
    noise = torch.zeros_like(clip)
    # Rayleigh = sqrt(X^2 + Y^2) where X, Y ~ N(0, sigma^2)
    x = torch.randn_like(clip) * sigma
    y = torch.randn_like(clip) * sigma
    noise = torch.sqrt(x**2 + y**2)
    return torch.clamp(clip * noise, 0.0, 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Validation CSV (space-delimited: path label)")
    parser.add_argument("--n_videos", type=int, default=200)
    parser.add_argument("--output", default="scripts/rebuttal/perturbed_cache.pt")
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

    # Generate perturbed versions
    perturbed = {}
    for sigma in SIGMA_LEVELS:
        print(f"Generating speckle sigma={sigma}...")
        noisy = torch.stack([apply_rayleigh_speckle(c, sigma) for c in clean_clips])
        perturbed[sigma] = noisy

    cache = {
        "clean": clean_tensor,
        "perturbed": perturbed,
        "sigma_levels": SIGMA_LEVELS,
        "paths": valid_paths,
    }
    torch.save(cache, args.output)
    print(f"Saved cache to {args.output} ({len(valid_paths)} videos, {len(SIGMA_LEVELS)} noise levels)")


if __name__ == "__main__":
    main()
