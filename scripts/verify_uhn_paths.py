#!/usr/bin/env python
"""
Verify UHN embedding-path alignment by re-encoding random clips through each model
and comparing cosine similarity with the stored embeddings.

For each model:
  - Sample N random clips
  - Load stored embedding from NPZ (via zipfile random-access for huge files)
  - Download video from S3, run through model, get fresh embedding
  - Compute cosine similarity: match (same clip) vs mismatch (random different clip)

A PASS requires match >> mismatch for all clips.
"""

import os
import sys
import tempfile

import numpy as np
import torch
import yaml

# Setup paths
BASE = "/mnt/custom-file-systems/efs/fs-0049217cdf69186d7_fsap-0fa7145b64eaa046b/vjepa2"
UHN = f"{BASE}/experiments/nature_medicine/uhn"
sys.path.insert(0, BASE)

from evals.video_classification_frozen.models import init_module
from evals.video_classification_frozen.utils import make_transforms
from src.datasets.video_dataset import VideoDataset

DEFAULT_NORMALIZATION = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


def cosine_similarity(a, b):
    a, b = a.flatten(), b.flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


class MmapNpzReader:
    """Memory-mapped reader for large uncompressed NPZ files.
    Opens once, reads individual rows without loading the full array."""

    def __init__(self, npz_path):
        self.npz = np.load(npz_path, mmap_mode="r", allow_pickle=True)

    def read_row(self, array_name, row_idx):
        return np.array(self.npz[array_name][row_idx])

    def close(self):
        del self.npz


def load_video_and_encode(model, s3_path, transform, frames_per_clip=16, frame_step=2, device="cuda:0"):
    """Download a single video from S3 and encode it through the model."""
    # Create a minimal VideoDataset to reuse S3 download + decord logic
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(f"{s3_path} 0\n")
        csv_path = f.name

    try:
        ds = VideoDataset(
            data_paths=[csv_path],
            transform=transform,
            frames_per_clip=frames_per_clip,
            frame_step=frame_step,
            num_clips=1,
            random_clip_sampling=True,
            allow_clip_overlap=True,
        )
        data = ds[0]
        if data is None:
            return None

        clips = data[0]  # list of clips, each clip is list of view tensors
        clip_indices = data[2]  # clip indices tensor

        # clips is [[view_tensor], ...] for num_clips clips, each view is [C, F, H, W]
        # Model expects [[batch_tensor_per_view], ...] per clip
        clips_batched = [[v.unsqueeze(0).to(device) for v in clip] for clip in clips]
        if not isinstance(clip_indices, list):
            clip_indices = [clip_indices]
        clip_indices_batched = [torch.as_tensor(ci).unsqueeze(0).to(device) for ci in clip_indices]

        with torch.no_grad():
            outputs = model(clips_batched, clip_indices_batched)
            # Mean pool over tokens
            pooled_segments = [o.mean(dim=1) for o in outputs]
            if len(pooled_segments) > 1:
                pooled = torch.stack(pooled_segments, dim=1).mean(dim=1)
            else:
                pooled = pooled_segments[0]

        return pooled.float().cpu().numpy().flatten()
    finally:
        os.unlink(csv_path)


MODELS = [
    {
        "name": "EchoJEPA-G",
        "config": f"{BASE}/configs/inference/vitg-384/extract_uhn.yaml",
        "npz": f"{UHN}/echojepa_g_embeddings/clip_embeddings.npz",
        "n_clips": 18111232,
    },
    {
        "name": "EchoJEPA-L",
        "config": f"{BASE}/configs/inference/vitl/extract_uhn.yaml",
        "npz": f"{UHN}/echojepa_l_embeddings/clip_embeddings.npz",
        "n_clips": 18110464,
    },
    {
        "name": "EchoJEPA-L-K",
        "config": f"{BASE}/configs/inference/vitl/extract_uhn_kinetics.yaml",
        "npz": f"{UHN}/echojepa_l_kinetics_embeddings/clip_embeddings.npz",
        "n_clips": 18111416,
    },
    {
        "name": "EchoMAE",
        "config": f"{BASE}/configs/inference/vitl/extract_uhn_echomae.yaml",
        "npz": f"{UHN}/echomae_embeddings/clip_embeddings.npz",
        "n_clips": 18111416,
    },
]


def verify_model(model_info, n_samples=3, device="cuda:0"):
    print(f"\n{'='*70}")
    print(f"  {model_info['name']}")
    print(f"{'='*70}")

    # Load config
    with open(model_info["config"]) as f:
        config = yaml.safe_load(f)

    model_kwargs = config["model_kwargs"]
    data_cfg = config["experiment"]["data"]

    # Initialize model
    print(f"  Loading model...")
    use_bf16 = not model_kwargs.get("wrapper_kwargs", {}).get("force_fp32", False)

    encoder = init_module(
        module_name=model_kwargs["module_name"],
        frames_per_clip=data_cfg.get("frames_per_clip", 16),
        resolution=data_cfg.get("resolution", 224),
        checkpoint=model_kwargs.get("checkpoint"),
        model_kwargs=model_kwargs.get("pretrain_kwargs", {}),
        wrapper_kwargs=model_kwargs.get("wrapper_kwargs", {}),
        device=device,
    )
    encoder.eval()
    if use_bf16:
        encoder = encoder.to(dtype=torch.bfloat16)

    autocast_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else torch.amp.autocast("cuda", enabled=False)

    # Build transform (same as extraction)
    transform = make_transforms(
        training=False,
        num_views_per_clip=1,
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(0.75, 4 / 3),
        random_resize_scale=(0.08, 1.0),
        reprob=0.25,
        auto_augment=True,
        motion_shift=False,
        crop_size=data_cfg.get("resolution", 224),
        normalize=data_cfg.get("normalization", DEFAULT_NORMALIZATION),
    )

    npz_path = model_info["npz"]
    n_clips = model_info["n_clips"]

    # Sample random clip indices (avoid padding at the end)
    rng = np.random.RandomState(42)
    max_real = min(n_clips, 18111412)  # don't pick padding rows
    sample_indices = rng.choice(max_real, size=n_samples, replace=False)
    mismatch_indices = rng.choice(max_real, size=n_samples, replace=False)
    # Make sure mismatch indices are different from sample indices
    while any(s == m for s, m in zip(sample_indices, mismatch_indices)):
        mismatch_indices = rng.choice(max_real, size=n_samples, replace=False)

    # Load paths
    print(f"  Reading paths from NPZ...")
    clip_index = np.load(f"{UHN}/uhn_clip_index.npz", allow_pickle=True)
    s3_paths = clip_index["s3_paths"]

    # Open mmap reader for embeddings
    print(f"  Opening mmap reader for {npz_path}...")
    reader = MmapNpzReader(npz_path)

    match_cosines = []
    mismatch_cosines = []

    for i, (idx, mismatch_idx) in enumerate(zip(sample_indices, mismatch_indices)):
        path = s3_paths[idx]
        print(f"\n  Clip {i+1}: index={idx}")
        print(f"    Path: ...{path[-80:]}")

        # Read stored embedding
        print(f"    Reading stored embedding...")
        stored_emb = reader.read_row("embeddings", idx)
        mismatch_emb = reader.read_row("embeddings", mismatch_idx)

        # Encode fresh
        print(f"    Downloading & encoding video...")
        with autocast_ctx:
            fresh_emb = load_video_and_encode(
                encoder,
                path,
                transform,
                frames_per_clip=data_cfg.get("frames_per_clip", 16),
                frame_step=data_cfg.get("frame_step", 2),
                device=device,
            )

        if fresh_emb is None:
            print(f"    SKIP - failed to load video")
            continue

        match_cos = cosine_similarity(stored_emb, fresh_emb)
        mismatch_cos = cosine_similarity(mismatch_emb, fresh_emb)
        match_cosines.append(match_cos)
        mismatch_cosines.append(mismatch_cos)

        status = "PASS" if match_cos > mismatch_cos else "FAIL"
        print(f"    Match cosine:    {match_cos:.4f}")
        print(f"    Mismatch cosine: {mismatch_cos:.4f}")
        print(f"    Gap:             {match_cos - mismatch_cos:.4f}  [{status}]")

    # Cleanup
    reader.close()
    del encoder
    torch.cuda.empty_cache()

    # Summary
    if match_cosines:
        mean_match = np.mean(match_cosines)
        mean_mismatch = np.mean(mismatch_cosines)
        gap = mean_match - mean_mismatch
        verdict = "PASS" if all(m > mm for m, mm in zip(match_cosines, mismatch_cosines)) else "FAIL"
        print(f"\n  --- {model_info['name']} Summary ---")
        print(f"  Mean match:    {mean_match:.4f}")
        print(f"  Mean mismatch: {mean_mismatch:.4f}")
        print(f"  Mean gap:      {gap:.4f}")
        print(f"  Verdict:       {verdict}")
        return {"name": model_info["name"], "match": mean_match, "mismatch": mean_mismatch, "gap": gap, "verdict": verdict}

    return None


if __name__ == "__main__":
    device = "cuda:0"
    n_samples = 3

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False

    results = []
    for model_info in MODELS:
        result = verify_model(model_info, n_samples=n_samples, device=device)
        if result:
            results.append(result)
        # Free GPU before next model
        torch.cuda.empty_cache()

    # Final table
    print(f"\n{'='*70}")
    print(f"  FINAL RESULTS")
    print(f"{'='*70}")
    print(f"  {'Model':<20} {'Match':>8} {'Mismatch':>10} {'Gap':>8} {'Verdict':>8}")
    print(f"  {'-'*56}")
    for r in results:
        print(f"  {r['name']:<20} {r['match']:>8.4f} {r['mismatch']:>10.4f} {r['gap']:>8.4f} {r['verdict']:>8}")
