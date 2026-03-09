#!/usr/bin/env python
"""
Verify MIMIC embedding-path alignment for all 7 models.
Same protocol as verify_uhn_paths.py: re-encode random clips, compare cosine similarity.
"""

import os
import sys
import tempfile

import numpy as np
import torch
import yaml

BASE = "/mnt/custom-file-systems/efs/fs-0049217cdf69186d7_fsap-0fa7145b64eaa046b/vjepa2"
MIMIC = f"{BASE}/experiments/nature_medicine/mimic"
sys.path.insert(0, BASE)

from evals.video_classification_frozen.models import init_module
from evals.video_classification_frozen.utils import make_transforms
from src.datasets.video_dataset import VideoDataset

DEFAULT_NORMALIZATION = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


def cosine_similarity(a, b):
    a, b = a.flatten(), b.flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def load_video_and_encode(model, s3_path, transform, frames_per_clip=16, frame_step=2, device="cuda:0"):
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
        clips = data[0]
        clip_indices = data[2]
        clips_batched = [[v.unsqueeze(0).to(device) for v in clip] for clip in clips]
        if not isinstance(clip_indices, list):
            clip_indices = [clip_indices]
        clip_indices_batched = [torch.as_tensor(ci).unsqueeze(0).to(device) for ci in clip_indices]
        with torch.no_grad():
            outputs = model(clips_batched, clip_indices_batched)
            pooled_segments = [o.mean(dim=1) for o in outputs]
            if len(pooled_segments) > 1:
                pooled = torch.stack(pooled_segments, dim=1).mean(dim=1)
            else:
                pooled = pooled_segments[0]
        return pooled.float().cpu().numpy().flatten()
    finally:
        os.unlink(csv_path)


VIEW = f"{BASE}/configs/inference/vitg-384/view"
MODELS = [
    {"name": "EchoJEPA-G", "config": f"{VIEW}/echojepa_224px.yaml", "npz": f"{MIMIC}/echojepa_g_mimic_embeddings.npz"},
    {"name": "EchoJEPA-L", "config": f"{VIEW}/echojepa_large_224px.yaml", "npz": f"{MIMIC}/echojepa_l_mimic_embeddings.npz"},
    {"name": "EchoJEPA-L-K", "config": f"{VIEW}/echojepa_large_kinetics_224px.yaml", "npz": f"{MIMIC}/echojepa_l_kinetics_mimic_embeddings.npz"},
    {"name": "EchoMAE", "config": f"{VIEW}/videomae_224px.yaml", "npz": f"{MIMIC}/echomae_mimic_embeddings.npz"},
    {"name": "PanEcho", "config": f"{VIEW}/panecho_224px.yaml", "npz": f"{MIMIC}/panecho_mimic_embeddings.npz"},
    {"name": "EchoPrime", "config": f"{VIEW}/echoprime_224px.yaml", "npz": f"{MIMIC}/echoprime_mimic_embeddings.npz"},
    {"name": "EchoFM", "config": f"{VIEW}/echofm_224px.yaml", "npz": f"{MIMIC}/echofm_mimic_embeddings.npz"},
]


def verify_model(model_info, n_samples=3, device="cuda:0"):
    print(f"\n{'='*70}", flush=True)
    print(f"  {model_info['name']}", flush=True)
    print(f"{'='*70}", flush=True)

    with open(model_info["config"]) as f:
        config = yaml.safe_load(f)

    model_kwargs = config["model_kwargs"]
    data_cfg = config["experiment"]["data"]

    print(f"  Loading model...", flush=True)
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

    # Load MIMIC NPZ (small enough to load fully)
    print(f"  Loading embeddings NPZ...", flush=True)
    data = np.load(model_info["npz"], allow_pickle=True)
    embeddings = data["embeddings"]
    paths = data["paths"]
    n_clips = len(embeddings)
    print(f"  {n_clips} clips, embed_dim={embeddings.shape[1]}", flush=True)

    # Sample random clips
    rng = np.random.RandomState(42)
    # Filter out padding entries
    real_mask = np.array([not str(p).startswith("padding") for p in paths])
    real_indices = np.where(real_mask)[0]
    sample_indices = rng.choice(real_indices, size=n_samples, replace=False)
    mismatch_indices = rng.choice(real_indices, size=n_samples, replace=False)
    while any(s == m for s, m in zip(sample_indices, mismatch_indices)):
        mismatch_indices = rng.choice(real_indices, size=n_samples, replace=False)

    match_cosines = []
    mismatch_cosines = []

    for i, (idx, mismatch_idx) in enumerate(zip(sample_indices, mismatch_indices)):
        path = str(paths[idx])
        print(f"\n  Clip {i+1}: index={idx}", flush=True)
        print(f"    Path: ...{path[-80:]}", flush=True)

        stored_emb = embeddings[idx]
        mismatch_emb = embeddings[mismatch_idx]

        print(f"    Downloading & encoding video...", flush=True)
        with autocast_ctx:
            fresh_emb = load_video_and_encode(
                encoder, path, transform,
                frames_per_clip=data_cfg.get("frames_per_clip", 16),
                frame_step=data_cfg.get("frame_step", 2),
                device=device,
            )

        if fresh_emb is None:
            print(f"    SKIP - failed to load video", flush=True)
            continue

        match_cos = cosine_similarity(stored_emb, fresh_emb)
        mismatch_cos = cosine_similarity(mismatch_emb, fresh_emb)
        match_cosines.append(match_cos)
        mismatch_cosines.append(mismatch_cos)

        status = "PASS" if match_cos > mismatch_cos else "FAIL"
        print(f"    Match cosine:    {match_cos:.4f}", flush=True)
        print(f"    Mismatch cosine: {mismatch_cos:.4f}", flush=True)
        print(f"    Gap:             {match_cos - mismatch_cos:.4f}  [{status}]", flush=True)

    # Cleanup
    del data, embeddings, paths, encoder
    torch.cuda.empty_cache()

    if match_cosines:
        mean_match = np.mean(match_cosines)
        mean_mismatch = np.mean(mismatch_cosines)
        gap = mean_match - mean_mismatch
        verdict = "PASS" if all(m > mm for m, mm in zip(match_cosines, mismatch_cosines)) else "FAIL"
        print(f"\n  --- {model_info['name']} Summary ---", flush=True)
        print(f"  Mean match:    {mean_match:.4f}", flush=True)
        print(f"  Mean mismatch: {mean_mismatch:.4f}", flush=True)
        print(f"  Mean gap:      {gap:.4f}", flush=True)
        print(f"  Verdict:       {verdict}", flush=True)
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
        torch.cuda.empty_cache()

    print(f"\n{'='*70}", flush=True)
    print(f"  FINAL RESULTS", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  {'Model':<20} {'Match':>8} {'Mismatch':>10} {'Gap':>8} {'Verdict':>8}", flush=True)
    print(f"  {'-'*56}", flush=True)
    for r in results:
        print(f"  {r['name']:<20} {r['match']:>8.4f} {r['mismatch']:>10.4f} {r['gap']:>8.4f} {r['verdict']:>8}", flush=True)
