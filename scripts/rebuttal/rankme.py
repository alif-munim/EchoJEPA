"""
RankMe: effective dimensionality via spectral entropy of SVD.

Computes eff_dim = exp(-sum(sigma_bar * log(sigma_bar))) where sigma_bar
are normalized singular values of the [N_videos, D] mean-pooled embedding matrix.
Higher = more diverse representations. Lower = feature collapse.

Reference numbers (from Goodfire/e100 analysis):
  BYOL e100: 209, JEPA IN21K e100: 197, MAE e99: 63

This script adds SALT to test the dimensionality-collapse hypothesis:
  If SALT's frozen pixel-reconstruction teacher constrains the student,
  SALT's eff_dim should be closer to MAE (~63) than JEPA (~197).

Usage:
    python scripts/rebuttal/rankme.py \
        --models SALT-S2v1-e79 JEPA-IN21K-e100 MAE-L-e99 BYOL-L-e100 \
        --device cuda:0 --n_videos 500

    # Single model
    python scripts/rebuttal/rankme.py --models SALT-S2v1-e79 --device cuda:0
"""

import argparse
import io
import os
import sys

import boto3
import numpy as np
import torch
from botocore.config import Config as BotoConfig
from decord import VideoReader, cpu

sys.path.insert(0, ".")
from scripts.rebuttal.model_registry import add_model_args, get_models

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])

_s3_client = None


def _get_s3_client():
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client(
            "s3", config=BotoConfig(max_pool_connections=32,
                                    retries={"max_attempts": 3, "mode": "standard"}),
        )
    return _s3_client


def load_encoder(cfg, device, resolution=224, frames=16):
    """Load frozen encoder. Returns (model, is_videomae, token_hook_storage)."""
    import src.models.vision_transformer as vit

    model_type = cfg.get("type", "vjepa")
    token_storage = {}

    if model_type in ("vjepa", "byol"):
        model_name = cfg.get("model_name", "vit_large")
        model = vit.__dict__[model_name](
            img_size=resolution, num_frames=frames, patch_size=16, tubelet_size=2,
            uniform_power=True, use_rope=True,
        )
        ckpt = torch.load(cfg["checkpoint"], map_location="cpu", weights_only=False)
        key = cfg.get("checkpoint_key", "target_encoder")
        state = ckpt[key]
        state = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state.items()}
        model_sd = model.state_dict()
        for k in list(state.keys()):
            if k in model_sd and state[k].shape != model_sd[k].shape:
                del state[k]
        model.load_state_dict(state, strict=False)
        model.eval().to(device)
        return model, False, token_storage
    elif model_type == "videomae":
        from evals.video_classification_frozen.modelcustom.videomae_encoder import (
            _convert_pretrain_to_finetune_state_dict,
            _import_modeling_finetune,
        )
        mf = _import_modeling_finetune()
        model = mf.vit_large_patch16_224(
            img_size=resolution, all_frames=frames, tubelet_size=2, num_classes=1000,
        )
        ckpt = torch.load(cfg["checkpoint"], map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt)
        state = _convert_pretrain_to_finetune_state_dict(state, model.state_dict())
        model.load_state_dict(state, strict=False)
        model.eval().to(device)

        def capture_tokens(module, input, output):
            token_storage["tokens"] = output.detach()
        model.norm.register_forward_hook(capture_tokens)

        return model, True, token_storage
    else:
        raise ValueError(f"Unsupported: {model_type}")


def load_video(video_path, resolution=224, num_frames=16):
    """Load video as [1, C, T, H, W] tensor. Supports local and S3 paths."""
    try:
        if video_path.startswith("s3://"):
            s3 = _get_s3_client()
            bucket, key = video_path.replace("s3://", "").split("/", 1)
            obj = s3.get_object(Bucket=bucket, Key=key)
            data = obj["Body"].read()
            vr = VideoReader(io.BytesIO(data), num_threads=-1, ctx=cpu(0))
        else:
            vr = VideoReader(video_path, ctx=cpu(0))
    except Exception:
        return None

    total = len(vr)
    step = max(1, total // num_frames)
    indices = list(range(0, min(total, step * num_frames), step))[:num_frames]
    while len(indices) < num_frames:
        indices.append(indices[-1])
    indices = indices[:num_frames]

    clip = vr.get_batch(indices).asnumpy()
    clip = torch.from_numpy(clip).permute(3, 0, 1, 2).float() / 255.0  # [C, T, H, W]
    clip = torch.nn.functional.interpolate(
        clip.unsqueeze(0), size=(num_frames, resolution, resolution), mode="trilinear"
    ).squeeze(0)
    clip = (clip - IMAGENET_MEAN.view(3, 1, 1, 1)) / IMAGENET_STD.view(3, 1, 1, 1)
    return clip.unsqueeze(0)


@torch.no_grad()
def extract_embedding(model, clip, device, is_videomae, token_storage):
    """Extract mean-pooled embedding [D] from a single video clip."""
    clip = clip.to(device)

    if is_videomae:
        token_storage.clear()
        model.forward_features(clip)
        if "tokens" not in token_storage:
            return None
        tokens = token_storage["tokens"].squeeze(0)  # [N, D]
    else:
        tokens = model(clip).squeeze(0)  # [N, D]

    # Mean-pool all spatial-temporal tokens → [D]
    return tokens.mean(dim=0).cpu()


def compute_rankme(embeddings):
    """
    Compute RankMe (effective dimensionality) from embedding matrix.

    Args:
        embeddings: [N, D] numpy array

    Returns:
        eff_dim: exp(-sum(p * log(p))) where p = normalized singular values
    """
    # Center embeddings
    embeddings = embeddings - embeddings.mean(axis=0, keepdims=True)

    # SVD
    _, S, _ = np.linalg.svd(embeddings, full_matrices=False)

    # Normalize to probability distribution
    S_pos = S[S > 1e-10]  # remove near-zero
    p = S_pos / S_pos.sum()

    # Spectral entropy → effective dimensionality
    entropy = -np.sum(p * np.log(p))
    eff_dim = np.exp(entropy)

    return eff_dim, S


def main():
    parser = argparse.ArgumentParser(description="RankMe effective dimensionality")
    add_model_args(parser)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--csv", default="data/csv/echonet_dynamic_test_local.csv")
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--n_videos", type=int, default=500,
                        help="Number of videos to sample (500 is sufficient for stable RankMe)")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    models = get_models(args.models)

    import pandas as pd
    df = pd.read_csv(args.csv, header=None, delimiter=" ")
    video_paths = list(df[0])
    # S3 paths can't be checked with os.path.exists — trust them
    if video_paths and str(video_paths[0]).startswith("s3://"):
        valid = video_paths
        print(f"Found {len(valid)} S3 video paths")
    else:
        valid = [p for p in video_paths if os.path.exists(p)]
        print(f"Found {len(valid)}/{len(video_paths)} videos on disk")

    if args.n_videos and args.n_videos < len(valid):
        import random
        random.seed(42)
        valid = random.sample(valid, args.n_videos)
    print(f"Using {len(valid)} videos for RankMe computation")

    all_results = []

    for model_name, cfg in models.items():
        print(f"\n{'=' * 60}")
        print(f"Model: {model_name}")
        print(f"Checkpoint: {cfg['checkpoint']}")
        print(f"{'=' * 60}")

        model, is_videomae, token_storage = load_encoder(cfg, device, args.resolution, args.num_frames)

        embeddings = []
        n_skipped = 0

        for i, path in enumerate(valid):
            clip = load_video(path, args.resolution, args.num_frames)
            if clip is None:
                n_skipped += 1
                continue
            emb = extract_embedding(model, clip, device, is_videomae, token_storage)
            if emb is None:
                n_skipped += 1
                continue
            embeddings.append(emb.numpy())

            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(valid)} extracted")

        embeddings = np.stack(embeddings)  # [N, D]
        print(f"\n  Embedding matrix: {embeddings.shape}")

        eff_dim, singular_values = compute_rankme(embeddings)
        embed_dim = embeddings.shape[1]
        usage_pct = eff_dim / embed_dim * 100

        # Also compute participation ratio for comparison
        S2 = singular_values ** 2
        participation_ratio = (S2.sum() ** 2) / (S2 ** 2).sum()

        # Top-k singular value energy
        total_energy = (singular_values ** 2).sum()
        top10_energy = (singular_values[:10] ** 2).sum() / total_energy * 100
        top50_energy = (singular_values[:50] ** 2).sum() / total_energy * 100

        print(f"\n  RankMe (spectral entropy):  {eff_dim:.1f} / {embed_dim} ({usage_pct:.1f}%)")
        print(f"  Participation ratio:        {participation_ratio:.1f}")
        print(f"  Top-10 SV energy:           {top10_energy:.1f}%")
        print(f"  Top-50 SV energy:           {top50_energy:.1f}%")
        print(f"  Videos used:                {len(embeddings)} (skipped {n_skipped})")

        all_results.append({
            "model": model_name,
            "eff_dim_rankme": eff_dim,
            "participation_ratio": participation_ratio,
            "embed_dim": embed_dim,
            "usage_pct": usage_pct,
            "top10_energy_pct": top10_energy,
            "top50_energy_pct": top50_energy,
            "n_videos": len(embeddings),
        })

        del model
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'=' * 60}")
    print("EFFECTIVE DIMENSIONALITY SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Model':<25} {'RankMe':>8} {'PR':>8} {'Usage':>8} {'Top-10':>8} {'Top-50':>8}")
    print("-" * 67)
    for r in all_results:
        print(f"{r['model']:<25} {r['eff_dim_rankme']:>8.1f} {r['participation_ratio']:>8.1f} "
              f"{r['usage_pct']:>7.1f}% {r['top10_energy_pct']:>7.1f}% {r['top50_energy_pct']:>7.1f}%")

    # Save
    if args.output is None:
        os.makedirs("scripts/rebuttal/samples", exist_ok=True)
        args.output = "scripts/rebuttal/samples/rankme_salt.csv"
    with open(args.output, "w") as f:
        f.write("model,eff_dim_rankme,participation_ratio,embed_dim,usage_pct,"
                "top10_energy_pct,top50_energy_pct,n_videos\n")
        for r in all_results:
            f.write(f"{r['model']},{r['eff_dim_rankme']:.4f},{r['participation_ratio']:.4f},"
                    f"{r['embed_dim']},{r['usage_pct']:.4f},"
                    f"{r['top10_energy_pct']:.4f},{r['top50_energy_pct']:.4f},{r['n_videos']}\n")
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
