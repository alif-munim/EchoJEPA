"""
CKA layer-wise feature extraction — single model, single GPU, with sharding.

Loads one encoder, registers forward hooks on every ViT block, runs a pass
over its assigned slice of the EchoNet-Dynamic test set, and caches
mean-pooled features at every layer as a [L, N_shard, D] tensor in NPZ.

Sharding: --shard_id (0..num_shards-1) selects every Nth video starting
at shard_id. Set --num_shards=1 to process everything on one GPU.

Across 2×8-GPU nodes: 4 models × 4 shards each = 16 GPUs → ~5 min/model.
After all shards complete, run cka_merge_shards.py to concatenate.

Usage:
    # Single GPU (no sharding)
    python scripts/neurips/cka_extract.py --model JEPA-IN21K-e100 --device cuda:0

    # 4-way shard, GPU 0 gets shard 0 of 4
    python scripts/neurips/cka_extract.py --model JEPA-IN21K-e100 \
        --device cuda:0 --shard_id 0 --num_shards 4

Output:
    scripts/neurips/cka/features/{MODEL}.shard{id}of{N}.npz   (sharded)
    scripts/neurips/cka/features/{MODEL}.npz                  (num_shards=1)
"""

import argparse
import io
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from decord import VideoReader, cpu

sys.path.insert(0, ".")
from scripts.neurips.model_registry import ALL_MODELS  # noqa: E402

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])

# Canonical e100 checkpoint paths (override registry where stale).
CANONICAL_OVERRIDES = {
    "JEPA-IN21K-e100": "checkpoints/jepa_in21k_vitl_e100.pt",
    "SALT-S2v1-e79": "checkpoints/salt_s2v1_e79.pt",
}


def load_encoder_all_layers(cfg, device, n_layers, resolution=224, frames=16):
    """Load encoder and register forward hooks on every ViT block."""
    import src.models.vision_transformer as vit

    model_type = cfg.get("type", "vjepa")
    layer_outputs = {}

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
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"  [load] missing={len(missing)} unexpected={len(unexpected)}")
        model.eval().to(device)

        for layer_idx in range(n_layers):
            def make_hook(idx):
                def hook(_module, _input, output):
                    # Unparametric LayerNorm over last dim — symmetric across
                    # all models (SALT's norm.weight is missing; using a fresh
                    # weight-free LN keeps the comparison fair).
                    D_ = output.shape[-1]
                    layer_outputs[idx] = torch.nn.functional.layer_norm(
                        output, (D_,)).detach()
                return hook
            model.blocks[layer_idx].register_forward_hook(make_hook(layer_idx))

        return model, layer_outputs, False

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
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"  [load] missing={len(missing)} unexpected={len(unexpected)}")
        model.eval().to(device)

        for layer_idx in range(n_layers):
            def make_hook(idx):
                def hook(_module, _input, output):
                    # Same unparametric LayerNorm as for the JEPA/BYOL/SALT path.
                    D_ = output.shape[-1]
                    layer_outputs[idx] = torch.nn.functional.layer_norm(
                        output, (D_,)).detach()
                return hook
            model.blocks[layer_idx].register_forward_hook(make_hook(layer_idx))

        return model, layer_outputs, True

    else:
        raise ValueError(f"Unsupported model type: {model_type}")


_S3_CLIENT = None


def _get_s3_client():
    global _S3_CLIENT
    if _S3_CLIENT is None:
        import boto3
        _S3_CLIENT = boto3.client("s3")
    return _S3_CLIENT


def _open_videoreader(path):
    """Return a decord.VideoReader for a local path or s3:// URI, or None."""
    try:
        if isinstance(path, str) and path.startswith("s3://"):
            bucket, key = path[len("s3://"):].split("/", 1)
            obj = _get_s3_client().get_object(Bucket=bucket, Key=key)
            data = obj["Body"].read()
            if not data:
                return None
            return VideoReader(io.BytesIO(data), ctx=cpu(0))
        return VideoReader(path, ctx=cpu(0))
    except Exception:
        return None


def load_clip(video_path, resolution=224, num_frames=16):
    """Load one video as a normalized [1, C, T, H, W] tensor, or None on error."""
    vr = _open_videoreader(video_path)
    if vr is None:
        return None
    total = len(vr)
    step = max(1, total // num_frames)
    indices = list(range(0, min(total, step * num_frames), step))[:num_frames]
    while len(indices) < num_frames:
        indices.append(indices[-1])
    indices = indices[:num_frames]

    clip = vr.get_batch(indices).asnumpy()  # [T, H, W, C]
    clip = torch.from_numpy(clip).permute(3, 0, 1, 2).float() / 255.0
    clip = torch.nn.functional.interpolate(
        clip.unsqueeze(0), size=(num_frames, resolution, resolution), mode="trilinear"
    ).squeeze(0)
    clip = (clip - IMAGENET_MEAN.view(3, 1, 1, 1)) / IMAGENET_STD.view(3, 1, 1, 1)
    return clip.unsqueeze(0)  # [1, C, T, H, W]


@torch.no_grad()
def extract(model, layer_outputs, is_videomae, clip, device, n_layers,
            tokens_per_video=None, token_rng=None):
    """Run forward pass, return per-layer features.

    If tokens_per_video is None: mean-pool tokens → [L, D].
    Otherwise: subsample `tokens_per_video` tokens (same indices across layers
    for a given clip) → [L, tokens_per_video, D].
    """
    layer_outputs.clear()
    if is_videomae:
        model.forward_features(clip.to(device))
    else:
        model(clip.to(device))

    sample_emb = next(iter(layer_outputs.values()))  # [1, N_tok, D]
    N_tok = sample_emb.shape[1]
    D = sample_emb.shape[-1]

    if tokens_per_video is None:
        out = np.empty((n_layers, D), dtype=np.float32)
        for l in range(n_layers):
            out[l] = layer_outputs[l].mean(dim=1).squeeze(0).cpu().float().numpy()
        return out  # [L, D]

    # Shared token indices across layers so pairs of (layer_i, layer_j) see
    # the same underlying tokens — the CKA comparison we want.
    k = min(tokens_per_video, N_tok)
    if token_rng is None:
        token_rng = np.random.default_rng()
    idx = token_rng.choice(N_tok, size=k, replace=False)
    idx_t = torch.as_tensor(idx, device=sample_emb.device, dtype=torch.long)

    out = np.empty((n_layers, k, D), dtype=np.float32)
    for l in range(n_layers):
        emb = layer_outputs[l][0]  # [N_tok, D]
        out[l] = emb.index_select(0, idx_t).cpu().float().numpy()
    return out  # [L, K, D]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(ALL_MODELS.keys()))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--csv", default="data/csv/echonet_dynamic_test_local.csv")
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--n_layers", type=int, default=24)
    parser.add_argument("--output_dir", default="scripts/neurips/cka/features")
    parser.add_argument("--max_videos", type=int, default=None,
                        help="Cap on videos (for smoke tests)")
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--tokens_per_video", type=int, default=32,
                        help="Subsampled tokens per video; 0 = mean-pool (legacy mode)")
    parser.add_argument("--token_seed", type=int, default=0,
                        help="Seed for the token subsample. Must match across models.")
    args = parser.parse_args()

    assert 0 <= args.shard_id < args.num_shards, \
        f"shard_id={args.shard_id} out of range for num_shards={args.num_shards}"

    cfg = dict(ALL_MODELS[args.model])
    if args.model in CANONICAL_OVERRIDES:
        override = CANONICAL_OVERRIDES[args.model]
        print(f"[info] overriding checkpoint for {args.model}: {cfg['checkpoint']} → {override}")
        cfg["checkpoint"] = override

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.csv, header=None, delimiter=" ")
    # Keep a global index so shards can be merged back in CSV order. S3 URIs
    # pass through; local paths are existence-checked.
    def _keep(p):
        return isinstance(p, str) and (p.startswith("s3://") or os.path.exists(p))
    full = [(i, p) for i, p in enumerate(df[0]) if _keep(p)]
    if args.max_videos is not None:
        full = full[: args.max_videos]
    # Strided shard: indices args.shard_id, args.shard_id+num_shards, ...
    shard = full[args.shard_id :: args.num_shards]
    global_idx = np.array([gi for gi, _ in shard], dtype=np.int64)
    video_paths = [p for _, p in shard]
    print(f"[{args.model}] shard {args.shard_id}/{args.num_shards}  "
          f"videos: {len(video_paths)} (of {len(full)} total)  device: {args.device}")

    model, layer_outputs, is_videomae = load_encoder_all_layers(
        cfg, device, args.n_layers, args.resolution, args.num_frames
    )

    tokens_per_video = args.tokens_per_video if args.tokens_per_video > 0 else None

    # Probe first valid clip to learn D + token count
    first = None
    for p in video_paths:
        first = load_clip(p, args.resolution, args.num_frames)
        if first is not None:
            break
    if first is None:
        raise RuntimeError("No readable videos in CSV")

    # Per-video RNG: seeded identically across models so token subsample is
    # aligned (same tokens picked from the same clip for JEPA/BYOL/MAE/SALT).
    def _rng_for(global_index):
        return np.random.default_rng([args.token_seed, int(global_index)])

    first_rng = _rng_for(global_idx[0]) if tokens_per_video else None
    feats_first = extract(model, layer_outputs, is_videomae, first, device,
                           args.n_layers, tokens_per_video, first_rng)
    if tokens_per_video:
        _, K, D = feats_first.shape
        print(f"[{args.model}] per-token mode: K={K} tokens/video, D={D}, layers={args.n_layers}")
    else:
        K = 1
        D = feats_first.shape[1]
        print(f"[{args.model}] mean-pool mode: D={D}, layers={args.n_layers}")

    N = len(video_paths)
    # Storage: [L, N, K, D] for per-token mode, or [L, N, D] for mean-pool.
    if tokens_per_video:
        all_feats = np.zeros((args.n_layers, N, K, D), dtype=np.float16)
    else:
        all_feats = np.zeros((args.n_layers, N, D), dtype=np.float16)
    valid_idx = []

    t0 = time.time()
    for i, path in enumerate(video_paths):
        clip = load_clip(path, args.resolution, args.num_frames)
        if clip is None:
            continue
        rng_i = _rng_for(global_idx[i]) if tokens_per_video else None
        feats = extract(model, layer_outputs, is_videomae, clip, device,
                        args.n_layers, tokens_per_video, rng_i)
        if tokens_per_video:
            all_feats[:, i, :, :] = feats.astype(np.float16)
        else:
            all_feats[:, i, :] = feats.astype(np.float16)
        valid_idx.append(i)
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (N - i - 1) / rate
            print(f"[{args.model}] {i + 1}/{N}  {rate:.2f} vid/s  ETA {eta / 60:.1f} min")

    valid_local = np.array(valid_idx, dtype=np.int64)
    if tokens_per_video:
        all_feats = all_feats[:, valid_local, :, :]
    else:
        all_feats = all_feats[:, valid_local, :]
    # Map local shard indices → global CSV indices so merge preserves order.
    valid_global = global_idx[valid_local]

    if args.num_shards == 1:
        out_path = os.path.join(args.output_dir, f"{args.model}.npz")
    else:
        out_path = os.path.join(
            args.output_dir,
            f"{args.model}.shard{args.shard_id}of{args.num_shards}.npz",
        )
    np.savez_compressed(
        out_path,
        features=all_feats,
        model=args.model,
        valid_idx=valid_global,
        csv_path=args.csv,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
    )
    dt = time.time() - t0
    print(f"[{args.model}] wrote {out_path}  shape={all_feats.shape}  {dt / 60:.1f} min")


if __name__ == "__main__":
    main()
