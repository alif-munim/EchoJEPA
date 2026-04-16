"""
Temporal consistency: frame-to-frame embedding similarity.

If JEPA filters frame-specific stochastic noise, its representations of
adjacent frames (same anatomy, different speckle realization) should be
more similar than MAE's. This directly tests the temporal averaging hypothesis
without conflating anatomy with noise.

For each video:
  1. Extract per-frame embeddings (mean-pooled spatial tokens per temporal position)
  2. Compute cosine similarity between consecutive frame embeddings
  3. Report mean ± std across all frame pairs and videos

Higher consistency = model discards frame-specific noise (speckle).
Lower consistency = model retains frame-specific variation.

Usage:
    python scripts/neurips/temporal_consistency.py \
        --models JEPA-IN21K-e100 --device cuda:0

    # All 3 in parallel
    python scripts/neurips/temporal_consistency.py \
        --models JEPA-IN21K-e100 --device cuda:0 &
    python scripts/neurips/temporal_consistency.py \
        --models BYOL-L-e100 --device cuda:1 &
    python scripts/neurips/temporal_consistency.py \
        --models MAE-L-e99 --device cuda:2 &
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu

sys.path.insert(0, ".")
from scripts.neurips.model_registry import add_model_args, get_models

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])


def load_encoder(cfg, device, resolution=224, frames=16):
    """Load frozen encoder. Returns (model, is_videomae, token_hook_storage).
    For VideoMAE, registers a hook to capture pre-pooled tokens since
    forward_features() returns mean-pooled output."""
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

        # Hook to capture pre-pooled tokens from norm layer
        # VideoMAE forward_features does: x = self.norm(x); return self.fc_norm(x.mean(1))
        # We want x after norm but before mean-pooling
        def capture_tokens(module, input, output):
            token_storage["tokens"] = output.detach()
        model.norm.register_forward_hook(capture_tokens)

        return model, True, token_storage
    else:
        raise ValueError(f"Unsupported: {model_type}")


@torch.no_grad()
def compute_temporal_consistency(model, video_path, device, is_videomae,
                                  token_storage, resolution=224, num_frames=16):
    """
    Compute frame-to-frame cosine similarity of embeddings.

    For ViT with tubelet_size=2, tokens are grouped into T/2 temporal positions.
    We compute mean-pooled embedding per temporal position, then cosine similarity
    between consecutive positions.

    Returns: list of cosine similarities [cos(t, t+1) for t in range(T_tokens-1)]
    """
    try:
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
    clip = clip.unsqueeze(0).to(device)

    if is_videomae:
        # forward_features returns mean-pooled [1, D], but our hook captures
        # pre-pooled tokens [1, N, D] from the norm layer
        token_storage.clear()
        model.forward_features(clip)
        if "tokens" not in token_storage:
            return None
        tokens = token_storage["tokens"].squeeze(0)  # [N_total, D]
    else:
        out = model(clip)  # [1, N, D]
        tokens = out.squeeze(0)  # [N_total, D]

    # Reshape to temporal positions
    n_h = resolution // 16
    n_w = resolution // 16
    n_t = num_frames // 2  # tubelet_size=2
    n_spatial = n_h * n_w
    n_total = tokens.shape[0]

    if n_total != n_t * n_spatial:
        if n_total % n_spatial == 0:
            n_t = n_total // n_spatial
        else:
            return None

    # [T_tokens, H*W, D]
    tokens = tokens.view(n_t, n_spatial, -1)

    # Mean-pool spatial tokens per temporal position → [T_tokens, D]
    temporal_embeddings = tokens.mean(dim=1)

    # Cosine similarity between consecutive temporal positions
    similarities = []
    for t in range(n_t - 1):
        cos_sim = F.cosine_similarity(
            temporal_embeddings[t].unsqueeze(0),
            temporal_embeddings[t + 1].unsqueeze(0)
        ).item()
        similarities.append(cos_sim)

    return similarities


def main():
    parser = argparse.ArgumentParser(description="Temporal consistency analysis")
    add_model_args(parser)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--csv", default="data/csv/echonet_dynamic_test_local.csv")
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--n_videos", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    models = get_models(args.models)

    import pandas as pd
    df = pd.read_csv(args.csv, header=None, delimiter=" ")
    video_paths = list(df[0])
    valid = [p for p in video_paths if os.path.exists(p)]
    if args.n_videos and args.n_videos < len(valid):
        import random
        random.seed(42)
        valid = random.sample(valid, args.n_videos)
    print(f"Videos: {len(valid)}")

    all_results = []

    for model_name, cfg in models.items():
        print(f"\n{'=' * 60}")
        print(f"Model: {model_name}")
        print(f"{'=' * 60}")

        model, is_videomae, token_storage = load_encoder(cfg, device, args.resolution, args.num_frames)

        all_sims = []
        n_skipped = 0

        for i, path in enumerate(valid):
            sims = compute_temporal_consistency(
                model, path, device, is_videomae, token_storage,
                args.resolution, args.num_frames
            )
            if sims is None:
                n_skipped += 1
                continue
            all_sims.extend(sims)

            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(valid)}")

        all_sims = np.array(all_sims)
        mean_sim = float(all_sims.mean())
        std_sim = float(all_sims.std())
        median_sim = float(np.median(all_sims))

        print(f"\n  Frame-to-frame cosine similarity:")
        print(f"    Mean:   {mean_sim:.6f} ± {std_sim:.6f}")
        print(f"    Median: {median_sim:.6f}")
        print(f"    Min:    {all_sims.min():.6f}")
        print(f"    Max:    {all_sims.max():.6f}")
        print(f"    Videos: {len(valid) - n_skipped} (skipped {n_skipped})")
        print(f"    Pairs:  {len(all_sims)}")

        all_results.append({
            "model": model_name,
            "mean_cosine_sim": mean_sim,
            "std_cosine_sim": std_sim,
            "median_cosine_sim": median_sim,
            "n_pairs": len(all_sims),
        })

        del model
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'=' * 60}")
    print("TEMPORAL CONSISTENCY SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Model':<25} {'Mean Cosine Sim':>18} {'Std':>10}")
    print("-" * 55)
    for r in all_results:
        print(f"{r['model']:<25} {r['mean_cosine_sim']:>18.6f} {r['std_cosine_sim']:>10.6f}")

    # Save
    if args.output is None:
        args.output = "scripts/neurips/samples/temporal_consistency.csv"
    with open(args.output, "w") as f:
        f.write("model,mean_cosine_sim,std_cosine_sim,median_cosine_sim,n_pairs\n")
        for r in all_results:
            f.write(f"{r['model']},{r['mean_cosine_sim']:.8f},{r['std_cosine_sim']:.8f},"
                    f"{r['median_cosine_sim']:.8f},{r['n_pairs']}\n")
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
