"""
Token-level speckle probing: do individual tokens encode local noise?

Instead of mean-pooling all tokens → predict whole-clip speckle,
predict per-patch speckle energy from each spatial token individually.

Computes speckle energy for each 16×16 image patch, then trains
a Ridge probe: token embedding → patch speckle energy.

Usage:
    python scripts/rebuttal/token_speckle_probing.py \
        --models JEPA-IN21K-e100 --device cuda:0
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from decord import VideoReader, cpu
from scipy.ndimage import laplace
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, ".")
from scripts.rebuttal.model_registry import add_model_args, get_models

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])


def compute_patch_speckle(video_path, num_frames=16, resolution=224, patch_size=16):
    """Compute per-patch speckle energy for each spatial patch."""
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

    clip = vr.get_batch(indices).asnumpy()  # [T, H, W, C]
    # Resize to target resolution
    from PIL import Image
    resized = []
    for t in range(clip.shape[0]):
        img = Image.fromarray(clip[t]).resize((resolution, resolution))
        resized.append(np.array(img))
    clip = np.stack(resized)  # [T, H, W, C]

    gray = np.mean(clip.astype(np.float32) / 255.0, axis=-1)  # [T, H, W]

    n_patches_h = resolution // patch_size
    n_patches_w = resolution // patch_size

    # Compute per-patch speckle energy (averaged over frames)
    patch_speckle = np.zeros((n_patches_h, n_patches_w), dtype=np.float32)
    for ph in range(n_patches_h):
        for pw in range(n_patches_w):
            h_start = ph * patch_size
            w_start = pw * patch_size
            patch_energies = []
            for t in range(gray.shape[0]):
                patch = gray[t, h_start:h_start + patch_size, w_start:w_start + patch_size]
                lap = laplace(patch)
                patch_energies.append(np.mean(lap ** 2))
            patch_speckle[ph, pw] = np.mean(patch_energies)

    return patch_speckle.flatten()  # [n_patches_h * n_patches_w]


def load_encoder(cfg, device, resolution=224, frames=16):
    """Load frozen encoder."""
    import src.models.vision_transformer as vit

    model_type = cfg.get("type", "vjepa")
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
        return model, False
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

        # Hook to capture pre-pooled tokens (forward_features returns mean-pooled)
        model._token_storage = {}
        def capture_tokens(module, input, output):
            model._token_storage["tokens"] = output.detach()
        model.norm.register_forward_hook(capture_tokens)

        return model, True
    else:
        raise ValueError(f"Unsupported: {model_type}")


@torch.no_grad()
def extract_spatial_tokens(model, video_path, device, is_videomae,
                           resolution=224, num_frames=16):
    """Extract spatial token embeddings, mean-pooled over temporal dimension.
    Returns [N_spatial, D] where N_spatial = (H/patch)*(W/patch)."""
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
    clip = torch.from_numpy(clip).permute(3, 0, 1, 2).float() / 255.0
    clip = torch.nn.functional.interpolate(
        clip.unsqueeze(0), size=(num_frames, resolution, resolution), mode="trilinear"
    ).squeeze(0)
    clip = (clip - IMAGENET_MEAN.view(3, 1, 1, 1)) / IMAGENET_STD.view(3, 1, 1, 1)
    clip = clip.unsqueeze(0).to(device)

    if is_videomae:
        model._token_storage.clear()
        model.forward_features(clip)
        if "tokens" not in model._token_storage:
            return None
        tokens = model._token_storage["tokens"].squeeze(0)  # [N_total, D]
    else:
        out = model(clip)  # [1, N, D]
        tokens = out.squeeze(0)  # [N_total, D]

    # Reshape to [T_tokens, H_patches, W_patches, D] and mean over temporal
    n_h = resolution // 16
    n_w = resolution // 16
    n_t = num_frames // 2  # tubelet_size=2
    n_total = tokens.shape[0]

    if n_total == n_t * n_h * n_w:
        tokens = tokens.view(n_t, n_h, n_w, -1)  # [T, H, W, D]
        spatial_tokens = tokens.mean(dim=0)  # [H, W, D] — mean over temporal
        spatial_tokens = spatial_tokens.view(n_h * n_w, -1)  # [N_spatial, D]
    else:
        # Fallback: just reshape as best we can
        spatial_tokens = tokens[:n_h * n_w]  # take first spatial slice

    return spatial_tokens.cpu().numpy()  # [N_spatial, D]


def main():
    parser = argparse.ArgumentParser(description="Token-level speckle probing")
    add_model_args(parser)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--csv", default="data/csv/echonet_dynamic_test_local.csv")
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument("--n_videos", type=int, default=500, help="Subsample for speed")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    models = get_models(args.models)

    df = pd.read_csv(args.csv, header=None, delimiter=" ")
    video_paths = list(df[0])
    valid = [p for p in video_paths if os.path.exists(p)]
    if args.n_videos and args.n_videos < len(valid):
        import random
        random.seed(42)
        valid = random.sample(valid, args.n_videos)
    print(f"Videos: {len(valid)}")

    n_patches = (args.resolution // 16) ** 2  # 14*14 = 196

    for model_name, cfg in models.items():
        print(f"\n{'=' * 60}")
        print(f"Model: {model_name}")
        print(f"{'=' * 60}")

        model, is_videomae = load_encoder(cfg, device, args.resolution, args.num_frames)

        # Collect token embeddings and patch speckle for all videos
        all_token_embs = []  # list of [N_spatial, D]
        all_patch_speckle = []  # list of [N_spatial]

        for i, path in enumerate(valid):
            ps = compute_patch_speckle(path, args.num_frames, args.resolution)
            if ps is None or len(ps) != n_patches:
                continue
            tokens = extract_spatial_tokens(
                model, path, device, is_videomae, args.resolution, args.num_frames
            )
            if tokens is None or tokens.shape[0] != n_patches:
                continue

            all_token_embs.append(tokens)
            all_patch_speckle.append(ps)

            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(valid)}")

        # Stack: [N_videos * N_patches, D] and [N_videos * N_patches]
        X = np.concatenate(all_token_embs, axis=0)
        Y = np.concatenate(all_patch_speckle, axis=0)
        print(f"\n  Total token-patch pairs: {X.shape[0]} ({len(all_token_embs)} videos × {n_patches} patches)")

        # Train Ridge probe: token embedding → patch speckle
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        ridge = Ridge(alpha=1.0)
        scores = cross_val_score(ridge, X_scaled, Y, cv=args.cv_folds, scoring="r2")
        r2 = float(scores.mean())
        r2_std = float(scores.std())
        print(f"\n  Token-level speckle R²: {r2:.4f} ± {r2_std:.4f}")

        # Save
        if args.output is None:
            args.output = f"scripts/rebuttal/samples/token_speckle_{model_name}.csv"
        with open(args.output, "w") as f:
            f.write("model,token_speckle_r2,token_speckle_r2_std\n")
            f.write(f"{model_name},{r2:.6f},{r2_std:.6f}\n")
        print(f"  Saved: {args.output}")
        args.output = None  # reset for next model

        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
