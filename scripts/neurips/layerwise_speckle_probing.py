"""
Layer-wise speckle probing: does speckle encoding vary across depth?

Extracts mean-pooled embeddings at layers 1, 6, 12, 18, 24 (ViT-L has 24 blocks).
Trains Ridge regression to predict speckle energy from each layer's embedding.
Reports partial R² (controlling for mean intensity) per layer.

Usage:
    python scripts/neurips/layerwise_speckle_probing.py \
        --models JEPA-IN21K-e100 --device cuda:0

    # All 3 in parallel
    python scripts/neurips/layerwise_speckle_probing.py \
        --models JEPA-IN21K-e100 --device cuda:0 &
    python scripts/neurips/layerwise_speckle_probing.py \
        --models BYOL-L-e100 --device cuda:1 &
    python scripts/neurips/layerwise_speckle_probing.py \
        --models MAE-L-e99 --device cuda:2 &
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from decord import VideoReader, cpu
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, ".")
from scripts.neurips.model_registry import add_model_args, get_models

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])

PROBE_LAYERS = [0, 5, 11, 17, 23]  # 0-indexed: layers 1, 6, 12, 18, 24


def compute_nuisance_vars(video_path, num_frames=16):
    """Compute speckle energy and mean intensity from raw pixels."""
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        total = len(vr)
        step = max(1, total // num_frames)
        indices = list(range(0, min(total, step * num_frames), step))[:num_frames]
        clip_raw = vr.get_batch(indices).asnumpy()  # [T, H, W, C]
    except Exception:
        return None

    gray = np.mean(clip_raw.astype(np.float32) / 255.0, axis=-1)
    mean_intensity = float(gray.mean())

    # Speckle energy: mean Laplacian energy across frames
    from scipy.ndimage import laplace
    hf_energies = []
    for t in range(gray.shape[0]):
        lap = laplace(gray[t])
        hf_energies.append(np.mean(lap ** 2))
    speckle_energy = float(np.mean(hf_energies))

    return {"mean_intensity": mean_intensity, "speckle_energy": speckle_energy}


def load_encoder_with_hooks(cfg, device, probe_layers, resolution=224, frames=16):
    """Load encoder and register hooks to capture intermediate layer outputs."""
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
        model.load_state_dict(state, strict=False)
        model.eval().to(device)

        # Register hooks on specified blocks
        for layer_idx in probe_layers:
            def make_hook(idx):
                def hook(module, input, output):
                    # Apply norm to match what the final output gets
                    layer_outputs[idx] = model.norm(output).detach()
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
        model.load_state_dict(state, strict=False)
        model.eval().to(device)

        # VideoMAE blocks
        for layer_idx in probe_layers:
            def make_hook(idx):
                def hook(module, input, output):
                    layer_outputs[idx] = model.fc_norm(output).detach()
                return hook
            model.blocks[layer_idx].register_forward_hook(make_hook(layer_idx))

        return model, layer_outputs, True

    else:
        raise ValueError(f"Unsupported model type: {model_type}")


@torch.no_grad()
def extract_layerwise_embeddings(model, layer_outputs, video_path, device,
                                  is_videomae, probe_layers,
                                  resolution=224, num_frames=16):
    """Extract mean-pooled embeddings at each probed layer."""
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

    clip = vr.get_batch(indices).asnumpy()  # [T, H, W, C] numpy
    clip = torch.from_numpy(clip).permute(3, 0, 1, 2).float() / 255.0  # [C, T, H, W]
    clip = torch.nn.functional.interpolate(
        clip.unsqueeze(0), size=(num_frames, resolution, resolution), mode="trilinear"
    ).squeeze(0)
    clip = (clip - IMAGENET_MEAN.view(3, 1, 1, 1)) / IMAGENET_STD.view(3, 1, 1, 1)
    clip = clip.unsqueeze(0).to(device)

    # Clear previous outputs
    layer_outputs.clear()

    # Forward pass triggers hooks
    if is_videomae:
        model.forward_features(clip)
    else:
        model(clip)

    # Extract mean-pooled embeddings per layer
    embeddings = {}
    for layer_idx in probe_layers:
        if layer_idx in layer_outputs:
            emb = layer_outputs[layer_idx]  # [1, N, D]
            emb = emb.mean(dim=1).squeeze(0).cpu().numpy()  # [D]
            embeddings[layer_idx] = emb

    return embeddings


def main():
    parser = argparse.ArgumentParser(description="Layer-wise speckle probing")
    add_model_args(parser)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--csv", default="data/csv/echonet_dynamic_test_local.csv")
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    models = get_models(args.models)

    # Load video paths
    df = pd.read_csv(args.csv, header=None, delimiter=" ")
    video_paths = list(df[0])
    valid = [p for p in video_paths if os.path.exists(p)]
    print(f"Videos: {len(valid)} (from {len(video_paths)} in CSV)")

    for model_name, cfg in models.items():
        print(f"\n{'=' * 60}")
        print(f"Model: {model_name}")
        print(f"Layers: {[l + 1 for l in PROBE_LAYERS]}")
        print(f"{'=' * 60}")

        model, layer_outputs, is_videomae = load_encoder_with_hooks(
            cfg, device, PROBE_LAYERS, args.resolution, args.num_frames
        )

        # Extract embeddings + nuisance vars
        all_embeddings = {l: [] for l in PROBE_LAYERS}
        all_speckle = []
        all_intensity = []

        for i, path in enumerate(valid):
            nuisance = compute_nuisance_vars(path, args.num_frames)
            if nuisance is None:
                continue
            embs = extract_layerwise_embeddings(
                model, layer_outputs, path, device, is_videomae,
                PROBE_LAYERS, args.resolution, args.num_frames
            )
            if embs is None:
                continue

            all_speckle.append(nuisance["speckle_energy"])
            all_intensity.append(nuisance["mean_intensity"])
            for l in PROBE_LAYERS:
                if l in embs:
                    all_embeddings[l].append(embs[l])

            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(valid)}")

        Y_speckle = np.array(all_speckle)
        Y_intensity = np.array(all_intensity).reshape(-1, 1)

        print(f"\n  Extracted {len(Y_speckle)} videos")
        print(f"\n  Layer-wise speckle partial R² (controlling for intensity):")
        print(f"  {'Layer':>8}  {'Speckle R²':>12}  {'Partial R²':>12}")
        print(f"  {'-' * 36}")

        results = []
        for l in PROBE_LAYERS:
            X = np.array(all_embeddings[l])
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Full speckle R²
            ridge = Ridge(alpha=1.0)
            scores = cross_val_score(ridge, X_scaled, Y_speckle, cv=args.cv_folds, scoring="r2")
            speckle_r2 = float(scores.mean())

            # Partial R²: residualize both X and Y on intensity
            from sklearn.linear_model import LinearRegression
            lr_x = LinearRegression().fit(Y_intensity, X_scaled)
            X_resid = X_scaled - lr_x.predict(Y_intensity)
            lr_y = LinearRegression().fit(Y_intensity, Y_speckle)
            Y_resid = Y_speckle - lr_y.predict(Y_intensity)

            ridge2 = Ridge(alpha=1.0)
            partial_scores = cross_val_score(ridge2, X_resid, Y_resid, cv=args.cv_folds, scoring="r2")
            partial_r2 = float(partial_scores.mean())

            print(f"  {l + 1:>8d}  {speckle_r2:>12.3f}  {partial_r2:>12.3f}")
            results.append({"layer": l + 1, "speckle_r2": speckle_r2, "partial_r2": partial_r2})

        # Save results
        if args.output is None:
            args.output = f"scripts/neurips/samples/layerwise_speckle_{model_name}.csv"
        with open(args.output, "w") as f:
            f.write("model,layer,speckle_r2,partial_r2\n")
            for r in results:
                f.write(f"{model_name},{r['layer']},{r['speckle_r2']:.6f},{r['partial_r2']:.6f}\n")
        print(f"\n  Saved: {args.output}")

        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
