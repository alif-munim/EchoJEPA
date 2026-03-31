"""
Information probing: what do frozen representations encode?

Trains linear probes on clip-level embeddings to predict:
  - Nuisance variables (from raw pixels): speckle energy, mean intensity, texture variance
  - Target variables (clinical labels): EF, ESV, EDV

Reports R² per (model, variable) as a table + heatmap.
If JEPA filters noise: low nuisance R², high target R².
If MAE retains pixels: high nuisance R², variable target R².

Usage:
    TMPDIR=/tmp LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH \
        python scripts/rebuttal/information_probing.py \
        --models pt50 --device cuda:0 \
        --output scripts/rebuttal/samples/information_probing.png
"""

import argparse
import hashlib
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, ".")
from scripts.rebuttal.model_registry import add_model_args, get_models

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])


# ── Model loading ──────────────────────────────────────────────────────────


def load_encoder(cfg, device, resolution=224, frames=16):
    """Load frozen encoder. Same as umap script."""
    import src.models.vision_transformer as vit

    model_type = cfg.get("type", "vjepa")
    if model_type == "videomae":
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
    else:
        model = vit.__dict__[cfg["model_name"]](
            img_size=resolution, num_frames=frames, patch_size=16, tubelet_size=2,
            **cfg.get("kwargs", {}),
        )
        ckpt = torch.load(cfg["checkpoint"], map_location="cpu", weights_only=False)
        state = ckpt[cfg["checkpoint_key"]]
        state = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state.items()}
        model_sd = model.state_dict()
        for k in list(state.keys()):
            if k in model_sd and state[k].shape != model_sd[k].shape:
                del state[k]
        model.load_state_dict(state, strict=False)

    model.eval().to(device)
    return model


# ── Video loading ──────────────────────────────────────────────────────────


def load_clip(video_path, start_frame=0, num_frames=16, resolution=224):
    """
    Load a single clip: num_frames consecutive frames starting at start_frame.
    Returns:
        clip_normalized: [C, T, H, W] ImageNet-normalized tensor
        clip_raw: [T, H, W, C] uint8 numpy array (for pixel statistics)
    """
    vr = VideoReader(video_path, num_threads=1, ctx=cpu(0))
    total = len(vr)

    indices = list(range(start_frame, min(start_frame + num_frames, total)))
    while len(indices) < num_frames:
        indices.append(indices[-1])

    frames = vr.get_batch(indices).asnumpy()  # [T, H, W, C] uint8
    clip_raw = frames.copy()

    # Normalize for model
    clip = torch.from_numpy(frames).float().permute(3, 0, 1, 2) / 255.0  # [C, T, H, W]
    C, T, H, W = clip.shape
    clip = clip.reshape(C * T, H, W).unsqueeze(0)
    clip = F.interpolate(clip, size=(resolution, resolution), mode="bilinear", align_corners=False)
    clip = clip.squeeze(0).reshape(C, T, resolution, resolution)
    mean = IMAGENET_MEAN.view(3, 1, 1, 1)
    std = IMAGENET_STD.view(3, 1, 1, 1)
    clip_normalized = (clip - mean) / std

    return clip_normalized, clip_raw


# ── Pixel-level nuisance statistics ───────────────────────────────────────


def compute_nuisance_stats(clip_raw):
    """
    Compute nuisance variables from raw pixel clip [T, H, W, C] uint8.
    Returns dict of scalar values.
    """
    # Convert to grayscale float [0, 1]
    gray = np.mean(clip_raw.astype(np.float32) / 255.0, axis=-1)  # [T, H, W]

    # 1. Mean intensity (global brightness)
    mean_intensity = float(gray.mean())

    # 2. Speckle energy: average high-frequency power across frames
    # High-freq = Laplacian magnitude (proxy for speckle texture)
    from scipy.ndimage import laplace
    hf_energies = []
    for t in range(gray.shape[0]):
        lap = laplace(gray[t])
        hf_energies.append(np.mean(lap ** 2))
    speckle_energy = float(np.mean(hf_energies))

    # 3. Texture variance: variance of local standard deviation
    # Compute std in 5x5 patches
    from scipy.ndimage import uniform_filter
    tex_vars = []
    for t in range(gray.shape[0]):
        local_mean = uniform_filter(gray[t], size=5)
        local_sq_mean = uniform_filter(gray[t] ** 2, size=5)
        local_std = np.sqrt(np.maximum(local_sq_mean - local_mean ** 2, 0))
        tex_vars.append(float(np.var(local_std)))
    texture_variance = float(np.mean(tex_vars))

    return {
        "mean_intensity": mean_intensity,
        "speckle_energy": speckle_energy,
        "texture_variance": texture_variance,
    }


# ── Feature extraction ────────────────────────────────────────────────────


@torch.no_grad()
def extract_clip_embedding(model, clip, device, is_videomae=False):
    """Extract single clip embedding: [D] vector."""
    clip = clip.unsqueeze(0).to(device)  # [1, C, T, H, W]
    if is_videomae:
        out = model.forward_features(clip)  # [1, D] or [1, N, D]
        if out.dim() == 3:
            out = out.mean(dim=1)
    else:
        out = model(clip)  # [1, N, D]
        out = out.mean(dim=1)  # [1, D]
    return out.squeeze(0).cpu().numpy()  # [D]


# ── Main ──────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Information probing of frozen representations")
    add_model_args(parser)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--csv", default="data/csv/echonet_dynamic_test_local.csv")
    parser.add_argument("--filelist", default="data/sample_data/echonet/echonetdynamic-2/EchoNet-Dynamic/FileList.csv")
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--clips_per_video", type=int, default=2, help="Number of clips per video")
    parser.add_argument("--output", default="scripts/rebuttal/samples/information_probing.png")
    parser.add_argument("--cv_folds", type=int, default=5, help="Cross-validation folds")
    args = parser.parse_args()

    device = torch.device(args.device)

    # ── Load video paths + labels ──
    df_csv = pd.read_csv(args.csv, header=None, delimiter=" ")
    video_paths = list(df_csv[0])
    ef_labels = list(df_csv[1].astype(float))

    # Load extended labels from FileList
    df_fl = pd.read_csv(args.filelist)
    fl_lookup = {}
    for _, row in df_fl.iterrows():
        fl_lookup[row["FileName"]] = {
            "EF": row["EF"], "ESV": row["ESV"], "EDV": row["EDV"],
        }

    # Filter to existing files
    valid = [(p, ef) for p, ef in zip(video_paths, ef_labels) if os.path.exists(p)]
    print(f"Videos: {len(valid)} (from {len(video_paths)} in CSV)")

    # ── Extract clips + nuisance stats ──
    print(f"\nExtracting {args.clips_per_video} clips × {len(valid)} videos...")
    all_clips = []  # (clip_normalized, video_idx, clip_idx)
    all_nuisance = []  # dicts
    all_targets = []  # dicts
    clip_raw_list = []

    for vi, (vpath, ef) in enumerate(valid):
        vr = VideoReader(vpath, num_threads=1, ctx=cpu(0))
        total = len(vr)
        del vr

        # Get extended labels
        fname = os.path.basename(vpath).replace(".avi", "")
        ext = fl_lookup.get(fname, {})

        # Sample clips: evenly spaced start positions
        clip_len = args.num_frames
        if total <= clip_len:
            starts = [0]
        else:
            spacing = (total - clip_len) // max(args.clips_per_video, 1)
            starts = [i * spacing for i in range(args.clips_per_video)]

        for ci, start in enumerate(starts):
            try:
                clip_norm, clip_raw = load_clip(vpath, start, args.num_frames, args.resolution)
                nuisance = compute_nuisance_stats(clip_raw)
                targets = {"EF": ef}
                if "ESV" in ext:
                    targets["ESV"] = ext["ESV"]
                if "EDV" in ext:
                    targets["EDV"] = ext["EDV"]

                all_clips.append(clip_norm)
                all_nuisance.append(nuisance)
                all_targets.append(targets)
            except Exception as e:
                if vi < 5:
                    print(f"  Skip {vpath} clip {ci}: {e}")

        if (vi + 1) % 200 == 0:
            print(f"  {vi + 1}/{len(valid)} videos processed ({len(all_clips)} clips)")

    print(f"  Total clips: {len(all_clips)}")

    # Build nuisance/target arrays
    nuisance_names = ["speckle_energy", "mean_intensity", "texture_variance"]
    target_names = ["EF", "ESV", "EDV"]

    Y_nuisance = np.array([[n[k] for k in nuisance_names] for n in all_nuisance])
    Y_target = np.array([[t.get(k, np.nan) for k in target_names] for t in all_targets])

    print(f"\nNuisance shape: {Y_nuisance.shape}, Target shape: {Y_target.shape}")
    for i, name in enumerate(nuisance_names):
        print(f"  {name}: mean={Y_nuisance[:, i].mean():.4f}, std={Y_nuisance[:, i].std():.4f}")
    for i, name in enumerate(target_names):
        valid_mask = ~np.isnan(Y_target[:, i])
        print(f"  {name}: mean={Y_target[valid_mask, i].mean():.2f}, std={Y_target[valid_mask, i].std():.2f}, N={valid_mask.sum()}")

    # ── Extract embeddings per model ──
    models = get_models(args.models)
    all_probe_results = {}

    for mname, cfg in models.items():
        print(f"\n{'=' * 60}")
        print(f"Model: {mname}")
        print(f"{'=' * 60}")

        if not os.path.exists(cfg["checkpoint"]):
            print(f"  SKIP: checkpoint not found")
            continue

        is_videomae = cfg.get("type") == "videomae"
        model = load_encoder(cfg, device, args.resolution, args.num_frames)

        # Extract embeddings
        embeddings = []
        for ci, clip in enumerate(all_clips):
            emb = extract_clip_embedding(model, clip, device, is_videomae)
            embeddings.append(emb)
            if (ci + 1) % 500 == 0:
                print(f"  {ci + 1}/{len(all_clips)} clips embedded")

        X = np.array(embeddings)
        print(f"  Embeddings: {X.shape}")

        del model
        torch.cuda.empty_cache()

        # ── Train linear probes ──
        scaler_X = StandardScaler()
        X_scaled = scaler_X.fit_transform(X)

        probe_results = {}

        # Nuisance probes
        for i, name in enumerate(nuisance_names):
            y = Y_nuisance[:, i]
            scaler_y = StandardScaler()
            y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

            scores = cross_val_score(
                Ridge(alpha=1.0), X_scaled, y_scaled,
                cv=args.cv_folds, scoring="r2",
            )
            r2 = float(scores.mean())
            probe_results[name] = r2
            print(f"  Nuisance | {name:20s}: R² = {r2:.3f} (±{scores.std():.3f})")

        # Partial R² for speckle energy after regressing out mean intensity
        # Tests whether the speckle gap (JEPA < MAE) is confounded by intensity
        y_speckle = StandardScaler().fit_transform(Y_nuisance[:, 0:1]).ravel()  # speckle_energy
        y_intensity = StandardScaler().fit_transform(Y_nuisance[:, 1:2]).ravel()  # mean_intensity

        # Residualize embeddings against intensity
        from sklearn.linear_model import LinearRegression
        lr_x = LinearRegression().fit(y_intensity.reshape(-1, 1), X_scaled)
        X_resid = X_scaled - lr_x.predict(y_intensity.reshape(-1, 1))

        # Residualize speckle against intensity
        lr_y = LinearRegression().fit(y_intensity.reshape(-1, 1), y_speckle)
        y_speckle_resid = y_speckle - lr_y.predict(y_intensity.reshape(-1, 1))

        # Partial R²: predict residualized speckle from residualized embeddings
        partial_scores = cross_val_score(
            Ridge(alpha=1.0), X_resid, y_speckle_resid,
            cv=args.cv_folds, scoring="r2",
        )
        partial_r2 = float(partial_scores.mean())
        probe_results["speckle_partial"] = partial_r2
        print(f"  Partial  | speckle|intensity  : R² = {partial_r2:.3f} (±{partial_scores.std():.3f})")

        # Speckle-intensity correlation for reference
        corr = float(np.corrcoef(Y_nuisance[:, 0], Y_nuisance[:, 1])[0, 1])
        if mname == list(models.keys())[0]:
            print(f"  (Speckle-intensity Pearson correlation: {corr:.3f})")

        # Target probes
        for i, name in enumerate(target_names):
            y = Y_target[:, i]
            valid_mask = ~np.isnan(y)
            if valid_mask.sum() < 50:
                print(f"  Target  | {name:20s}: SKIP (N={valid_mask.sum()})")
                continue

            y_valid = y[valid_mask]
            X_valid = X_scaled[valid_mask]
            scaler_y = StandardScaler()
            y_scaled = scaler_y.fit_transform(y_valid.reshape(-1, 1)).ravel()

            scores = cross_val_score(
                Ridge(alpha=1.0), X_valid, y_scaled,
                cv=args.cv_folds, scoring="r2",
            )
            r2 = float(scores.mean())
            probe_results[name] = r2
            print(f"  Target  | {name:20s}: R² = {r2:.3f} (±{scores.std():.3f})")

        all_probe_results[mname] = probe_results

        # Save embeddings for this model
        npz_path = args.output.replace(".png", f"_{mname}.npz")
        np.savez_compressed(npz_path, embeddings=X, nuisance=Y_nuisance, targets=Y_target)
        print(f"  Saved: {npz_path}")

    # ── Print summary table ──
    print(f"\n{'=' * 80}")
    print("INFORMATION PROBING RESULTS")
    print(f"{'=' * 80}")

    all_vars = nuisance_names + [n for n in target_names if any(n in r for r in all_probe_results.values())]
    model_names = list(all_probe_results.keys())

    header = f"{'Variable':>22s}"
    for m in model_names:
        header += f"  {m:>14s}"
    print(header)
    print("-" * len(header))

    for var in all_vars:
        is_nuisance = var in nuisance_names
        label = f"{'[N]' if is_nuisance else '[T]'} {var}"
        row = f"{label:>22s}"
        for m in model_names:
            r2 = all_probe_results[m].get(var)
            if r2 is not None:
                row += f"  {r2:>14.3f}"
            else:
                row += f"  {'N/A':>14s}"
        print(row)

    # ── Generate heatmap ──
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))

    data = []
    ylabels = []
    for var in all_vars:
        is_nuisance = var in nuisance_names
        row = []
        for m in model_names:
            r2 = all_probe_results[m].get(var, 0)
            row.append(r2)
        data.append(row)
        prefix = "(Nuisance) " if is_nuisance else "(Target) "
        ylabels.append(prefix + var.replace("_", " ").title())

    data = np.array(data)

    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=-0.05, vmax=max(0.5, data.max() + 0.05))
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, fontsize=10, fontweight="bold")
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels, fontsize=9)

    # Add horizontal line separating nuisance from target
    ax.axhline(y=len(nuisance_names) - 0.5, color="black", linewidth=2)

    # Annotate cells
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            color = "white" if abs(val) > 0.3 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=10, color=color, fontweight="bold")

    ax.set_title("Information Probing: What Do Representations Encode?", fontsize=12, fontweight="bold", pad=10)
    plt.colorbar(im, ax=ax, label="R² (5-fold CV)", shrink=0.8)
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
