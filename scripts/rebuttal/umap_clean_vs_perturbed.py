"""
UMAP visualization: clean vs perturbed echo embeddings.

Loads each encoder, extracts mean-pooled features from clean and perturbed
versions of the same test videos, runs UMAP, and generates a publication-quality
figure showing whether embeddings overlap (invariant) or separate (sensitive).

Single perturbation -> 1xN figure (one panel per model).
Multiple perturbations -> MxN grid (M perturbation types x N models).

Usage:
    TMPDIR=/tmp LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH \
        python scripts/rebuttal/umap_clean_vs_perturbed.py \
        --models pt50 \
        --perturbation depth_attenuation \
        --severity severe \
        --n_samples 200 \
        --device cuda:6 \
        --output scripts/rebuttal/samples/umap_clean_vs_perturbed.png

    # Multiple perturbations (3x3 grid):
    TMPDIR=/tmp LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH \
        python scripts/rebuttal/umap_clean_vs_perturbed.py \
        --models pt50 \
        --perturbation depth_attenuation gaussian_shadow haze_artifact \
        --severity severe \
        --n_samples 200 \
        --device cuda:6 \
        --output scripts/rebuttal/samples/umap_clean_vs_perturbed_3x3.png

Dependencies:
    pip install umap-learn   # optional; falls back to t-SNE (sklearn) if missing
"""

import argparse
import hashlib
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu

sys.path.insert(0, ".")

from scripts.rebuttal.echo_perturbations import (
    PERTURBATIONS,
    apply_perturbation,
    create_scan_mask,
)
from scripts.rebuttal.model_registry import add_model_args, get_models

# ImageNet normalization constants
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])


# ---------------------------------------------------------------------------
# Model loading (reuses diagnose_p0 pattern)
# ---------------------------------------------------------------------------


def load_encoder(cfg, device, resolution=224, frames=16):
    """Load a frozen encoder from checkpoint. Supports vjepa and videomae types."""
    import src.models.vision_transformer as vit

    model_type = cfg.get("type", "vjepa")
    if model_type == "videomae":
        from evals.video_classification_frozen.modelcustom.videomae_encoder import (
            _convert_pretrain_to_finetune_state_dict,
            _import_modeling_finetune,
        )

        mf = _import_modeling_finetune()
        model = mf.vit_large_patch16_224(
            img_size=resolution,
            all_frames=frames,
            tubelet_size=2,
            num_classes=1000,
        )
        ckpt = torch.load(cfg["checkpoint"], map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt)
        state = _convert_pretrain_to_finetune_state_dict(state, model.state_dict())
        model.load_state_dict(state, strict=False)
    else:
        model = vit.__dict__[cfg["model_name"]](
            img_size=resolution,
            num_frames=frames,
            patch_size=16,
            tubelet_size=2,
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


# ---------------------------------------------------------------------------
# Video loading and preprocessing
# ---------------------------------------------------------------------------


def load_video_frames(video_path, num_frames=16, resolution=224):
    """
    Load a video, sample num_frames uniformly, resize to resolution, normalize.

    Returns:
        clip: [C, T, H, W] float32, ImageNet-normalized
    """
    vr = VideoReader(video_path, num_threads=1, ctx=cpu(0))
    total = len(vr)

    # Sample num_frames consecutive frames with step=1 from the start
    indices = list(range(min(num_frames, total)))
    # Pad if video is shorter than requested frames
    while len(indices) < num_frames:
        indices.append(indices[-1])
    frames = vr.get_batch(indices).asnumpy()  # [T, H, W, C] uint8

    # Convert to float [0, 1] and to tensor
    clip = torch.from_numpy(frames).float()  # [T, H, W, C]
    clip = clip.permute(3, 0, 1, 2)  # [C, T, H, W]
    clip = clip / 255.0

    # Resize spatial dims to resolution (center crop via resize)
    C, T, H, W = clip.shape
    clip = clip.reshape(C * T, H, W).unsqueeze(0)  # [1, C*T, H, W]
    clip = F.interpolate(clip, size=(resolution, resolution), mode="bilinear", align_corners=False)
    clip = clip.squeeze(0).reshape(C, T, resolution, resolution)

    # ImageNet normalize
    mean = IMAGENET_MEAN.view(3, 1, 1, 1)
    std = IMAGENET_STD.view(3, 1, 1, 1)
    clip = (clip - mean) / std

    return clip


def apply_perturbation_to_normalized(clip, ptype, severity, video_path):
    """
    Apply perturbation to an ImageNet-normalized clip.
    Un-normalizes -> perturbs in [0,1] space -> re-normalizes.
    Same flow as VideoDataset.get_item_video perturbation hook.
    """
    mean = IMAGENET_MEAN.view(3, 1, 1, 1)
    std = IMAGENET_STD.view(3, 1, 1, 1)

    # Un-normalize to [0, 1]
    pixel = (clip * std + mean).clamp(0, 1)

    # Create scan mask from first frame
    mask = create_scan_mask(pixel[:, 0, :, :])

    # Deterministic seed from video path
    seed = int(hashlib.md5(str(video_path).encode()).hexdigest()[:8], 16)

    # Apply perturbation
    pixel = apply_perturbation(pixel, ptype, severity, scan_mask=mask, seed=seed, transducer_pos=(0.5, 0.0))

    # Re-normalize
    return (pixel - mean) / std


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


@torch.no_grad()
def extract_features(model, clips, device, is_videomae=False, batch_size=8):
    """
    Extract mean-pooled features from a list of clips.

    Args:
        model: frozen encoder
        clips: list of [C, T, H, W] tensors (ImageNet-normalized)
        device: torch device
        is_videomae: whether model is VideoMAE (uses forward_features)
        batch_size: inference batch size

    Returns:
        [N, D] float32 tensor of features
    """
    all_feats = []
    for i in range(0, len(clips), batch_size):
        batch = torch.stack(clips[i : i + batch_size]).to(device)
        if is_videomae:
            out = model.forward_features(batch)
            if out.dim() == 3:
                out = out.mean(dim=1)
        else:
            out = model(batch)
            out = out.mean(dim=1)
        all_feats.append(out.cpu())
    return torch.cat(all_feats, dim=0)


# ---------------------------------------------------------------------------
# UMAP + plotting
# ---------------------------------------------------------------------------


def run_umap(features, n_neighbors=15, min_dist=0.1, random_state=42):
    """
    Run UMAP dimensionality reduction on features.
    Falls back to t-SNE (sklearn) if umap-learn is not installed.
    """
    try:
        import umap

        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=2,
            random_state=random_state,
            metric="cosine",
        )
        embedding = reducer.fit_transform(features.numpy())
        return embedding
    except ImportError:
        print("  [WARN] umap-learn not installed, falling back to t-SNE (pip install umap-learn)")
        from sklearn.manifold import TSNE

        reducer = TSNE(
            n_components=2,
            perplexity=min(30, len(features) - 1),
            random_state=random_state,
            metric="cosine",
            init="pca",
            learning_rate="auto",
        )
        embedding = reducer.fit_transform(features.numpy())
        return embedding


def make_figure(
    embeddings_dict,
    model_names,
    perturbation_types,
    severity,
    n_samples,
    output_path,
    color_by_label=False,
):
    """
    Generate the UMAP figure.

    Args:
        embeddings_dict: {(model_name, ptype): {"clean": [N,2], "perturbed": [N,2], "labels": [N]}}
        model_names: list of model display names
        perturbation_types: list of perturbation type strings
        severity: severity level string
        n_samples: number of videos
        output_path: path to save PNG
        color_by_label: if True, color by LVEF label (continuous colormap),
            clean=circles, perturbed=triangles. Shows task-relevant structure preservation.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    n_rows = len(perturbation_types)
    n_cols = len(model_names)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.5 * n_cols, 4.0 * n_rows),
        squeeze=False,
    )

    # For label-colored mode, compute global label range for consistent colormap
    if color_by_label:
        all_labels = []
        for key in embeddings_dict.values():
            if key.get("labels") is not None:
                all_labels.extend(key["labels"].tolist())
        vmin = np.percentile(all_labels, 2) if all_labels else 0
        vmax = np.percentile(all_labels, 98) if all_labels else 1
        cmap = plt.cm.RdYlBu_r  # low LVEF = red, high LVEF = blue

    for row, ptype in enumerate(perturbation_types):
        for col, mname in enumerate(model_names):
            ax = axes[row, col]
            key = (mname, ptype)
            clean_emb = embeddings_dict[key]["clean"]
            pert_emb = embeddings_dict[key]["perturbed"]
            labels = embeddings_dict[key].get("labels")

            if color_by_label and labels is not None:
                # Color by LVEF label, distinguish clean/perturbed by marker shape
                sc_pert = ax.scatter(
                    pert_emb[:, 0], pert_emb[:, 1],
                    c=labels, cmap=cmap, vmin=vmin, vmax=vmax,
                    alpha=0.6, s=18, marker="^", edgecolors="none", rasterized=True,
                )
                sc_clean = ax.scatter(
                    clean_emb[:, 0], clean_emb[:, 1],
                    c=labels, cmap=cmap, vmin=vmin, vmax=vmax,
                    alpha=0.6, s=18, marker="o", edgecolors="none", rasterized=True,
                )
            else:
                # Default: color by clean/perturbed
                ax.scatter(
                    pert_emb[:, 0], pert_emb[:, 1],
                    c="#E74C3C", alpha=0.5, s=12, edgecolors="none",
                    label="Perturbed", rasterized=True,
                )
                ax.scatter(
                    clean_emb[:, 0], clean_emb[:, 1],
                    c="#3498DB", alpha=0.5, s=12, edgecolors="none",
                    label="Clean", rasterized=True,
                )

            # Title and labels
            ptype_display = ptype.replace("_", " ").title()
            if n_rows == 1:
                ax.set_title(f"{mname}", fontsize=13, fontweight="bold")
            else:
                if row == 0:
                    ax.set_title(f"{mname}", fontsize=13, fontweight="bold")
                if col == 0:
                    ax.set_ylabel(f"{ptype_display}", fontsize=11, fontweight="bold")

            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
                spine.set_color("#CCCCCC")

            # Add cosine similarity annotation
            cos_sim = embeddings_dict[key].get("cosine_sim", None)
            if cos_sim is not None:
                ax.text(
                    0.03, 0.97, f"cos={cos_sim:.3f}",
                    transform=ax.transAxes, fontsize=9, verticalalignment="top",
                    fontfamily="monospace",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="#CCCCCC"),
                )

            # Legend
            if row == n_rows - 1 and col == n_cols - 1:
                if color_by_label:
                    legend_elements = [
                        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
                               markersize=7, label="Clean"),
                        Line2D([0], [0], marker="^", color="w", markerfacecolor="gray",
                               markersize=7, label="Perturbed"),
                    ]
                    ax.legend(handles=legend_elements, loc="lower right", fontsize=8,
                              framealpha=0.8, edgecolor="#CCCCCC")
                else:
                    ax.legend(loc="lower right", fontsize=8, framealpha=0.8,
                              edgecolor="#CCCCCC", markerscale=1.5)

    # Colorbar for label-colored mode
    if color_by_label:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label("LVEF (%)", fontsize=10)

    sev_display = severity.title()
    mode_label = "LVEF-Colored" if color_by_label else "Clean vs Perturbed"
    fig.suptitle(
        f"UMAP: {mode_label} Embeddings ({sev_display}, N={n_samples})",
        fontsize=14, fontweight="bold", y=1.01,
    )

    plt.tight_layout(rect=[0, 0, 0.91 if color_by_label else 1, 1])
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved figure to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="UMAP visualization of clean vs perturbed echo embeddings"
    )
    add_model_args(parser)
    parser.add_argument(
        "--perturbation",
        nargs="+",
        default=["depth_attenuation"],
        choices=list(PERTURBATIONS.keys()),
        help="Perturbation type(s). Multiple types create a grid.",
    )
    parser.add_argument(
        "--severity",
        default="severe",
        choices=["mild", "moderate", "severe"],
        help="Perturbation severity level",
    )
    parser.add_argument(
        "--csv",
        default="data/csv/echonet_dynamic_test_local.csv",
        help="Path to CSV with video paths (space-delimited, col 0 = path)",
    )
    parser.add_argument("--n_samples", type=int, default=200, help="Number of videos to sample")
    parser.add_argument("--device", default="cuda:0", help="Torch device")
    parser.add_argument("--batch_size", type=int, default=8, help="Inference batch size")
    parser.add_argument("--num_frames", type=int, default=16, help="Frames per clip")
    parser.add_argument("--resolution", type=int, default=224, help="Spatial resolution")
    parser.add_argument(
        "--output",
        default="scripts/rebuttal/samples/umap_clean_vs_perturbed.png",
        help="Output PNG path",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for video sampling")
    parser.add_argument(
        "--umap_neighbors", type=int, default=15, help="UMAP n_neighbors parameter"
    )
    parser.add_argument(
        "--umap_min_dist", type=float, default=0.1, help="UMAP min_dist parameter"
    )
    parser.add_argument(
        "--color_by_label",
        action="store_true",
        help="Color points by LVEF label (continuous colormap) instead of clean/perturbed. "
        "Clean=circles, perturbed=triangles. Shows whether task-relevant structure is preserved.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    rng = np.random.RandomState(args.seed)

    # --- Load video paths and labels ---
    import pandas as pd

    df = pd.read_csv(args.csv, header=None, delimiter=" ")
    all_paths = list(df.values[:, 0])
    all_labels = list(df.values[:, 1].astype(float)) if df.shape[1] > 1 else [None] * len(all_paths)
    print(f"Loaded {len(all_paths)} video paths from {args.csv}")

    # Filter to existing files (keep path-label pairs)
    existing = [(p, l) for p, l in zip(all_paths, all_labels) if os.path.exists(p)]
    if len(existing) < len(all_paths):
        print(f"  {len(existing)}/{len(all_paths)} files exist locally")
    if len(existing) == 0:
        print("ERROR: No video files found. Check CSV paths.")
        sys.exit(1)

    # Random subset
    n = min(args.n_samples, len(existing))
    indices = rng.choice(len(existing), size=n, replace=False)
    video_paths = [existing[i][0] for i in indices]
    video_labels = [existing[i][1] for i in indices]
    print(f"Selected {n} videos for UMAP")

    # --- Load all video clips (clean) ---
    print(f"\nLoading {n} video clips ({args.num_frames} frames, {args.resolution}px)...")
    clean_clips = []
    valid_paths = []
    valid_labels = []
    for i, (vpath, vlabel) in enumerate(zip(video_paths, video_labels)):
        try:
            clip = load_video_frames(vpath, num_frames=args.num_frames, resolution=args.resolution)
            clean_clips.append(clip)
            valid_paths.append(vpath)
            valid_labels.append(vlabel)
        except Exception as e:
            print(f"  Failed to load {vpath}: {e}")
        if (i + 1) % 50 == 0:
            print(f"  Loaded {i + 1}/{n} videos")
    print(f"  Successfully loaded {len(clean_clips)}/{n} videos")

    if len(clean_clips) < 10:
        print("ERROR: Too few videos loaded. Need at least 10 for meaningful UMAP.")
        sys.exit(1)

    # --- Prepare perturbed clips for each perturbation type ---
    perturbed_clips_dict = {}
    for ptype in args.perturbation:
        print(f"\nApplying perturbation: {ptype}/{args.severity}...")
        perturbed = []
        for i, (clip, vpath) in enumerate(zip(clean_clips, valid_paths)):
            pclip = apply_perturbation_to_normalized(clip, ptype, args.severity, vpath)
            perturbed.append(pclip)
        perturbed_clips_dict[ptype] = perturbed
        print(f"  Done ({len(perturbed)} clips)")

    # --- Extract features for each model ---
    models = get_models(args.models)
    model_names = list(models.keys())
    print(f"\nModels: {model_names}")

    embeddings_dict = {}

    for mname, cfg in models.items():
        print(f"\n{'='*60}")
        print(f"Model: {mname}")
        print(f"{'='*60}")

        ckpt_path = cfg["checkpoint"]
        if not os.path.exists(ckpt_path):
            print(f"  SKIP: checkpoint not found at {ckpt_path}")
            continue

        is_videomae = cfg.get("type") == "videomae"
        model = load_encoder(cfg, device, resolution=args.resolution, frames=args.num_frames)

        # Extract clean features
        print(f"  Extracting clean features...")
        feat_clean = extract_features(model, clean_clips, device, is_videomae, args.batch_size)
        print(f"  Clean features: {feat_clean.shape}")

        # Extract perturbed features for each perturbation type
        for ptype in args.perturbation:
            print(f"  Extracting {ptype}/{args.severity} features...")
            feat_pert = extract_features(
                model, perturbed_clips_dict[ptype], device, is_videomae, args.batch_size
            )

            # Cosine similarity between paired clean/perturbed
            cos_sim = F.cosine_similarity(feat_clean, feat_pert, dim=-1).mean().item()
            print(f"  Mean cosine similarity (clean vs {ptype}): {cos_sim:.4f}")

            # Concatenate and run UMAP
            combined = torch.cat([feat_clean, feat_pert], dim=0)  # [2N, D]
            print(f"  Running UMAP on {combined.shape[0]} points ({combined.shape[1]}-d)...")
            umap_emb = run_umap(
                combined,
                n_neighbors=args.umap_neighbors,
                min_dist=args.umap_min_dist,
                random_state=args.seed,
            )

            N = feat_clean.shape[0]
            embeddings_dict[(mname, ptype)] = {
                "clean": umap_emb[:N],
                "perturbed": umap_emb[N:],
                "cosine_sim": cos_sim,
                "labels": np.array(valid_labels),
            }

        del model
        torch.cuda.empty_cache()

    # --- Check that we got results for all requested combinations ---
    complete_models = []
    for mname in model_names:
        if all((mname, pt) in embeddings_dict for pt in args.perturbation):
            complete_models.append(mname)

    if not complete_models:
        print("\nERROR: No models completed successfully.")
        sys.exit(1)

    # --- Generate figure ---
    print(f"\nGenerating figure ({len(args.perturbation)}x{len(complete_models)} grid)...")
    make_figure(
        embeddings_dict=embeddings_dict,
        model_names=complete_models,
        perturbation_types=args.perturbation,
        severity=args.severity,
        n_samples=len(clean_clips),
        output_path=args.output,
        color_by_label=args.color_by_label,
    )

    # --- Summary ---
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for ptype in args.perturbation:
        ptype_display = ptype.replace("_", " ").title()
        print(f"\n  {ptype_display} ({args.severity}):")
        for mname in complete_models:
            cos = embeddings_dict[(mname, ptype)]["cosine_sim"]
            print(f"    {mname:20s}  cosine={cos:.4f}")


if __name__ == "__main__":
    main()
