"""
Cardiac Cycle Trajectory Analysis — signature rebuttal figure.

Extracts per-frame embeddings from frozen encoders (JEPA, BYOL, MAE) on a
single echocardiogram spanning multiple heartbeats. Projects to 2D via PCA.
Colors by time (frame index) to reveal cardiac cycle loops.

Prediction: JEPA traces clean, closed loops (representation captures
physiological state). MAE traces noisy, jittery paths (encodes frame-specific
texture). BYOL is in between.

Quantitative metric: inter-cycle variance / intra-cycle variance ratio.
High ratio = consistent cycles with distinct phases. Low = noisy/irregular.

Usage:
    # Quick test with one video, one model
    python scripts/neurips/cardiac_trajectory.py \
        --videos 0X18610C37F29E80E4 \
        --models JEPA-L-pt50 \
        --device cuda:0

    # Full figure: 3 models × 2 videos × clean+perturbed
    python scripts/neurips/cardiac_trajectory.py \
        --videos 0X18610C37F29E80E4 0X6767F4359E20EE24 \
        --models JEPA-L-pt50 BYOL-L-pt50 MAE-L-pt50 \
        --perturb depth_attenuation severe \
        --device cuda:0

    # Use specific video paths instead of EchoNet IDs
    python scripts/neurips/cardiac_trajectory.py \
        --video_paths /path/to/video1.avi /path/to/video2.avi \
        --models pt50 \
        --device cuda:0
"""

import argparse
import os
import sys

import decord
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF
from matplotlib.collections import LineCollection
from scipy.signal import find_peaks

sys.path.insert(0, ".")

from scripts.neurips.echo_perturbations import (
    PERTURBATIONS,
    TRANSDUCER_PRESETS,
    apply_perturbation,
    create_scan_mask,
)
from scripts.neurips.model_registry import add_model_args, get_models

decord.bridge.set_bridge("torch")

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

ECHONET_ROOT = "data/sample_data/echonet/echonetdynamic-2/EchoNet-Dynamic"


# --- Video Loading ---


def load_full_video(video_path, resolution=224):
    """Load all frames of a video as [T, C, H, W] float32 in [0, 1]."""
    vr = decord.VideoReader(video_path, num_threads=1)
    total = len(vr)
    indices = list(range(total))
    frames = vr.get_batch(indices)  # [T, H, W, C]
    frames = frames.permute(0, 3, 1, 2).float() / 255.0  # [T, C, H, W]
    frames = torch.stack([TF.resize(f, [resolution, resolution], antialias=True) for f in frames])
    return frames  # [T, C, H, W]


def normalize_frames(frames):
    """Apply ImageNet normalization. frames: [T, C, H, W]."""
    mean = IMAGENET_MEAN.unsqueeze(0)  # [1, 3, 1, 1]
    std = IMAGENET_STD.unsqueeze(0)
    return (frames - mean) / std


# --- Encoder Loading ---


def load_encoder(model_cfg, device, resolution=224):
    """Load a frozen encoder from model registry config. Returns (encoder, embed_dim, is_videomae)."""
    mtype = model_cfg["type"]

    if mtype in ("vjepa",):
        import src.models.vision_transformer as vit_module
        model_name = model_cfg["model_name"] or "vit_large"
        model = vit_module.__dict__[model_name](
            img_size=resolution, num_frames=16, patch_size=16, tubelet_size=2,
            **model_cfg.get("kwargs", {}),
        )
        ckpt = torch.load(model_cfg["checkpoint"], map_location="cpu", weights_only=False)
        key = model_cfg.get("checkpoint_key", "target_encoder")
        if key and key in ckpt:
            sd = ckpt[key]
        else:
            for k in ["encoder", "model", "state_dict"]:
                if k in ckpt:
                    sd = ckpt[k]
                    break
            else:
                sd = ckpt
        sd = {k.replace("module.", "").replace("backbone.", ""): v for k, v in sd.items()}
        model_sd = model.state_dict()
        for k in list(sd.keys()):
            if k in model_sd and sd[k].shape != model_sd[k].shape:
                del sd[k]
        model.load_state_dict(sd, strict=False)
        model.eval().to(device)
        for p in model.parameters():
            p.requires_grad = False
        return model, model.embed_dim, False

    elif mtype == "videomae":
        from evals.video_classification_frozen.modelcustom.videomae_encoder import (
            _convert_pretrain_to_finetune_state_dict,
            _import_modeling_finetune,
        )
        mf = _import_modeling_finetune()
        model = mf.vit_large_patch16_224(
            img_size=resolution, all_frames=16, tubelet_size=2, num_classes=1000,
        )
        ckpt = torch.load(model_cfg["checkpoint"], map_location="cpu", weights_only=False)
        sd = ckpt.get("model", ckpt)
        sd = _convert_pretrain_to_finetune_state_dict(sd, model.state_dict())
        model.load_state_dict(sd, strict=False)
        model.eval().to(device)
        for p in model.parameters():
            p.requires_grad = False
        return model, model.embed_dim, True

    else:
        raise ValueError(f"Unsupported model type for trajectory analysis: {mtype}")


# --- Per-Frame Embedding Extraction ---


@torch.no_grad()
def extract_per_frame_embeddings(encoder, frames, device, is_videomae=False,
                                 clip_len=16, stride=2, resolution=224):
    """
    Extract one embedding vector per frame using a sliding window.

    For each frame t, we build a clip of `clip_len` frames centered on t
    (with edge padding), encode it, and extract the representation at t's
    temporal position. For ViT encoders, we mean-pool spatially.

    Args:
        encoder: frozen encoder
        frames: [T_total, C, H, W] normalized tensor
        device: torch device
        is_videomae: if True, use forward_features (returns pooled)
        clip_len: frames per clip (default 16)
        stride: temporal stride for frame sampling within clip

    Returns:
        embeddings: [T_total, D] numpy array
    """
    T_total = frames.shape[0]
    tubelet_size = 2
    T_tokens = clip_len // tubelet_size  # 8 temporal tokens per clip
    embeddings = []

    for t in range(T_total):
        # Build a clip of clip_len frames centered on t with stride
        half_span = (clip_len * stride) // 2
        center = t
        start = center - half_span
        end = start + clip_len * stride

        # Gather frame indices with stride, clamping to valid range
        indices = list(range(start, end, stride))
        indices = [max(0, min(T_total - 1, i)) for i in indices]
        indices = indices[:clip_len]

        # Which temporal token does frame t map to?
        # Find which sampled index is closest to t
        dists = [abs(idx - t) for idx in indices]
        closest_sample = int(np.argmin(dists))
        t_token = closest_sample // tubelet_size

        clip = frames[indices]  # [clip_len, C, H, W]
        clip = clip.permute(1, 0, 2, 3).unsqueeze(0).to(device)  # [1, C, T, H, W]

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            if is_videomae:
                # VideoMAE forward_features returns mean-pooled [B, D]
                out = encoder.forward_features(clip)
                emb = out.float().squeeze(0)  # [D]
            else:
                # V-JEPA returns [B, N, D] spatial+temporal tokens
                tokens = encoder(clip)  # [1, N, D]
                tokens = tokens.float()
                B, N, D = tokens.shape
                spatial_per_t = N // T_tokens
                H_patch = int(spatial_per_t ** 0.5)
                # Reshape to [T_tokens, H, W, D]
                tokens = tokens.squeeze(0).reshape(T_tokens, H_patch, H_patch, D)
                # Extract spatial tokens at t's temporal position, mean-pool
                emb = tokens[t_token].reshape(-1, D).mean(dim=0)  # [D]

        embeddings.append(emb.cpu().numpy())

    return np.stack(embeddings)  # [T_total, D]


# --- Cycle Analysis ---


def estimate_cycle_period(embeddings_2d):
    """Estimate cardiac cycle period from PCA trajectory autocorrelation."""
    # Use first PC
    signal = embeddings_2d[:, 0]
    signal = signal - signal.mean()

    # Autocorrelation
    n = len(signal)
    autocorr = np.correlate(signal, signal, mode="full")[n - 1:]
    autocorr = autocorr / autocorr[0]

    # Find first peak after initial decay (skip first few frames)
    min_period = 8  # at least 8 frames per cycle
    peaks, properties = find_peaks(autocorr[min_period:], height=0.2, distance=min_period)

    if len(peaks) == 0:
        return None
    return peaks[0] + min_period


def compute_cycle_variance_ratio(embeddings_2d, period):
    """
    Compute inter-cycle / intra-cycle variance ratio.

    High ratio = cycles are consistent, phases are distinct (good structure).
    Low ratio = noisy/irregular trajectories.

    Splits the trajectory into full cycles, then compares:
    - Intra-cycle variance: average variance within each cycle (phase discrimination)
    - Inter-cycle variance: variance of corresponding phase points across cycles
    """
    n_frames = len(embeddings_2d)
    n_full_cycles = n_frames // period
    if n_full_cycles < 2:
        return None, 0

    # Trim to full cycles
    trimmed = embeddings_2d[:n_full_cycles * period]
    cycles = trimmed.reshape(n_full_cycles, period, 2)  # [n_cycles, period, 2]

    # Intra-cycle: how spread is each cycle in PCA space (phase discrimination)
    intra_vars = [np.var(cycles[i], axis=0).sum() for i in range(n_full_cycles)]
    mean_intra = np.mean(intra_vars)

    # Inter-cycle: for each phase point, how much do cycles differ?
    inter_vars = [np.var(cycles[:, t, :], axis=0).sum() for t in range(period)]
    mean_inter = np.mean(inter_vars)

    # Ratio: intra / inter. High = good phase structure, consistent across cycles
    ratio = mean_intra / (mean_inter + 1e-8)
    return ratio, n_full_cycles


# --- Visualization ---


def plot_trajectory(ax, emb_2d, title, period=None, ratio=None, n_cycles=None):
    """Plot a 2D trajectory colored by time with line segments."""
    n = len(emb_2d)
    t = np.arange(n)

    # Create line segments
    points = emb_2d.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Color by time using cyclic colormap
    norm = plt.Normalize(0, n - 1)
    lc = LineCollection(segments, cmap="twilight", norm=norm, linewidth=0.8, alpha=0.7)
    lc.set_array(t[:-1])
    ax.add_collection(lc)

    # Start/end markers
    ax.scatter(*emb_2d[0], c="green", s=40, zorder=5, marker="o", edgecolors="black", linewidths=0.5)
    ax.scatter(*emb_2d[-1], c="red", s=40, zorder=5, marker="s", edgecolors="black", linewidths=0.5)

    ax.set_xlim(emb_2d[:, 0].min() - 0.5, emb_2d[:, 0].max() + 0.5)
    ax.set_ylim(emb_2d[:, 1].min() - 0.5, emb_2d[:, 1].max() + 0.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    label = title
    if ratio is not None:
        label += f"\nCVR={ratio:.1f} ({n_cycles} cycles)"
    ax.set_title(label, fontsize=9)


def make_figure(all_results, output_path):
    """
    Generate the full grid figure.

    all_results: list of dicts with keys:
        model_name, video_id, condition, emb_2d, period, ratio, n_cycles
    """
    # Organize into grid: rows=models, cols=video×condition
    model_names = sorted(set(r["model_name"] for r in all_results),
                         key=lambda m: ["JEPA", "BYOL", "MAE"].index(m.split("-")[0]) if m.split("-")[0] in ["JEPA", "BYOL", "MAE"] else 99)
    col_keys = []
    seen = set()
    for r in all_results:
        key = (r["video_id"], r["condition"])
        if key not in seen:
            col_keys.append(key)
            seen.add(key)

    n_rows = len(model_names)
    n_cols = len(col_keys)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3.5 * n_rows),
                             squeeze=False)

    # Index results
    result_map = {}
    for r in all_results:
        result_map[(r["model_name"], r["video_id"], r["condition"])] = r

    for i, model in enumerate(model_names):
        for j, (vid, cond) in enumerate(col_keys):
            ax = axes[i, j]
            key = (model, vid, cond)
            if key in result_map:
                r = result_map[key]
                title = model if j == 0 else ""
                plot_trajectory(ax, r["emb_2d"], model,
                                r.get("period"), r.get("ratio"), r.get("n_cycles"))
            else:
                ax.set_visible(False)

            # Column headers
            if i == 0:
                ef_label = f"EF={vid}" if not vid.startswith("0X") else vid[:8]
                cond_label = cond.replace("_", " ").title() if cond != "clean" else "Clean"
                ax.set_xlabel(f"{cond_label}", fontsize=8)

        # Row labels
        axes[i, 0].set_ylabel(model, fontsize=10, fontweight="bold")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


# --- Main ---


def main():
    parser = argparse.ArgumentParser(description="Cardiac Cycle Trajectory Analysis")
    # Videos
    parser.add_argument("--videos", nargs="+", default=["0X18610C37F29E80E4", "0X6767F4359E20EE24"],
                        help="EchoNet-Dynamic video IDs (filenames without .avi)")
    parser.add_argument("--video_paths", nargs="+", default=None,
                        help="Override: direct paths to video files")
    parser.add_argument("--echonet_root", default=ECHONET_ROOT)
    # Models
    add_model_args(parser)
    # Perturbation
    parser.add_argument("--perturb", nargs=2, default=None, metavar=("TYPE", "SEVERITY"),
                        help="Perturbation type and severity (e.g., depth_attenuation severe)")
    # Options
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--max_frames", type=int, default=300,
                        help="Max frames to process per video (for speed)")
    parser.add_argument("--stride", type=int, default=2,
                        help="Temporal stride for frame sampling within each clip window")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default="scripts/neurips/samples/cardiac_trajectory.png")
    parser.add_argument("--output_npz", default="scripts/neurips/samples/cardiac_trajectory.npz",
                        help="Save raw embeddings for downstream analysis")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Resolve video paths
    if args.video_paths:
        video_paths = {os.path.basename(p).replace(".avi", ""): p for p in args.video_paths}
    else:
        video_paths = {}
        for vid in args.videos:
            path = os.path.join(args.echonet_root, "Videos", f"{vid}.avi")
            if not os.path.exists(path):
                print(f"WARNING: Video not found: {path}")
                continue
            video_paths[vid] = path

    if not video_paths:
        print("No videos found!")
        return

    # Get models
    models = args.models if args.models else ["JEPA-L-pt50", "BYOL-L-pt50", "MAE-L-pt50"]
    model_cfgs = get_models(models)
    print(f"Models: {list(model_cfgs.keys())}")
    print(f"Videos: {list(video_paths.keys())}")

    # Conditions to run
    conditions = ["clean"]
    if args.perturb:
        conditions.append(f"{args.perturb[0]}/{args.perturb[1]}")

    # Load all videos once
    print("\nLoading videos...")
    video_data = {}
    for vid, path in video_paths.items():
        frames = load_full_video(path, args.resolution)
        if frames.shape[0] > args.max_frames:
            frames = frames[:args.max_frames]
        video_data[vid] = frames
        print(f"  {vid}: {frames.shape[0]} frames")

    # Prepare perturbed versions
    perturbed_data = {}
    if args.perturb:
        ptype, severity = args.perturb
        cond_key = f"{ptype}/{severity}"
        for vid, frames in video_data.items():
            mask = create_scan_mask(frames[0])  # [C, H, W] -> mask
            perturbed = apply_perturbation(
                frames.permute(1, 0, 2, 3),  # [C, T, H, W]
                ptype, severity, scan_mask=mask, seed=42,
                transducer_pos=TRANSDUCER_PRESETS["standard"],
            )
            perturbed_data[(vid, cond_key)] = perturbed.permute(1, 0, 2, 3)  # back to [T, C, H, W]
            print(f"  Perturbed {vid}: {ptype}/{severity}")

    # Extract embeddings for all models × videos × conditions
    all_embeddings = {}  # (model, vid, cond) -> [T, D]
    for model_name, model_cfg in model_cfgs.items():
        print(f"\nLoading {model_name}...")
        encoder, embed_dim, is_videomae = load_encoder(model_cfg, device, args.resolution)
        print(f"  embed_dim={embed_dim}, videomae={is_videomae}")

        for vid in video_data:
            for cond in conditions:
                print(f"  Extracting: {model_name} / {vid[:12]} / {cond}")

                if cond == "clean":
                    frames = video_data[vid]
                else:
                    frames = perturbed_data[(vid, cond)]

                norm_frames = normalize_frames(frames)
                emb = extract_per_frame_embeddings(
                    encoder, norm_frames, device, is_videomae,
                    clip_len=16, stride=args.stride, resolution=args.resolution,
                )
                all_embeddings[(model_name, vid, cond)] = emb
                print(f"    -> {emb.shape}")

        # Free GPU memory between models
        del encoder
        torch.cuda.empty_cache()

    # Per-model PCA (each model gets its own projection to maximize within-model variance)
    print("\nFitting per-model PCA...")
    from sklearn.decomposition import PCA

    # Group embeddings by model
    model_names_seen = []
    for (model_name, vid, cond) in all_embeddings:
        if model_name not in model_names_seen:
            model_names_seen.append(model_name)

    model_pcas = {}
    for mname in model_names_seen:
        model_embs = [emb for (m, v, c), emb in all_embeddings.items() if m == mname]
        all_emb_flat = np.concatenate(model_embs, axis=0)
        pca = PCA(n_components=2, random_state=42)
        pca.fit(all_emb_flat)
        model_pcas[mname] = pca
        print(f"  {mname}: explained variance = {pca.explained_variance_ratio_}")

    # Project and compute metrics
    all_results = []
    npz_data = {}
    for (model_name, vid, cond), emb in all_embeddings.items():
        pca = model_pcas[model_name]
        emb_2d = pca.transform(emb)
        period = estimate_cycle_period(emb_2d)
        ratio, n_cycles = (None, 0)
        if period:
            ratio, n_cycles = compute_cycle_variance_ratio(emb_2d, period)

        all_results.append({
            "model_name": model_name,
            "video_id": vid,
            "condition": cond,
            "emb_2d": emb_2d,
            "period": period,
            "ratio": ratio,
            "n_cycles": n_cycles,
        })

        tag = f"{model_name}_{vid[:12]}_{cond.replace('/', '_')}"
        npz_data[f"{tag}_emb"] = emb
        npz_data[f"{tag}_pca"] = emb_2d

        period_str = f"period={period}" if period else "no period detected"
        ratio_str = f"CVR={ratio:.2f}" if ratio else "N/A"
        print(f"  {model_name} / {vid[:12]} / {cond}: {period_str}, {ratio_str} ({n_cycles} cycles)")

    # Save raw data
    np.savez_compressed(args.output_npz, **npz_data)
    print(f"\nSaved embeddings: {args.output_npz}")

    # Generate figure
    make_figure(all_results, args.output)

    # Print summary table
    print(f"\n{'Model':<18} {'Video':<15} {'Condition':<20} {'Period':<8} {'CVR':<8} {'Cycles'}")
    print("-" * 85)
    for r in all_results:
        p = str(r["period"]) if r["period"] else "-"
        cvr = f"{r['ratio']:.2f}" if r["ratio"] else "-"
        print(f"{r['model_name']:<18} {r['video_id'][:12]:<15} {r['condition']:<20} {p:<8} {cvr:<8} {r['n_cycles']}")


if __name__ == "__main__":
    main()
