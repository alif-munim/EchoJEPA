"""Anomaly detection via representation distance (zero-shot OOD).

Instead of prediction error, measures how far each sample's encoder
representation is from a "normal" reference distribution.

Two modes:
  **Mean-pooled** (default): Pool 1568 tokens → 1408-dim vector, then Mahalanobis/cosine.
  **Token-level** (--token_level): Keep full [1568, 1408] token grid, compute per-token
    distances from normal reference, aggregate via max/percentile to detect spatially
    localized abnormalities (e.g., thick septum in HCM, dilated chambers in DCM).

Usage:
    # Mean-pooled (original)
    python -m evals.forward_prediction.anomaly_repr \
        --checkpoint checkpoints/vitg-384.pt \
        --csv experiments/nature_medicine/uhn/probe_csvs/disease_hcm/test.csv \
        --output results/anomaly_repr/disease_hcm.csv \
        --device cuda:0

    # Token-level (preserves spatial structure)
    python -m evals.forward_prediction.anomaly_repr \
        --checkpoint checkpoints/vitg-384.pt \
        --csv experiments/nature_medicine/uhn/probe_csvs/disease_hcm/test.csv \
        --output results/anomaly_repr_token/disease_hcm.csv \
        --device cuda:0 --token_level
"""

import argparse
import csv
import json
import logging
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evals.video_classification_frozen.models import init_module
from evals.video_classification_frozen.utils import make_transforms
from src.datasets.video_dataset import VideoDataset
import src.models.vision_transformer as vit

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Model registry: model_key -> (module_name, default_checkpoint, model_kwargs, wrapper_kwargs, normalization)
MODEL_REGISTRY = {
    "echojepa": {
        "loader": "jepa",  # use custom JEPA loader (needs target_encoder key)
    },
    "echoprime": {
        "module_name": "evals.video_classification_frozen.modelcustom.echo_prime_encoder",
        "checkpoint": None,
        "model_kwargs": {},
        "wrapper_kwargs": {
            "echo_prime_root": str(Path(__file__).resolve().parents[1] / "video_classification_frozen" / "modelcustom" / "EchoPrime"),
            "force_fp32": True,
            "bin_size": 50,
        },
        "normalization": [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],  # EchoPrime handles its own norm
    },
    "panecho": {
        "module_name": "evals.video_classification_frozen.modelcustom.panecho_encoder",
        "checkpoint": None,
        "model_kwargs": {},
        "wrapper_kwargs": {},
        "normalization": None,  # uses default ImageNet
    },
    "videomae": {
        "module_name": "evals.video_classification_frozen.modelcustom.videomae_encoder",
        "checkpoint": "checkpoints/videomae-ep163.pth",
        "model_kwargs": {"encoder": {"model_name": "vit_large_patch16_224", "tubelet_size": 2}},
        "wrapper_kwargs": {},
        "normalization": None,
    },
    "echofm": {
        "module_name": "evals.video_classification_frozen.modelcustom.echofm_encoder",
        "checkpoint": str(Path(__file__).resolve().parents[1] / "video_classification_frozen" / "modelcustom" / "EchoFM" / "weights" / "EchoFM" / "EchoFM_latest.pth"),
        "model_kwargs": {"encoder": {"num_frames": 32, "embed_dim": 1024, "t_patch_size": 4}},
        "wrapper_kwargs": {},
        "normalization": None,
    },
}


def load_encoder(checkpoint_path, img_size=224, num_frames=16, patch_size=16,
                 tubelet_size=2, model_name="vit_giant_xformers",
                 use_rope=True, uniform_power=True, device="cuda"):
    """Load JEPA encoder (no predictor/target_encoder needed)."""
    logger.info(f"Loading JEPA encoder from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    encoder = vit.__dict__[model_name](
        img_size=img_size, num_frames=num_frames, patch_size=patch_size,
        tubelet_size=tubelet_size, uniform_power=uniform_power, use_rope=use_rope,
    )
    sd = {k.replace("module.", "").replace("backbone.", ""): v
          for k, v in ckpt["target_encoder"].items()}
    msg = encoder.load_state_dict(sd, strict=False)
    logger.info(f"Loaded encoder: {msg}")

    encoder.eval().requires_grad_(False).to(device)
    n_params = sum(p.numel() for p in encoder.parameters()) / 1e6
    logger.info(f"Encoder: {n_params:.0f}M params, embed_dim={encoder.embed_dim}")
    del ckpt
    return encoder


def load_model(model_key, checkpoint=None, resolution=224, frames_per_clip=16, device="cuda", **jepa_kwargs):
    """Load any supported model by key."""
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_key}. Available: {list(MODEL_REGISTRY.keys())}")

    cfg = MODEL_REGISTRY[model_key]

    if cfg.get("loader") == "jepa":
        return load_encoder(checkpoint, img_size=resolution, num_frames=frames_per_clip,
                            device=device, **jepa_kwargs)

    ckpt = checkpoint or cfg.get("checkpoint")
    encoder = init_module(
        module_name=cfg["module_name"],
        device=device,
        frames_per_clip=frames_per_clip,
        resolution=resolution,
        checkpoint=ckpt,
        model_kwargs=cfg.get("model_kwargs", {}),
        wrapper_kwargs=cfg.get("wrapper_kwargs", {}),
    )
    n_params = sum(p.numel() for p in encoder.parameters()) / 1e6
    logger.info(f"{model_key}: {n_params:.0f}M params, embed_dim={encoder.embed_dim}")
    return encoder, cfg.get("normalization")


def extract_representations(encoder, loader, device="cuda", max_batches=None,
                            token_level=False):
    """Extract encoder representations for all clips.

    Args:
        token_level: If True, return full [N, num_tokens, D] token tensors.
                     If False, return mean-pooled [N, D] vectors.

    Returns:
        reprs: [N, D] or [N, T, D] numpy array
        labels: [N] numpy array
        paths: list of N strings
    """
    all_reprs, all_labels, all_paths = [], [], []

    for batch_idx, batch in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break

        buffer, labels, _clip_indices, paths = batch
        clips = buffer[0][0].to(device)  # [B, C, T, H, W]

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            tokens = encoder(clips)  # [B, N, D]
            if isinstance(tokens, list):
                tokens = tokens[-1]
            tokens = tokens.float()

            if token_level:
                reprs = tokens  # [B, N, D] — keep full spatial structure
            else:
                reprs = tokens.mean(dim=1)  # [B, D]

        all_reprs.append(reprs.cpu().numpy())
        all_labels.append(labels.numpy())
        all_paths.extend(paths)

        if (batch_idx + 1) % 100 == 0:
            logger.info(f"  Batch {batch_idx + 1}/{len(loader)}")

    return np.concatenate(all_reprs), np.concatenate(all_labels), all_paths


def mahalanobis_scores(reprs, ref_reprs, regularize=1e-4):
    """Compute Mahalanobis distance from reference distribution.

    Args:
        reprs: [N, D] — test samples
        ref_reprs: [M, D] — reference (normal) samples
        regularize: regularization for covariance matrix

    Returns:
        scores: [N] — Mahalanobis distances
    """
    mu = ref_reprs.mean(axis=0)  # [D]
    centered = ref_reprs - mu  # [M, D]

    # Use PCA to handle high-dimensional covariance (D=1408)
    # Keep top-k components that explain 95% variance
    D = reprs.shape[1]
    logger.info(f"Computing covariance: {ref_reprs.shape[0]} samples x {D} dims")

    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    cumvar = np.cumsum(S ** 2) / np.sum(S ** 2)
    k = min(np.searchsorted(cumvar, 0.95) + 1, len(S), 512)
    logger.info(f"Using top {k} PCA components (95% variance)")

    # Project to PCA space
    V_k = Vt[:k].T  # [D, k]
    S_k = S[:k]

    # Mahalanobis in PCA space: d = sqrt(sum((z_i / sigma_i)^2))
    test_centered = reprs - mu  # [N, D]
    test_proj = test_centered @ V_k  # [N, k]
    scores = np.sqrt(np.sum((test_proj / (S_k / np.sqrt(ref_reprs.shape[0] - 1) + regularize)) ** 2, axis=1))

    return scores


def cosine_scores(reprs, ref_reprs):
    """Compute negative mean cosine similarity to reference distribution.

    Higher = more anomalous (less similar to reference).
    """
    mu = ref_reprs.mean(axis=0)
    mu_norm = mu / (np.linalg.norm(mu) + 1e-8)

    # Cosine similarity to reference centroid
    norms = np.linalg.norm(reprs, axis=1, keepdims=True) + 1e-8
    reprs_norm = reprs / norms
    cos_sim = reprs_norm @ mu_norm  # [N]
    return -cos_sim  # negate: higher = more anomalous


def token_level_scores(token_reprs, ref_token_reprs, top_k=50):
    """Compute per-token anomaly scores preserving spatial structure.

    Instead of collapsing tokens to a single vector, compute per-token
    distances from the normal reference distribution and aggregate via
    max/percentile to detect spatially localized abnormalities.

    Args:
        token_reprs: [N, T, D] — test samples (T=1568 tokens, D=1408)
        ref_token_reprs: [M, T, D] — reference (normal) samples
        top_k: number of most-anomalous tokens to average for top-k score

    Returns:
        dict of {method_name: [N] scores}
    """
    N, T, D = token_reprs.shape
    M = ref_token_reprs.shape[0]
    logger.info(f"Token-level scoring: {N} test × {T} tokens × {D} dims, {M} ref samples")

    # Per-token reference centroid: [T, D]
    ref_centroid = ref_token_reprs.mean(axis=0)  # [T, D]

    # Per-token L2 distance from reference centroid: [N, T]
    diff = token_reprs - ref_centroid[None, :, :]  # [N, T, D]
    per_token_l2 = np.linalg.norm(diff, axis=2)  # [N, T]

    # Per-token cosine distance from reference centroid: [N, T]
    ref_norms = np.linalg.norm(ref_centroid, axis=1, keepdims=True) + 1e-8  # [T, 1]
    ref_normed = ref_centroid / ref_norms  # [T, D]
    test_norms = np.linalg.norm(token_reprs, axis=2, keepdims=True) + 1e-8  # [N, T, 1]
    test_normed = token_reprs / test_norms  # [N, T, D]
    # Batched dot product: [N, T]
    per_token_cos = np.sum(test_normed * ref_normed[None, :, :], axis=2)

    # Per-token Mahalanobis via PCA on each token position is too expensive
    # (1568 separate SVDs). Instead, use token-pooled PCA:
    # Flatten ref to [M*T, D], fit one PCA, project all tokens
    logger.info("Fitting shared PCA across all token positions...")
    ref_flat = ref_token_reprs.reshape(-1, D)  # [M*T, D]
    # Subsample for SVD (M*T can be huge)
    max_svd_samples = 50000
    if ref_flat.shape[0] > max_svd_samples:
        rng = np.random.RandomState(42)
        svd_idx = rng.choice(ref_flat.shape[0], max_svd_samples, replace=False)
        ref_subsample = ref_flat[svd_idx]
    else:
        ref_subsample = ref_flat
    mu_all = ref_subsample.mean(axis=0)
    centered = ref_subsample - mu_all
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    cumvar = np.cumsum(S ** 2) / np.sum(S ** 2)
    k = min(np.searchsorted(cumvar, 0.95) + 1, len(S), 256)
    logger.info(f"Shared PCA: {k} components (95% variance)")

    V_k = Vt[:k].T  # [D, k]
    S_k = S[:k]
    sigma_k = S_k / np.sqrt(ref_subsample.shape[0] - 1) + 1e-4

    # Project test tokens: [N, T, D] → [N, T, k] → per-token Mahalanobis
    test_flat = token_reprs.reshape(-1, D)  # [N*T, D]
    test_centered = test_flat - mu_all
    test_proj = test_centered @ V_k  # [N*T, k]
    per_token_mahal = np.sqrt(np.sum((test_proj / sigma_k) ** 2, axis=1))  # [N*T]
    per_token_mahal = per_token_mahal.reshape(N, T)  # [N, T]

    # Aggregate per-token scores into per-sample scores
    k_actual = min(top_k, T)
    scores = {}

    for name, per_token in [("l2", per_token_l2), ("cosine", 1.0 - per_token_cos), ("mahal", per_token_mahal)]:
        scores[f"token_{name}_max"] = per_token.max(axis=1)  # [N]
        scores[f"token_{name}_p95"] = np.percentile(per_token, 95, axis=1)  # [N]
        scores[f"token_{name}_p99"] = np.percentile(per_token, 99, axis=1)  # [N]
        scores[f"token_{name}_mean"] = per_token.mean(axis=1)  # [N]
        # Top-k mean: average of k most anomalous tokens
        topk_idx = np.argpartition(per_token, -k_actual, axis=1)[:, -k_actual:]
        topk_vals = np.take_along_axis(per_token, topk_idx, axis=1)
        scores[f"token_{name}_topk{k_actual}"] = topk_vals.mean(axis=1)  # [N]

    return scores


def run_anomaly_repr(args):
    """Main representation-distance anomaly detection pipeline."""
    model_key = getattr(args, "model", "echojepa")
    custom_norm = None

    if model_key == "echojepa":
        encoder = load_encoder(
            args.checkpoint, img_size=args.resolution, num_frames=args.frames_per_clip,
            patch_size=args.patch_size, tubelet_size=args.tubelet_size,
            model_name=args.model_name, use_rope=args.use_rope,
            uniform_power=args.uniform_power, device=args.device,
        )
    else:
        encoder, custom_norm = load_model(
            model_key, checkpoint=args.checkpoint, resolution=args.resolution,
            frames_per_clip=args.frames_per_clip, device=args.device,
        )

    token_level = getattr(args, "token_level", False)

    # Token-level mode uses ~8.8 MB per clip ([1568, 1408] × float32)
    # Cap at 2000 clips to stay under ~18 GB RAM
    if token_level:
        max_token_samples = 2000
        if args.max_samples is None or args.max_samples > max_token_samples:
            logger.warning(f"Token-level mode: capping samples to {max_token_samples} for memory "
                           f"(was {args.max_samples})")
            args.max_samples = max_token_samples

    # Use model-specific normalization if provided
    norm_kwargs = {}
    if custom_norm is not None:
        norm_kwargs["normalize"] = (custom_norm[0], custom_norm[1])

    transform = make_transforms(training=False, crop_size=args.resolution, **norm_kwargs)
    dataset = VideoDataset(
        data_paths=[args.csv], frames_per_clip=args.frames_per_clip,
        frame_step=args.frame_step, num_clips=1,
        random_clip_sampling=False, allow_clip_overlap=False, transform=transform,
    )
    if args.max_samples and args.max_samples < len(dataset):
        rng = np.random.RandomState(42)
        indices = rng.choice(len(dataset), args.max_samples, replace=False)
        dataset = Subset(dataset, indices)
        logger.info(f"Subsampled to {args.max_samples} clips")

    batch_size = min(args.batch_size, 4) if token_level else args.batch_size
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
    )

    # Extract representations
    logger.info(f"Extracting representations for {len(dataset)} clips (token_level={token_level})...")
    reprs, labels, paths = extract_representations(
        encoder, loader, device=args.device, token_level=token_level,
    )
    logger.info(f"Extracted: {reprs.shape}")

    if token_level:
        return _run_token_level(reprs, labels, paths, args)
    else:
        return _run_mean_pooled(reprs, labels, paths, args)


def _run_mean_pooled(reprs, labels, paths, args):
    """Original mean-pooled anomaly detection pipeline."""
    # --- Study-level aggregation ---
    study_data = defaultdict(lambda: {"reprs": [], "label": None})
    for i, path in enumerate(paths):
        sid = _extract_study_id(path)
        study_data[sid]["reprs"].append(reprs[i])
        study_data[sid]["label"] = int(labels[i])

    study_ids = sorted(study_data.keys())
    study_reprs = np.array([np.mean(study_data[sid]["reprs"], axis=0) for sid in study_ids])
    study_labels = np.array([study_data[sid]["label"] for sid in study_ids])
    study_nclips = [len(study_data[sid]["reprs"]) for sid in study_ids]

    n_pos = int(study_labels.sum())
    n_neg = len(study_labels) - n_pos
    logger.info(f"Study-level: {len(study_ids)} studies ({n_pos} pos, {n_neg} neg)")

    # Reference = negative class (label=0) studies
    ref_mask = study_labels == 0
    ref_reprs = study_reprs[ref_mask]
    logger.info(f"Reference distribution: {ref_reprs.shape[0]} normal studies")

    # Score with multiple methods
    from sklearn.metrics import roc_auc_score

    results = {
        "mode": "mean_pooled",
        "n_clips": len(reprs),
        "n_studies": len(study_ids),
        "n_positive": n_pos,
        "n_negative": n_neg,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    for method_name, score_fn in [
        ("mahalanobis", lambda r, ref: mahalanobis_scores(r, ref)),
        ("cosine", lambda r, ref: cosine_scores(r, ref)),
    ]:
        scores = score_fn(study_reprs, ref_reprs)

        # Binarize labels for AUROC (0 vs non-zero)
        binary_study = (study_labels > 0).astype(int)
        if len(np.unique(binary_study)) == 2:
            auroc = roc_auc_score(binary_study, scores)
            auroc_inv = roc_auc_score(binary_study, -scores)
            best = max(auroc, auroc_inv)
            results[f"{method_name}_auroc"] = float(auroc)
            results[f"{method_name}_auroc_inv"] = float(auroc_inv)
            results[f"{method_name}_best_auroc"] = float(best)
            logger.info(f"{method_name}: AUROC={auroc:.4f} (inv={auroc_inv:.4f}, best={best:.4f})")

        # Also clip-level scores
        clip_scores = score_fn(reprs, ref_reprs[:min(1000, len(ref_reprs))])  # subsample ref for speed
        binary_clip = (labels > 0).astype(int)
        if len(np.unique(binary_clip)) == 2:
            clip_auroc = roc_auc_score(binary_clip, clip_scores)
            clip_auroc_inv = roc_auc_score(binary_clip, -clip_scores)
            clip_best = max(clip_auroc, clip_auroc_inv)
            results[f"{method_name}_clip_auroc"] = float(clip_auroc)
            results[f"{method_name}_clip_best"] = float(clip_best)
            logger.info(f"{method_name} (clip): AUROC={clip_auroc:.4f} (best={clip_best:.4f})")

    # Save study-level CSV
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["study_id", "label", "mahal_score", "cosine_score", "n_clips"])
        mahal_s = mahalanobis_scores(study_reprs, ref_reprs)
        cos_s = cosine_scores(study_reprs, ref_reprs)
        for i, sid in enumerate(study_ids):
            writer.writerow([sid, study_labels[i], f"{mahal_s[i]:.6f}",
                             f"{cos_s[i]:.6f}", study_nclips[i]])
    logger.info(f"Saved study-level scores to {args.output}")

    # Save summary
    summary_path = args.output.replace(".csv", "_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Summary: {summary_path}")

    return results


def _run_token_level(reprs, labels, paths, args):
    """Token-level anomaly detection preserving spatial structure.

    reprs: [N, T, D] where T=1568 tokens, D=1408 dims
    """
    from sklearn.metrics import roc_auc_score

    # --- Study-level aggregation (average token maps across clips) ---
    study_data = defaultdict(lambda: {"reprs": [], "label": None})
    for i, path in enumerate(paths):
        sid = _extract_study_id(path)
        study_data[sid]["reprs"].append(reprs[i])  # [T, D]
        study_data[sid]["label"] = int(labels[i])

    study_ids = sorted(study_data.keys())
    # Average token maps across clips per study: [n_studies, T, D]
    study_reprs = np.array([np.mean(study_data[sid]["reprs"], axis=0) for sid in study_ids])
    study_labels = np.array([study_data[sid]["label"] for sid in study_ids])
    study_nclips = [len(study_data[sid]["reprs"]) for sid in study_ids]

    n_pos = int(study_labels.sum())
    n_neg = len(study_labels) - n_pos
    logger.info(f"Study-level: {len(study_ids)} studies ({n_pos} pos, {n_neg} neg), "
                f"token shape: {study_reprs.shape}")

    # Reference = negative class studies
    ref_mask = study_labels == 0
    ref_reprs = study_reprs[ref_mask]
    logger.info(f"Reference distribution: {ref_reprs.shape[0]} normal studies")

    # Compute token-level scores
    top_k = getattr(args, "top_k", 50)
    all_scores = token_level_scores(study_reprs, ref_reprs, top_k=top_k)

    results = {
        "mode": "token_level",
        "n_clips": len(reprs),
        "n_studies": len(study_ids),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "top_k": top_k,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Evaluate each scoring method
    if len(np.unique(study_labels)) == 2:
        logger.info("--- Token-level AUROC results ---")
        for method_name, scores in sorted(all_scores.items()):
            auroc = roc_auc_score(study_labels, scores)
            auroc_inv = roc_auc_score(study_labels, -scores)
            best = max(auroc, auroc_inv)
            results[f"{method_name}_auroc"] = float(auroc)
            results[f"{method_name}_best_auroc"] = float(best)
            logger.info(f"  {method_name}: AUROC={auroc:.4f} (inv={auroc_inv:.4f}, best={best:.4f})")

    # Also compute clip-level token scores
    logger.info("Computing clip-level token scores...")
    ref_clip_reprs = reprs[labels == 0]
    max_ref = min(500, len(ref_clip_reprs))  # subsample ref for memory
    if max_ref < len(ref_clip_reprs):
        rng = np.random.RandomState(42)
        ref_idx = rng.choice(len(ref_clip_reprs), max_ref, replace=False)
        ref_clip_reprs = ref_clip_reprs[ref_idx]
    clip_scores = token_level_scores(reprs, ref_clip_reprs, top_k=top_k)

    if len(np.unique(labels)) == 2:
        logger.info("--- Clip-level token AUROC results ---")
        for method_name, scores in sorted(clip_scores.items()):
            auroc = roc_auc_score(labels, scores)
            auroc_inv = roc_auc_score(labels, -scores)
            best = max(auroc, auroc_inv)
            results[f"clip_{method_name}_auroc"] = float(auroc)
            results[f"clip_{method_name}_best_auroc"] = float(best)
            logger.info(f"  clip_{method_name}: best={best:.4f}")

    # Save study-level CSV with best scoring methods
    best_method = max(
        [(k, v) for k, v in results.items() if k.endswith("_best_auroc") and not k.startswith("clip_")],
        key=lambda x: x[1],
        default=("none", 0),
    )
    logger.info(f"Best study-level method: {best_method[0]} = {best_method[1]:.4f}")

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        score_cols = sorted(all_scores.keys())
        writer.writerow(["study_id", "label", "n_clips"] + score_cols)
        for i, sid in enumerate(study_ids):
            row = [sid, study_labels[i], study_nclips[i]]
            row += [f"{all_scores[col][i]:.6f}" for col in score_cols]
            writer.writerow(row)
    logger.info(f"Saved study-level token scores to {args.output}")

    # Save summary
    summary_path = args.output.replace(".csv", "_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Summary: {summary_path}")

    return results


def _extract_study_id(path):
    match = re.search(r"/s(\d+)/\d+_\d+\.mp4$", path)
    if match:
        return match.group(1)
    return os.path.basename(os.path.dirname(path))


def main():
    parser = argparse.ArgumentParser(description="Anomaly detection via representation distance")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint (required for echojepa/videomae/echofm, ignored for panecho)")
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--model", type=str, default="echojepa",
                        choices=list(MODEL_REGISTRY.keys()),
                        help="Model to use for representation extraction")

    parser.add_argument("--model_name", type=str, default="vit_giant_xformers")
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--frames_per_clip", type=int, default=16)
    parser.add_argument("--frame_step", type=int, default=2)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--tubelet_size", type=int, default=2)
    parser.add_argument("--use_rope", action="store_true", default=True)
    parser.add_argument("--uniform_power", action="store_true", default=True)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_samples", type=int, default=None)

    parser.add_argument("--token_level", action="store_true", default=False,
                        help="Use full token-level representations instead of mean-pooling")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Number of most-anomalous tokens to average for top-k score")

    args = parser.parse_args()
    run_anomaly_repr(args)


if __name__ == "__main__":
    main()
