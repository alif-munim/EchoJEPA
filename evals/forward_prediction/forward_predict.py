"""Temporal forward prediction: predict future frames from past frames.

Given first T/2 frames, predict representations of remaining T/2 frames.
Measures the JEPA world model's ability to anticipate future cardiac states.

Key outputs:
- Overall forward prediction error per study
- Per-frame error curve (error vs frames-into-future)
- Correlation of forward prediction error with disease labels

This is JEPA-unique — EchoPrime, PanEcho, and MAE have no predictor network.

Usage:
    python -m evals.forward_prediction.forward_predict \
        --checkpoint checkpoints/vitg-384.pt \
        --csv data/csv/nature_medicine/uhn/disease_hcm.csv \
        --output results/forward_prediction/disease_hcm.csv \
        --device cuda:0
"""

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evals.forward_prediction.masking import FutureFrameMask
from evals.forward_prediction.models import load_jepa_models
from evals.video_classification_frozen.utils import make_transforms
from src.datasets.video_dataset import VideoDataset
from src.masks.utils import apply_masks

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def compute_forward_prediction_error(
    clip,
    encoder,
    predictor,
    target_encoder,
    mask_generator,
    device="cuda",
):
    """Compute forward prediction error: predict future from past.

    Args:
        clip: [B, C, T, H, W] video tensor
        encoder, predictor, target_encoder: frozen JEPA models
        mask_generator: FutureFrameMask instance
        device: computation device

    Returns:
        total_error: [B] mean prediction error per sample
        per_frame_errors: [B, num_pred_frames] error per predicted frame
    """
    clip = clip.to(device)
    B = clip.shape[0]

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        # Target representations
        h = target_encoder(clip)
        if isinstance(h, list):
            h = h[-1]
        h = F.layer_norm(h, (h.size(-1),))

        # Generate temporal split masks
        masks_enc, masks_pred = mask_generator(B)
        masks_enc = masks_enc.to(device)
        masks_pred = masks_pred.to(device)

        # Encode context (past frames)
        z = encoder(clip, [masks_enc])

        # Predict target (future frames)
        pred = predictor(z, [masks_enc], [masks_pred])

        # Ground-truth future tokens
        h_target = apply_masks(h, [masks_pred])

        # Overall L1 error per sample
        total_error = torch.mean(torch.abs(pred - h_target), dim=(1, 2))  # [B]

        # Per-frame error
        frame_indices = mask_generator.per_frame_indices()
        spatial_tokens = mask_generator.H * mask_generator.W
        per_frame_errors = []

        for frame_idx_offset, _ in enumerate(frame_indices):
            start = frame_idx_offset * spatial_tokens
            end = start + spatial_tokens
            frame_pred = pred[:, start:end, :]
            frame_target = h_target[:, start:end, :]
            frame_error = torch.mean(torch.abs(frame_pred - frame_target), dim=(1, 2))  # [B]
            per_frame_errors.append(frame_error)

        per_frame_errors = torch.stack(per_frame_errors, dim=1)  # [B, num_pred_frames]

    return total_error.cpu(), per_frame_errors.cpu()


def run_forward_prediction(args):
    """Main forward prediction pipeline."""
    # Load models
    encoder, predictor, target_encoder = load_jepa_models(
        checkpoint_path=args.checkpoint,
        img_size=args.resolution,
        num_frames=args.frames_per_clip,
        patch_size=args.patch_size,
        tubelet_size=args.tubelet_size,
        model_name=args.model_name,
        predictor_depth=args.predictor_depth,
        predictor_embed_dim=args.predictor_embed_dim,
        use_rope=args.use_rope,
        uniform_power=args.uniform_power,
        device=args.device,
    )

    # Temporal split mask
    mask_gen = FutureFrameMask(
        num_frames=args.frames_per_clip,
        img_size=args.resolution,
        patch_size=args.patch_size,
        tubelet_size=args.tubelet_size,
        context_ratio=args.context_ratio,
    )

    # Load dataset with eval transforms (center crop, normalize)
    transform = make_transforms(training=False, crop_size=args.resolution)
    dataset = VideoDataset(
        data_paths=[args.csv],
        frames_per_clip=args.frames_per_clip,
        frame_step=args.frame_step,
        num_clips=1,
        random_clip_sampling=False,
        allow_clip_overlap=False,
        transform=transform,
    )
    if args.max_samples and args.max_samples < len(dataset):
        rng = np.random.RandomState(42)
        indices = rng.choice(len(dataset), args.max_samples, replace=False)
        dataset = Subset(dataset, indices)
        logger.info(f"Subsampled to {args.max_samples} clips")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    all_paths = []
    all_labels = []
    all_total_errors = []
    all_per_frame_errors = []

    logger.info(f"Forward prediction for {len(dataset)} clips (context_ratio={args.context_ratio})...")
    for batch_idx, batch in enumerate(loader):
        # VideoDataset returns (buffer, label, clip_indices, sample_uri)
        # buffer structure: list[num_clips][num_views] -> tensor [B, C, T, H, W]
        buffer, labels, _clip_indices, paths = batch
        clips = buffer[0][0]  # first clip, first view: [B, C, T, H, W]

        total_err, per_frame_err = compute_forward_prediction_error(
            clips, encoder, predictor, target_encoder,
            mask_gen, device=args.device,
        )

        all_total_errors.append(total_err.numpy())
        all_per_frame_errors.append(per_frame_err.numpy())
        all_labels.append(labels.numpy() if isinstance(labels, torch.Tensor) else np.array(labels))
        all_paths.extend(paths)

        if (batch_idx + 1) % 50 == 0:
            logger.info(f"  Batch {batch_idx + 1}/{len(loader)}")

    all_total_errors = np.concatenate(all_total_errors)
    all_per_frame_errors = np.concatenate(all_per_frame_errors, axis=0)
    all_labels = np.concatenate(all_labels)

    # Save per-sample results
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        n_pred_frames = all_per_frame_errors.shape[1]
        header = ["path", "label", "total_error"] + [f"frame_{i}_error" for i in range(n_pred_frames)]
        writer.writerow(header)
        for i in range(len(all_total_errors)):
            row = [all_paths[i], int(all_labels[i]), f"{all_total_errors[i]:.6f}"]
            row += [f"{all_per_frame_errors[i, j]:.6f}" for j in range(n_pred_frames)]
            writer.writerow(row)
    logger.info(f"Saved {len(all_total_errors)} predictions to {args.output}")

    # Summary stats
    logger.info(f"Mean forward prediction error: {all_total_errors.mean():.4f} +/- {all_total_errors.std():.4f}")
    logger.info(f"Per-frame errors (mean): {all_per_frame_errors.mean(axis=0)}")

    # Save summary
    summary_path = args.output.replace(".csv", "_summary.json")
    summary = {
        "n_samples": len(all_total_errors),
        "context_ratio": args.context_ratio,
        "mean_error": float(all_total_errors.mean()),
        "std_error": float(all_total_errors.std()),
        "per_frame_mean": all_per_frame_errors.mean(axis=0).tolist(),
        "per_frame_std": all_per_frame_errors.std(axis=0).tolist(),
    }

    # AUROC if binary labels
    unique_labels = np.unique(all_labels)
    if len(unique_labels) == 2:
        from sklearn.metrics import roc_auc_score

        auroc = roc_auc_score(all_labels, all_total_errors)
        auroc_inv = roc_auc_score(all_labels, -all_total_errors)
        summary["auroc"] = float(auroc)
        summary["auroc_inverted"] = float(auroc_inv)
        summary["best_auroc"] = float(max(auroc, auroc_inv))
        logger.info(f"Forward prediction AUROC: {auroc:.4f} (inverted: {auroc_inv:.4f})")

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary to {summary_path}")

    return all_total_errors, all_per_frame_errors, all_labels


def main():
    parser = argparse.ArgumentParser(description="JEPA temporal forward prediction")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)

    # Model config
    parser.add_argument("--model_name", type=str, default="vit_giant_xformers")
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--frames_per_clip", type=int, default=16)
    parser.add_argument("--frame_step", type=int, default=2)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--tubelet_size", type=int, default=2)
    parser.add_argument("--predictor_depth", type=int, default=12)
    parser.add_argument("--predictor_embed_dim", type=int, default=384)
    parser.add_argument("--use_rope", action="store_true", default=True)
    parser.add_argument("--uniform_power", action="store_true", default=True)

    # Forward prediction config
    parser.add_argument("--context_ratio", type=float, default=0.5, help="Fraction of frames used as context")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_samples", type=int, default=None, help="Max clips to process (subsample)")

    args = parser.parse_args()
    run_forward_prediction(args)


if __name__ == "__main__":
    main()
