"""Anomaly detection via JEPA prediction error (zero-shot).

Computes prediction error as anomaly score:
  1. Encode visible tokens with frozen encoder
  2. Predict masked token representations with frozen predictor
  3. Compare predicted vs actual (from target encoder) representations
  4. High error = the world model is surprised = likely abnormal

No training or labels needed — pure zero-shot anomaly detection.

Usage:
    python -m evals.forward_prediction.eval \
        --checkpoint checkpoints/vitg-384.pt \
        --csv data/csv/nature_medicine/uhn/disease_hcm.csv \
        --output results/anomaly_detection/disease_hcm.csv \
        --num_masks 10 --device cuda:0
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

from evals.forward_prediction.masking import RandomBlockMask
from evals.forward_prediction.models import load_jepa_models
from evals.video_classification_frozen.utils import make_transforms
from src.datasets.video_dataset import VideoDataset
from src.masks.utils import apply_masks

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def compute_prediction_error(
    clip,
    encoder,
    predictor,
    target_encoder,
    mask_generator,
    num_masks=10,
    device="cuda",
):
    """Compute average JEPA prediction error across multiple random masks.

    Args:
        clip: [B, C, T, H, W] video tensor
        encoder, predictor, target_encoder: frozen JEPA models
        mask_generator: RandomBlockMask instance
        num_masks: number of random masks to average over
        device: computation device

    Returns:
        errors: [B] tensor of mean prediction errors per sample
    """
    clip = clip.to(device)
    B = clip.shape[0]
    all_errors = []

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        # Target representations (compute once, reuse across masks)
        h = target_encoder(clip)
        if isinstance(h, list):
            h = h[-1]  # use last layer if multi-layer output
        h = F.layer_norm(h, (h.size(-1),))

        for mask_idx in range(num_masks):
            masks_enc, masks_pred = mask_generator(B, seed=mask_idx)
            masks_enc = masks_enc.to(device)
            masks_pred = masks_pred.to(device)

            # Encode context tokens
            z = encoder(clip, [masks_enc])

            # Predict target representations
            pred = predictor(z, [masks_enc], [masks_pred])

            # Extract ground-truth target tokens
            h_target = apply_masks(h, [masks_pred])

            # L1 prediction error per sample
            error = torch.mean(torch.abs(pred - h_target), dim=(1, 2))  # [B]
            all_errors.append(error)

    # Average across masks: [num_masks, B] -> [B]
    errors = torch.stack(all_errors, dim=0).mean(dim=0)
    return errors.cpu()


def run_anomaly_detection(args):
    """Main anomaly detection pipeline."""
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

    # Setup mask generator (training-time masking config)
    mask_gen = RandomBlockMask(
        num_frames=args.frames_per_clip,
        img_size=args.resolution,
        patch_size=args.patch_size,
        tubelet_size=args.tubelet_size,
        spatial_scale=(0.15, 0.15),
        temporal_scale=(1.0, 1.0),
        aspect_ratio=(0.75, 1.5),
        num_blocks=8,
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

    # Run anomaly scoring
    all_paths = []
    all_labels = []
    all_errors = []

    logger.info(f"Computing anomaly scores for {len(dataset)} clips with {args.num_masks} masks each...")
    for batch_idx, batch in enumerate(loader):
        # VideoDataset returns (buffer, label, clip_indices, sample_uri)
        # buffer structure: list[num_clips][num_views] -> tensor [B, C, T, H, W]
        buffer, labels, _clip_indices, paths = batch
        clips = buffer[0][0]  # first clip, first view: [B, C, T, H, W]

        errors = compute_prediction_error(
            clips, encoder, predictor, target_encoder,
            mask_gen, num_masks=args.num_masks, device=args.device,
        )

        all_errors.append(errors.numpy())
        all_labels.append(labels.numpy() if isinstance(labels, torch.Tensor) else np.array(labels))
        all_paths.extend(paths)

        if (batch_idx + 1) % 50 == 0:
            logger.info(f"  Batch {batch_idx + 1}/{len(loader)}")

    all_errors = np.concatenate(all_errors)
    all_labels = np.concatenate(all_labels)

    # Save results
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label", "anomaly_score"])
        for path, label, error in zip(all_paths, all_labels, all_errors):
            writer.writerow([path, int(label), f"{error:.6f}"])
    logger.info(f"Saved {len(all_errors)} anomaly scores to {args.output}")

    # Compute clip-level AUROC
    unique_labels = np.unique(all_labels)
    results = {"n_clips": len(all_errors)}
    if len(unique_labels) == 2:
        from sklearn.metrics import roc_auc_score

        auroc = roc_auc_score(all_labels, all_errors)
        auroc_inv = roc_auc_score(all_labels, -all_errors)
        results["clip_auroc"] = float(auroc)
        results["clip_auroc_inv"] = float(auroc_inv)
        results["clip_best_auroc"] = float(max(auroc, auroc_inv))
        logger.info(f"Clip-level AUROC: {auroc:.4f} (inverted: {auroc_inv:.4f}, best: {max(auroc, auroc_inv):.4f})")

    # --- Study-level aggregation ---
    study_scores = defaultdict(list)
    study_labels = {}
    for path, label, error in zip(all_paths, all_labels, all_errors):
        sid = _extract_study_id(path)
        study_scores[sid].append(float(error))
        study_labels[sid] = int(label)

    study_ids = sorted(study_scores.keys())
    study_mean_errors = np.array([np.mean(study_scores[sid]) for sid in study_ids])
    study_label_arr = np.array([study_labels[sid] for sid in study_ids])
    results["n_studies"] = len(study_ids)
    logger.info(f"Aggregated {len(all_errors)} clips into {len(study_ids)} studies")

    # Save study-level CSV
    study_output = args.output.replace(".csv", "_study.csv")
    with open(study_output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["study_id", "label", "anomaly_score", "n_clips"])
        for sid in study_ids:
            writer.writerow([sid, study_labels[sid], f"{np.mean(study_scores[sid]):.6f}", len(study_scores[sid])])
    logger.info(f"Saved study-level scores to {study_output}")

    if len(np.unique(study_label_arr)) == 2:
        auroc_study = roc_auc_score(study_label_arr, study_mean_errors)
        auroc_study_inv = roc_auc_score(study_label_arr, -study_mean_errors)
        results["study_auroc"] = float(auroc_study)
        results["study_auroc_inv"] = float(auroc_study_inv)
        results["study_best_auroc"] = float(max(auroc_study, auroc_study_inv))
        logger.info(
            f"Study-level AUROC: {auroc_study:.4f} (inverted: {auroc_study_inv:.4f}, "
            f"best: {max(auroc_study, auroc_study_inv):.4f})"
        )

        # Label distribution at study level
        n_pos = int(study_label_arr.sum())
        n_neg = len(study_label_arr) - n_pos
        results["n_positive_studies"] = n_pos
        results["n_negative_studies"] = n_neg
        logger.info(f"Study-level: {n_pos} positive, {n_neg} negative")

    # Save summary JSON
    summary_path = args.output.replace(".csv", "_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Summary: {summary_path}")

    return all_errors, all_labels


def _extract_study_id(path):
    """Extract study ID from S3 path (matches DistributedStudySampler logic)."""
    # MIMIC: .../s90001295/90001295_0054.mp4 -> "90001295"
    match = re.search(r"/s(\d+)/\d+_\d+\.mp4$", path)
    if match:
        return match.group(1)
    # UHN: parent directory is the study UID
    return os.path.basename(os.path.dirname(path))


def main():
    parser = argparse.ArgumentParser(description="JEPA anomaly detection via prediction error")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to JEPA checkpoint")
    parser.add_argument("--csv", type=str, required=True, help="Path to dataset CSV (path label)")
    parser.add_argument("--output", type=str, required=True, help="Output CSV path")

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

    # Eval config
    parser.add_argument("--num_masks", type=int, default=10, help="Random masks per clip")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_samples", type=int, default=None, help="Max clips to process (subsample)")

    args = parser.parse_args()
    run_anomaly_detection(args)


if __name__ == "__main__":
    main()
