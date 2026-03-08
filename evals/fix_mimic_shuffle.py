#!/usr/bin/env python
"""Fix shuffle ordering in MIMIC embedding NPZs.

The original extraction used DistributedSampler(shuffle=True), which permuted
clip order. Downstream scripts (remap_embeddings, pool_embeddings) assume CSV
order. This script reorders embeddings to match source CSV order.

Usage:
    # Fix a single model's master NPZ
    python -m evals.fix_mimic_shuffle \
        --npz experiments/nature_medicine/mimic/echojepa_g_mimic_embeddings.npz \
        --source_csv data/csv/nature_medicine/mimic/mortality_1yr.csv

    # Fix all MIMIC models at once
    python -m evals.fix_mimic_shuffle \
        --mimic_dir experiments/nature_medicine/mimic \
        --source_csv data/csv/nature_medicine/mimic/mortality_1yr.csv

    # Dry run (verify only, don't modify files)
    python -m evals.fix_mimic_shuffle \
        --npz experiments/nature_medicine/mimic/echojepa_g_mimic_embeddings.npz \
        --source_csv data/csv/nature_medicine/mimic/mortality_1yr.csv \
        --dry_run
"""

import argparse
import logging
import os

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def count_csv_rows(csv_path):
    """Count non-empty lines in a VJEPA-format CSV."""
    n = 0
    with open(csv_path) as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def load_csv_labels(csv_path, n=None):
    """Load labels from VJEPA-format CSV (space-delimited: path label)."""
    labels = []
    with open(csv_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            labels.append(float(line.rsplit(" ", 1)[1]))
            if n and len(labels) >= n:
                break
    return np.array(labels)


def fix_single_npz(npz_path, n_dataset, dry_run=False, verify_csv=None):
    """Reorder a single master NPZ from shuffled to CSV order.

    Args:
        npz_path: Path to the master embedding NPZ
        n_dataset: The dataset size used during extraction (for permutation reconstruction)
        dry_run: If True, verify only without modifying files
        verify_csv: Optional CSV path to verify label correctness after reordering
    """
    logger.info(f"Loading {npz_path}...")
    data = np.load(npz_path, allow_pickle=True)
    embeddings = data["embeddings"]
    labels = data["labels"]
    paths = data["paths"]
    n_emb = len(embeddings)

    logger.info(f"  Embeddings: {embeddings.shape}, Labels: {labels.shape}")

    # Reconstruct the permutation (seed=0, epoch=0)
    g = torch.Generator()
    g.manual_seed(0)
    perm = torch.randperm(n_dataset, generator=g).numpy()

    # Only use the first n_emb elements of the permutation
    perm_n = perm[:n_emb]
    valid = perm_n < n_emb
    n_valid = int(valid.sum())
    n_invalid = n_emb - n_valid
    if n_invalid > 0:
        logger.warning(f"  {n_invalid}/{n_emb} positions have perm target >= {n_emb}, will be zero-filled")

    logger.info(f"  Reordering {n_valid} embeddings to CSV order...")

    reordered_emb = np.zeros_like(embeddings)
    reordered_labels = np.zeros_like(labels)
    reordered_paths = np.empty_like(paths)

    valid_idx = np.where(valid)[0]
    reordered_emb[perm_n[valid]] = embeddings[valid_idx]
    reordered_labels[perm_n[valid]] = labels[valid_idx]
    reordered_paths[perm_n[valid]] = paths[valid_idx]

    # Verify if CSV labels provided
    if verify_csv:
        csv_labels = load_csv_labels(verify_csv, n=n_emb)
        n_check = min(len(csv_labels), n_emb)
        match_pct = np.mean(reordered_labels[:n_check] == csv_labels[:n_check]) * 100
        logger.info(f"  Label verification: {match_pct:.2f}% match with CSV")
        if match_pct < 99.0:
            logger.error(f"  LOW MATCH — permutation reconstruction may be wrong!")
            return False

    if dry_run:
        logger.info("  Dry run — no files modified")
        return True

    # Backup and save
    backup_path = npz_path + ".shuffled_backup"
    if not os.path.exists(backup_path):
        os.rename(npz_path, backup_path)
        logger.info(f"  Backed up to {backup_path}")
    else:
        logger.info(f"  Backup already exists at {backup_path}")

    np.savez(npz_path, embeddings=reordered_emb, labels=reordered_labels, paths=reordered_paths)
    logger.info(f"  Saved reordered NPZ: {reordered_emb.shape}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Fix MIMIC embedding shuffle ordering")
    parser.add_argument("--npz", help="Single master NPZ to fix")
    parser.add_argument("--mimic_dir", help="MIMIC embeddings directory (fix all *_mimic_embeddings.npz)")
    parser.add_argument("--source_csv", required=True, help="Source CSV used during extraction")
    parser.add_argument("--dry_run", action="store_true", help="Verify only, don't modify")
    args = parser.parse_args()

    if not args.npz and not args.mimic_dir:
        parser.error("Provide either --npz or --mimic_dir")

    # Count CSV rows to get dataset size
    n_dataset = count_csv_rows(args.source_csv)
    logger.info(f"Source CSV has {n_dataset} rows")

    if args.npz:
        npz_files = [args.npz]
    else:
        npz_files = sorted(
            os.path.join(args.mimic_dir, f)
            for f in os.listdir(args.mimic_dir)
            if f.endswith("_mimic_embeddings.npz")
        )
        logger.info(f"Found {len(npz_files)} master NPZs in {args.mimic_dir}")

    for npz_path in npz_files:
        logger.info(f"\n{'='*60}")
        logger.info(f"Fixing: {os.path.basename(npz_path)}")
        logger.info(f"{'='*60}")
        fix_single_npz(npz_path, n_dataset, dry_run=args.dry_run, verify_csv=args.source_csv)

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
