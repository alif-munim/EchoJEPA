"""Create per-task label files that reference a master embedding NPZ by index.

The master NPZ contains embeddings for all clips (extracted once), ordered by the
source CSV rows. Each task CSV maps a subset of those same S3 paths to task-specific
labels. This script creates lightweight label-only NPZs (indices + labels) that
reference the master without duplicating embeddings.

Output NPZs contain:
    - indices: int array of row positions into the master NPZ
    - labels: float array of task-specific labels

Use with: python -m evals.train_probe --data master.npz --labels task_labels.npz

Usage:
    # Single task
    python -m evals.remap_embeddings \
        --embeddings embeddings/nature_medicine/mimic/echojepa_g_mimic_embeddings.npz \
        --source_csv data/csv/nature_medicine/mimic/mortality_1yr.csv \
        --task_csv data/csv/nature_medicine/mimic/creatinine.csv \
        --output embeddings/nature_medicine/mimic/labels/creatinine.npz

    # All tasks in a directory
    python -m evals.remap_embeddings \
        --embeddings embeddings/nature_medicine/mimic/echojepa_g_mimic_embeddings.npz \
        --source_csv data/csv/nature_medicine/mimic/mortality_1yr.csv \
        --task_dir data/csv/nature_medicine/mimic/ \
        --output_dir embeddings/nature_medicine/mimic/labels/
"""

import argparse
import os
from pathlib import Path

import numpy as np


def load_csv_paths_and_labels(csv_path):
    """Read VJEPA-format CSV (space-delimited: s3_path label, no header)."""
    paths = []
    labels = []
    with open(csv_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.rsplit(" ", 1)
            paths.append(parts[0])
            labels.append(float(parts[1]))
    return paths, np.array(labels)


def remap_single(source_paths, task_csv, output_path):
    """Create a labels-only NPZ with indices into the master embedding NPZ."""
    task_paths, task_labels = load_csv_paths_and_labels(task_csv)

    # Build lookup: s3_path -> index in master embeddings
    path_to_idx = {p: i for i, p in enumerate(source_paths)}

    # Match task paths to embedding indices
    matched_indices = []
    matched_labels = []
    missing = 0
    for path, label in zip(task_paths, task_labels):
        idx = path_to_idx.get(path)
        if idx is not None:
            matched_indices.append(idx)
            matched_labels.append(label)
        else:
            missing += 1

    matched_indices = np.array(matched_indices, dtype=np.int64)
    matched_labels = np.array(matched_labels)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.savez(output_path, indices=matched_indices, labels=matched_labels)

    task_name = Path(task_csv).stem
    n_unique = len(np.unique(matched_labels))
    label_range = f"[{matched_labels.min():.3f}, {matched_labels.max():.3f}]"
    print(f"  {task_name}: {len(matched_indices)} clips, {n_unique} unique labels, range {label_range}, {missing} missing")
    return len(matched_indices)


def main():
    parser = argparse.ArgumentParser(description="Remap master embeddings to per-task NPZs")
    parser.add_argument("--embeddings", required=True, help="Master NPZ file")
    parser.add_argument("--source_csv", required=True, help="CSV used during extraction (defines path ordering)")
    parser.add_argument("--task_csv", help="Single task CSV to remap")
    parser.add_argument("--output", help="Output NPZ path (for single task)")
    parser.add_argument("--task_dir", help="Directory of task CSVs to remap")
    parser.add_argument("--output_dir", help="Output directory (for batch mode)")
    args = parser.parse_args()

    if not args.task_csv and not args.task_dir:
        parser.error("Provide either --task_csv or --task_dir")

    print(f"Loading master embeddings metadata from {args.embeddings}...")
    data = np.load(args.embeddings, mmap_mode="r")
    n_embeddings = data["embeddings"].shape[0]
    print(f"  Embeddings: {data['embeddings'].shape}")

    # Load source CSV paths (first N to match embedding count)
    source_paths, _ = load_csv_paths_and_labels(args.source_csv)
    source_paths = source_paths[:n_embeddings]  # DistributedSampler may drop last few
    print(f"  Source paths: {len(source_paths)}")

    if args.task_csv:
        output = args.output or args.task_csv.replace(".csv", ".npz")
        remap_single(source_paths, args.task_csv, output)
    else:
        output_dir = args.output_dir or args.task_dir
        os.makedirs(output_dir, exist_ok=True)
        csvs = sorted(Path(args.task_dir).glob("*.csv"))
        print(f"\nRemapping {len(csvs)} task CSVs:")
        for csv_path in csvs:
            output_path = os.path.join(output_dir, csv_path.stem + ".npz")
            remap_single(source_paths, str(csv_path), output_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
