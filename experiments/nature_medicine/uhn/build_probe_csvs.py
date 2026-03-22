#!/usr/bin/env python3
"""Build training/val/test CSVs for d=1 attentive probe evaluation on UHN tasks.

For each task, produces three CSVs (all containing ALL clips per study):
  - train.csv: ALL clips per study (DistributedStudySampler selects 1 per epoch)
  - val.csv:   ALL clips per study (for HP selection via prediction averaging)
  - test.csv:  ALL clips per study (for final evaluation via prediction averaging)

Builds a study-to-clips index from uhn_all_clips.csv on first run, then
reuses it for all tasks. The index maps study_id -> list of S3 paths.

Usage:
    # Single task
    python build_probe_csvs.py --task tapse

    # Multiple tasks
    python build_probe_csvs.py --tasks tapse rv_sp rv_fac rv_function rv_size

    # All tasks
    python build_probe_csvs.py --all
"""

import argparse
import json
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

RANGES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "physiological_ranges.json")


def load_physiological_ranges():
    """Load plausible value ranges for regression tasks."""
    with open(RANGES_PATH) as f:
        ranges = json.load(f)
    # Strip metadata keys
    return {k: v for k, v in ranges.items() if not k.startswith("_")}


def build_study_to_clips_index(clips_csv_path, index_cache_path):
    """Build {study_id: [s3_path, ...]} mapping from the 18M clip CSV.

    Caches to pickle for reuse across tasks.
    """
    if os.path.exists(index_cache_path):
        print(f"Loading cached study-to-clips index from {index_cache_path}")
        with open(index_cache_path, "rb") as f:
            return pickle.load(f)

    print(f"Building study-to-clips index from {clips_csv_path} (18M lines, ~2 min)...")
    study_to_clips = defaultdict(list)
    with open(clips_csv_path) as f:
        for i, line in enumerate(f):
            s3_path = line.strip().split(" ")[0]
            parts = s3_path.split("/")
            study_id = parts[5]  # s3://echodata25/results/echo-study*/{study_id}/...
            study_to_clips[study_id].append(s3_path)
            if (i + 1) % 5_000_000 == 0:
                print(f"  ...processed {i+1:,} lines ({len(study_to_clips):,} studies)")

    study_to_clips = dict(study_to_clips)
    print(f"  Done: {i+1:,} clips, {len(study_to_clips):,} studies")

    print(f"Saving index cache to {index_cache_path}")
    with open(index_cache_path, "wb") as f:
        pickle.dump(study_to_clips, f)

    return study_to_clips


def build_csvs_for_task(task, labels_dir, study_to_clips, output_dir, phys_ranges=None,
                        max_neg_pos_ratio=None):
    """Build train/val/test CSVs for a single task."""
    label_path = os.path.join(labels_dir, f"{task}.npz")
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Label file not found: {label_path}")

    lab = np.load(label_path, allow_pickle=True)
    study_ids = lab["study_ids"]
    labels = lab["labels"]
    splits = lab["splits"]

    task_dir = os.path.join(output_dir, task)
    os.makedirs(task_dir, exist_ok=True)

    # Detect regression vs classification
    unique_labels = np.unique(labels)
    is_regression = not all(l == int(l) for l in unique_labels) or len(unique_labels) > 20

    # Filter by physiological range (regression only)
    n_dropped_range = 0
    if is_regression and phys_ranges and task in phys_ranges:
        r = phys_ranges[task]
        valid = (labels >= r["min"]) & (labels <= r["max"])
        n_dropped_range = int((~valid).sum())
        if n_dropped_range > 0:
            study_ids = study_ids[valid]
            labels = labels[valid]
            splits = splits[valid]
            print(f"  Physiological filter [{r['min']}-{r['max']} {r['unit']}]: dropped {n_dropped_range} studies")

    # Compute Z-score params (saved for runtime normalization, labels stored raw)
    z_mean, z_std = None, None
    if is_regression:
        train_mask = splits == "train"
        train_labels = labels[train_mask]
        z_mean = float(train_labels.mean())
        z_std = float(train_labels.std())
        print(f"  Regression: mean={z_mean:.4f}, std={z_std:.4f} (labels stored in raw units)")

    stats = {}
    missing_studies = 0

    for split_name in ["train", "val", "test"]:
        mask = splits == split_name
        split_sids = study_ids[mask]
        split_labels = labels[mask]

        # Downsample majority class in train split only (val/test stay intact)
        n_capped = 0
        if split_name == "train" and max_neg_pos_ratio is not None and not is_regression:
            unique_cls = np.unique(split_labels)
            if len(unique_cls) == 2:
                pos_count = int((split_labels == 1).sum())
                neg_count = int((split_labels == 0).sum())
                max_neg = int(pos_count * max_neg_pos_ratio)
                if neg_count > max_neg:
                    rng = np.random.RandomState(42)
                    pos_idx = np.where(split_labels == 1)[0]
                    neg_idx = np.where(split_labels == 0)[0]
                    neg_keep = rng.choice(neg_idx, size=max_neg, replace=False)
                    keep_idx = np.sort(np.concatenate([pos_idx, neg_keep]))
                    n_capped = neg_count - max_neg
                    split_sids = split_sids[keep_idx]
                    split_labels = split_labels[keep_idx]
                    print(f"  Train neg cap ({max_neg_pos_ratio}:1): kept {max_neg}/{neg_count} neg "
                          f"({pos_count} pos, {n_capped} neg removed)")

        rows = []
        n_studies = 0
        for sid, label in zip(split_sids, split_labels):
            clips = study_to_clips.get(str(sid))
            if clips is None:
                missing_studies += 1
                continue
            n_studies += 1
            label_str = str(int(label)) if not is_regression else f"{label:.6f}"
            for s3_path in clips:
                rows.append(f"{s3_path} {label_str}")

        csv_path = os.path.join(task_dir, f"{split_name}.csv")
        with open(csv_path, "w") as f:
            f.write("\n".join(rows) + "\n")

        clips_per_study = len(rows) / n_studies if n_studies > 0 else 0
        stats[split_name] = {
            "studies": n_studies,
            "clips": len(rows),
            "clips_per_study": clips_per_study,
        }

        # Class distribution for classification tasks
        unique_labels = np.unique(split_labels)
        if len(unique_labels) <= 10 and all(l == int(l) for l in unique_labels):
            dist = {int(u): int((split_labels == u).sum()) for u in unique_labels}
            stats[split_name]["class_distribution"] = dist

    if missing_studies > 0:
        print(f"  Warning: {missing_studies} studies not found in clip index")

    # Save z-score params for regression configs
    if is_regression:
        params_path = os.path.join(task_dir, "zscore_params.json")
        with open(params_path, "w") as f:
            json.dump({"target_mean": z_mean, "target_std": z_std}, f)
        stats["zscore"] = {"target_mean": z_mean, "target_std": z_std}

    stats["is_regression"] = is_regression
    return stats


def main():
    parser = argparse.ArgumentParser(description="Build probe CSVs for UHN tasks")
    parser.add_argument("--task", type=str, help="Single task name")
    parser.add_argument("--tasks", nargs="+", help="Multiple task names")
    parser.add_argument("--all", action="store_true", help="Build CSVs for all tasks")
    parser.add_argument(
        "--base_dir",
        type=str,
        default="experiments/nature_medicine/uhn",
        help="Base directory for UHN data",
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--labels_dir", type=str, default=None, help="Override labels directory (default: {base_dir}/labels)")
    parser.add_argument("--max_neg_pos_ratio", type=float, default=None,
                        help="Cap negative:positive ratio in train split (e.g. 10 for 10:1). "
                             "Only applies to binary classification tasks. Val/test unaffected.")
    args = parser.parse_args()

    if not args.task and not args.tasks and not args.all:
        parser.error("Specify --task, --tasks, or --all")

    base_dir = args.base_dir
    labels_dir = args.labels_dir or os.path.join(base_dir, "labels")
    clips_csv = os.path.join(base_dir, "uhn_all_clips.csv")
    index_cache = os.path.join(base_dir, "study_to_clips_index.pkl")
    output_dir = args.output_dir or os.path.join(base_dir, "probe_csvs")

    # Build or load study-to-clips index
    study_to_clips = build_study_to_clips_index(clips_csv, index_cache)

    # Load physiological ranges
    phys_ranges = load_physiological_ranges()

    # Determine tasks
    if args.all:
        tasks = sorted([p.stem for p in Path(labels_dir).glob("*.npz")])
    elif args.tasks:
        tasks = args.tasks
    else:
        tasks = [args.task]

    print(f"\nBuilding probe CSVs for {len(tasks)} task(s)")
    print(f"Output: {output_dir}\n")

    for task in tasks:
        print(f"--- {task} ---")
        try:
            stats = build_csvs_for_task(task, labels_dir, study_to_clips, output_dir, phys_ranges,
                                       max_neg_pos_ratio=args.max_neg_pos_ratio)
        except FileNotFoundError as e:
            print(f"  SKIPPED: {e}")
            continue
        for split_name in ["train", "val", "test"]:
            s = stats[split_name]
            dist_str = ""
            if "class_distribution" in s:
                dist_str = f"  classes={s['class_distribution']}"
            print(
                f"  {split_name}: {s['studies']} studies, {s['clips']} clips "
                f"({s['clips_per_study']:.1f}/study){dist_str}"
            )
        print()


if __name__ == "__main__":
    main()
