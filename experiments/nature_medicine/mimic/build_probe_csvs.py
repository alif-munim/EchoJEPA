#!/usr/bin/env python3
"""Build training/val/test CSVs for d=1 attentive probe evaluation on MIMIC tasks.

For each task, produces three CSVs (all containing ALL clips per study):
  - train.csv: ALL clips per study (DistributedStudySampler selects 1 per epoch)
  - val.csv:   ALL clips per study (for HP selection via prediction averaging)
  - test.csv:  ALL clips per study (for final evaluation via prediction averaging)

Training uses study_sampling=true in the YAML config, which activates
DistributedStudySampler to pick 1 random clip per study per epoch, providing
cross-view augmentation across epochs.

CSV format: `s3_path label` (space-delimited), compatible with VideoDataset.

Usage:
    # Single task
    python build_probe_csvs.py --task mortality_1yr

    # All 23 tasks
    python build_probe_csvs.py --all

    # Custom output directory
    python build_probe_csvs.py --task mortality_1yr --output_dir /path/to/output
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

RANGES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "physiological_ranges.json")


def load_physiological_ranges():
    """Load plausible value ranges for regression tasks."""
    with open(RANGES_PATH) as f:
        ranges = json.load(f)
    return {k: v for k, v in ranges.items() if not k.startswith("_")}


def build_csvs_for_task(
    task: str,
    clip_index_path: str,
    labels_dir: str,
    patient_split_path: str,
    output_dir: str,
    phys_ranges=None,
):
    """Build train/val/test CSVs for a single task.

    All splits (including train) contain ALL clips per study.
    DistributedStudySampler handles 1-clip-per-study selection at training time.
    """

    # Load shared data
    ci = np.load(clip_index_path, allow_pickle=True)
    s3_paths = ci["s3_paths"]
    study_ids = ci["study_ids"]
    patient_ids = ci["patient_ids"]

    with open(patient_split_path) as f:
        patient_split = json.load(f)

    # Load task labels
    label_path = os.path.join(labels_dir, f"{task}.npz")
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Label file not found: {label_path}")

    lab = np.load(label_path, allow_pickle=True)
    indices = lab["indices"]
    labels = lab["labels"]

    # Subset to clips that have labels for this task
    task_s3 = s3_paths[indices]
    task_study = study_ids[indices]
    task_patient = patient_ids[indices]
    task_labels = labels

    # Group clips by study
    study_clips = {}  # study_id -> [(s3_path, label), ...]
    study_patient = {}  # study_id -> patient_id
    for i in range(len(task_s3)):
        sid = int(task_study[i])
        if sid not in study_clips:
            study_clips[sid] = []
            study_patient[sid] = int(task_patient[i])
        study_clips[sid].append((task_s3[i], task_labels[i]))

    # Split studies by patient
    splits = {"train": [], "val": [], "test": []}
    missing_patients = 0
    for sid, clips in study_clips.items():
        pid = study_patient[sid]
        split = patient_split.get(str(pid))
        if split is None:
            missing_patients += 1
            continue
        label = clips[0][1]  # All clips in a study share the same label
        splits[split].append((sid, label, clips))

    if missing_patients > 0:
        print(f"  Warning: {missing_patients} studies skipped (patient not in split)")

    # Filter by physiological range (regression tasks)
    if phys_ranges and task in phys_ranges:
        r = phys_ranges[task]
        n_before = sum(len(v) for v in splits.values())
        for split_name in splits:
            splits[split_name] = [(sid, lbl, clips) for sid, lbl, clips in splits[split_name]
                                  if r["min"] <= lbl <= r["max"]]
        n_after = sum(len(v) for v in splits.values())
        n_dropped = n_before - n_after
        if n_dropped > 0:
            print(f"  Physiological filter [{r['min']}-{r['max']} {r['unit']}]: dropped {n_dropped} studies")

    # Detect regression vs classification
    all_labels = np.array([s[1] for s in splits["train"] + splits["val"] + splits["test"]])
    unique_labels = np.unique(all_labels)
    is_regression = not all(l == int(l) for l in unique_labels) or len(unique_labels) > 20

    # Compute Z-score params (saved for runtime normalization, labels stored raw)
    z_mean, z_std = None, None
    if is_regression:
        train_labels = np.array([s[1] for s in splits["train"]])
        z_mean = float(train_labels.mean())
        z_std = float(train_labels.std())
        print(f"  Regression: mean={z_mean:.4f}, std={z_std:.4f} (labels stored in raw units)")

    # Create output directory
    task_dir = os.path.join(output_dir, task)
    os.makedirs(task_dir, exist_ok=True)

    # Build CSVs — all splits contain ALL clips per study
    stats = {}
    for split_name, studies in splits.items():
        rows = []
        for sid, label, clips in studies:
            if is_regression:
                label_str = f"{label:.6f}"
            else:
                label_str = str(int(label))
            for s3_path, lbl in clips:
                rows.append(f"{s3_path} {label_str}")

        csv_path = os.path.join(task_dir, f"{split_name}.csv")
        with open(csv_path, "w") as f:
            f.write("\n".join(rows) + "\n")

        # Compute stats
        labels_arr = np.array([s[1] for s in studies])
        n_studies = len(studies)
        n_clips = len(rows)
        stats[split_name] = {
            "studies": n_studies,
            "clips": n_clips,
            "clips_per_study": n_clips / n_studies if n_studies > 0 else 0,
        }

        # Classification stats
        if not is_regression and len(unique_labels) <= 10:
            dist = {int(u): int((labels_arr == u).sum()) for u in unique_labels}
            stats[split_name]["class_distribution"] = dist

    # Save z-score params for regression tasks
    if is_regression:
        params_path = os.path.join(task_dir, "zscore_params.json")
        with open(params_path, "w") as f:
            json.dump({"target_mean": z_mean, "target_std": z_std}, f)

    stats["is_regression"] = is_regression
    return stats


def main():
    parser = argparse.ArgumentParser(description="Build probe CSVs for MIMIC tasks")
    parser.add_argument("--task", type=str, help="Task name (e.g., mortality_1yr)")
    parser.add_argument("--all", action="store_true", help="Build CSVs for all 23 tasks")
    parser.add_argument(
        "--base_dir",
        type=str,
        default="experiments/nature_medicine/mimic",
        help="Base directory for MIMIC data",
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (default: {base_dir}/probe_csvs)")
    args = parser.parse_args()

    if not args.task and not args.all:
        parser.error("Specify --task or --all")

    base_dir = args.base_dir
    clip_index_path = os.path.join(base_dir, "clip_index.npz")
    labels_dir = os.path.join(base_dir, "labels")
    patient_split_path = os.path.join(base_dir, "patient_split.json")
    output_dir = args.output_dir or os.path.join(base_dir, "probe_csvs")

    if args.all:
        tasks = sorted([p.stem for p in Path(labels_dir).glob("*.npz")])
    else:
        tasks = [args.task]

    # Load physiological ranges
    phys_ranges = load_physiological_ranges()

    print(f"Building probe CSVs for {len(tasks)} task(s)")
    print(f"Output: {output_dir}")
    print()

    for task in tasks:
        print(f"--- {task} ---")
        stats = build_csvs_for_task(
            task=task,
            clip_index_path=clip_index_path,
            labels_dir=labels_dir,
            patient_split_path=patient_split_path,
            output_dir=output_dir,
            phys_ranges=phys_ranges,
        )
        for split_name in ["train", "val", "test"]:
            s = stats[split_name]
            dist_str = ""
            if "class_distribution" in s:
                dist_str = f"  classes={s['class_distribution']}"
            print(f"  {split_name}: {s['studies']} studies, {s['clips']} clips ({s['clips_per_study']:.1f}/study){dist_str}")
        print()


if __name__ == "__main__":
    main()
