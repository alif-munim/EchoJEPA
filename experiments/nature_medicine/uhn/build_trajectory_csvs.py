#!/usr/bin/env python3
"""Build training/val/test CSVs for trajectory (delta prediction) tasks.

Each trajectory label file contains pairs of studies (study_1, study_2) with a delta
label representing the change in a clinical measurement between the two echos.

Supports two modes:
  - Regression (default): predict raw delta. 1 clip per pair, zscore_params.json.
  - Classification (--classification --threshold N): predict declined/stable/improved.
    All clips per study_1 written (for study_sampling). No zscore_params.json.
    Classes: 0=declined (delta <= -N), 1=stable, 2=improved (delta >= +N).

Usage:
    python build_trajectory_csvs.py --all
    python build_trajectory_csvs.py --task trajectory_lvef
    python build_trajectory_csvs.py --task trajectory_lvef --classification --threshold 10
"""

import argparse
import json
import os
import pickle
import random
import sys

import numpy as np

RANGES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "physiological_ranges.json")


def load_physiological_ranges():
    """Load plausible value ranges for regression tasks."""
    with open(RANGES_PATH) as f:
        ranges = json.load(f)
    return {k: v for k, v in ranges.items() if not k.startswith("_")}

TRAJECTORY_TASKS = [
    "trajectory_lvef",
    "trajectory_tapse",
    "trajectory_lv_mass",
    "trajectory_rv_sp",
    "trajectory_mr_severity",
]

# View filters for trajectory tasks (same as the base measurement).
# No B-mode filter — predicting trajectory, not demonstrating structure→flow.
TRAJECTORY_VIEW_FILTERS = {
    "trajectory_lvef": ["A4C", "A2C"],
    "trajectory_tapse": ["A4C"],
    "trajectory_lv_mass": ["PLAX", "A4C", "A2C", "PSAX-AV", "PSAX-MV", "PSAX-PM", "PSAX-AP"],
    "trajectory_rv_sp": ["A4C", "Subcostal"],
    "trajectory_mr_severity": ["A4C", "A2C", "A3C", "PLAX"],
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LABELS_DIR = os.path.join(BASE_DIR, "labels", "trajectory")
OUTPUT_DIR = os.path.join(BASE_DIR, "probe_csvs")
CLIPS_INDEX = os.path.join(BASE_DIR, "study_to_clips_index.pkl")
VIEW_PREDICTIONS = os.path.join(
    os.path.expanduser("~"), "user-default-efs/vjepa2/classifier/output/view_inference_18m/master_predictions.csv"
)


def load_study_to_clips():
    """Load cached study -> clips mapping."""
    print(f"Loading study-to-clips index from {CLIPS_INDEX}")
    with open(CLIPS_INDEX, "rb") as f:
        return pickle.load(f)


def load_view_lookup(uris_needed):
    """Load view predictions for the given URIs."""
    print(f"  Loading view predictions for {len(uris_needed):,} URIs...")
    lookup = {}
    with open(VIEW_PREDICTIONS) as f:
        next(f)  # skip header
        for line in f:
            parts = line.rstrip("\n").split(",")
            uri = parts[0]
            if uri in uris_needed:
                lookup[uri] = parts[1]
                if len(lookup) == len(uris_needed):
                    break
    print(f"  Matched {len(lookup):,} / {len(uris_needed):,} URIs")
    return lookup


def filter_clips_by_view(clips, allowed_views, view_lookup):
    """Filter clips to only those matching allowed views."""
    allowed = set(allowed_views)
    return [c for c in clips if view_lookup.get(c) in allowed]


def build_csvs_for_task(task, study_to_clips, view_lookup=None, seed=42, phys_ranges=None,
                        min_days=None, max_days=None):
    """Build train/val/test CSVs for a trajectory task."""
    label_path = os.path.join(LABELS_DIR, f"{task}.npz")
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Label file not found: {label_path}")

    d = np.load(label_path, allow_pickle=True)
    study_id_1 = d["study_id_1"]
    study_id_2 = d["study_id_2"]
    patient_ids = d["patient_ids"]
    delta = d["delta"].astype(float)
    days_between = d["days_between"]
    splits = d["splits"]

    # Filter by physiological range
    if phys_ranges and task in phys_ranges:
        r = phys_ranges[task]
        valid = (delta >= r["min"]) & (delta <= r["max"])
        n_dropped = int((~valid).sum())
        if n_dropped > 0:
            study_id_1 = study_id_1[valid]
            study_id_2 = study_id_2[valid]
            patient_ids = patient_ids[valid]
            delta = delta[valid]
            days_between = days_between[valid]
            splits = splits[valid]
            print(f"  Physiological filter [{r['min']}-{r['max']} {r['unit']}]: dropped {n_dropped} pairs")

    # Filter by time window
    if min_days is not None or max_days is not None:
        db_int = days_between.astype(int)
        time_valid = np.ones(len(db_int), dtype=bool)
        if min_days is not None:
            time_valid &= db_int >= min_days
        if max_days is not None:
            time_valid &= db_int <= max_days
        n_dropped = int((~time_valid).sum())
        if n_dropped > 0:
            study_id_1 = study_id_1[time_valid]
            study_id_2 = study_id_2[time_valid]
            patient_ids = patient_ids[time_valid]
            delta = delta[time_valid]
            days_between = days_between[time_valid]
            splits = splits[time_valid]
            lo = min_days or "any"
            hi = max_days or "any"
            print(f"  Time window [{lo}-{hi}d]: dropped {n_dropped}, kept {len(delta)} pairs")

    task_dir = os.path.join(OUTPUT_DIR, task)
    os.makedirs(task_dir, exist_ok=True)

    # Compute Z-score params (saved for runtime normalization, labels stored raw)
    train_mask = splits == "train"
    train_deltas = delta[train_mask]
    z_mean = float(train_deltas.mean())
    z_std = float(train_deltas.std())

    print(f"  Delta stats: mean={z_mean:.4f}, std={z_std:.4f} (labels stored in raw units)")

    # Get view filter for this task
    allowed_views = TRAJECTORY_VIEW_FILTERS.get(task)

    rng = random.Random(seed)
    stats = {}
    all_pairs_meta = []

    for split_name in ["train", "val", "test"]:
        mask = splits == split_name
        s1 = study_id_1[mask]
        s2 = study_id_2[mask]
        pids = patient_ids[mask]
        raw_delta = delta[mask]
        db = days_between[mask]

        rows = []
        pairs_meta = []
        missing = 0
        empty_after_filter = 0

        for i in range(len(s1)):
            sid1 = str(s1[i])
            clips = study_to_clips.get(sid1)
            if clips is None:
                missing += 1
                continue

            # Apply view filter if available
            if allowed_views and view_lookup:
                filtered = filter_clips_by_view(clips, allowed_views, view_lookup)
                if not filtered:
                    empty_after_filter += 1
                    continue
                selected = rng.choice(filtered)
            else:
                selected = rng.choice(clips)

            label_str = f"{raw_delta[i]:.6f}"
            rows.append(f"{selected} {label_str}")

            pairs_meta.append({
                "pair_idx": len(all_pairs_meta) + len(pairs_meta),
                "study_id_1": sid1,
                "study_id_2": str(s2[i]),
                "patient_id": str(pids[i]),
                "delta": float(raw_delta[i]),
                "days_between": int(db[i]),
                "split": split_name,
                "clip_selected": selected,
            })

        csv_path = os.path.join(task_dir, f"{split_name}.csv")
        with open(csv_path, "w") as f:
            f.write("\n".join(rows) + "\n")

        all_pairs_meta.extend(pairs_meta)

        n_pairs = mask.sum()
        print(f"  {split_name}: {len(rows)} / {n_pairs} pairs written"
              f" (missing: {missing}, empty after view filter: {empty_after_filter})")
        stats[split_name] = {"pairs_total": int(n_pairs), "pairs_written": len(rows)}

    # Save Z-score params
    params_path = os.path.join(task_dir, "zscore_params.json")
    with open(params_path, "w") as f:
        json.dump({"target_mean": z_mean, "target_std": z_std}, f)

    # Save pairs metadata
    meta_path = os.path.join(task_dir, "pairs_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(all_pairs_meta, f)

    # Save view filter info
    if allowed_views:
        filter_meta = {
            "task": task,
            "allowed_views": allowed_views,
            "bmode_only": False,
        }
        filter_path = os.path.join(task_dir, "viewfilter_meta.json")
        with open(filter_path, "w") as f:
            json.dump(filter_meta, f, indent=2)

    return stats


def classify_delta(delta, threshold):
    """Map delta to class: 0=declined, 1=stable, 2=improved."""
    if delta <= -threshold:
        return 0
    elif delta >= threshold:
        return 2
    return 1


def build_classification_csvs_for_task(task, study_to_clips, view_lookup=None, seed=42,
                                       phys_ranges=None, threshold=10,
                                       min_days=None, max_days=None):
    """Build classification CSVs for a trajectory task.

    Writes ALL clips per study_1 (for study_sampling). Each study_1 gets one label
    based on its single pair (dataset must have unique study_1s).
    Classes: 0=declined (delta <= -threshold), 1=stable, 2=improved (delta >= +threshold).
    """
    label_path = os.path.join(LABELS_DIR, f"{task}.npz")
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Label file not found: {label_path}")

    d = np.load(label_path, allow_pickle=True)
    study_id_1 = d["study_id_1"]
    study_id_2 = d["study_id_2"]
    patient_ids = d["patient_ids"]
    delta = d["delta"].astype(float)
    label_1 = d["label_1"].astype(float)
    label_2 = d["label_2"].astype(float)
    days_between = d["days_between"]
    splits = d["splits"]

    # Filter by physiological range
    if phys_ranges and task in phys_ranges:
        r = phys_ranges[task]
        valid = (delta >= r["min"]) & (delta <= r["max"])
        n_dropped = int((~valid).sum())
        if n_dropped > 0:
            study_id_1 = study_id_1[valid]
            study_id_2 = study_id_2[valid]
            patient_ids = patient_ids[valid]
            delta = delta[valid]
            label_1 = label_1[valid]
            label_2 = label_2[valid]
            days_between = days_between[valid]
            splits = splits[valid]
            print(f"  Physiological filter: dropped {n_dropped} pairs")

    # Filter by time window
    if min_days is not None or max_days is not None:
        db_int = days_between.astype(int)
        time_valid = np.ones(len(db_int), dtype=bool)
        if min_days is not None:
            time_valid &= db_int >= min_days
        if max_days is not None:
            time_valid &= db_int <= max_days
        n_dropped = int((~time_valid).sum())
        if n_dropped > 0:
            study_id_1 = study_id_1[time_valid]
            study_id_2 = study_id_2[time_valid]
            patient_ids = patient_ids[time_valid]
            delta = delta[time_valid]
            label_1 = label_1[time_valid]
            label_2 = label_2[time_valid]
            days_between = days_between[time_valid]
            splits = splits[time_valid]
            lo = min_days or "any"
            hi = max_days or "any"
            print(f"  Time window [{lo}-{hi}d]: dropped {n_dropped}, kept {len(delta)} pairs")

    task_dir = os.path.join(OUTPUT_DIR, task)
    os.makedirs(task_dir, exist_ok=True)

    # Get view filter for this task
    allowed_views = TRAJECTORY_VIEW_FILTERS.get(task)

    rng = random.Random(seed)
    stats = {}
    all_pairs_meta = []

    # Class distribution tracking
    class_names = {0: "declined", 1: "stable", 2: "improved"}

    for split_name in ["train", "val", "test"]:
        mask = splits == split_name
        s1 = study_id_1[mask]
        s2 = study_id_2[mask]
        pids = patient_ids[mask]
        raw_delta = delta[mask]
        l1 = label_1[mask]
        l2 = label_2[mask]
        db = days_between[mask]

        rows = []
        pairs_meta = []
        missing = 0
        empty_after_filter = 0
        class_counts = {0: 0, 1: 0, 2: 0}

        for i in range(len(s1)):
            sid1 = str(s1[i])
            clips = study_to_clips.get(sid1)
            if clips is None:
                missing += 1
                continue

            # Apply view filter if available
            if allowed_views and view_lookup:
                filtered = filter_clips_by_view(clips, allowed_views, view_lookup)
                if not filtered:
                    empty_after_filter += 1
                    continue
                clip_list = filtered
            else:
                clip_list = clips

            cls = classify_delta(raw_delta[i], threshold)
            class_counts[cls] += 1

            # Write ALL clips for this study (study_sampling picks 1 per epoch)
            for clip in clip_list:
                rows.append(f"{clip} {cls}")

            pairs_meta.append({
                "pair_idx": len(all_pairs_meta) + len(pairs_meta),
                "study_id_1": sid1,
                "study_id_2": str(s2[i]),
                "patient_id": str(pids[i]),
                "delta_raw": float(raw_delta[i]),
                "label_1": float(l1[i]),
                "label_2": float(l2[i]),
                "days_between": int(db[i]),
                "class": cls,
                "class_name": class_names[cls],
                "split": split_name,
            })

        csv_path = os.path.join(task_dir, f"{split_name}.csv")
        with open(csv_path, "w") as f:
            f.write("\n".join(rows) + "\n")

        all_pairs_meta.extend(pairs_meta)

        n_pairs = mask.sum()
        n_written = sum(class_counts.values())
        print(f"  {split_name}: {n_written} studies, {len(rows)} clips"
              f" (missing: {missing}, empty after view filter: {empty_after_filter})")
        for c in [0, 1, 2]:
            pct = 100 * class_counts[c] / n_written if n_written else 0
            print(f"    {class_names[c]}: {class_counts[c]} ({pct:.1f}%)")
        stats[split_name] = {"studies": n_written, "clips": len(rows), "class_counts": class_counts}

    # Remove zscore_params.json if it exists (classification, not regression)
    params_path = os.path.join(task_dir, "zscore_params.json")
    if os.path.exists(params_path):
        os.remove(params_path)
        print(f"  Removed {params_path} (classification mode)")

    # Save pairs metadata
    meta_path = os.path.join(task_dir, "pairs_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(all_pairs_meta, f)

    # Save view filter + classification info
    filter_meta = {
        "task": task,
        "mode": "classification",
        "threshold": threshold,
        "classes": {str(k): v for k, v in class_names.items()},
        "allowed_views": allowed_views,
        "bmode_only": False,
    }
    filter_path = os.path.join(task_dir, "viewfilter_meta.json")
    with open(filter_path, "w") as f:
        json.dump(filter_meta, f, indent=2)

    return stats


def build_onset_csvs_for_task(task, study_to_clips, view_lookup=None, seed=42,
                              phys_ranges=None, baseline_min=50, future_below=50,
                              min_days=None, max_days=None, output_suffix="_onset"):
    """Build binary onset-prediction CSVs for a trajectory task.

    Filters to patients with baseline measurement >= baseline_min, then labels
    binary: 0 = future value >= future_below (stable), 1 = future value < future_below (decline).
    Writes ALL clips per study_1 (for study_sampling).
    """
    label_path = os.path.join(LABELS_DIR, f"{task}.npz")
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Label file not found: {label_path}")

    d = np.load(label_path, allow_pickle=True)
    study_id_1 = d["study_id_1"]
    study_id_2 = d["study_id_2"]
    patient_ids = d["patient_ids"]
    delta = d["delta"].astype(float)
    label_1 = d["label_1"].astype(float)
    label_2 = d["label_2"].astype(float)
    days_between = d["days_between"]
    splits = d["splits"]

    # Filter to baseline >= baseline_min
    baseline_mask = label_1 >= baseline_min
    n_excluded = int((~baseline_mask).sum())
    study_id_1 = study_id_1[baseline_mask]
    study_id_2 = study_id_2[baseline_mask]
    patient_ids = patient_ids[baseline_mask]
    delta = delta[baseline_mask]
    label_1 = label_1[baseline_mask]
    label_2 = label_2[baseline_mask]
    days_between = days_between[baseline_mask]
    splits = splits[baseline_mask]
    print(f"  Baseline filter (>= {baseline_min}): kept {len(delta)}, excluded {n_excluded}")

    # Filter by physiological range
    if phys_ranges and task in phys_ranges:
        r = phys_ranges[task]
        valid = (delta >= r["min"]) & (delta <= r["max"])
        n_dropped = int((~valid).sum())
        if n_dropped > 0:
            study_id_1 = study_id_1[valid]
            study_id_2 = study_id_2[valid]
            patient_ids = patient_ids[valid]
            delta = delta[valid]
            label_1 = label_1[valid]
            label_2 = label_2[valid]
            days_between = days_between[valid]
            splits = splits[valid]
            print(f"  Physiological filter: dropped {n_dropped} pairs")

    # Filter by time window
    if min_days is not None or max_days is not None:
        db_int = days_between.astype(int)
        time_valid = np.ones(len(db_int), dtype=bool)
        if min_days is not None:
            time_valid &= db_int >= min_days
        if max_days is not None:
            time_valid &= db_int <= max_days
        n_dropped = int((~time_valid).sum())
        if n_dropped > 0:
            study_id_1 = study_id_1[time_valid]
            study_id_2 = study_id_2[time_valid]
            patient_ids = patient_ids[time_valid]
            delta = delta[time_valid]
            label_1 = label_1[time_valid]
            label_2 = label_2[time_valid]
            days_between = days_between[time_valid]
            splits = splits[time_valid]
            lo = min_days or "any"
            hi = max_days or "any"
            print(f"  Time window [{lo}-{hi}d]: dropped {n_dropped}, kept {len(delta)} pairs")

    # Output directory
    task_dir = os.path.join(OUTPUT_DIR, task + output_suffix)
    os.makedirs(task_dir, exist_ok=True)

    # Get view filter for this task
    allowed_views = TRAJECTORY_VIEW_FILTERS.get(task)

    rng = random.Random(seed)
    stats = {}
    all_pairs_meta = []
    class_names = {0: "stable", 1: "decline"}

    for split_name in ["train", "val", "test"]:
        mask = splits == split_name
        s1 = study_id_1[mask]
        s2 = study_id_2[mask]
        pids = patient_ids[mask]
        l1 = label_1[mask]
        l2 = label_2[mask]
        raw_delta = delta[mask]
        db = days_between[mask]

        rows = []
        pairs_meta = []
        missing = 0
        empty_after_filter = 0
        class_counts = {0: 0, 1: 0}

        for i in range(len(s1)):
            sid1 = str(s1[i])
            clips = study_to_clips.get(sid1)
            if clips is None:
                missing += 1
                continue

            # Apply view filter if available
            if allowed_views and view_lookup:
                filtered = filter_clips_by_view(clips, allowed_views, view_lookup)
                if not filtered:
                    empty_after_filter += 1
                    continue
                clip_list = filtered
            else:
                clip_list = clips

            cls = 1 if l2[i] < future_below else 0
            class_counts[cls] += 1

            # Write ALL clips for this study (study_sampling picks 1 per epoch)
            for clip in clip_list:
                rows.append(f"{clip} {cls}")

            pairs_meta.append({
                "pair_idx": len(all_pairs_meta) + len(pairs_meta),
                "study_id_1": sid1,
                "study_id_2": str(s2[i]),
                "patient_id": str(pids[i]),
                "baseline_ef": float(l1[i]),
                "followup_ef": float(l2[i]),
                "delta": float(raw_delta[i]),
                "days_between": int(db[i]),
                "class": cls,
                "class_name": class_names[cls],
                "split": split_name,
            })

        csv_path = os.path.join(task_dir, f"{split_name}.csv")
        with open(csv_path, "w") as f:
            f.write("\n".join(rows) + "\n")

        all_pairs_meta.extend(pairs_meta)

        n_pairs = mask.sum()
        n_written = sum(class_counts.values())
        print(f"  {split_name}: {n_written} studies, {len(rows)} clips"
              f" (missing: {missing}, empty after view filter: {empty_after_filter})")
        for c in [0, 1]:
            pct = 100 * class_counts[c] / n_written if n_written else 0
            print(f"    {class_names[c]}: {class_counts[c]} ({pct:.1f}%)")
        stats[split_name] = {"studies": n_written, "clips": len(rows), "class_counts": class_counts}

    # Remove zscore_params.json if it exists (classification, not regression)
    params_path = os.path.join(task_dir, "zscore_params.json")
    if os.path.exists(params_path):
        os.remove(params_path)

    # Save pairs metadata
    meta_path = os.path.join(task_dir, "pairs_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(all_pairs_meta, f)

    # Save filter + classification info
    filter_meta = {
        "task": task,
        "output_task": task + output_suffix,
        "mode": "onset_binary",
        "baseline_min": baseline_min,
        "future_below": future_below,
        "classes": {str(k): v for k, v in class_names.items()},
        "allowed_views": allowed_views,
        "bmode_only": False,
        "min_days": min_days,
        "max_days": max_days,
    }
    filter_path = os.path.join(task_dir, "viewfilter_meta.json")
    with open(filter_path, "w") as f:
        json.dump(filter_meta, f, indent=2)

    return stats


def main():
    parser = argparse.ArgumentParser(description="Build trajectory probe CSVs")
    parser.add_argument("--task", type=str, help="Single task name")
    parser.add_argument("--all", action="store_true", help="Build all trajectory tasks")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for clip selection")
    parser.add_argument("--classification", action="store_true",
                        help="Build as 3-class classification (declined/stable/improved)")
    parser.add_argument("--threshold", type=float, default=10,
                        help="Delta threshold for classification (default: 10)")
    parser.add_argument("--onset", action="store_true",
                        help="Build as binary onset prediction (baseline >= min, future < threshold)")
    parser.add_argument("--baseline_min", type=float, default=50,
                        help="Minimum baseline value for onset mode (default: 50)")
    parser.add_argument("--future_below", type=float, default=50,
                        help="Future value threshold for onset mode (default: 50)")
    parser.add_argument("--output_suffix", type=str, default="_onset",
                        help="Suffix for output directory in onset mode (default: _onset)")
    parser.add_argument("--min_days", type=int, default=None,
                        help="Minimum days between studies (inclusive)")
    parser.add_argument("--max_days", type=int, default=None,
                        help="Maximum days between studies (inclusive)")
    args = parser.parse_args()

    if not args.task and not args.all:
        parser.error("Specify --task or --all")

    tasks = TRAJECTORY_TASKS if args.all else [args.task]

    # Load study-to-clips index
    study_to_clips = load_study_to_clips()

    # Collect all study_1 URIs that need view predictions
    print("Collecting URIs for view lookup...")
    all_uris = set()
    for task in tasks:
        d = np.load(os.path.join(LABELS_DIR, f"{task}.npz"), allow_pickle=True)
        for sid in set(d["study_id_1"]):
            clips = study_to_clips.get(str(sid))
            if clips:
                all_uris.update(clips)
    print(f"  {len(all_uris):,} unique URIs from study_1 clips")

    # Load view predictions
    view_lookup = load_view_lookup(all_uris)

    # Load physiological ranges
    phys_ranges = load_physiological_ranges()

    if args.onset:
        mode = "onset"
    elif args.classification:
        mode = "classification"
    else:
        mode = "regression"
    print(f"\nBuilding trajectory CSVs for {len(tasks)} task(s) [{mode}]")
    if args.classification:
        print(f"  Threshold: ±{args.threshold} (0=declined, 1=stable, 2=improved)")
    if args.onset:
        print(f"  Onset mode: baseline >= {args.baseline_min}, future < {args.future_below}")
        print(f"  Output suffix: {args.output_suffix}")
    if args.min_days or args.max_days:
        print(f"  Time window: {args.min_days or 'any'}-{args.max_days or 'any'} days")
    print()

    for task in tasks:
        print(f"--- {task} ---")
        try:
            if args.onset:
                stats = build_onset_csvs_for_task(
                    task, study_to_clips, view_lookup, seed=args.seed,
                    phys_ranges=phys_ranges, baseline_min=args.baseline_min,
                    future_below=args.future_below,
                    min_days=args.min_days, max_days=args.max_days,
                    output_suffix=args.output_suffix)
            elif args.classification:
                stats = build_classification_csvs_for_task(
                    task, study_to_clips, view_lookup, seed=args.seed,
                    phys_ranges=phys_ranges, threshold=args.threshold,
                    min_days=args.min_days, max_days=args.max_days)
            else:
                stats = build_csvs_for_task(
                    task, study_to_clips, view_lookup, seed=args.seed,
                    phys_ranges=phys_ranges,
                    min_days=args.min_days, max_days=args.max_days)
        except FileNotFoundError as e:
            print(f"  SKIPPED: {e}")
            continue
        print()


if __name__ == "__main__":
    main()
