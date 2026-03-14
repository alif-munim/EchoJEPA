#!/usr/bin/env python3
"""Build training/val/test CSVs for trajectory (delta prediction) tasks.

Each trajectory label file contains pairs of studies (study_1, study_2) with a delta
label representing the change in a clinical measurement between the two echos.

The probe sees clips from study_1 and predicts the delta. Since the same study_1 can
appear in multiple pairs with different deltas (different follow-up echos), we select
1 random clip from study_1 per pair. No study_sampling is used — each pair is one
training example.

Produces:
  - train.csv, val.csv, test.csv: 1 clip per pair, raw delta label
  - zscore_params.json: {target_mean, target_std} for the delta
  - pairs_metadata.json: full pair info for future multi-clip aggregation

Usage:
    python build_trajectory_csvs.py --all
    python build_trajectory_csvs.py --task trajectory_lvef
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


def build_csvs_for_task(task, study_to_clips, view_lookup=None, seed=42, phys_ranges=None):
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


def main():
    parser = argparse.ArgumentParser(description="Build trajectory probe CSVs")
    parser.add_argument("--task", type=str, help="Single task name")
    parser.add_argument("--all", action="store_true", help="Build all trajectory tasks")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for clip selection")
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

    print(f"\nBuilding trajectory CSVs for {len(tasks)} task(s)\n")

    for task in tasks:
        print(f"--- {task} ---")
        try:
            stats = build_csvs_for_task(task, study_to_clips, view_lookup, seed=args.seed, phys_ranges=phys_ranges)
        except FileNotFoundError as e:
            print(f"  SKIPPED: {e}")
            continue
        print()


if __name__ == "__main__":
    main()
