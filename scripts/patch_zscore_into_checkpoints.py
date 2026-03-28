#!/usr/bin/env python3
"""Patch target_mean/target_std/task_type into existing probe checkpoints.

One-time migration: makes old checkpoints self-contained for inference
by embedding the z-score normalization parameters that were previously
stored in separate zscore_params.json files or sklearn scaler pkl files.

Handles two checkpoint directories:
  checkpoints/probes/       — Nature Medicine pipeline (best.pt/latest.pt)
  checkpoints/eval_probes/  — ICML preprint (flat .pt files, pre-z-scored CSVs)

Usage:
    python scripts/patch_zscore_into_checkpoints.py --dry-run   # preview
    python scripts/patch_zscore_into_checkpoints.py              # patch
"""

import argparse
import json
import os
import pickle

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Classification tasks never have z-score params
CLASSIFICATION_TASKS = {
    "ar_severity", "as_severity", "mr_severity", "tr_severity",
    "diastolic_function", "rv_function",
    "disease_amyloidosis", "disease_bicuspid_av", "disease_dcm",
    "disease_hcm", "disease_myxomatous_mv", "disease_rheumatic_mv",
    "disease_stemi", "disease_takotsubo",
    "trajectory_lvef_onset", "trajectory_mr_severity_onset",
    # MIMIC classification
    "discharge_destination", "in_hospital_mortality",
    "mort_30d", "mort_90d", "mort_1yr",
    "mortality_30d", "mortality_90d", "mortality_1yr",
    "readmission_30d",
    "disease_afib",
    # Trajectory classification (onset prediction)
    "trajectory_lvef", "trajectory_lvef_v1",
}

# Where to find zscore_params.json for each task
ZSCORE_SEARCH_PATHS = [
    os.path.join(ROOT, "experiments/nature_medicine/uhn/probe_csvs/{task}"),
    os.path.join(ROOT, "experiments/nature_medicine/mimic/probe_csvs/{task}"),
    os.path.join(ROOT, "experiments/nature_medicine/mimic/probe_csvs/misc/{task}"),
    os.path.join(ROOT, "data/csv/rebuttal/{task}"),
    os.path.join(ROOT, "data/csv/{task}"),
]


def find_zscore_params(task_name):
    """Look up zscore_params.json for a given task name."""
    for pattern in ZSCORE_SEARCH_PATHS:
        path = os.path.join(pattern.format(task=task_name), "zscore_params.json")
        if os.path.exists(path):
            with open(path) as f:
                params = json.load(f)
            return params["target_mean"], params["target_std"], path
    return None, None, None


def infer_task_type(task_name, checkpoint):
    """Infer whether a task is classification or regression."""
    # Check if checkpoint already has task_type
    if "task_type" in checkpoint and checkpoint["task_type"] is not None:
        return checkpoint["task_type"]
    # Known classification tasks
    if task_name in CLASSIFICATION_TASKS:
        return "classification"
    # If we can find zscore params, it's regression
    mean, std, _ = find_zscore_params(task_name)
    if mean is not None:
        return "regression"
    # Default: unknown — skip
    return None


def patch_checkpoint(ckpt_path, dry_run=False):
    """Patch a single checkpoint file. Returns (status, message)."""
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Already patched? Check for target_mean key (not value — classification has None)
    if "target_mean" in checkpoint:
        return "skip", "already patched"

    # Extract task name from path: .../probes/{task}/{model}/best.pt
    # or .../probes/mimic/{task}/{model}/best.pt
    parts = ckpt_path.split(os.sep)
    try:
        probes_idx = parts.index("probes")
    except ValueError:
        return "skip", "not under probes/"

    remaining = parts[probes_idx + 1:]  # e.g. ['lvef', 'echojepa-g', 'best.pt']
    if len(remaining) < 3:
        return "skip", f"unexpected path structure: {remaining}"

    # Handle mimic/ subdirectory
    if remaining[0] == "mimic":
        task_name = remaining[1]
    else:
        task_name = remaining[0]

    task_type = infer_task_type(task_name, checkpoint)
    if task_type is None:
        return "skip", f"cannot infer task_type for '{task_name}'"

    if task_type == "classification":
        checkpoint["task_type"] = "classification"
        checkpoint["target_mean"] = None
        checkpoint["target_std"] = None
        if not dry_run:
            torch.save(checkpoint, ckpt_path)
        return "patched", "classification (mean=None, std=None)"

    # Regression: need zscore params
    mean, std, source = find_zscore_params(task_name)
    if mean is None:
        return "skip", f"regression task '{task_name}' but no zscore_params.json found"

    checkpoint["task_type"] = "regression"
    checkpoint["target_mean"] = mean
    checkpoint["target_std"] = std
    if not dry_run:
        torch.save(checkpoint, ckpt_path)
    return "patched", f"regression (mean={mean:.4f}, std={std:.4f}) from {source}"


def load_scaler(name):
    """Load mean/std from a sklearn StandardScaler pkl file."""
    path = os.path.join(ROOT, f"data/scalers/{name}.pkl")
    if not os.path.exists(path):
        return None, None
    with open(path, "rb") as f:
        s = pickle.load(f)
    return float(s.mean_[0]), float(s.scale_[0])


def patch_eval_probes(eval_probes_dir, dry_run=False):
    """Patch checkpoints/eval_probes/ (ICML preprint, pre-z-scored CSVs).

    These used sklearn StandardScalers to pre-normalize CSVs, so z-score
    params come from data/scalers/*.pkl, not zscore_params.json.
    """
    if not os.path.isdir(eval_probes_dir):
        return {"patched": 0, "skip": 0, "error": 0}

    ef_mean, ef_std = load_scaler("ef_scaler")
    ped_mean, ped_std = load_scaler("pediatric_ef_scaler")
    lvef_mean, lvef_std = load_scaler("lvef_scaler")
    rvsp_mean, rvsp_std = load_scaler("rvsp_scaler")

    # Ordered rules: first prefix match wins
    rules = [
        ("classification/", "classification", None, None, "view classification"),
        ("lvef/echonet-dynamic/", "regression", ef_mean, ef_std, "ef_scaler.pkl"),
        ("lvef/echonet-pediatric/", "regression", ped_mean, ped_std, "pediatric_ef_scaler.pkl"),
        ("lvef/", "regression", lvef_mean, lvef_std, "lvef_scaler.pkl"),
        ("rvsp/", "regression", rvsp_mean, rvsp_std, "rvsp_scaler.pkl"),
    ]

    ckpt_files = []
    for dirpath, _, filenames in os.walk(eval_probes_dir):
        for fname in filenames:
            if fname.endswith(".pt"):
                ckpt_files.append(os.path.join(dirpath, fname))
    ckpt_files.sort()

    stats = {"patched": 0, "skip": 0, "error": 0}
    for path in ckpt_files:
        rel = os.path.relpath(path, eval_probes_dir)

        matched = None
        for prefix, task_type, mean, std, source in rules:
            if rel.startswith(prefix):
                matched = (task_type, mean, std, source)
                break

        if matched is None:
            print(f"  [!] {rel}: no matching rule")
            stats["error"] += 1
            continue

        task_type, mean, std, source = matched
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            if "target_mean" in ckpt:
                print(f"  [.] {rel}: already patched")
                stats["skip"] += 1
                continue

            ckpt["task_type"] = task_type
            ckpt["target_mean"] = mean
            ckpt["target_std"] = std
            if not dry_run:
                torch.save(ckpt, path)

            if mean is not None:
                print(f"  [+] {rel}: {task_type} (mean={mean:.4f}, std={std:.4f}) from {source}")
            else:
                print(f"  [+] {rel}: classification (mean=None, std=None)")
            stats["patched"] += 1
        except Exception as e:
            print(f"  [!] {rel}: ERROR {e}")
            stats["error"] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Preview without modifying files")
    parser.add_argument("--probe-dir", default=os.path.join(ROOT, "checkpoints/probes"),
                        help="Root directory for Nature Medicine probe checkpoints")
    parser.add_argument("--eval-probes-dir", default=os.path.join(ROOT, "checkpoints/eval_probes"),
                        help="Root directory for ICML preprint eval_probes checkpoints")
    args = parser.parse_args()

    if args.dry_run:
        print("=== DRY RUN (no files will be modified) ===\n")

    total = {"patched": 0, "skip": 0, "error": 0}

    # --- Part 1: checkpoints/probes/ (Nature Medicine) ---
    print("=== checkpoints/probes/ (Nature Medicine) ===")
    ckpt_files = []
    for dirpath, _, filenames in os.walk(args.probe_dir):
        for fname in filenames:
            if fname in ("best.pt", "latest.pt"):
                ckpt_files.append(os.path.join(dirpath, fname))
    ckpt_files.sort()
    print(f"Found {len(ckpt_files)} checkpoint files\n")

    for path in ckpt_files:
        rel = os.path.relpath(path, args.probe_dir)
        try:
            status, msg = patch_checkpoint(path, dry_run=args.dry_run)
        except Exception as e:
            status, msg = "error", str(e)
        total[status] = total.get(status, 0) + 1
        marker = {"patched": "+", "skip": ".", "error": "!"}[status]
        print(f"  [{marker}] {rel}: {msg}")

    # --- Part 2: checkpoints/eval_probes/ (ICML preprint) ---
    print("\n=== checkpoints/eval_probes/ (ICML preprint) ===")
    ep_files = []
    if os.path.isdir(args.eval_probes_dir):
        for dirpath, _, filenames in os.walk(args.eval_probes_dir):
            for fname in filenames:
                if fname.endswith(".pt"):
                    ep_files.append(os.path.join(dirpath, fname))
    print(f"Found {len(ep_files)} checkpoint files\n")
    ep_stats = patch_eval_probes(args.eval_probes_dir, dry_run=args.dry_run)
    for k in total:
        total[k] += ep_stats.get(k, 0)

    print(f"\nTotal: {total['patched']} patched, {total['skip']} skipped, {total['error']} errors")
    if args.dry_run and total["patched"] > 0:
        print("Re-run without --dry-run to apply patches.")


if __name__ == "__main__":
    main()
