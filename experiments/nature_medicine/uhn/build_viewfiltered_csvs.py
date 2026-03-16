"""Build view-filtered probe CSVs for Nature Medicine tasks.

Joins all-clip probe CSVs with view classifier predictions (and optionally
color classifier predictions) to produce filtered CSVs containing only
task-relevant clips.

Usage:
    # A4C only (TAPSE, RV S', RV FAC)
    python build_viewfiltered_csvs.py --task tapse --views A4C

    # Multi-view (RV function)
    python build_viewfiltered_csvs.py --task rv_function --views A4C Subcostal PLAX

    # B-mode only hemodynamics (MR severity)
    python build_viewfiltered_csvs.py --task mr_severity --views A4C A2C A3C PLAX --bmode_only

    # All tasks with recommended filters
    python build_viewfiltered_csvs.py --all
"""

import argparse
import json
import os
import sys
import time

# Task -> (allowed_views, bmode_only)
# Task names must match probe_csvs/ subdirectory names (= label NPZ stems)
TASK_FILTERS = {
    # --- RV mechanics (Wendy pillar 1) ---
    "tapse": (["A4C"], False),
    "rv_sp": (["A4C", "Subcostal"], False),
    "rv_fac": (["A4C"], False),
    "rv_function": (["A4C", "Subcostal", "PLAX"], False),
    "rv_size": (["A4C", "Subcostal", "PLAX", "PSAX-AV", "PSAX-MV", "PSAX-PM", "PSAX-AP"], False),
    # --- Standard tasks ---
    "lvef": (["A4C", "A2C"], False),
    "edv": (["A4C", "A2C"], False),
    "esv": (["A4C", "A2C"], False),
    "lv_systolic_function": (["A4C", "A2C", "PLAX"], False),
    "lv_mass": (["PLAX", "A4C", "A2C", "PSAX-AV", "PSAX-MV", "PSAX-PM", "PSAX-AP"], False),
    "lv_cavity_size": (["PLAX", "A4C", "A2C", "PSAX-AV", "PSAX-MV", "PSAX-PM", "PSAX-AP"], False),
    "ivsd": (["PLAX"], False),
    "ao_root": (["PLAX"], False),
    "rwma": (["A4C", "A2C", "A3C", "PSAX-PM", "PSAX-AP"], False),
    "la_size": (["A4C", "A2C", "PLAX"], False),
    "la_vol": (["A4C", "A2C"], False),
    "ra_size": (["A4C", "Subcostal"], False),
    "lv_hypertrophy": (["PLAX", "A4C", "PSAX-PM", "PSAX-MV"], False),
    "pericardial_effusion": (["A4C", "PLAX", "Subcostal"], False),
    # --- Hemodynamics (Wendy pillar 2) — B-mode only ---
    "mr_severity": (["A4C", "A2C", "A3C", "PLAX"], True),
    "ar_severity": (["A4C", "A2C", "A3C", "PLAX"], True),
    "as_severity": (["PLAX", "PSAX-AV", "A3C"], True),
    "aov_vmax": (["PLAX", "A3C", "PSAX-AV"], True),
    "aov_mean_grad": (["PLAX", "A3C", "PSAX-AV"], True),
    "aov_area": (["PLAX", "A3C", "PSAX-AV"], True),
    "tr_severity": (["A4C", "Subcostal", "PLAX"], True),
    "pr_severity": (["PSAX-AV", "Subcostal", "PLAX"], True),
    # --- Pressure / hemodynamic measurements (B-mode for Pillar 2) ---
    "rvsp": (["A4C", "Subcostal"], True),
    "pa_pressure": (["A4C", "Subcostal", "PLAX"], False),
    # --- Diastolic function ---
    "mv_ea": (["A4C", "A2C", "A3C"], False),
    "mv_ee": (["A4C"], True),  # B-mode for Pillar 2 (E/e' from structure)
    "mv_ee_medial": (["A4C"], False),
    "mv_dt": (["A4C", "A2C", "A3C"], False),
    "diastolic_function": (["A4C", "A2C", "A3C", "PLAX"], False),
    "lvot_vti": (["A5C", "A3C"], False),
    "cardiac_output": (["A5C", "A3C"], False),
    # --- Disease detection ---
    "disease_hcm": (["PLAX", "PSAX-PM", "PSAX-MV", "A4C"], False),
    "disease_amyloidosis": (["PLAX", "A4C", "PSAX-AV", "PSAX-MV", "PSAX-PM", "PSAX-AP"], False),
    "disease_bicuspid_av": (["PLAX", "PSAX-AV", "A3C"], False),
    "disease_myxomatous_mv": (["A4C", "A2C", "PLAX"], False),
    "disease_rheumatic_mv": (["A4C", "A2C", "PLAX"], False),
    # --- Trajectory (Wendy pillar 3) — same filter as base measurement ---
    # (trajectory CSVs are in labels/trajectory/, handled separately if needed)
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROBE_CSV_DIR = os.path.join(BASE_DIR, "probe_csvs")
VIEW_PREDICTIONS = os.path.join(
    os.path.expanduser("~"), "user-default-efs/vjepa2/classifier/output/view_inference_18m/master_predictions.csv"
)
COLOR_PREDICTIONS = os.path.join(
    os.path.expanduser("~"), "user-default-efs/vjepa2/classifier/output/color_inference_18m/master_predictions.csv"
)


def load_view_lookup(uris_needed):
    """Load view predictions for the given URIs."""
    print(f"Loading view predictions for {len(uris_needed):,} URIs...")
    t0 = time.time()
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
    elapsed = time.time() - t0
    print(f"  Matched {len(lookup):,} / {len(uris_needed):,} URIs ({elapsed:.1f}s)")
    return lookup


def load_color_lookup(uris_needed):
    """Load color predictions for the given URIs."""
    print(f"Loading color predictions for {len(uris_needed):,} URIs...")
    t0 = time.time()
    lookup = {}
    with open(COLOR_PREDICTIONS) as f:
        next(f)  # skip header
        for line in f:
            parts = line.rstrip("\n").split(",")
            uri = parts[0]
            if uri in uris_needed:
                lookup[uri] = parts[1]  # "Yes" or "No"
                if len(lookup) == len(uris_needed):
                    break
    elapsed = time.time() - t0
    print(f"  Matched {len(lookup):,} / {len(uris_needed):,} URIs ({elapsed:.1f}s)")
    return lookup


def collect_uris(task_dir):
    """Collect all unique URIs from train/val/test CSVs."""
    uris = set()
    for split in ["train.csv", "val.csv", "test.csv"]:
        path = os.path.join(task_dir, split)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                uri = line.strip().split(" ")[0]
                uris.add(uri)
    return uris


def filter_csv(input_path, output_path, view_lookup, allowed_views, color_lookup=None):
    """Filter a probe CSV to keep only clips matching allowed views (and optionally B-mode only)."""
    allowed = set(allowed_views)
    total = 0
    kept = 0
    missing_view = 0
    missing_color = 0
    studies = set()
    kept_studies = set()

    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            total += 1
            parts = line.strip().split(" ")
            uri = parts[0]
            # Extract study ID (parent directory of clip)
            study_id = "/".join(uri.split("/")[:-1])
            studies.add(study_id)

            view = view_lookup.get(uri)
            if view is None:
                missing_view += 1
                continue
            if view not in allowed:
                continue

            if color_lookup is not None:
                color = color_lookup.get(uri)
                if color is None:
                    missing_color += 1
                    continue
                if color != "No":  # Keep B-mode only
                    continue

            fout.write(line)
            kept += 1
            kept_studies.add(study_id)

    dropped_studies = len(studies) - len(kept_studies)
    return total, kept, len(studies), len(kept_studies), missing_view, missing_color


def process_task(task_name, allowed_views, bmode_only):
    """Process a single task: filter train/val/test CSVs."""
    task_dir = os.path.join(PROBE_CSV_DIR, task_name)
    if not os.path.exists(task_dir):
        print(f"  SKIP: {task_dir} does not exist")
        return False

    view_str = "+".join(allowed_views)
    bmode_str = "_bmode" if bmode_only else ""
    print(f"\n{'='*60}")
    print(f"Task: {task_name}")
    print(f"Views: {view_str}{' (B-mode only)' if bmode_only else ''}")
    print(f"{'='*60}")

    # Collect all URIs first
    uris = collect_uris(task_dir)

    # Load lookups (only for URIs in this task)
    view_lookup = load_view_lookup(uris)
    color_lookup = load_color_lookup(uris) if bmode_only else None

    # Filter each split
    for split in ["train", "val", "test"]:
        input_path = os.path.join(task_dir, f"{split}.csv")
        if not os.path.exists(input_path):
            continue
        output_path = os.path.join(task_dir, f"{split}_vf.csv")

        total, kept, n_studies, n_kept_studies, miss_v, miss_c = filter_csv(
            input_path, output_path, view_lookup, allowed_views, color_lookup
        )
        pct = 100 * kept / total if total > 0 else 0
        dropped = n_studies - n_kept_studies
        print(f"  {split}: {kept:,} / {total:,} clips kept ({pct:.1f}%)")
        print(f"    Studies: {n_kept_studies:,} / {n_studies:,} (dropped {dropped:,} with no matching clips)")
        if miss_v > 0:
            print(f"    Warning: {miss_v:,} clips missing view prediction")
        if miss_c > 0:
            print(f"    Warning: {miss_c:,} clips missing color prediction")

    # Write filter metadata
    meta = {
        "task": task_name,
        "allowed_views": allowed_views,
        "bmode_only": bmode_only,
        "view_predictions": VIEW_PREDICTIONS,
        "color_predictions": COLOR_PREDICTIONS if bmode_only else None,
    }
    meta_path = os.path.join(task_dir, "viewfilter_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return True


def main():
    parser = argparse.ArgumentParser(description="Build view-filtered probe CSVs")
    parser.add_argument("--task", type=str, help="Task name (must match probe_csvs/ subdirectory)")
    parser.add_argument("--views", nargs="+", help="Allowed view classes")
    parser.add_argument("--bmode_only", action="store_true", help="Keep B-mode only (exclude color Doppler)")
    parser.add_argument("--all", action="store_true", help="Process all tasks with recommended filters")
    parser.add_argument("--list", action="store_true", help="List available task filters")
    args = parser.parse_args()

    if args.list:
        print("Available task filters:")
        for task, (views, bmode) in sorted(TASK_FILTERS.items()):
            bmode_str = " [B-mode only]" if bmode else ""
            print(f"  {task:25s} -> {', '.join(views)}{bmode_str}")
        return

    if args.all:
        processed = 0
        skipped = 0
        for task_name, (views, bmode) in sorted(TASK_FILTERS.items()):
            if process_task(task_name, views, bmode):
                processed += 1
            else:
                skipped += 1
        print(f"\nDone: {processed} tasks processed, {skipped} skipped (CSVs not yet built)")
        return

    if args.task:
        if args.views:
            views = args.views
            bmode = args.bmode_only
        elif args.task in TASK_FILTERS:
            views, bmode = TASK_FILTERS[args.task]
        else:
            print(f"Error: no default filter for '{args.task}'. Provide --views explicitly.")
            sys.exit(1)
        process_task(args.task, views, bmode)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
