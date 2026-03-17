"""Fix troponin_t.csv and creatinine.csv labels using minimum positive timegap from lab CSVs."""

import re
import shutil
import sys

import pandas as pd
from pathlib import Path

BASE = Path("/mnt/pool/datasets/CY/EchoJEPA")
LOG_FILE = BASE / "EchoJEPA" / "fix_lab_labels.log"
MAX_TIMEGAP_HOURS = 48

TASKS = [
    {
        "task_csv": BASE / "embeddings/mimic/data/csv/nature_medicine/mimic/troponin_t.csv",
        "timegap_csv": BASE / "mimic-iv-echo/lab_troponin_t_timegap.csv",
        "name": "troponin_t",
    },
    {
        "task_csv": BASE / "embeddings/mimic/data/csv/nature_medicine/mimic/creatinine.csv",
        "timegap_csv": BASE / "mimic-iv-echo/lab_creatinine_timegap.csv",
        "name": "creatinine",
    },
]

STUDY_ID_RE = re.compile(r"/s(\d+)/")


def extract_study_id(s3_path):
    m = STUDY_ID_RE.search(s3_path)
    return int(m.group(1)) if m else None


def build_best_label_map(timegap_csv):
    """For each study_id, find the valuenum and timegap with the minimum positive time_gap_hours."""
    df = pd.read_csv(timegap_csv)
    # Keep only positive timegap with valid valuenum
    df = df[(df["time_gap_hours"] > 0) & df["valuenum"].notna()].copy()
    if df.empty:
        return {}, {}
    # For each study_id, pick the row with minimum positive time_gap_hours
    idx = df.groupby("study_id")["time_gap_hours"].idxmin()
    best = df.loc[idx, ["study_id", "valuenum", "time_gap_hours"]].set_index("study_id")
    label_map = best["valuenum"].to_dict()
    timegap_map = best["time_gap_hours"].to_dict()
    return label_map, timegap_map


def log(f, msg=""):
    print(msg)
    f.write(msg + "\n")


def fix_task(task_csv, timegap_csv, name, logf):
    log(logf, f"\n{'='*60}")
    log(logf, f"Processing: {name}")
    log(logf, f"{'='*60}")

    # Build lookup: study_id -> best label, study_id -> timegap
    label_map, timegap_map = build_best_label_map(timegap_csv)
    log(logf, f"Timegap CSV: {len(label_map)} study_ids with valid positive timegap")

    # Read task CSV (space-delimited, no header)
    lines = Path(task_csv).read_text().strip().split("\n")
    log(logf, f"Task CSV: {len(lines)} rows")

    # Collect unique study_ids in task CSV
    study_ids_in_task = set()
    for line in lines:
        s3_path = line.rsplit(" ", 1)[0]
        sid = extract_study_id(s3_path)
        if sid:
            study_ids_in_task.add(sid)
    log(logf, f"Unique study_ids in task CSV: {len(study_ids_in_task)}")

    updated_lines = []
    updated_studies = {}  # study_id -> (old_label, new_label, timegap)
    removed_no_timegap = set()
    removed_too_far = {}  # study_id -> timegap

    for line in lines:
        s3_path, old_label_str = line.rsplit(" ", 1)
        sid = extract_study_id(s3_path)
        old_label = float(old_label_str)

        if sid not in label_map:
            # No valid positive timegap — remove
            removed_no_timegap.add(sid)
            continue

        gap = timegap_map[sid]
        if gap > MAX_TIMEGAP_HOURS:
            # Minimum timegap > 48h — remove
            removed_too_far[sid] = gap
            continue

        new_label = label_map[sid]
        updated_lines.append(f"{s3_path} {new_label}")
        if sid not in updated_studies and old_label != new_label:
            updated_studies[sid] = (old_label, new_label, gap)

    # Log summary
    log(logf, f"\nStudies updated (label changed): {len(updated_studies)}")
    log(logf, f"Studies removed (no positive timegap): {len(removed_no_timegap)}")
    log(logf, f"Studies removed (min timegap > {MAX_TIMEGAP_HOURS}h): {len(removed_too_far)}")
    log(logf, f"Rows kept: {len(updated_lines)} (was {len(lines)})")

    log(logf, f"\n--- Updated labels (study_id: old -> new [timegap_hours]) ---")
    for sid in sorted(updated_studies):
        old, new, gap = updated_studies[sid]
        log(logf, f"  {sid}: {old} -> {new} [{gap:.1f}h]")

    log(logf, f"\n--- Removed study_ids (no positive timegap) ---")
    for sid in sorted(removed_no_timegap):
        log(logf, f"  {sid}")

    log(logf, f"\n--- Removed study_ids (min timegap > {MAX_TIMEGAP_HOURS}h) ---")
    for sid in sorted(removed_too_far):
        log(logf, f"  {sid}: {removed_too_far[sid]:.1f}h")

    # Write back
    backup = Path(str(task_csv) + ".bak")
    shutil.copy2(task_csv, backup)
    log(logf, f"\nBackup saved to {backup}")

    Path(task_csv).write_text("\n".join(updated_lines) + "\n")
    log(logf, f"Written {len(updated_lines)} rows to {task_csv}")


if __name__ == "__main__":
    with open(LOG_FILE, "w") as logf:
        for task in TASKS:
            fix_task(task["task_csv"], task["timegap_csv"], task["name"], logf)
        log(logf, f"\nLog saved to {LOG_FILE}")
