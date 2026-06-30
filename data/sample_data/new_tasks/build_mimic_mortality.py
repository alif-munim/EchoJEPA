"""Build MIMIC mortality labels directly from mimic.db and compare to existing CSVs.

Constructs 30-day, 90-day, and 1-year all-cause mortality labels by joining
echo_study_list with hosp_patients (date of death). Outputs per-task CSVs in
probe pipeline format: `<s3_path> <int_label>` (space-delimited).

Also compares resulting labels to the prebuilt CSVs in
data_exploration/mimic/csv/mortality_{30d,90d,1yr}.csv to verify consistency.
"""

from __future__ import annotations

import csv
import json
import re
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path("/mnt/custom-file-systems/efs/fs-0049217cdf69186d7_fsap-0fa7145b64eaa046b/vjepa2")
MIMIC_DB = REPO_ROOT / "uhn_echo/nature_medicine/data_exploration/mimic/mimic.db"
VIEW_MANIFEST = Path("/home/sagemaker-user/user-default-efs/vjepa2/classifier/output/mimic_view_predictions.csv")
SRC_SPLIT_DIR = REPO_ROOT / "experiments/nature_medicine/mimic/probe_csvs/disease_hf_v4.1"
OUT_BASE = REPO_ROOT / "experiments/nature_medicine/mimic/probe_csvs"
EXISTING_CSV_DIR = REPO_ROOT / "uhn_echo/nature_medicine/data_exploration/mimic/csv"

STUDY_ID_RE = re.compile(r"/s(\d+)/")

TASKS = {
    "mortality_30d_v2": {"window_days": 30, "existing_csv": "mortality_30d.csv", "existing_col": "mortality_30d"},
    "mortality_90d_v2": {"window_days": 90, "existing_csv": "mortality_90d.csv", "existing_col": "mortality_90d"},
    "mortality_1yr_v2": {"window_days": 365, "existing_csv": "mortality_1yr.csv", "existing_col": "mortality_1yr"},
}


def study_id_from_path(path: str) -> str:
    m = STUDY_ID_RE.search(path)
    return m.group(1) if m else ""


def build_labels_from_db(window_days: int) -> dict[str, int]:
    """Build {study_id: binary_label} directly from mimic.db."""
    con = sqlite3.connect(str(MIMIC_DB))

    rows = con.execute("""
        SELECT e.study_id, e.study_datetime, p.dod
        FROM echo_study_list e
        JOIN hosp_patients p ON e.subject_id = p.subject_id
    """).fetchall()
    con.close()

    labels: dict[str, int] = {}
    for study_id, study_dt, dod in rows:
        if not study_dt:
            continue
        study_id = str(study_id).strip()

        if not dod or dod.strip() == "":
            labels[study_id] = 0
            continue

        try:
            sd = datetime.fromisoformat(study_dt).date()
            dd = datetime.fromisoformat(dod).date()
        except (ValueError, TypeError):
            labels[study_id] = 0
            continue

        days_to_death = (dd - sd).days
        if days_to_death < 0:
            labels[study_id] = 0
        elif days_to_death <= window_days:
            labels[study_id] = 1
        else:
            labels[study_id] = 0

    return labels


def load_existing_labels(csv_name: str, col_name: str) -> dict[str, int]:
    """Load {study_id: label} from prebuilt CSV for comparison."""
    path = EXISTING_CSV_DIR / csv_name
    labels: dict[str, int] = {}
    with path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            sid = row["study_id"].strip()
            val = row[col_name].strip()
            if sid and val in ("0", "1"):
                labels[sid] = int(val)
    return labels


def load_split_study_ids(split: str) -> set[str]:
    """Get study IDs from the patient-split skeleton."""
    path = SRC_SPLIT_DIR / f"{split}.csv"
    sids: set[str] = set()
    with path.open("r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            s3_path = line.rsplit(" ", 1)[0]
            sid = study_id_from_path(s3_path)
            if sid:
                sids.add(sid)
    return sids


def load_split_clips(split: str) -> list[str]:
    """Get all S3 clip paths from a split file."""
    path = SRC_SPLIT_DIR / f"{split}.csv"
    clips: list[str] = []
    with path.open("r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            s3_path = line.rsplit(" ", 1)[0]
            clips.append(s3_path)
    return clips


def compare_labels(db_labels: dict[str, int], existing_labels: dict[str, int], task_name: str) -> None:
    """Compare DB-derived labels with existing prebuilt CSV labels."""
    common = set(db_labels.keys()) & set(existing_labels.keys())
    only_db = set(db_labels.keys()) - set(existing_labels.keys())
    only_existing = set(existing_labels.keys()) - set(db_labels.keys())

    matches = sum(1 for s in common if db_labels[s] == existing_labels[s])
    mismatches = [(s, db_labels[s], existing_labels[s]) for s in common if db_labels[s] != existing_labels[s]]

    print(f"\n  Comparison with {task_name} existing CSV:")
    print(f"    Common studies: {len(common):,}")
    print(f"    Only in DB-derived: {len(only_db):,}")
    print(f"    Only in existing CSV: {len(only_existing):,}")
    print(f"    Matches: {matches:,} / {len(common):,} ({100*matches/len(common):.2f}%)")
    print(f"    Mismatches: {len(mismatches):,}")

    if mismatches:
        print(f"    First 5 mismatches (study_id, db_label, csv_label):")
        for sid, db_lbl, csv_lbl in mismatches[:5]:
            print(f"      {sid}: db={db_lbl}, csv={csv_lbl}")


def build_task(task_name: str, cfg: dict) -> None:
    window = cfg["window_days"]
    print(f"\n{'='*60}")
    print(f"  {task_name} — {window}-day all-cause mortality")
    print(f"{'='*60}")

    # Build labels from DB
    print(f"  Building labels from mimic.db (window={window} days)...")
    db_labels = build_labels_from_db(window)
    dist = Counter(db_labels.values())
    print(f"    Total studies labelled: {len(db_labels):,}")
    print(f"    Positive (died within {window}d): {dist[1]:,} ({100*dist[1]/len(db_labels):.1f}%)")
    print(f"    Negative: {dist[0]:,}")

    # Compare with existing
    existing_labels = load_existing_labels(cfg["existing_csv"], cfg["existing_col"])
    compare_labels(db_labels, existing_labels, cfg["existing_csv"])

    # Build split CSVs using disease_hf_v4.1 skeleton
    out_dir = OUT_BASE / f"{task_name}_all"
    out_dir.mkdir(parents=True, exist_ok=True)

    total_clips = 0
    total_studies = 0
    split_stats = {}

    for split in ("train", "val", "test"):
        clips = load_split_clips(split)
        rows: list[tuple[str, int]] = []
        studies_seen: set[str] = set()

        for clip_path in clips:
            sid = study_id_from_path(clip_path)
            if not sid or sid not in db_labels:
                continue
            rows.append((clip_path, db_labels[sid]))
            studies_seen.add(sid)

        rows.sort(key=lambda x: x[0])

        out_path = out_dir / f"{split}.csv"
        with out_path.open("w") as fh:
            for path, lbl in rows:
                fh.write(f"{path} {lbl}\n")

        pos_clips = sum(1 for _, l in rows if l == 1)
        pos_studies = sum(1 for s in studies_seen if db_labels[s] == 1)
        total_clips += len(rows)
        total_studies += len(studies_seen)

        split_stats[split] = {
            "clips": len(rows),
            "studies": len(studies_seen),
            "pos_studies": pos_studies,
            "neg_studies": len(studies_seen) - pos_studies,
            "pos_clips": pos_clips,
            "neg_clips": len(rows) - pos_clips,
        }

        print(f"\n  [{split}] {len(rows):,} clips, {len(studies_seen):,} studies "
              f"(pos: {pos_studies}, neg: {len(studies_seen) - pos_studies})")
        print(f"    -> {out_path}")

    # Also write a single all.csv (all clips, no split)
    all_rows: list[tuple[str, int]] = []
    all_studies: set[str] = set()
    for split in ("train", "val", "test"):
        clips = load_split_clips(split)
        for clip_path in clips:
            sid = study_id_from_path(clip_path)
            if not sid or sid not in db_labels:
                continue
            all_rows.append((clip_path, db_labels[sid]))
            all_studies.add(sid)

    all_rows.sort(key=lambda x: x[0])
    all_path = out_dir / "all.csv"
    with all_path.open("w") as fh:
        for path, lbl in all_rows:
            fh.write(f"{path} {lbl}\n")
    print(f"\n  [all] {len(all_rows):,} clips, {len(all_studies):,} studies -> {all_path}")

    # Metadata
    meta = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "task_type": "classification",
        "num_classes": 2,
        "class_labels": ["alive", f"dead_within_{window}d"],
        "target": task_name,
        "source": "mimic.db (echo_study_list JOIN hosp_patients on dod)",
        "window_days": window,
        "cohort_skeleton": str(SRC_SPLIT_DIR.relative_to(REPO_ROOT)),
        "total_clips": total_clips,
        "total_studies": total_studies,
        "label_distribution": {"positive": dist[1], "negative": dist[0]},
        "splits": split_stats,
    }
    meta_path = out_dir / "task_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"  -> {meta_path}")


def main() -> None:
    print("=" * 60)
    print("Building MIMIC mortality labels from mimic.db")
    print("and comparing to existing prebuilt CSVs")
    print("=" * 60)

    for task_name, cfg in TASKS.items():
        build_task(task_name, cfg)

    print(f"\n\n{'='*60}")
    print("DONE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
