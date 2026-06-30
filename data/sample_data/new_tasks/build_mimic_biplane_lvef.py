"""Build MIMIC Simpson's biplane LVEF regression labels from mimic.db.

Extracts 'biplane_lvef' measurements from echo_structured_measurement (TTE only),
matches to echo studies via subject_id + ±1 day window, and produces probe-pipeline
CSVs. Filters to plausible range [10, 85]% and A2C/A4C views (Simpson's method uses
these views).

Also compares to the existing lvef_structured splits (which use visual-estimate 'lvef'
field, heavily rounded to 5% increments) to quantify overlap and label differences.

Output: experiments/nature_medicine/mimic/probe_csvs/biplane_lvef_structured/
  - train.csv, val.csv, test.csv (format: `<s3_path> <float_label>`)
  - all.csv (unsplit, all clips)
  - zscore_params.json
  - task_meta.json
"""

from __future__ import annotations

import csv
import json
import re
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path("/mnt/custom-file-systems/efs/fs-0049217cdf69186d7_fsap-0fa7145b64eaa046b/vjepa2")
MIMIC_DB = REPO_ROOT / "uhn_echo/nature_medicine/data_exploration/mimic/mimic.db"
VIEW_MANIFEST = Path("/home/sagemaker-user/user-default-efs/vjepa2/classifier/output/mimic_view_predictions.csv")
SRC_SPLIT_DIR = REPO_ROOT / "experiments/nature_medicine/mimic/probe_csvs/disease_hf_v4.1"
EXISTING_LVEF_DIR = REPO_ROOT / "experiments/nature_medicine/mimic/probe_csvs/lvef_structured"
OUT_DIR = REPO_ROOT / "experiments/nature_medicine/mimic/probe_csvs/biplane_lvef_structured"

STUDY_ID_RE = re.compile(r"/s(\d+)/")
ALLOWED_VIEWS = {"A2C", "A4C"}
MATCHING_WINDOW_DAYS = 1
VALUE_RANGE = (10.0, 85.0)


def study_id_from_path(path: str) -> str:
    m = STUDY_ID_RE.search(path)
    return m.group(1) if m else ""


def build_labels_from_db() -> dict[str, float]:
    """Build {study_id: biplane_lvef} from mimic.db using ±1 day matching."""
    con = sqlite3.connect(str(MIMIC_DB))

    echo_studies = con.execute(
        "SELECT study_id, subject_id, study_datetime FROM echo_study_list"
    ).fetchall()

    measurements = con.execute(
        "SELECT subject_id, measurement_datetime, result "
        "FROM echo_structured_measurement "
        "WHERE measurement = 'biplane_lvef' AND test_type = 'tte' "
        "AND result IS NOT NULL AND result != ''"
    ).fetchall()
    con.close()

    by_subject: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for subj, mdt, result in measurements:
        try:
            val = float(result)
        except (ValueError, TypeError):
            continue
        if val < VALUE_RANGE[0] or val > VALUE_RANGE[1]:
            continue
        try:
            ts = datetime.fromisoformat(mdt).timestamp()
        except (ValueError, TypeError):
            continue
        by_subject[subj].append((ts, val))

    window_sec = MATCHING_WINDOW_DAYS * 86400.0
    labels: dict[str, float] = {}
    for study_id, subj, sdt in echo_studies:
        ms = by_subject.get(subj)
        if not ms:
            continue
        try:
            t0 = datetime.fromisoformat(sdt).timestamp()
        except (ValueError, TypeError):
            continue

        best_dt = None
        best_val = None
        for t, val in ms:
            d = abs(t - t0)
            if d > window_sec:
                continue
            if best_dt is None or d < best_dt:
                best_dt = d
                best_val = val

        if best_val is not None:
            labels[str(study_id).strip()] = best_val

    return labels


def load_view_filtered_uris() -> set[str]:
    """Load S3 URIs for clips with allowed views."""
    uris: set[str] = set()
    with VIEW_MANIFEST.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row["view_status"] == "OK" and row["view"] in ALLOWED_VIEWS:
                uris.add(row["s3_uri"])
    return uris


def load_split_clips(split: str) -> list[str]:
    """Get all S3 clip paths from the patient-split skeleton."""
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


def load_existing_labels() -> dict[str, float]:
    """Load study-level labels from existing lvef_structured for comparison."""
    labels: dict[str, float] = {}
    for split in ("train", "val", "test"):
        path = EXISTING_LVEF_DIR / f"{split}.csv"
        if not path.exists():
            continue
        with path.open("r") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                s3_path, val = line.rsplit(" ", 1)
                sid = study_id_from_path(s3_path)
                if sid and sid not in labels:
                    try:
                        labels[sid] = float(val)
                    except ValueError:
                        pass
    return labels


def stats_summary(vals: list[float]) -> dict:
    if not vals:
        return {"n": 0}
    s = sorted(vals)
    n = len(s)
    mean = sum(s) / n
    var = sum((v - mean) ** 2 for v in s) / n
    std = var ** 0.5
    return {
        "n": n,
        "min": round(s[0], 2),
        "p25": round(s[n // 4], 2),
        "median": round(s[n // 2], 2),
        "p75": round(s[3 * n // 4], 2),
        "max": round(s[-1], 2),
        "mean": round(mean, 4),
        "std": round(std, 4),
    }


def main() -> None:
    print("=" * 60)
    print("Building MIMIC Simpson's biplane LVEF from mimic.db")
    print("=" * 60)

    # Step 1: Build labels
    print("\nStep 1: Extracting biplane_lvef from echo_structured_measurement...")
    labels = build_labels_from_db()
    vals = list(labels.values())
    s = stats_summary(vals)
    print(f"  Studies with biplane_lvef [10-85]: {len(labels):,}")
    print(f"  Mean: {s['mean']}, Std: {s['std']}, Median: {s['median']}")
    print(f"  Range: [{s['min']}, {s['max']}]")

    # Step 2: Load view filter
    print(f"\nStep 2: Loading view manifest (views: {sorted(ALLOWED_VIEWS)})...")
    view_uris = load_view_filtered_uris()
    print(f"  Eligible A2C/A4C clips: {len(view_uris):,}")

    # Step 3: Compare with existing visual-estimate LVEF
    print("\nStep 3: Comparing with existing lvef_structured (visual estimate)...")
    existing = load_existing_labels()
    common = set(labels.keys()) & set(existing.keys())
    only_biplane = set(labels.keys()) - set(existing.keys())
    only_visual = set(existing.keys()) - set(labels.keys())
    print(f"  Studies in both: {len(common):,}")
    print(f"  Only biplane (no visual): {len(only_biplane):,}")
    print(f"  Only visual (no biplane): {len(only_visual):,}")

    if common:
        diffs = [abs(labels[s] - existing[s]) for s in common]
        mean_diff = sum(diffs) / len(diffs)
        exact = sum(1 for d in diffs if d < 0.5)
        close = sum(1 for d in diffs if d < 5.0)
        print(f"  Mean absolute difference: {mean_diff:.2f}%")
        print(f"  Within 0.5%: {exact:,} ({100*exact/len(common):.1f}%)")
        print(f"  Within 5.0%: {close:,} ({100*close/len(common):.1f}%)")

    # Step 4: Build split CSVs
    print("\nStep 4: Building train/val/test CSVs...")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows: list[tuple[str, float]] = []
    split_stats = {}
    train_vals: list[float] = []

    for split in ("train", "val", "test"):
        clips = load_split_clips(split)
        rows: list[tuple[str, float]] = []
        studies_seen: set[str] = set()

        for clip_path in clips:
            if clip_path not in view_uris:
                continue
            sid = study_id_from_path(clip_path)
            if not sid or sid not in labels:
                continue
            rows.append((clip_path, labels[sid]))
            studies_seen.add(sid)

        rows.sort(key=lambda x: x[0])
        all_rows.extend(rows)

        out_path = OUT_DIR / f"{split}.csv"
        with out_path.open("w") as fh:
            for path, val in rows:
                fh.write(f"{path} {val:.6f}\n")

        study_vals = [labels[s] for s in studies_seen]
        if split == "train":
            train_vals = [v for _, v in rows]

        ss = stats_summary(study_vals)
        cps = defaultdict(int)
        for p, _ in rows:
            cps[study_id_from_path(p)] += 1
        cps_vals = sorted(cps.values()) if cps else [0]

        split_stats[split] = {
            "clips": len(rows),
            "studies": len(studies_seen),
            "lvef_stats": ss,
            "clips_per_study": {
                "min": cps_vals[0],
                "median": cps_vals[len(cps_vals) // 2],
                "max": cps_vals[-1],
                "mean": round(sum(cps_vals) / len(cps_vals), 2),
            },
        }

        print(f"\n  [{split}] {len(rows):,} clips, {len(studies_seen):,} studies")
        print(f"    LVEF: mean={ss['mean']}, std={ss['std']}, range=[{ss['min']}, {ss['max']}]")
        print(f"    -> {out_path}")

    # Write all.csv
    all_rows.sort(key=lambda x: x[0])
    all_path = OUT_DIR / "all.csv"
    with all_path.open("w") as fh:
        for path, val in all_rows:
            fh.write(f"{path} {val:.6f}\n")
    all_studies = set(study_id_from_path(p) for p, _ in all_rows)
    print(f"\n  [all] {len(all_rows):,} clips, {len(all_studies):,} studies -> {all_path}")

    # Z-score params from train
    if train_vals:
        n = len(train_vals)
        mean = sum(train_vals) / n
        std = (sum((v - mean) ** 2 for v in train_vals) / n) ** 0.5
        zscore = {"target_mean": mean, "target_std": std}
        zscore_path = OUT_DIR / "zscore_params.json"
        with zscore_path.open("w") as fh:
            json.dump(zscore, fh)
        print(f"\n  Z-score (train clips): mean={mean:.4f}, std={std:.4f}")
        print(f"  -> {zscore_path}")
    else:
        zscore = {}

    # Metadata
    meta = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "task_type": "regression",
        "target": "biplane_lvef_percent",
        "description": "Simpson's biplane LVEF from echo_structured_measurement",
        "source": "mimic.db: echo_structured_measurement.measurement='biplane_lvef', test_type='tte'",
        "matching_window_days": MATCHING_WINDOW_DAYS,
        "value_range": list(VALUE_RANGE),
        "view_filter": sorted(ALLOWED_VIEWS),
        "cohort_skeleton": str(SRC_SPLIT_DIR.relative_to(REPO_ROOT)),
        "total_labelled_studies": len(labels),
        "zscore_params": zscore,
        "splits": split_stats,
    }
    meta_path = OUT_DIR / "task_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"  -> {meta_path}")


if __name__ == "__main__":
    main()
