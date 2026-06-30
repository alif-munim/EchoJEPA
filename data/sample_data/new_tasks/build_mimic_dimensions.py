"""Build MIMIC test CSVs for LA AP diam, Ao Root diam, and RV Basal diam.

All three are B-mode trained tasks, so we create B-mode-only test sets.
Extracts labels from echo_structured_measurement, matches to studies within
±1 day, filters to correct views + B-mode only.

Tasks:
  - LA AP diam: la_dimen, PLAX, B-mode only, range [1.5, 7.0] cm
  - Ao Root diam: ascending_diam, PLAX, B-mode only, range [1.5, 6.0] cm
  - RV Basal diam: rv_diam, A4C, B-mode only, range [1.0, 7.0] cm

Output per task: experiments/nature_medicine/mimic/probe_csvs/{task}_all/all.csv
Format: <s3_path> <raw_float_label> (space-delimited)
"""

from __future__ import annotations

import csv
import json
import re
import sqlite3
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path("/mnt/custom-file-systems/efs/fs-0049217cdf69186d7_fsap-0fa7145b64eaa046b/vjepa2")
MIMIC_DB = REPO_ROOT / "uhn_echo/nature_medicine/data_exploration/mimic/mimic.db"
VIEW_MANIFEST = Path("/home/sagemaker-user/user-default-efs/vjepa2/classifier/output/mimic_view_predictions.csv")
COLOR_MANIFEST = Path("/home/sagemaker-user/user-default-efs/vjepa2/classifier/output/mimic_color_predictions.csv")
OUT_BASE = REPO_ROOT / "experiments/nature_medicine/mimic/probe_csvs"

MATCH_WINDOW_DAYS = 1
STUDY_ID_RE = re.compile(r"/s(\d+)/")

TASKS = {
    "la_ap_diam": {
        "db_measurement": "la_dimen",
        "views": {"PLAX"},
        "bmode_only": True,
        "value_range": (1.5, 7.0),
        "unit": "cm",
        "description": "LA anterior-posterior diameter",
    },
    "ao_root_diam": {
        "db_measurement": "ascending_diam",
        "views": {"PLAX"},
        "bmode_only": True,
        "value_range": (1.5, 6.0),
        "unit": "cm",
        "description": "Aortic root diameter (ascending aorta at sinus level)",
    },
    "rv_basal_diam": {
        "db_measurement": "rv_diam",
        "views": {"A4C"},
        "bmode_only": True,
        "value_range": (1.0, 7.0),
        "unit": "cm",
        "description": "RV basal diameter",
    },
}


def study_id_from_path(path: str) -> str:
    m = STUDY_ID_RE.search(path)
    return m.group(1) if m else ""


def load_view_manifest() -> dict[str, str]:
    """Load {s3_uri: view} for all OK clips."""
    uri_to_view: dict[str, str] = {}
    with VIEW_MANIFEST.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row["view_status"] == "OK":
                uri_to_view[row["s3_uri"]] = row["view"]
    return uri_to_view


def load_bmode_uris() -> set[str]:
    """Load S3 URIs classified as B-mode."""
    uris: set[str] = set()
    with COLOR_MANIFEST.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row["color_status"] == "OK" and row["color"] == "No":
                uris.add(row["s3_uri"])
    return uris


def load_study_list() -> list[tuple[str, str, str]]:
    """Load (study_id, subject_id, study_datetime) from echo_study_list."""
    con = sqlite3.connect(str(MIMIC_DB))
    rows = con.execute("SELECT study_id, subject_id, study_datetime FROM echo_study_list").fetchall()
    con.close()
    return rows


def build_label_map(measurement: str, value_range: tuple[float, float]) -> dict[str, float]:
    """Build {study_id: float_value} from mimic.db, closest match within ±1 day."""
    import datetime as _dt

    lo, hi = value_range
    con = sqlite3.connect(str(MIMIC_DB))

    meas_rows = con.execute(
        "SELECT subject_id, measurement_datetime, CAST(result AS REAL) "
        "FROM echo_structured_measurement "
        "WHERE test_type='tte' AND measurement = ? AND result IS NOT NULL AND result != ''",
        (measurement,),
    ).fetchall()

    study_rows = con.execute(
        "SELECT study_id, subject_id, study_datetime FROM echo_study_list"
    ).fetchall()
    con.close()

    def parse_dt(s: str) -> float:
        try:
            return _dt.datetime.fromisoformat(s).timestamp()
        except Exception:
            return float("nan")

    by_subject: dict[str, list[tuple[float, float]]] = {}
    n_out_of_range = 0
    for subj, mdt, val in meas_rows:
        try:
            v = float(val)
        except (TypeError, ValueError):
            continue
        if not (lo <= v <= hi):
            n_out_of_range += 1
            continue
        t = parse_dt(mdt)
        if t != t:
            continue
        by_subject.setdefault(subj, []).append((t, v))

    window_sec = MATCH_WINDOW_DAYS * 86400.0
    out: dict[str, float] = {}
    for study_id, subj, sdt in study_rows:
        ms = by_subject.get(subj)
        if not ms:
            continue
        t0 = parse_dt(sdt)
        if t0 != t0:
            continue
        best_dt = None
        best_val = None
        for t, v in ms:
            d = abs(t - t0)
            if d > window_sec:
                continue
            if best_dt is None or d < best_dt:
                best_dt = d
                best_val = v
        if best_val is not None:
            out[study_id] = best_val

    print(f"    DB rows: {len(meas_rows):,}, out of range: {n_out_of_range:,}")
    print(f"    Subjects with measurements: {len(by_subject):,}")
    print(f"    Studies matched: {len(out):,}")
    return out


def numeric_stats(vals: list[float]) -> dict:
    if not vals:
        return {"n": 0}
    s = sorted(vals)
    n = len(s)
    mean = sum(s) / n
    std = (sum((v - mean) ** 2 for v in s) / n) ** 0.5
    return {
        "n": n,
        "min": round(s[0], 3),
        "p5": round(s[n // 20], 3) if n >= 20 else round(s[0], 3),
        "p25": round(s[n // 4], 3),
        "median": round(s[n // 2], 3),
        "p75": round(s[3 * n // 4], 3),
        "p95": round(s[19 * n // 20], 3) if n >= 20 else round(s[-1], 3),
        "max": round(s[-1], 3),
        "mean": round(mean, 4),
        "std": round(std, 4),
    }


def clips_per_study_stats(rows: list[tuple[str, float]]) -> dict:
    c: dict[str, int] = defaultdict(int)
    for p, _ in rows:
        sid = study_id_from_path(p)
        if sid:
            c[sid] += 1
    vals = sorted(c.values())
    if not vals:
        return {"min": 0, "median": 0, "max": 0, "mean": 0.0}
    n = len(vals)
    median = vals[n // 2] if n % 2 == 1 else (vals[n // 2 - 1] + vals[n // 2]) / 2
    return {"min": vals[0], "median": median, "max": vals[-1], "mean": round(sum(vals) / n, 2)}


def build_task(task_name: str, cfg: dict, uri_to_view: dict[str, str], bmode_uris: set[str]) -> None:
    print(f"\n{'='*60}")
    print(f"  {task_name} — {cfg['description']}")
    print(f"  DB measurement: {cfg['db_measurement']}, views: {sorted(cfg['views'])}")
    print(f"  B-mode only: {cfg['bmode_only']}, range: {cfg['value_range']} {cfg['unit']}")
    print(f"{'='*60}")

    # Step 1: Build label map
    print(f"  Step 1: Extracting labels from mimic.db...")
    labels = build_label_map(cfg["db_measurement"], cfg["value_range"])

    # Step 2: Filter clips by view + bmode
    print(f"  Step 2: Filtering clips (view + B-mode)...")
    eligible_uris: list[str] = []
    for uri, view in uri_to_view.items():
        if view not in cfg["views"]:
            continue
        if cfg["bmode_only"] and uri not in bmode_uris:
            continue
        eligible_uris.append(uri)
    print(f"    Eligible clips (view + bmode filter): {len(eligible_uris):,}")

    # Step 3: Join with labels
    print(f"  Step 3: Joining clips with labels...")
    all_rows: list[tuple[str, float]] = []
    study_ids: set[str] = set()
    for uri in eligible_uris:
        sid = study_id_from_path(uri)
        if not sid:
            continue
        val = labels.get(sid)
        if val is None:
            continue
        all_rows.append((uri, val))
        study_ids.add(sid)

    all_rows.sort(key=lambda x: x[0])
    print(f"    Matched: {len(all_rows):,} clips, {len(study_ids):,} studies")

    # Step 4: Write output
    out_dir = OUT_BASE / f"{task_name}_all"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "all.csv"
    with out_csv.open("w") as fh:
        for path, val in all_rows:
            fh.write(f"{path} {val:.6f}\n")
    print(f"    -> {out_csv}")

    # Write metadata
    vals = [v for _, v in all_rows]
    stats = numeric_stats(vals)
    cps = clips_per_study_stats(all_rows)

    meta = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "task_type": "regression",
        "target": task_name,
        "description": cfg["description"],
        "source": "echo_structured_measurement",
        "db_measurement": cfg["db_measurement"],
        "matching_window_days": MATCH_WINDOW_DAYS,
        "view_filter": sorted(cfg["views"]),
        "bmode_only": cfg["bmode_only"],
        "value_range": list(cfg["value_range"]),
        "unit": cfg["unit"],
        "total_clips": len(all_rows),
        "total_studies": len(study_ids),
        "labelled_studies_in_db": len(labels),
        "label_stats": stats,
        "clips_per_study": cps,
    }
    meta_path = out_dir / "task_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"    Label: mean={stats['mean']}, std={stats['std']}, median={stats['median']}, range=[{stats['min']}, {stats['max']}]")
    print(f"    Clips/study: min={cps['min']}, median={cps['median']}, max={cps['max']}, mean={cps['mean']}")


def main() -> None:
    print("=" * 60)
    print("Building MIMIC dimension CSVs (B-mode only, zero-shot transfer)")
    print("=" * 60)

    print("\nLoading view manifest...")
    uri_to_view = load_view_manifest()
    print(f"  Total OK clips: {len(uri_to_view):,}")

    print("Loading B-mode color manifest...")
    bmode_uris = load_bmode_uris()
    print(f"  B-mode clips: {len(bmode_uris):,}")

    for task_name, cfg in TASKS.items():
        build_task(task_name, cfg, uri_to_view, bmode_uris)

    # Summary
    print(f"\n\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for task_name in TASKS:
        meta_path = OUT_BASE / f"{task_name}_all" / "task_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            print(f"  {task_name:15s}: {meta['total_clips']:>7,} clips, {meta['total_studies']:>5,} studies, "
                  f"mean={meta['label_stats']['mean']:.2f} {TASKS[task_name]['unit']}")


if __name__ == "__main__":
    main()
