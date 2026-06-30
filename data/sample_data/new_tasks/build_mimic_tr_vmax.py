"""Build MIMIC test CSV for TR Vmax (zero-shot transfer from UHN probes).

Extracts tr_velocity from echo_structured_measurement, matches to studies within
±1 day, filters to A4C view clips (matching UHN probe training views).

Output: experiments/nature_medicine/mimic/probe_csvs/tr_vmax_a4c/all.csv
Format: <s3_path> <raw_float_label> (space-delimited)

The UHN probes were trained on A4C, color allowed (not B-mode restricted).
At inference we supply UHN zscore_params (not MIMIC-derived).
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
OUT_DIR = REPO_ROOT / "experiments/nature_medicine/mimic/probe_csvs/tr_vmax_a4c"

ALLOWED_VIEWS = {"A4C"}
BMODE_ONLY = False
DB_MEASUREMENT = "tr_velocity"
VALUE_RANGE = (0.5, 5.0)
UNIT = "m/s"
MATCH_WINDOW_DAYS = 1

STUDY_ID_RE = re.compile(r"/s(\d+)/")


def study_id_from_path(path: str) -> str:
    m = STUDY_ID_RE.search(path)
    return m.group(1) if m else ""


def load_view_filtered_uris() -> set[str]:
    uris: set[str] = set()
    with VIEW_MANIFEST.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row["view_status"] == "OK" and row["view"] in ALLOWED_VIEWS:
                uris.add(row["s3_uri"])
    return uris


def build_label_map() -> dict[str, float]:
    lo, hi = VALUE_RANGE
    con = sqlite3.connect(str(MIMIC_DB))

    meas_rows = con.execute(
        "SELECT subject_id, measurement_datetime, CAST(result AS REAL) "
        "FROM echo_structured_measurement "
        "WHERE test_type='tte' AND measurement = ? AND result IS NOT NULL AND result != ''",
        (DB_MEASUREMENT,),
    ).fetchall()

    study_rows = con.execute(
        "SELECT study_id, subject_id, study_datetime FROM echo_study_list"
    ).fetchall()
    con.close()

    import datetime as _dt

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

    print(f"  DB measurements: {len(meas_rows):,} rows, {n_out_of_range:,} out of range [{lo}, {hi}]")
    print(f"  Subjects with measurements: {len(by_subject):,}")
    print(f"  Studies with matched labels: {len(out):,}")
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


def main() -> None:
    print("=" * 60)
    print("Building MIMIC TR Vmax test CSV (A4C, zero-shot transfer)")
    print("=" * 60)

    print("\nStep 1: Loading A4C view manifest...")
    a4c_uris = load_view_filtered_uris()
    print(f"  A4C clips: {len(a4c_uris):,}")

    print("\nStep 2: Extracting TR velocity labels from mimic.db...")
    labels = build_label_map()

    print("\nStep 3: Joining clips with labels (by study_id)...")
    all_rows: list[tuple[str, float]] = []
    study_ids: set[str] = set()
    for uri in a4c_uris:
        sid = study_id_from_path(uri)
        if not sid:
            continue
        val = labels.get(sid)
        if val is None:
            continue
        all_rows.append((uri, val))
        study_ids.add(sid)

    all_rows.sort(key=lambda x: x[0])
    print(f"  Matched: {len(all_rows):,} clips, {len(study_ids):,} studies")

    print("\nStep 4: Writing output...")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / "all.csv"
    with out_csv.open("w") as fh:
        for path, val in all_rows:
            fh.write(f"{path} {val:.6f}\n")
    print(f"  -> {out_csv}")

    vals = [v for _, v in all_rows]
    stats = numeric_stats(vals)
    cps = clips_per_study_stats(all_rows)

    meta = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "task_type": "regression",
        "target": "tr_vmax",
        "source": "echo_structured_measurement",
        "db_measurement": DB_MEASUREMENT,
        "matching_window_days": MATCH_WINDOW_DAYS,
        "view_filter": sorted(ALLOWED_VIEWS),
        "bmode_only": BMODE_ONLY,
        "value_range": list(VALUE_RANGE),
        "unit": UNIT,
        "total_clips": len(all_rows),
        "total_studies": len(study_ids),
        "labelled_studies_in_db": len(labels),
        "label_stats": stats,
        "clips_per_study": cps,
    }
    meta_path = OUT_DIR / "task_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"  -> {meta_path}")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Task:          TR Vmax (tricuspid regurgitation peak velocity)")
    print(f"  Views:         {sorted(ALLOWED_VIEWS)}")
    print(f"  B-mode only:   {BMODE_ONLY}")
    print(f"  Value range:   {VALUE_RANGE} {UNIT}")
    print(f"  Total clips:   {len(all_rows):,}")
    print(f"  Total studies: {len(study_ids):,}")
    print(f"  Label stats:   mean={stats['mean']}, std={stats['std']}, median={stats['median']}")
    print(f"  Clips/study:   min={cps['min']}, median={cps['median']}, max={cps['max']}, mean={cps['mean']}")


if __name__ == "__main__":
    main()
