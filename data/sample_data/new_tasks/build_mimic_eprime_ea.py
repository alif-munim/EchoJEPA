"""Build MIMIC test CSVs for E' medial and MV E/A ratio (zero-shot transfer).

E' medial:
  - DB measurement: sept_e_prime (septal = medial in echo terminology)
  - Views: A4C
  - B-mode only
  - Range: [0.02, 0.25] cm/s
  - Output: e_prime_medial_all/all.csv

MV E/A ratio:
  - DB measurement: mv_peak_e_a
  - Views: A4C, A2C
  - Color-trained task → two test sets:
    - color: all clips (B-mode + color Doppler) at A4C/A2C
    - bmode: B-mode only clips at A4C/A2C
  - Range: [0.3, 5.0]
  - Output: mv_ea_ratio_all/all_color.csv, mv_ea_ratio_all/all_bmode.csv

Format: <s3_path> <raw_float_label> (space-delimited)
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
COLOR_MANIFEST = Path("/home/sagemaker-user/user-default-efs/vjepa2/classifier/output/mimic_color_predictions.csv")
OUT_BASE = REPO_ROOT / "experiments/nature_medicine/mimic/probe_csvs"

MATCH_WINDOW_DAYS = 1
STUDY_ID_RE = re.compile(r"/s(\d+)/")


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
        "min": round(s[0], 4),
        "p5": round(s[n // 20], 4) if n >= 20 else round(s[0], 4),
        "p25": round(s[n // 4], 4),
        "median": round(s[n // 2], 4),
        "p75": round(s[3 * n // 4], 4),
        "p95": round(s[19 * n // 20], 4) if n >= 20 else round(s[-1], 4),
        "max": round(s[-1], 4),
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


def write_csv(rows: list[tuple[str, float]], dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w") as fh:
        for path, val in rows:
            fh.write(f"{path} {val:.6f}\n")


def main() -> None:
    print("=" * 60)
    print("Building MIMIC E' medial + MV E/A ratio CSVs")
    print("=" * 60)

    # Load manifests (shared)
    print("\nLoading view manifest...")
    uri_to_view = load_view_manifest()
    print(f"  Total OK clips: {len(uri_to_view):,}")

    print("Loading B-mode color manifest...")
    bmode_uris = load_bmode_uris()
    print(f"  B-mode clips: {len(bmode_uris):,}")

    # ============================================================
    # E' medial: A4C, B-mode only
    # ============================================================
    print(f"\n{'='*60}")
    print("  E' medial — sept_e_prime, A4C, B-mode only, [0.02, 0.25] cm/s")
    print(f"{'='*60}")

    print("  Step 1: Extracting labels...")
    eprime_labels = build_label_map("sept_e_prime", (0.02, 0.25))

    print("  Step 2: Filtering clips (A4C + B-mode)...")
    eprime_rows: list[tuple[str, float]] = []
    eprime_studies: set[str] = set()
    for uri, view in uri_to_view.items():
        if view != "A4C":
            continue
        if uri not in bmode_uris:
            continue
        sid = study_id_from_path(uri)
        if not sid:
            continue
        val = eprime_labels.get(sid)
        if val is None:
            continue
        eprime_rows.append((uri, val))
        eprime_studies.add(sid)

    eprime_rows.sort(key=lambda x: x[0])
    print(f"    Matched: {len(eprime_rows):,} clips, {len(eprime_studies):,} studies")

    out_dir = OUT_BASE / "e_prime_medial_all"
    write_csv(eprime_rows, out_dir / "all.csv")

    stats = numeric_stats([v for _, v in eprime_rows])
    cps = clips_per_study_stats(eprime_rows)
    meta = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "task_type": "regression",
        "target": "e_prime_medial",
        "source": "echo_structured_measurement",
        "db_measurement": "sept_e_prime",
        "matching_window_days": MATCH_WINDOW_DAYS,
        "view_filter": ["A4C"],
        "bmode_only": True,
        "value_range": [0.02, 0.25],
        "unit": "cm/s",
        "total_clips": len(eprime_rows),
        "total_studies": len(eprime_studies),
        "labelled_studies_in_db": len(eprime_labels),
        "label_stats": stats,
        "clips_per_study": cps,
    }
    (out_dir / "task_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"    -> {out_dir / 'all.csv'}")
    print(f"    Label: mean={stats['mean']}, std={stats['std']}, median={stats['median']}")
    print(f"    Clips/study: min={cps['min']}, median={cps['median']}, max={cps['max']}")

    # ============================================================
    # MV E/A ratio: A4C + A2C, color-trained → color + bmode test
    # ============================================================
    print(f"\n{'='*60}")
    print("  MV E/A ratio — mv_peak_e_a, A4C+A2C, color-trained")
    print("  Creating: all_color.csv (all clips) + all_bmode.csv (B-mode only)")
    print(f"{'='*60}")

    print("  Step 1: Extracting labels...")
    ea_labels = build_label_map("mv_peak_e_a", (0.3, 5.0))

    print("  Step 2a: Building color test set (A4C+A2C, all clips)...")
    ea_color_rows: list[tuple[str, float]] = []
    ea_color_studies: set[str] = set()
    for uri, view in uri_to_view.items():
        if view not in ("A4C", "A2C"):
            continue
        sid = study_id_from_path(uri)
        if not sid:
            continue
        val = ea_labels.get(sid)
        if val is None:
            continue
        ea_color_rows.append((uri, val))
        ea_color_studies.add(sid)

    ea_color_rows.sort(key=lambda x: x[0])
    print(f"    Color: {len(ea_color_rows):,} clips, {len(ea_color_studies):,} studies")

    print("  Step 2b: Building B-mode test set (A4C+A2C, B-mode only)...")
    ea_bmode_rows: list[tuple[str, float]] = []
    ea_bmode_studies: set[str] = set()
    for uri, view in uri_to_view.items():
        if view not in ("A4C", "A2C"):
            continue
        if uri not in bmode_uris:
            continue
        sid = study_id_from_path(uri)
        if not sid:
            continue
        val = ea_labels.get(sid)
        if val is None:
            continue
        ea_bmode_rows.append((uri, val))
        ea_bmode_studies.add(sid)

    ea_bmode_rows.sort(key=lambda x: x[0])
    print(f"    B-mode: {len(ea_bmode_rows):,} clips, {len(ea_bmode_studies):,} studies")

    out_dir = OUT_BASE / "mv_ea_ratio_all"
    write_csv(ea_color_rows, out_dir / "all_color.csv")
    write_csv(ea_bmode_rows, out_dir / "all_bmode.csv")

    color_stats = numeric_stats([v for _, v in ea_color_rows])
    bmode_stats = numeric_stats([v for _, v in ea_bmode_rows])
    color_cps = clips_per_study_stats(ea_color_rows)
    bmode_cps = clips_per_study_stats(ea_bmode_rows)

    meta = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "task_type": "regression",
        "target": "mv_ea_ratio",
        "source": "echo_structured_measurement",
        "db_measurement": "mv_peak_e_a",
        "matching_window_days": MATCH_WINDOW_DAYS,
        "view_filter": ["A4C", "A2C"],
        "bmode_only": False,
        "training_includes_color": True,
        "value_range": [0.3, 5.0],
        "unit": "ratio",
        "color_test": {
            "csv": "all_color.csv",
            "total_clips": len(ea_color_rows),
            "total_studies": len(ea_color_studies),
            "label_stats": color_stats,
            "clips_per_study": color_cps,
        },
        "bmode_test": {
            "csv": "all_bmode.csv",
            "total_clips": len(ea_bmode_rows),
            "total_studies": len(ea_bmode_studies),
            "label_stats": bmode_stats,
            "clips_per_study": bmode_cps,
        },
        "labelled_studies_in_db": len(ea_labels),
    }
    (out_dir / "task_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"    -> {out_dir / 'all_color.csv'}")
    print(f"    -> {out_dir / 'all_bmode.csv'}")
    print(f"    Color label: mean={color_stats['mean']}, std={color_stats['std']}")
    print(f"    B-mode label: mean={bmode_stats['mean']}, std={bmode_stats['std']}")

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  E' medial:        {len(eprime_rows):>7,} clips, {len(eprime_studies):>5,} studies (A4C, B-mode only)")
    print(f"  MV E/A (color):   {len(ea_color_rows):>7,} clips, {len(ea_color_studies):>5,} studies (A4C+A2C, all)")
    print(f"  MV E/A (bmode):   {len(ea_bmode_rows):>7,} clips, {len(ea_bmode_studies):>5,} studies (A4C+A2C, B-mode only)")


if __name__ == "__main__":
    main()
