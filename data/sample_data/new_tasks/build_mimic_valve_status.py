"""Build MIMIC test CSVs for MV Status and AV Status (zero-shot transfer).

Classification tasks using mv_leaflets and av_leaflets from echo_structured_measurement.
Color allowed (not B-mode restricted), matching UHN training protocol.

MV Status (4-class): mechanical / bioprosthetic / repair / native
  - Views: PLAX, A4C
  - Color allowed
  - mechanical: "Bileaflet", "Mechanical", "Ball and cage", "SIngle tilting disc"
  - bioprosthetic: "Bioprosthesis", "CardiaAQ-Edwards", "Edwards-Sapien", "Sapien 3 TMVR"
  - repair: "Annular ring", "MitraClip", "PASCAL"
  - native: "Normal", "Mild thick", "Mod thick", "Severe thick", "Myxomatous",
            "Elongated", "Mild thick/Myxomatous", "Mild thick/Elongated",
            "Mod thick/Elongated", "Mild thick, elongated", "partial flail mitral leaflet",
            "Abnormal", "Other"
  - Excluded: "Not well seen"

AV Status (4-class): mechanical / surgical_bioprosthetic / tavr / native
  - Views: PLAX, A4C, A3C
  - Color allowed
  - mechanical: "Bileaflet mechanical", "Mechanical", "Single tilting disk",
                "SIngle tilting disc", "Ball and Cage"
  - surgical_bioprosthetic: "Bioprosthesis", "AVR homograft", "Homograft", "Other prosthesis"
  - tavr: "Sapien 3", "CoreValve", "Evolut", "Edwards-Sapien", "Lotus"
  - native: "Nl (3 leaflets)", "Mild thickened (3)", "Mild thickened (?#)", "Mod thickened",
            "Nl (?# leaflets)", "Severe thickened", "can't determine # AV leaflets",
            "Mod thickened (3)", "Bicuspid", "Unicuspid", "three AV leaflets",
            "Bileaflet, mild thickened", "Bicuspid, mod thickened",
            "Bicuspid thin, mobile", "Quadricuspid thickened",
            "Unicuspid/thickened", "Quadricuspid, thin/mobile",
            "abnl aortic valve-no qualifiers"
  - Excluded: "Not well seen"

Output per task: experiments/nature_medicine/mimic/probe_csvs/{task}_all/all.csv
Format: <s3_path> <int_label> (space-delimited)
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
OUT_BASE = REPO_ROOT / "experiments/nature_medicine/mimic/probe_csvs"

STUDY_ID_RE = re.compile(r"/s(\d+)/")

# ---- MV Status class mapping ----
MV_CLASS_MAP = {
    # 0: mechanical
    "Bileaflet": 0,
    "Mechanical": 0,
    "Ball and cage": 0,
    "SIngle tilting disc": 0,
    # 1: bioprosthetic
    "Bioprosthesis": 1,
    "CardiaAQ-Edwards": 1,
    "Edwards-Sapien": 1,
    "Sapien 3 TMVR": 1,
    # 2: repair
    "Annular ring": 2,
    "MitraClip": 2,
    "PASCAL": 2,
    # 3: native
    "Normal": 3,
    "Mild thick": 3,
    "Mod thick": 3,
    "Severe thick": 3,
    "Myxomatous": 3,
    "Elongated": 3,
    "Mild thick/Myxomatous": 3,
    "Mild thick/Elongated": 3,
    "Mod thick/Elongated": 3,
    "Mild thick, elongated": 3,
    "partial flail mitral leaflet": 3,
    "Abnormal": 3,
    "Other": 3,
}

# ---- AV Status class mapping ----
AV_CLASS_MAP = {
    # 0: mechanical
    "Bileaflet mechanical": 0,
    "Mechanical": 0,
    "Single tilting disk": 0,
    "SIngle tilting disc": 0,
    "Ball and Cage": 0,
    # 1: surgical bioprosthetic
    "Bioprosthesis": 1,
    "AVR homograft": 1,
    "Homograft": 1,
    "Other prosthesis": 1,
    # 2: TAVR
    "Sapien 3": 2,
    "CoreValve": 2,
    "Evolut": 2,
    "Edwards-Sapien": 2,
    "Lotus": 2,
    # 3: native
    "Nl (3 leaflets)": 3,
    "Mild thickened (3)": 3,
    "Mild thickened (?#)": 3,
    "Mod thickened": 3,
    "Nl (?# leaflets)": 3,
    "Severe thickened": 3,
    "can't determine # AV leaflets": 3,
    "Mod thickened (3)": 3,
    "Bicuspid": 3,
    "Unicuspid": 3,
    "three AV leaflets": 3,
    "Bileaflet, mild thickened": 3,
    "Bicuspid, mod thickened": 3,
    "Bicuspid thin, mobile": 3,
    "Quadricuspid thickened": 3,
    "Unicuspid/thickened": 3,
    "Quadricuspid, thin/mobile": 3,
    "abnl aortic valve-no qualifiers": 3,
}

TASKS = {
    "mv_status": {
        "db_measurement": "mv_leaflets",
        "class_map": MV_CLASS_MAP,
        "class_names": {0: "mechanical", 1: "bioprosthetic", 2: "repair", 3: "native"},
        "views": {"PLAX", "A4C"},
        "bmode_only": False,
        "description": "Mitral valve status (mechanical/bio/repair/native)",
    },
    "av_status": {
        "db_measurement": "av_leaflets",
        "class_map": AV_CLASS_MAP,
        "class_names": {0: "mechanical", 1: "surgical_bioprosthetic", 2: "tavr", 3: "native"},
        "views": {"PLAX", "A4C", "A3C"},
        "bmode_only": False,
        "description": "Aortic valve status (mechanical/surg-bio/TAVR/native)",
    },
}


def study_id_from_path(path: str) -> str:
    m = STUDY_ID_RE.search(path)
    return m.group(1) if m else ""


def load_view_filtered_uris(allowed_views: set[str]) -> set[str]:
    uris: set[str] = set()
    with VIEW_MANIFEST.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row["view_status"] == "OK" and row["view"] in allowed_views:
                uris.add(row["s3_uri"])
    return uris


def build_label_map(measurement: str, class_map: dict[str, int]) -> dict[str, int]:
    """Build {study_id: class_label} from mimic.db using the class mapping."""
    con = sqlite3.connect(str(MIMIC_DB))

    meas_rows = con.execute(
        "SELECT subject_id, measurement_datetime, result "
        "FROM echo_structured_measurement "
        "WHERE test_type='tte' AND measurement = ? AND result IS NOT NULL AND result != ''",
        (measurement,),
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

    # Index measurements by subject — keep those that map to a class
    by_subject: dict[str, list[tuple[float, str]]] = {}
    n_unmapped = 0
    mapped_counts: dict[int, int] = defaultdict(int)
    for subj, mdt, result in meas_rows:
        if result not in class_map:
            n_unmapped += 1
            continue
        t = parse_dt(mdt)
        if t != t:
            continue
        by_subject.setdefault(subj, []).append((t, result))
        mapped_counts[class_map[result]] += 1

    # Match to closest study within ±1 day
    window_sec = 86400.0
    out: dict[str, int] = {}
    for study_id, subj, sdt in study_rows:
        ms = by_subject.get(subj)
        if not ms:
            continue
        t0 = parse_dt(sdt)
        if t0 != t0:
            continue
        best_dt = None
        best_result = None
        for t, result in ms:
            d = abs(t - t0)
            if d > window_sec:
                continue
            if best_dt is None or d < best_dt:
                best_dt = d
                best_result = result
        if best_result is not None:
            out[study_id] = class_map[best_result]

    print(f"    DB rows: {len(meas_rows):,}, unmapped (excl 'Not well seen' etc): {n_unmapped:,}")
    print(f"    Mapped measurement counts: {dict(sorted(mapped_counts.items()))}")
    print(f"    Subjects with measurements: {len(by_subject):,}")
    print(f"    Studies matched: {len(out):,}")
    return out


def build_task(task_name: str, cfg: dict) -> None:
    print(f"\n{'='*60}")
    print(f"  {task_name} — {cfg['description']}")
    print(f"  DB measurement: {cfg['db_measurement']}, views: {sorted(cfg['views'])}")
    print(f"  B-mode only: {cfg['bmode_only']}, classes: {cfg['class_names']}")
    print(f"{'='*60}")

    # Step 1: Build label map
    print(f"  Step 1: Extracting labels from mimic.db...")
    labels = build_label_map(cfg["db_measurement"], cfg["class_map"])

    # Step 2: Load view-filtered clips
    print(f"  Step 2: Loading view-filtered clips ({sorted(cfg['views'])})...")
    eligible_uris = load_view_filtered_uris(cfg["views"])
    print(f"    Eligible clips: {len(eligible_uris):,}")

    # Step 3: Join with labels
    print(f"  Step 3: Joining clips with labels...")
    all_rows: list[tuple[str, int]] = []
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

    # Class distribution
    class_counts: dict[int, int] = defaultdict(int)
    study_classes: dict[str, int] = {}
    for uri, label in all_rows:
        sid = study_id_from_path(uri)
        study_classes[sid] = label
    for sid, label in study_classes.items():
        class_counts[label] += 1

    print(f"    Matched: {len(all_rows):,} clips, {len(study_ids):,} studies")
    print(f"    Class distribution (studies):")
    for cls in sorted(class_counts):
        print(f"      {cls} ({cfg['class_names'][cls]}): {class_counts[cls]:,} ({100*class_counts[cls]/len(study_ids):.1f}%)")

    # Step 4: Write output
    out_dir = OUT_BASE / f"{task_name}_all"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "all.csv"
    with out_csv.open("w") as fh:
        for path, val in all_rows:
            fh.write(f"{path} {val}\n")
    print(f"    -> {out_csv}")

    # Write metadata
    cps_counts: dict[str, int] = defaultdict(int)
    for p, _ in all_rows:
        sid = study_id_from_path(p)
        if sid:
            cps_counts[sid] += 1
    cps_vals = sorted(cps_counts.values())
    n = len(cps_vals)
    cps_median = cps_vals[n // 2] if n else 0

    meta = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "task_type": "classification",
        "target": task_name,
        "description": cfg["description"],
        "source": "echo_structured_measurement",
        "db_measurement": cfg["db_measurement"],
        "matching_window_days": 1,
        "view_filter": sorted(cfg["views"]),
        "bmode_only": cfg["bmode_only"],
        "num_classes": len(cfg["class_names"]),
        "class_names": cfg["class_names"],
        "total_clips": len(all_rows),
        "total_studies": len(study_ids),
        "labelled_studies_in_db": len(labels),
        "class_distribution_studies": {cfg["class_names"][k]: v for k, v in sorted(class_counts.items())},
        "clips_per_study": {"min": cps_vals[0] if cps_vals else 0, "median": cps_median,
                            "max": cps_vals[-1] if cps_vals else 0,
                            "mean": round(sum(cps_vals) / n, 2) if n else 0},
    }
    meta_path = out_dir / "task_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"    -> {meta_path}")


def main() -> None:
    print("=" * 60)
    print("Building MIMIC valve status CSVs (zero-shot transfer)")
    print("=" * 60)

    for task_name, cfg in TASKS.items():
        build_task(task_name, cfg)

    # Summary
    print(f"\n\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for task_name, cfg in TASKS.items():
        meta_path = OUT_BASE / f"{task_name}_all" / "task_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            dist = meta["class_distribution_studies"]
            print(f"  {task_name:12s}: {meta['total_clips']:>7,} clips, {meta['total_studies']:>5,} studies")
            for cls_name, count in dist.items():
                print(f"    {cls_name:25s}: {count:,}")


if __name__ == "__main__":
    main()
