"""Build A4C-only train/val/test splits for incident heart failure detection.

Label: HF hospital admission within 365 days AFTER the echo study.
  - Positive = ICD-10 I50.x or ICD-9 428.x on an admission with admittime > study_datetime and delta <= 365d.
  - Eligible = no HF admission with admittime <= study_datetime (patient is HF-naive at echo).
  - Negative = eligible AND no HF admission within 365d (includes right-censored patients;
    standard caveat for ICD-based incident labels from MIMIC).

Splits reuse the patient-level partition from disease_hf_v4.1 to avoid leakage
and keep cross-task comparability. Patients with prior HF are dropped outright.

Output: disease_hf_incident_1yr_a4c/{train,val,test}.csv + label_meta.json
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
SRC_SPLIT_DIR = REPO_ROOT / "experiments/nature_medicine/mimic/probe_csvs/disease_hf_v4.1"
OUT_SPLIT_DIR = REPO_ROOT / "experiments/nature_medicine/mimic/probe_csvs/disease_hf_incident_1yr_a4c"
VIEW_MANIFEST = Path("/home/sagemaker-user/user-default-efs/vjepa2/classifier/output/mimic_view_predictions.csv")

ALLOWED_VIEWS = {"A4C"}
WINDOW_DAYS = 365
SPLITS = ("train", "val", "test")

HF_FILTER = """
  (d.icd_code LIKE 'I50%' AND d.icd_version = '10')
  OR (d.icd_code LIKE '428%' AND d.icd_version = '9')
"""

STUDY_ID_RE = re.compile(r"/s(\d+)/")
SUBJECT_ID_RE = re.compile(r"/p\d+/p(\d+)/")


def build_incident_labels(db_path: Path) -> dict[str, dict]:
    """Return {study_id: {subject_id, study_datetime, prior_hf, future_hf, eligible, label_1y, days_to_hf}}."""
    con = sqlite3.connect(str(db_path))
    q = f"""
    WITH hf AS (
      SELECT DISTINCT a.subject_id, a.admittime AS hf_time
      FROM hosp_admissions a
      JOIN hosp_diagnoses_icd d ON a.hadm_id = d.hadm_id
      WHERE {HF_FILTER}
    )
    SELECT
      s.subject_id,
      s.study_id,
      s.study_datetime,
      MIN(CASE WHEN h.hf_time <= s.study_datetime THEN h.hf_time END) AS prior_hf,
      MIN(CASE WHEN h.hf_time >  s.study_datetime THEN h.hf_time END) AS future_hf
    FROM echo_study_list s
    LEFT JOIN hf h ON s.subject_id = h.subject_id
    GROUP BY s.subject_id, s.study_id, s.study_datetime
    """
    out: dict[str, dict] = {}
    for subject_id, study_id, study_dt, prior_hf, future_hf in con.execute(q):
        eligible = prior_hf is None
        days_to_hf = None
        label_1y = 0
        if future_hf is not None:
            try:
                d0 = datetime.fromisoformat(study_dt)
                d1 = datetime.fromisoformat(future_hf)
                days_to_hf = (d1 - d0).total_seconds() / 86400.0
                if eligible and days_to_hf <= WINDOW_DAYS:
                    label_1y = 1
            except ValueError:
                pass
        out[study_id] = {
            "subject_id": subject_id,
            "study_datetime": study_dt,
            "prior_hf": prior_hf,
            "future_hf": future_hf,
            "eligible": eligible,
            "label_1y": label_1y,
            "days_to_hf": days_to_hf,
        }
    con.close()
    return out


def load_a4c_uris(manifest_path: Path) -> set[str]:
    a4c: set[str] = set()
    with manifest_path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row["view_status"] == "OK" and row["view"] in ALLOWED_VIEWS:
                a4c.add(row["s3_uri"])
    return a4c


def extract_ids(path: str) -> tuple[str, str]:
    s = STUDY_ID_RE.search(path)
    p = SUBJECT_ID_RE.search(path)
    return (s.group(1) if s else ""), (p.group(1) if p else "")


def filter_split(src: Path, a4c: set[str], labels: dict[str, dict]) -> tuple[list[tuple[str, str]], dict]:
    clips_in = 0
    clips_a4c = 0
    clips_out = 0
    studies_in: set[str] = set()
    studies_a4c: set[str] = set()
    studies_out: set[str] = set()
    studies_ineligible: set[str] = set()
    studies_unmapped: set[str] = set()
    pos_studies: set[str] = set()
    neg_studies: set[str] = set()
    kept: list[tuple[str, str]] = []

    with src.open("r") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            path, _orig_label = line.rsplit(" ", 1)
            clips_in += 1
            study_id, _subject_id = extract_ids(path)
            if study_id:
                studies_in.add(study_id)
            label_rec = labels.get(study_id)
            if label_rec is None:
                if study_id:
                    studies_unmapped.add(study_id)
                continue
            if not label_rec["eligible"]:
                studies_ineligible.add(study_id)
                continue
            if path not in a4c:
                continue
            clips_a4c += 1
            studies_a4c.add(study_id)
            new_label = str(label_rec["label_1y"])
            kept.append((path, new_label))
            clips_out += 1
            studies_out.add(study_id)
            if new_label == "1":
                pos_studies.add(study_id)
            else:
                neg_studies.add(study_id)

    stats = {
        "clips_in": clips_in,
        "clips_a4c": clips_a4c,
        "clips_out": clips_out,
        "studies_in": len(studies_in),
        "studies_out": len(studies_out),
        "studies_ineligible_prior_hf": len(studies_ineligible),
        "studies_unmapped": len(studies_unmapped),
        "pos_studies": len(pos_studies),
        "neg_studies": len(neg_studies),
        "pos_clips": sum(1 for _, l in kept if l == "1"),
        "neg_clips": sum(1 for _, l in kept if l == "0"),
    }
    return kept, stats


def write_split(rows: list[tuple[str, str]], dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w") as fh:
        for path, label in rows:
            fh.write(f"{path} {label}\n")


def per_study_clip_counts(rows: list[tuple[str, str]]) -> dict:
    counts: dict[str, int] = defaultdict(int)
    for path, _ in rows:
        sid, _ = extract_ids(path)
        if sid:
            counts[sid] += 1
    vals = sorted(counts.values())
    if not vals:
        return {"min": 0, "median": 0, "max": 0, "mean": 0.0}
    n = len(vals)
    median = vals[n // 2] if n % 2 == 1 else (vals[n // 2 - 1] + vals[n // 2]) / 2
    return {"min": vals[0], "median": median, "max": vals[-1], "mean": round(sum(vals) / n, 2)}


def main() -> None:
    print(f"Building incident HF labels from {MIMIC_DB.name}")
    labels = build_incident_labels(MIMIC_DB)
    total_studies = len(labels)
    eligible = sum(1 for v in labels.values() if v["eligible"])
    pos = sum(1 for v in labels.values() if v["label_1y"] == 1)
    print(f"  total echo studies:           {total_studies:,}")
    print(f"  eligible (no prior HF):       {eligible:,}")
    print(f"  positive (HF within {WINDOW_DAYS}d): {pos:,}  ({pos / max(eligible, 1) * 100:.1f}% of eligible)")

    print(f"\nLoading A4C view manifest: {VIEW_MANIFEST.name}")
    a4c = load_a4c_uris(VIEW_MANIFEST)
    print(f"  A4C OK clips: {len(a4c):,}")

    meta = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "label_definition": {
            "positive": f"HF admission (ICD-10 I50.x or ICD-9 428.x) with admittime in (study_datetime, study_datetime + {WINDOW_DAYS}d]",
            "negative": "eligible AND no HF admission within window",
            "ineligible": "prior HF admission with admittime <= study_datetime",
            "window_days": WINDOW_DAYS,
        },
        "source_splits": str(SRC_SPLIT_DIR.relative_to(REPO_ROOT)),
        "source_db": str(MIMIC_DB.relative_to(REPO_ROOT)),
        "view_manifest": str(VIEW_MANIFEST),
        "view_filter": sorted(ALLOWED_VIEWS),
        "view_status_filter": "OK",
        "cohort_counts": {
            "total_echo_studies": total_studies,
            "eligible_studies": eligible,
            "positive_studies_pre_split": pos,
        },
        "splits": {},
    }

    for split in SPLITS:
        src = SRC_SPLIT_DIR / f"{split}.csv"
        print(f"\n[{split}] filtering {src.name}")
        rows, stats = filter_split(src, a4c, labels)
        stats["clips_per_study"] = per_study_clip_counts(rows)
        dst = OUT_SPLIT_DIR / f"{split}.csv"
        write_split(rows, dst)
        meta["splits"][split] = stats
        print(f"  clips:    {stats['clips_in']:>7,} -> A4C {stats['clips_a4c']:>6,} -> kept {stats['clips_out']:>6,}")
        print(f"  studies:  {stats['studies_in']:>7,} -> kept {stats['studies_out']:>6,}")
        print(f"    dropped prior HF (prevalent): {stats['studies_ineligible_prior_hf']:>4,}")
        print(f"    unmapped study_ids:           {stats['studies_unmapped']:>4,}")
        print(f"  labels:   pos {stats['pos_studies']:>3,} studies / {stats['pos_clips']:>4,} clips"
              f"   neg {stats['neg_studies']:>4,} studies / {stats['neg_clips']:>5,} clips")
        print(f"  clips/study: min={stats['clips_per_study']['min']} "
              f"median={stats['clips_per_study']['median']} max={stats['clips_per_study']['max']}")
        print(f"  -> {dst}")

    meta_path = OUT_SPLIT_DIR / "label_meta.json"
    with meta_path.open("w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"\nWrote metadata: {meta_path}")


if __name__ == "__main__":
    main()
