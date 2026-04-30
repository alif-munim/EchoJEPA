"""Audit study_id / subject_id mapping and acquisition_datetime spread in
phase_annotations.parquet, with optional train/val/test split leak check.

Checks:
  1. study_id uniqueness — every study_id exactly one subject_id (when
     subject_id exists).
  2. Subject-across-studies — number of subject_ids with multiple study_ids.
  3. Train/val/test subject_id leakage, if a split CSV is provided.
  4. Intra-study acquisition_datetime spread (per study, max-min seconds).
  5. If acquisition_datetime varies >threshold within a study, report how
     many records are affected — the sampler can be told to restrict
     pairing to same-session via ``same_session_only=True``.

Outputs a concise stdout summary and a JSON report.

Usage:
  python check_phase_split_integrity.py \\
      --parquet phase_annotations/phase_annotations.parquet \\
      --out /tmp/phase_split_integrity.json \\
      [--split-csv my_split.csv --subject-column subject_id \\
       --split-column split]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def summarize(df: pd.DataFrame) -> dict:
    out = {
        "n_rows": int(len(df)),
        "columns_present": sorted(df.columns.tolist()),
        "has_subject_id": "subject_id" in df.columns,
        "has_acquisition_datetime": "acquisition_datetime" in df.columns,
    }

    # 1. study_id uniqueness check
    if "subject_id" in df.columns:
        pairs = df[["study_id", "subject_id"]].dropna().drop_duplicates()
        per_study = pairs.groupby("study_id").size()
        out["study_to_subject_map"] = {
            "unique_studies": int(per_study.size),
            "studies_with_multiple_subjects": int((per_study > 1).sum()),
            "bad_studies_sample": per_study[per_study > 1].head(5).index.tolist(),
        }
    else:
        out["study_to_subject_map"] = None

    # 2. Subject-across-studies
    if "subject_id" in df.columns:
        per_subject = df[["subject_id", "study_id"]].drop_duplicates().groupby("subject_id").study_id.nunique()
        out["subject_to_study_map"] = {
            "unique_subjects": int(per_subject.size),
            "subjects_with_multiple_studies": int((per_subject > 1).sum()),
            "max_studies_per_subject": int(per_subject.max()) if per_subject.size else 0,
            "median_studies_per_subject": float(per_subject.median()) if per_subject.size else 0.0,
        }
    else:
        out["subject_to_study_map"] = None

    # 3. Intra-study acquisition_datetime spread
    if "acquisition_datetime" in df.columns:
        td = pd.to_datetime(df.acquisition_datetime, errors="coerce")
        tmp = pd.DataFrame({"study_id": df.study_id.values, "t": td.values})
        tmp = tmp.dropna(subset=["t"])
        if len(tmp):
            spread = tmp.groupby("study_id").t.agg(["min", "max"])
            spread["delta_seconds"] = (spread["max"] - spread["min"]).dt.total_seconds()
            affected = spread[spread.delta_seconds > 3600]
            out["acquisition_spread"] = {
                "studies_checked": int(len(spread)),
                "studies_with_any_spread_gt_0": int((spread.delta_seconds > 0).sum()),
                "studies_with_spread_gt_1h": int(len(affected)),
                "spread_seconds": {
                    "median": float(spread.delta_seconds.median()),
                    "p90": float(spread.delta_seconds.quantile(0.9)),
                    "max": float(spread.delta_seconds.max()),
                },
                "example_multi_session_studies": affected.index.tolist()[:5],
            }
        else:
            out["acquisition_spread"] = {"note": "acquisition_datetime present but all null"}
    else:
        out["acquisition_spread"] = None

    return out


def check_split_leakage(
    df: pd.DataFrame,
    split_csv: Path,
    subject_column: str = "subject_id",
    split_column: str = "split",
    study_column: str = "study_id",
) -> dict:
    sdf = pd.read_csv(split_csv)
    need = {subject_column, split_column}
    missing = need - set(sdf.columns)
    if missing:
        raise KeyError(f"split CSV missing columns: {missing}")
    # Collapse to subject -> {split}
    per_subj = sdf.groupby(subject_column)[split_column].agg(lambda s: sorted(set(s.astype(str))))
    leaked_subjects = per_subj[per_subj.apply(len) > 1]
    # Subjects present in both the split CSV and the parquet
    if subject_column in df.columns:
        shared = set(df[subject_column].dropna().astype(str)) & set(sdf[subject_column].astype(str))
    else:
        shared = None
    return {
        "split_csv": str(split_csv),
        "n_subjects_in_split": int(per_subj.size),
        "n_leaked_subjects": int(leaked_subjects.size),
        "leaked_subject_sample": leaked_subjects.head(5).to_dict(),
        "shared_subjects_with_parquet": (None if shared is None else len(shared)),
        "split_counts": sdf[split_column].value_counts().to_dict(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--split-csv", type=Path, default=None)
    ap.add_argument("--subject-column", default="subject_id")
    ap.add_argument("--split-column", default="split")
    ap.add_argument("--study-column", default="study_id")
    ap.add_argument("--fail-on-leakage", action="store_true",
                    help="Exit non-zero if any subject_id appears in >1 split.")
    args = ap.parse_args()

    # Pull only what we need; some columns may be absent.
    full_cols = pd.read_parquet(args.parquet, columns=["dicom_id"]).columns  # noqa: F841
    want = ["dicom_id", "study_id"]
    cols_available = pd.read_parquet(args.parquet, columns=want).columns
    extras = []
    for opt in ("subject_id", "acquisition_datetime"):
        try:
            pd.read_parquet(args.parquet, columns=[opt])
            extras.append(opt)
        except Exception:
            pass
    df = pd.read_parquet(args.parquet, columns=list(cols_available) + extras)

    report = summarize(df)
    if args.split_csv:
        report["split_leakage"] = check_split_leakage(
            df, args.split_csv, args.subject_column, args.split_column, args.study_column,
        )

    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        return str(o)

    print(json.dumps(report, indent=2, default=_default))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, default=_default))
        print(f"\nwrote {args.out}")

    if args.fail_on_leakage:
        leak = None
        if args.split_csv is not None:
            leak = report.get("split_leakage", {}).get("n_leaked_subjects", 0)
        if leak and int(leak) > 0:
            print(f"\nFAIL: {leak} subject_id(s) appear in multiple splits.", file=sys.stderr)
            sys.exit(2)


if __name__ == "__main__":
    main()
