#!/usr/bin/env python3
"""
Build selected_study_dirs.csv from labels_masked_inplace.csv

- Extract top-level study ID from each filename path in labels_masked_inplace.csv
  (the directory immediately after "h4h_ef_frames/").
- Determine which S3 prefix to use based on membership in es.txt, es1.txt, es2.txt
- Output CSV with: study_id,s3_uri
  (one row per unique study_id; duplicates removed)

Run:
  python make_selected_study_dirs.py \
    --labels labels_masked_inplace.csv \
    --es es.txt \
    --es1 es1.txt \
    --es2 es2.txt \
    --out selected_study_dirs.csv
"""

import argparse
import csv
import os
import sys
from pathlib import Path


S3_BASE = {
    "es":  "s3://echodata25/results/echo-study/",
    "es1": "s3://echodata25/results/echo-study-1/",
    "es2": "s3://echodata25/results/echo-study-2/",
}


def read_study_set(txt_path: Path) -> set[str]:
    """
    Each line like:
      1.2.276....10353285/
    or possibly without trailing slash.
    Return normalized set without trailing slash.
    """
    s = set()
    with txt_path.open("r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if not t:
                continue
            t = t.rstrip("/")  # normalize
            s.add(t)
    return s


def extract_study_id(filename_path: str) -> str | None:
    """
    Extract the segment after 'h4h_ef_frames/'.
    Example:
      .../h4h_ef_frames/<STUDY_ID>/<SERIES_ID>/mp4/<FILE>.jpg
    """
    key = "h4h_ef_frames/"
    idx = filename_path.find(key)
    if idx == -1:
        return None
    rest = filename_path[idx + len(key):]
    # rest starts with "<STUDY_ID>/..."
    parts = rest.split("/", 1)
    return parts[0] if parts and parts[0] else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, type=Path)
    ap.add_argument("--es", required=True, type=Path)
    ap.add_argument("--es1", required=True, type=Path)
    ap.add_argument("--es2", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--strict", action="store_true",
                    help="If set, error out when a study_id isn't found in any es*.txt")
    args = ap.parse_args()

    es = read_study_set(args.es)
    es1 = read_study_set(args.es1)
    es2 = read_study_set(args.es2)

    # optional sanity: ensure no overlaps (not required, but helpful)
    overlaps = (es & es1) | (es & es2) | (es1 & es2)
    if overlaps:
        print(f"[WARN] {len(overlaps)} study IDs appear in multiple es*.txt files. "
              f"First match priority: es > es1 > es2", file=sys.stderr)

    selected: dict[str, str] = {}  # study_id -> s3_uri
    missing = set()
    bad_rows = 0

    with args.labels.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "filename" not in reader.fieldnames:
            raise SystemExit(f"labels CSV missing 'filename' column. Columns: {reader.fieldnames}")

        for row in reader:
            fp = row["filename"]
            study_id = extract_study_id(fp)
            if not study_id:
                bad_rows += 1
                continue

            if study_id in selected:
                continue  # already recorded

            if study_id in es:
                base = S3_BASE["es"]
            elif study_id in es1:
                base = S3_BASE["es1"]
            elif study_id in es2:
                base = S3_BASE["es2"]
            else:
                missing.add(study_id)
                continue

            selected[study_id] = f"{base}{study_id}/"

    if args.strict and missing:
        print(f"[ERROR] {len(missing)} study IDs not found in any es*.txt. "
              f"Example: {next(iter(missing))}", file=sys.stderr)
        sys.exit(2)

    # write output
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["study_id", "s3_uri"])
        for study_id in sorted(selected.keys()):
            w.writerow([study_id, selected[study_id]])

    print(f"[OK] wrote {len(selected)} rows to {args.out}")
    if missing:
        print(f"[WARN] {len(missing)} study_ids were in labels but not in es/es1/es2; skipped.", file=sys.stderr)
    if bad_rows:
        print(f"[WARN] {bad_rows} rows had no parseable study_id (missing 'h4h_ef_frames/'); skipped.", file=sys.stderr)


if __name__ == "__main__":
    main()
