#!/usr/bin/env python3
"""
Create:
  1) mapping.txt  (label -> index)
  2) output txt/csv in "mp4_path index" format (space-separated, no header)

Uses labels_patient_split_mp4_s3.csv by default (expects columns: label, split, mp4_path).

Example:
  s3://.../video.mp4 0

Usage:
  python3 make_view_labels_space_sep.py \
    --in labels_patient_split_mp4_s3.csv \
    --out ../data/csv/uhn_views_train.txt \
    --mapping mapping.txt \
    --split train

Optional:
  --label-col label
  --path-col mp4_path
  --split-col split
"""

import argparse
import csv
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mapping", default="mapping.txt")
    ap.add_argument("--split", default=None, help="e.g., train/val/test; if omitted, use all rows")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--path-col", default="mp4_path")
    ap.add_argument("--split-col", default="split")
    args = ap.parse_args()

    # Load rows
    with open(args.inp, newline="") as f:
        r = csv.DictReader(f)
        for col in (args.label_col, args.path_col):
            if col not in r.fieldnames:
                raise SystemExit(f"Missing column '{col}'. Columns: {r.fieldnames}")
        rows = list(r)

    # Filter by split if requested
    if args.split is not None:
        if args.split_col not in rows[0]:
            raise SystemExit(f"--split provided but split column '{args.split_col}' not found.")
        rows = [row for row in rows if row.get(args.split_col) == args.split]

    if not rows:
        raise SystemExit("No rows after filtering.")

    # Determine labels (stable order = alphabetical)
    labels = sorted({row[args.label_col] for row in rows})
    label_to_idx = {lab: i for i, lab in enumerate(labels)}

    # Write mapping.txt
    with open(args.mapping, "w") as f:
        for lab in labels:
            f.write(f"{label_to_idx[lab]}\t{lab}\n")

    # Write output (space-separated)
    bad = 0
    with open(args.out, "w") as f:
        for row in rows:
            path = row[args.path_col].strip()
            lab = row[args.label_col].strip()
            if not path or not lab:
                bad += 1
                continue
            f.write(f"{path} {label_to_idx[lab]}\n")

    # Basic stats
    c = Counter(row[args.label_col] for row in rows)
    print(f"Wrote mapping: {args.mapping} ({len(labels)} classes)")
    print(f"Wrote data:    {args.out} ({len(rows)-bad} rows; dropped {bad} bad rows)")
    print("Class counts:")
    for lab in labels:
        print(f"  {label_to_idx[lab]:2d} {lab:10s} {c[lab]}")

if __name__ == "__main__":
    main()
