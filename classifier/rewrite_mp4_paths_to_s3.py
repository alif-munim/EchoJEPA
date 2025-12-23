#!/usr/bin/env python3
"""
Rewrite labels_patient_split_mp4.csv so mp4_path points to S3.

Input mp4_path example:
  /cluster/projects/bwanggroup/echo_reports/uhn_studies_22k_585/<STUDY>/<SERIES>/<SOP>.mp4

Output mp4_path example:
  s3://echodata25/results/uhn_studies_22k_585/uhn_studies_22k_585/<STUDY>/<SERIES>/<SOP>.mp4

Usage:
  python3 rewrite_mp4_paths_to_s3.py \
    --in labels_patient_split_mp4.csv \
    --out labels_patient_split_mp4_s3.csv \
    --s3-prefix s3://echodata25/results/uhn_studies_22k_585/uhn_studies_22k_585 \
    --root-marker uhn_studies_22k_585
"""

import argparse
import csv
from pathlib import PurePosixPath


def to_s3_path(mp4_path: str, s3_prefix: str, root_marker: str) -> str:
    """
    Keep the path suffix starting at the first occurrence of root_marker,
    then prepend s3_prefix (which should end with /<root_marker>).
    """
    if not mp4_path:
        return mp4_path

    p = mp4_path.replace("\\", "/")  # just in case
    marker = f"/{root_marker}/"
    idx = p.find(marker)
    if idx == -1:
        raise ValueError(f"root_marker '{root_marker}' not found in mp4_path: {mp4_path}")

    suffix = p[idx + 1 :]  # drop leading "/" -> "uhn_studies_22k_585/....mp4"
    # s3_prefix expected like: s3://bucket/prefix/uhn_studies_22k_585
    s3_prefix = s3_prefix.rstrip("/")
    return f"{s3_prefix}/{suffix[len(root_marker)+1:]}"  # append after root_marker/


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--s3-prefix", required=True,
                    help="e.g. s3://echodata25/results/uhn_studies_22k_585/uhn_studies_22k_585")
    ap.add_argument("--root-marker", default="uhn_studies_22k_585",
                    help="folder name to locate inside the existing mp4_path")
    args = ap.parse_args()

    n = 0
    with open(args.inp, "r", newline="") as fin, open(args.out, "w", newline="") as fout:
        reader = csv.DictReader(fin)
        if "mp4_path" not in reader.fieldnames:
            raise SystemExit(f"mp4_path column not found. Columns: {reader.fieldnames}")

        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()

        for row in reader:
            row["mp4_path"] = to_s3_path(row["mp4_path"], args.s3_prefix, args.root_marker)
            writer.writerow(row)
            n += 1

    print(f"Wrote {args.out} ({n} rows)")


if __name__ == "__main__":
    main()
