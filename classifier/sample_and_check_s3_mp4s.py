#!/usr/bin/env python3
"""
Randomly sample N rows from labels_patient_split_mp4_s3.csv and verify the mp4 exists on S3.

Fast existence check uses `aws s3api head-object` (no download).
Shows a progress bar and writes:
  - <out_prefix>_found.csv
  - <out_prefix>_missing.csv
  - <out_prefix>_errors.csv

Usage:
  python3 sample_and_check_s3_mp4s.py \
    --csv labels_patient_split_mp4_s3.csv \
    --n 500 \
    --seed 0 \
    --out-prefix sample_check

Optional:
  --profile <aws_profile>
  --region <aws_region>
  --mp4-col mp4_path
"""

import argparse
import csv
import random
import sys
import time
import subprocess
from urllib.parse import urlparse


def parse_s3_uri(s3_uri: str):
    # s3://bucket/key...
    u = urlparse(s3_uri)
    if u.scheme != "s3" or not u.netloc or not u.path:
        raise ValueError(f"Not a valid s3:// URI: {s3_uri}")
    bucket = u.netloc
    key = u.path.lstrip("/")
    return bucket, key


def head_object_exists(s3_uri: str, profile: str | None, region: str | None) -> tuple[bool, str]:
    bucket, key = parse_s3_uri(s3_uri)

    cmd = ["aws"]
    if profile:
        cmd += ["--profile", profile]
    if region:
        cmd += ["--region", region]
    cmd += ["s3api", "head-object", "--bucket", bucket, "--key", key]

    p = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if p.returncode == 0:
        return True, ""
    err = (p.stderr or "").strip()
    # Common patterns: 404/NotFound vs AccessDenied
    if "Not Found" in err or "404" in err or "NoSuchKey" in err:
        return False, "NotFound"
    if "AccessDenied" in err or "Access Denied" in err:
        return False, "AccessDenied"
    return False, err[:300]


def human(n: float) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    i = 0
    while n >= 1024 and i < len(units) - 1:
        n /= 1024.0
        i += 1
    return f"{n:.2f} {units[i]}"


def bar(p: float, width: int = 30) -> str:
    p = max(0.0, min(1.0, p))
    fill = int(p * width)
    return "[" + ("=" * fill) + (" " * (width - fill)) + "]"


def progress(i: int, total: int, found: int, missing: int, errors: int, start: float):
    now = time.time()
    elapsed = max(1e-6, now - start)
    rate = i / elapsed
    eta = (total - i) / rate if rate > 0 else float("inf")
    msg = (
        f"\r{i}/{total} {bar(i/total if total else 1.0)} {i/total*100:6.2f}%  "
        f"found={found} missing={missing} errors={errors}  "
        f"{rate:5.2f}/s ETA {eta/60:5.1f}m"
    )
    sys.stdout.write(msg)
    sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-prefix", default="sample_check")
    ap.add_argument("--mp4-col", default="mp4_path")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--region", default=None)
    args = ap.parse_args()

    # Read rows
    with open(args.csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        if args.mp4_col not in reader.fieldnames:
            raise SystemExit(f"Column '{args.mp4_col}' not found. Columns: {reader.fieldnames}")
        rows = list(reader)

    if args.n <= 0:
        raise SystemExit("--n must be > 0")
    if args.n > len(rows):
        raise SystemExit(f"--n ({args.n}) > number of rows ({len(rows)})")

    rnd = random.Random(args.seed)
    sample = rnd.sample(rows, args.n)

    found_rows, missing_rows, error_rows = [], [], []
    start = time.time()

    for i, row in enumerate(sample, start=1):
        s3_uri = row[args.mp4_col]
        ok, detail = head_object_exists(s3_uri, args.profile, args.region)
        if ok:
            found_rows.append(row)
        else:
            # keep row + error detail
            row2 = dict(row)
            row2["_check_error"] = detail
            if detail == "NotFound":
                missing_rows.append(row2)
            else:
                error_rows.append(row2)

        progress(i, args.n, len(found_rows), len(missing_rows), len(error_rows), start)

    sys.stdout.write("\n")
    sys.stdout.flush()

    # Write outputs
    base_fields = list(sample[0].keys())
    found_path = f"{args.out_prefix}_found.csv"
    miss_path = f"{args.out_prefix}_missing.csv"
    err_path = f"{args.out_prefix}_errors.csv"

    with open(found_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=base_fields)
        w.writeheader()
        w.writerows(found_rows)

    with open(miss_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=base_fields + ["_check_error"])
        w.writeheader()
        w.writerows(missing_rows)

    with open(err_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=base_fields + ["_check_error"])
        w.writeheader()
        w.writerows(error_rows)

    print("Done")
    print(f"Found:   {len(found_rows)}/{args.n} -> {found_path}")
    print(f"Missing: {len(missing_rows)}/{args.n} -> {miss_path}")
    print(f"Errors:  {len(error_rows)}/{args.n} -> {err_path}")


if __name__ == "__main__":
    main()
