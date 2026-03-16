"""
Parallel upload of local MP4 files to S3, preserving directory structure.

Usage:
    python preprocessing/upload_s3.py \
        --input_dir /path/to/mp4s_masked \
        --s3_prefix s3://bucket/dataset-name/ \
        --workers 16
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
from tqdm import tqdm


def upload_file(args):
    """Upload a single file to S3."""
    local_path, bucket, key = args
    try:
        s3 = boto3.client("s3")
        s3.upload_file(local_path, bucket, key)
        return "success"
    except Exception as e:
        print(f"Error uploading {local_path}: {e}")
        return "error"


def parse_s3_prefix(s3_prefix):
    """Parse s3://bucket/prefix/ into (bucket, prefix)."""
    if not s3_prefix.startswith("s3://"):
        raise ValueError(f"S3 prefix must start with s3://: {s3_prefix}")
    rest = s3_prefix[5:]
    bucket, _, prefix = rest.partition("/")
    return bucket, prefix.rstrip("/")


def main():
    parser = argparse.ArgumentParser(description="Upload MP4 files to S3")
    parser.add_argument("--input_dir", required=True, help="Local directory of MP4 files")
    parser.add_argument("--s3_prefix", required=True, help="S3 destination (e.g., s3://bucket/dataset/)")
    parser.add_argument("--workers", type=int, default=16, help="Number of upload threads")
    parser.add_argument("--extensions", nargs="+", default=[".mp4"], help="File extensions to upload")
    parser.add_argument("--dry_run", action="store_true", help="Print uploads without executing")
    args = parser.parse_args()

    bucket, prefix = parse_s3_prefix(args.s3_prefix)
    input_root = os.path.abspath(args.input_dir)

    # Find files
    files = []
    for root, _, filenames in os.walk(input_root):
        for f in filenames:
            if any(f.lower().endswith(ext) for ext in args.extensions):
                files.append(os.path.join(root, f))
    files.sort()

    print(f"Found {len(files)} files to upload.")
    print(f"Destination: s3://{bucket}/{prefix}/")

    if not files:
        return

    # Build tasks
    tasks = []
    for fp in files:
        rel = os.path.relpath(fp, input_root)
        key = f"{prefix}/{rel}" if prefix else rel
        tasks.append((fp, bucket, key))

    if args.dry_run:
        for local, b, k in tasks[:10]:
            print(f"  {local} -> s3://{b}/{k}")
        if len(tasks) > 10:
            print(f"  ... and {len(tasks) - 10} more")
        return

    # Upload
    success = 0
    errors = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(upload_file, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Uploading"):
            result = fut.result()
            if result == "success":
                success += 1
            else:
                errors += 1

    print(f"\nDone. Uploaded: {success}, Errors: {errors}")


if __name__ == "__main__":
    main()
