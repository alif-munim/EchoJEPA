#!/usr/bin/env python3
import os
import csv
import subprocess
import boto3
import shutil
import argparse
from pathlib import PurePosixPath
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from botocore.exceptions import ClientError
from tqdm import tqdm  # pip install tqdm

DEFAULT_TEMP_DIR = "./temp_processing"

def get_s3_client():
    return boto3.client("s3")

def parse_scale_arg(scale: str) -> tuple[int, int]:
    s = scale.strip().lower()
    if "x" in s:
        w_str, h_str = s.split("x", 1)
        w, h = int(w_str), int(h_str)
    else:
        w = h = int(s)
    if w <= 0 or h <= 0:
        raise ValueError("Scale must be positive.")
    return w, h

def parse_s3_uri(uri: str) -> tuple[str, str]:
    # s3://bucket/key...
    if not uri.startswith("s3://"):
        raise ValueError(f"Not an s3 uri: {uri}")
    rest = uri[len("s3://"):]
    bucket, _, key = rest.partition("/")
    if not bucket or not key:
        raise ValueError(f"Bad s3 uri: {uri}")
    return bucket, key

def build_output_root_from_in_root(s3_in_root: str, scale_w: int, scale_h: int) -> str:
    base = s3_in_root.rstrip("/")
    if scale_w == scale_h:
        return f"{base}_{scale_w}px"
    return f"{base}_{scale_w}x{scale_h}px"

def local_to_s3_uri(local_mp4: str, local_root_marker: str, s3_in_root: str) -> str:
    """
    Map local path .../<marker>/<suffix> to s3://bucket/<s3_in_root_key>/<suffix>
    IMPORTANT: suffix is AFTER marker (no duplication).
    """
    p = PurePosixPath(local_mp4)
    parts = p.parts
    if local_root_marker not in parts:
        raise ValueError(f"root marker '{local_root_marker}' not found in: {local_mp4}")
    idx = parts.index(local_root_marker)
    suffix = PurePosixPath(*parts[idx+1:]).as_posix()

    bucket, in_root_key = parse_s3_uri(s3_in_root)
    in_root_key = in_root_key.rstrip("/")
    return f"s3://{bucket}/{in_root_key}/{suffix}"

def resize_and_upload(row, *, mp4_col, target_dir_string, replacement_dir_string,
                      temp_dir, scale_w, scale_h, crf, preset, max_retries,
                      s3_in_root, local_root_marker):
    mp4_path = row.get(mp4_col)
    if not mp4_path:
        return {"status": "skipped", "msg": f"missing '{mp4_col}'"}

    # --- Determine input S3 URI ---
    if mp4_path.startswith("s3://"):
        in_s3_uri = mp4_path
    else:
        if not (s3_in_root and local_root_marker):
            return {"status": "skipped", "msg": "local mp4_path but --s3-prefix-in/--local-root-marker not provided"}
        try:
            in_s3_uri = local_to_s3_uri(mp4_path, local_root_marker, s3_in_root)
        except Exception as e:
            return {"status": "skipped", "msg": f"local->s3 mapping failed: {e}"}

    bucket, in_key = parse_s3_uri(in_s3_uri)

    # Sanity: ensure the key contains the target dir substring (S3 key is path w/o leading slash)
    # We check with leading slash to match your existing target_dir_string style.
    if target_dir_string not in ("/" + in_key):
        return {"status": "skipped", "msg": f"Path mismatch: {in_key}"}

    out_key = in_key.replace(target_dir_string.strip("/"), replacement_dir_string.strip("/"))

    safe_name = in_key.replace("/", "_")
    local_input = os.path.join(temp_dir, f"in_{safe_name}")
    local_output = os.path.join(temp_dir, f"out_{safe_name}")

    s3 = get_s3_client()
    attempt = 0
    try:
        while True:
            attempt += 1
            try:
                s3.download_file(bucket, in_key, local_input)

                vf = f"scale={scale_w}:{scale_h}"
                cmd = [
                    "ffmpeg", "-y",
                    "-i", local_input,
                    "-vf", vf,
                    "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
                    "-c:a", "copy",
                    "-loglevel", "error",
                    "-nostdin",
                    local_output,
                ]
                subprocess.run(cmd, check=True)

                s3.upload_file(local_output, bucket, out_key)
                return {"status": "success", "msg": f"s3://{bucket}/{out_key}"}

            except (ClientError, subprocess.CalledProcessError) as e:
                if attempt <= max_retries:
                    continue
                if isinstance(e, subprocess.CalledProcessError):
                    return {"status": "error", "msg": f"ffmpeg: {e}"}
                return {"status": "error", "msg": f"s3: {e}"}

    finally:
        for p in (local_input, local_output):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-csv", required=True)
    ap.add_argument("--mp4-col", default="mp4_path")

    # If your CSV mp4_path is local, set these:
    ap.add_argument("--s3-prefix-in", default="", help="Only used when mp4_path is local (not s3://...)")
    ap.add_argument("--local-root-marker", default="", help="Only used when mp4_path is local (not s3://...)")

    ap.add_argument("--target-dir", default="/results/uhn_studies_22k_607/",
                    help="Substring to match within S3 key, including leading/trailing slashes.")
    ap.add_argument("--scale", default="224x224")
    ap.add_argument("--crf", type=int, default=23)
    ap.add_argument("--preset", default="fast")
    ap.add_argument("--workers", type=int, default=(os.cpu_count() or 4) * 2)
    ap.add_argument("--temp-dir", default=DEFAULT_TEMP_DIR)
    ap.add_argument("--max-retries", type=int, default=0)
    args = ap.parse_args()

    scale_w, scale_h = parse_scale_arg(args.scale)

    # Replacement dir auto-derived from target-dir base
    base = args.target_dir.rstrip("/")  # /results/uhn_studies_22k_607
    if scale_w == scale_h:
        replacement_dir = f"{base}_{scale_w}px/"
    else:
        replacement_dir = f"{base}_{scale_w}x{scale_h}px/"

    os.makedirs(args.temp_dir, exist_ok=True)

    with open(args.in_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        tasks = list(reader)

    print(f"Reading {args.in_csv}...")
    print(f"Starting processing of {len(tasks)} videos with {args.workers} workers "
          f"(scale={scale_w}x{scale_h}, target={args.target_dir}, out_dir={replacement_dir})...")

    success = skipped = error = 0
    skip_examples = []

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [
            ex.submit(
                resize_and_upload,
                row,
                mp4_col=args.mp4_col,
                target_dir_string=args.target_dir,
                replacement_dir_string=replacement_dir,
                temp_dir=args.temp_dir,
                scale_w=scale_w,
                scale_h=scale_h,
                crf=args.crf,
                preset=args.preset,
                max_retries=args.max_retries,
                s3_in_root=args.s3_prefix_in,
                local_root_marker=args.local_root_marker,
            )
            for row in tasks
        ]

        for fut in tqdm(as_completed(futures), total=len(futures), unit="vid"):
            r = fut.result()
            if r["status"] == "success":
                success += 1
            elif r["status"] == "skipped":
                skipped += 1
                if len(skip_examples) < 10:
                    skip_examples.append(r["msg"])
            else:
                error += 1
                tqdm.write(f"Error: {r['msg']}")

    shutil.rmtree(args.temp_dir, ignore_errors=True)

    print(f"\nSUMMARY: success={success} skipped={skipped} error={error}")
    if skip_examples:
        print("First skip reasons:")
        for m in skip_examples:
            print(" -", m)

    print("Processing complete.")

if __name__ == "__main__":
    main()
