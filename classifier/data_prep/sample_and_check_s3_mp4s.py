#!/usr/bin/env python3
"""
Fast S3 MP4 Sampler & Verifier.

1. Randomly samples N rows from a CSV.
2. Uses boto3 + Threading for high-speed existence checks.
3. Generates presigned URLs to probe video metadata (Resolution, Frames) 
   without downloading the full file (uses OpenCV/FFmpeg streaming).
4. Outputs statistics and CSV reports.

Usage:
  python3 fast_sample_s3.py \
    --csv labels_patient_split_mp4_s3.csv \
    --n 500 \
    --threads 32
"""

import argparse
import csv
import random
import time
import statistics
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
import cv2
from botocore.exceptions import ClientError
from tqdm import tqdm

def parse_s3_uri(s3_uri):
    """Parses s3://bucket/key."""
    u = urlparse(s3_uri)
    return u.netloc, u.path.lstrip("/")

def get_s3_client(profile=None, region=None):
    """Creates a boto3 client with optional profile/region."""
    session = boto3.Session(profile_name=profile, region_name=region)
    return session.client("s3")

def process_row(row, s3_client, mp4_col):
    """
    Worker function to check existence and probe metadata.
    Returns (status, row_data, metadata).
    Status: 'found', 'missing', 'error'
    """
    s3_uri = row.get(mp4_col)
    if not s3_uri:
        return "error", row, {"error": "Empty URI"}

    bucket, key = parse_s3_uri(s3_uri)
    meta = {"width": 0, "height": 0, "frames": 0, "channels": 0}

    try:
        # 1. Fast Existence Check (HEAD request)
        s3_client.head_object(Bucket=bucket, Key=key)
        
        # 2. Probe Metadata via Presigned URL (Read Header only)
        # Expiration=60 seconds is enough for a quick probe
        try:
            url = s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': bucket, 'Key': key},
                ExpiresIn=60
            )
            
            # OpenCV can stream from HTTP/HTTPS if built with FFmpeg backend
            cap = cv2.VideoCapture(url)
            if cap.isOpened():
                meta["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                meta["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                meta["frames"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                meta["channels"] = 3 # Standard for loaded RGB video
                cap.release()
            else:
                # Could not open stream (codec issue or network)
                pass 
        except Exception:
            # Metadata probing failed, but file exists
            pass

        return "found", row, meta

    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == "404" or error_code == "NoSuchKey":
            row["_check_error"] = "NotFound"
            return "missing", row, {}
        else:
            row["_check_error"] = str(error_code)
            return "error", row, {}
    except Exception as e:
        row["_check_error"] = str(e)
        return "error", row, {}

def main():
    parser = argparse.ArgumentParser(description="Fast S3 Video Checker")
    parser.add_argument("--csv", required=True, help="Path to input CSV")
    parser.add_argument("--n", type=int, required=True, help="Number of rows to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--mp4-col", default="mp4_path", help="Column name for S3 URI")
    parser.add_argument("--out-prefix", default="sample_check", help="Output filename prefix")
    parser.add_argument("--profile", default=None, help="AWS Profile")
    parser.add_argument("--region", default=None, help="AWS Region")
    parser.add_argument("--threads", type=int, default=16, help="Concurrency level")
    args = parser.parse_args()

    # 1. Read and Sample
    print(f"Reading {args.csv}...")
    with open(args.csv, "r") as f:
        reader = csv.DictReader(f)
        if args.mp4_col not in reader.fieldnames:
            print(f"Error: Column '{args.mp4_col}' not found in CSV.")
            return
        all_rows = list(reader)

    if args.n > len(all_rows):
        print(f"Requested {args.n} samples, but only {len(all_rows)} rows available. Using all.")
        sample_rows = all_rows
    else:
        random.seed(args.seed)
        sample_rows = random.sample(all_rows, args.n)

    print(f"Processing {len(sample_rows)} files using {args.threads} threads...")

    # 2. Setup Boto3 (Thread-safe client strategy: create one per thread or share one)
    # Boto3 clients are thread-safe.
    s3_client = get_s3_client(args.profile, args.region)

    found_rows = []
    missing_rows = []
    error_rows = []
    
    # Stats containers
    widths = []
    heights = []
    frames = []

    # 3. Parallel Processing
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        # Submit all tasks
        future_to_row = {
            executor.submit(process_row, row, s3_client, args.mp4_col): row 
            for row in sample_rows
        }

        # Process as they complete
        for future in tqdm(as_completed(future_to_row), total=len(sample_rows), unit="file"):
            status, row, meta = future.result()
            
            if status == "found":
                # Add metadata to row for CSV output
                row["meta_width"] = meta.get("width")
                row["meta_height"] = meta.get("height")
                row["meta_frames"] = meta.get("frames")
                found_rows.append(row)
                
                # Collect stats (ignore 0/empty values)
                if meta.get("width"): widths.append(meta["width"])
                if meta.get("height"): heights.append(meta["height"])
                if meta.get("frames"): frames.append(meta["frames"])
                
            elif status == "missing":
                missing_rows.append(row)
            else:
                error_rows.append(row)

    # 4. Write Outputs
    base_fieldnames = list(sample_rows[0].keys())
    # Add metadata columns to found csv
    found_fieldnames = base_fieldnames + ["meta_width", "meta_height", "meta_frames"]
    
    with open(f"{args.out_prefix}_found.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=found_fieldnames)
        w.writeheader()
        w.writerows(found_rows)

    with open(f"{args.out_prefix}_missing.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=base_fieldnames + ["_check_error"])
        w.writeheader()
        w.writerows(missing_rows)

    with open(f"{args.out_prefix}_errors.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=base_fieldnames + ["_check_error"])
        w.writeheader()
        w.writerows(error_rows)

    # 5. Print Summary
    print("\n" + "="*40)
    print(f"SUMMARY (Sample Size: {len(sample_rows)})")
    print("="*40)
    print(f"✅ Found:   {len(found_rows)}")
    print(f"❌ Missing: {len(missing_rows)}")
    print(f"⚠️  Errors:  {len(error_rows)}")
    
    if found_rows:
        print("-" * 40)
        print("VIDEO METADATA STATISTICS (Found Files)")
        
        avg_w = statistics.mean(widths) if widths else 0
        avg_h = statistics.mean(heights) if heights else 0
        avg_f = statistics.mean(frames) if frames else 0
        
        # Most common resolution
        if widths and heights:
            res_pairs = list(zip(widths, heights))
            common_res = max(set(res_pairs), key=res_pairs.count)
            print(f"Common Resolution: {common_res[0]}x{common_res[1]} (Width x Height)")
            print(f"Average Resolution: {int(avg_w)}x{int(avg_h)}")
        
        print(f"Average Channels:   3 (Assumed RGB)")
        print(f"Average Frames:     {int(avg_f)}")
    print("="*40)

if __name__ == "__main__":
    main()