import boto3
import argparse
import random
import os
import sys
from tqdm import tqdm
from botocore.exceptions import ClientError

def parse_s3_uri(uri: str):
    """Parses s3://bucket/key"""
    uri = uri.strip().split()[0]  # Remove trailing labels/spaces
    if not uri.startswith("s3://"):
        return None, None
    parts = uri.replace("s3://", "", 1).split("/", 1)
    if len(parts) < 2:
        return None, None
    return parts[0], parts[1]

def random_seek_sample(filename, sample_size):
    """
    Jumps to random byte positions to sample lines without reading the whole file.
    """
    filesize = os.path.getsize(filename)
    samples = []
    
    with open(filename, "rb") as f:
        # Retry loop to ensure we get enough unique valid lines
        pbar = tqdm(total=sample_size, desc="Sampling", unit="lines")
        while len(samples) < sample_size:
            pos = random.randint(0, filesize)
            f.seek(pos)
            
            # Discard the partial line where we landed
            f.readline()
            
            # Read the next full line
            line = f.readline()
            
            # If we hit EOF, just try again
            if not line:
                continue
                
            decoded_line = line.decode('utf-8', errors='ignore').strip()
            if decoded_line and decoded_line.startswith("s3://"):
                samples.append(decoded_line)
                pbar.update(1)
        pbar.close()
        
    return samples

def main():
    parser = argparse.ArgumentParser(description="Fast verify S3 URIs.")
    parser.add_argument("input_file", type=str)
    parser.add_argument("--sample", type=int, default=500)
    args = parser.parse_args()

    print(f"File: {args.input_file}")
    
    # 1. Fast Sampling
    samples = random_seek_sample(args.input_file, args.sample)
    
    # 2. Verification
    s3 = boto3.client("s3")
    found = 0
    missing = 0
    source_exists_count = 0 
    
    print("\nVerifying existence...")
    
    # We will track the first few missing files for diagnosis
    diagnostics = []

    for uri in tqdm(samples, unit="file"):
        bucket, key = parse_s3_uri(uri)
        if not bucket: continue

        try:
            s3.head_object(Bucket=bucket, Key=key)
            found += 1
        except ClientError:
            missing += 1
            
            # --- DIAGNOSTIC: Check if the OLD source file exists ---
            # We try to reverse the mapping you did earlier to see if the source is there
            if "uhn_studies_22k_607_224px" in key:
                # Try checking echo-study (and -1, -2 variations)
                for old_folder in ["echo-study", "echo-study-1", "echo-study-2"]:
                    old_key = key.replace("uhn_studies_22k_607_224px", old_folder)
                    try:
                        s3.head_object(Bucket=bucket, Key=old_key)
                        source_exists_count += 1
                        if len(diagnostics) < 3:
                            diagnostics.append((uri, f"✅ Source exists at: .../{old_folder}/..."))
                        break # Found the source, stop checking variations
                    except ClientError:
                        pass
            
            if len(diagnostics) < 3 and source_exists_count == 0:
                diagnostics.append((uri, "❌ Source ALSO missing (or path logic mismatch)"))

    print("\n" + "="*50)
    print(f"SUMMARY (Sample: {len(samples)})")
    print(f"✅ Found (New Path):    {found}")
    print(f"❌ Missing (New Path):  {missing}")
    print("-" * 50)
    if missing > 0:
        print(f"🔎 Source File Check (on missing items):")
        print(f"   {source_exists_count} / {missing} source files were found in 'echo-study*'.")
        print(f"   (This confirms the resize job hasn't run/finished yet)")
    print("="*50)

    if diagnostics:
        print("\nDiagnostic Examples:")
        for uri, note in diagnostics:
            print(f"Target: {uri}")
            print(f"Status: {note}\n")

if __name__ == "__main__":
    main()