import boto3
import csv
import os
import time
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm  # pip install tqdm

# CONFIGURATION
# -----------------------------------------------------------------------------
TARGET_URIS = [
    "s3://echodata25/results/echo-study/",
    "s3://echodata25/results/echo-study-1/",
    "s3://echodata25/results/echo-study-2/",
]
OUTPUT_DIR = "./indices"
MASTER_FILENAME = "master_index_18M.csv"
# -----------------------------------------------------------------------------

def parse_s3_uri(uri):
    p = urlparse(uri)
    return p.netloc, p.path.lstrip('/')

def index_prefix(s3_uri, output_path, position_idx):
    """
    Paginates through S3 and writes to CSV.
    Uses position_idx to stack tqdm bars nicely in the terminal.
    """
    s3 = boto3.client("s3")
    bucket, prefix = parse_s3_uri(s3_uri)
    paginator = s3.get_paginator("list_objects_v2")
    
    # Extract a short name for the progress bar label (e.g., "echo-study-1")
    short_name = s3_uri.strip("/").split("/")[-1]
    
    count = 0
    
    # Open file and Progress Bar
    # position=position_idx ensures the bars stack vertically 0, 1, 2...
    with open(output_path, "w", newline="") as f, \
         tqdm(desc=f"{short_name:<15}", unit=" mp4s", position=position_idx, leave=True) as pbar:
        
        writer = csv.writer(f)
        writer.writerow(["s3_uri"])  # Header
        
        # Iterate through pages (each page = 1000 objects max)
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix, PaginationConfig={'PageSize': 1000}):
            if "Contents" not in page:
                continue
                
            batch = []
            for obj in page["Contents"]:
                key = obj["Key"]
                # Server-side filtering isn't possible for extensions, so we filter here
                if key.endswith(".mp4"):
                    batch.append([f"s3://{bucket}/{key}"])
            
            if batch:
                writer.writerows(batch)
                num_found = len(batch)
                count += num_found
                pbar.update(num_found)

    return count

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Indexing {len(TARGET_URIS)} buckets in parallel...\n")
    
    csv_files = []
    futures = []
    
    # Use ThreadPoolExecutor for concurrency
    with ThreadPoolExecutor(max_workers=len(TARGET_URIS)) as executor:
        for i, uri in enumerate(TARGET_URIS):
            # Create a safe filename
            safe_name = uri.replace("s3://", "").strip("/").replace("/", "_")
            out_file = os.path.join(OUTPUT_DIR, f"{safe_name}.csv")
            csv_files.append(out_file)
            
            # Submit task with index 'i' for the progress bar position
            futures.append(executor.submit(index_prefix, uri, out_file, i))

        # Wait for all to finish
        for fut in futures:
            fut.result()

    # Create Master List
    print("\n\nMerging into Master List...") # Double newline to clear progress bars
    master_path = os.path.join(OUTPUT_DIR, MASTER_FILENAME)
    
    with open(master_path, "w") as outfile:
        outfile.write("s3_uri\n") # Master Header
        
        for fname in tqdm(csv_files, desc="Merging Files"):
            with open(fname, "r") as infile:
                next(infile)  # Skip individual headers
                for line in infile:
                    outfile.write(line)

    print(f"--------------------------------------------------")
    print(f"Done. Master index: {master_path}")
    print(f"--------------------------------------------------")

if __name__ == "__main__":
    main()