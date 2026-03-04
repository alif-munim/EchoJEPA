import pandas as pd
import boto3
import argparse
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from botocore.config import Config

# Configure boto3 for high throughput
BOTO_CONFIG = Config(
    retries = {'max_attempts': 3, 'mode': 'standard'},
    max_pool_connections = 100
)

def check_file(row):
    """
    Checks if a file exists on S3.
    Returns the cleaned line (string) if it exists, None otherwise.
    """
    try:
        # Tuple unpacking handles the row regardless of pandas parsing oddities
        uri = str(row[0]).strip()
        label = int(row[1])
        
        # Robust URI parsing
        clean_uri = uri.replace("s3://", "")
        bucket, key = clean_uri.split("/", 1)
        
        # Create a thread-local client
        # Note: Boto3 sessions are not thread safe, but creating a client 
        # from a shared session usually works if read-only. 
        # To be 100% safe in threads, we create a fresh client.
        s3 = boto3.client('s3', config=BOTO_CONFIG)
        
        # Head object is fast and cheap
        s3.head_object(Bucket=bucket, Key=key)
        
        # Return valid line format
        return f"{uri} {label}"
        
    except Exception:
        # Any error (404, 403, parsing) means we drop this file
        return None

def process_csv(input_path, output_path, workers):
    print(f"\nProcessing: {input_path}")
    
    # 1. Load with regex separator to handle multiple spaces
    df = pd.read_csv(input_path, header=None, sep=r'\s+', engine='python')
    print(f"Loaded {len(df)} rows.")
    
    # Convert to list for threading
    data = df.values.tolist()
    
    valid_lines = []
    
    # 2. Multithreaded Verification
    with ThreadPoolExecutor(max_workers=workers) as executor:
        # We use tqdm to show progress
        results = list(tqdm(executor.map(check_file, data), total=len(data), unit="file"))
    
    # 3. Filter Nones
    valid_lines = [r for r in results if r is not None]
    
    # 4. Save
    with open(output_path, 'w') as f:
        for line in valid_lines:
            f.write(line + "\n")
            
    print(f"Original: {len(data)}")
    print(f"Valid:    {len(valid_lines)}")
    print(f"Removed:  {len(data) - len(valid_lines)}")
    print(f"Saved ->  {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_csv')
    parser.add_argument('val_csv')
    parser.add_argument('test_csv')
    parser.add_argument('--workers', type=int, default=64)
    args = parser.parse_args()

    # Process all three splits
    process_csv(args.train_csv, args.train_csv.replace('.csv', '_cleaned.csv'), args.workers)
    process_csv(args.val_csv, args.val_csv.replace('.csv', '_cleaned.csv'), args.workers)
    process_csv(args.test_csv, args.test_csv.replace('.csv', '_cleaned.csv'), args.workers)

if __name__ == '__main__':
    main()