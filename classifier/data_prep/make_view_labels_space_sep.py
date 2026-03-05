#!/usr/bin/env python3
import argparse
import csv
import os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mapping", required=True, help="Path to mapping file")
    ap.add_argument("--use-existing-mapping", action="store_true", help="If set, read mapping instead of creating it")
    ap.add_argument("--split", default=None)
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--path-col", default="mp4_path")
    ap.add_argument("--split-col", default="split")
    args = ap.parse_args()

    # 1. Load Data
    with open(args.inp, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if args.split:
        rows = [r for r in rows if r[args.split_col] == args.split]

    # 2. Handle Mapping
    label_to_idx = {}
    
    if args.use_existing_mapping:
        # READ mode: Load existing mapping
        print(f"Reading existing mapping from {args.mapping}...")
        with open(args.mapping, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    idx, label = parts[0], parts[1]
                    label_to_idx[label] = int(idx)
    else:
        # WRITE mode: Create new mapping from current data
        labels = sorted({r[args.label_col] for r in rows})
        label_to_idx = {lab: i for i, lab in enumerate(labels)}
        
        print(f"Creating new mapping at {args.mapping}...")
        with open(args.mapping, "w") as f:
            for lab, idx in label_to_idx.items():
                f.write(f"{idx}\t{lab}\n")

    # 3. Write Output
    written = 0
    skipped = 0
    with open(args.out, "w") as f:
        for r in rows:
            path = r[args.path_col].strip()
            label = r[args.label_col].strip()
            
            if label not in label_to_idx:
                print(f"Warning: Label '{label}' not in mapping. Skipping {path}")
                skipped += 1
                continue
                
            f.write(f"{path} {label_to_idx[label]}\n")
            written += 1

    print(f"Done. Wrote {written} rows to {args.out}. Skipped {skipped}.")

if __name__ == "__main__":
    main()