import argparse
import random
import os
import math
from collections import defaultdict

def make_stratified_subset(input_path, output_path, percentage, seed=42, min_per_class=1):
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        return

    print(f"Reading from: {input_path}")
    with open(input_path, 'r') as f:
        lines = f.readlines()

    # 1. Group lines by label
    label_to_lines = defaultdict(list)
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 2:
            continue # Skip malformed lines
        
        # Assuming format: path label_int ...
        label = parts[1] 
        label_to_lines[label].append(line)

    # 2. Sample from each group
    rng = random.Random(seed)
    final_subset = []
    
    print(f"Stratifying {len(lines)} samples across {len(label_to_lines)} classes...")
    
    for label, class_lines in label_to_lines.items():
        total_class = len(class_lines)
        
        # Calculate target count
        target_count = math.ceil(total_class * (percentage / 100.0))
        
        # Enforce minimum floor (unless class is smaller than floor)
        count = max(target_count, min(min_per_class, total_class))
        
        # Shuffle and pick
        rng.shuffle(class_lines)
        selected = class_lines[:count]
        final_subset.extend(selected)
        
        # Optional: Print distribution for sanity check
        # print(f"  Class {label}: {len(selected)}/{total_class} ({len(selected)/total_class:.1%})")

    # 3. Final Shuffle to mix classes
    rng.shuffle(final_subset)

    print(f"  - Original size: {len(lines)}")
    print(f"  - Subset size:   {len(final_subset)} ({len(final_subset)/len(lines):.2%})")
    print(f"  - Random Seed:   {seed}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        f.writelines(final_subset)
    
    print(f"Saved stratified subset to: {output_path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a stratified random subset of a dataset file.")
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV/TXT file")
    parser.add_argument("--out", type=str, required=True, help="Path to output file")
    parser.add_argument("--percent", type=float, required=True, help="Percentage of data to keep (e.g. 1.0)")
    parser.add_argument("--min", type=int, default=2, help="Minimum examples per class (default: 2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    make_stratified_subset(args.input, args.out, args.percent, args.seed, args.min)