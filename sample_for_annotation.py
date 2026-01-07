#!/usr/bin/env python3
"""
sample_for_annotation.py - Sample studies for manual annotation
Randomly selects 1500 studies from cohorts 1-9 classification results and copies
their unmasked PNGs to annotation directory, organized by predicted view class.

Input: CSV with classification results
Output: PNG files organized in subdirectories by predicted view
"""

import csv
import random
import shutil
from pathlib import Path
from collections import defaultdict
import sys

# ---------- Configuration ----------
CLASSIFICATION_CSV = Path("/gpfs/data/whitney-lab/echo-FM/RESULTS/classification/view_predictions_cohorts1-9_unmasked.csv")
ANNOTATION_DIR = Path("/gpfs/data/whitney-lab/echo-FM/CLASSIFICATION_ANNOTATION/TO_ANNOTATE")
NUM_STUDIES = 1500
RANDOM_SEED = 42  # For reproducibility

# View classes (must match CSV columns)
VIEW_CLASSES = ["a2c", "a3c", "a4c", "a5c", "plax", "plax-d", "tee", "subcostal", "exclude",
                "psax-av", "psax-mv", "psax-ap", "psax-pm"]

def main():
    print("=" * 70)
    print("Sample Studies for Manual Annotation")
    print("=" * 70)
    print()
    
    # Validate inputs
    if not CLASSIFICATION_CSV.exists():
        print(f"ERROR: Classification CSV not found: {CLASSIFICATION_CSV}")
        sys.exit(1)
    
    print(f"Classification CSV: {CLASSIFICATION_CSV}")
    print(f"Output directory: {ANNOTATION_DIR}")
    print(f"Number of studies to sample: {NUM_STUDIES}")
    print(f"Random seed: {RANDOM_SEED}")
    print()
    
    # Read CSV and group by study
    print("Reading classification results...")
    study_images = defaultdict(list)  # {(cohort, mrn, study_id): [row1, row2, ...]}
    total_images = 0
    
    with open(CLASSIFICATION_CSV, 'r', newline='') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            cohort = row['cohort']
            mrn = row['mrn']
            study_id = row['study_id']
            
            study_key = (cohort, mrn, study_id)
            study_images[study_key].append(row)
            total_images += 1
    
    total_studies = len(study_images)
    print(f"✓ Loaded {total_images} images from {total_studies} studies")
    print()
    
    # Validate sample size
    if NUM_STUDIES > total_studies:
        print(f"WARNING: Requested {NUM_STUDIES} studies, but only {total_studies} available")
        print(f"         Will sample all {total_studies} studies instead")
        sample_size = total_studies
    else:
        sample_size = NUM_STUDIES
    
    # Randomly sample studies
    print(f"Randomly sampling {sample_size} studies...")
    random.seed(RANDOM_SEED)
    all_study_keys = list(study_images.keys())
    sampled_keys = random.sample(all_study_keys, sample_size)
    print(f"✓ Selected {len(sampled_keys)} studies")
    print()
    
    # Create output directories
    print("Creating output directories...")
    for view_class in VIEW_CLASSES:
        view_dir = ANNOTATION_DIR / view_class
        view_dir.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {view_class}/")
    print()
    
    # Copy files organized by predicted view
    print("Copying PNG files...")
    files_copied = defaultdict(int)  # Count files per view class
    files_failed = []
    total_copied = 0
    
    for idx, study_key in enumerate(sampled_keys, 1):
        if idx % 100 == 0 or idx == 1:
            print(f"  Progress: {idx}/{sample_size} studies ({idx*100//sample_size}%)", flush=True)
        
        cohort, mrn, study_id = study_key
        images = study_images[study_key]
        
        for img_row in images:
            predicted_view = img_row['predicted_view']
            image_path = Path(img_row['image_path'])
            
            # Skip errors
            if predicted_view == "error" or not image_path.exists():
                files_failed.append(str(image_path))
                continue
            
            # Validate view class
            if predicted_view not in VIEW_CLASSES:
                print(f"WARNING: Unknown view class '{predicted_view}' for {image_path}", file=sys.stderr)
                files_failed.append(str(image_path))
                continue
            
            # Destination: ANNOTATION_DIR/view_class/cohort_mrn_studyid_seriesid.png
            series_id = img_row['series_id']
            dest_filename = f"{cohort}_{mrn}_{study_id}_{series_id}.png"
            dest_path = ANNOTATION_DIR / predicted_view / dest_filename
            
            # Copy file
            try:
                shutil.copy2(image_path, dest_path)
                files_copied[predicted_view] += 1
                total_copied += 1
            except Exception as e:
                print(f"ERROR copying {image_path} to {dest_path}: {e}", file=sys.stderr)
                files_failed.append(str(image_path))
    
    print()
    print("✓ File copying complete")
    print()
    
    # Summary statistics
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()
    print(f"Studies sampled: {sample_size}")
    print(f"Files copied: {total_copied}")
    print(f"Files failed: {len(files_failed)}")
    print()
    
    print("Files per View Class:")
    print("-" * 40)
    for view_class in VIEW_CLASSES:
        count = files_copied[view_class]
        if count > 0:
            percentage = count * 100 / total_copied if total_copied > 0 else 0
            print(f"  {view_class:15s}: {count:6d} files ({percentage:5.1f}%)")
    print()
    
    # Per-cohort breakdown
    print("Per-Cohort Breakdown:")
    print("-" * 40)
    cohort_counts = defaultdict(int)
    for (cohort, mrn, study_id) in sampled_keys:
        cohort_counts[cohort] += 1
    
    for cohort in sorted(cohort_counts.keys()):
        count = cohort_counts[cohort]
        percentage = count * 100 / sample_size if sample_size > 0 else 0
        print(f"  {cohort}: {count} studies ({percentage:.1f}%)")
    print()
    
    if files_failed:
        print(f"WARNING: {len(files_failed)} files failed to copy")
        if len(files_failed) <= 10:
            print("Failed files:")
            for f in files_failed:
                print(f"  - {f}")
        else:
            print(f"First 10 failed files:")
            for f in files_failed[:10]:
                print(f"  - {f}")
            print(f"  ... and {len(files_failed) - 10} more")
        print()
    
    print("=" * 70)
    print("Sampling Complete!")
    print("=" * 70)
    print()
    print(f"Output directory: {ANNOTATION_DIR}")
    print()
    print("Directory structure:")
    print(f"  {ANNOTATION_DIR}/")
    for view_class in VIEW_CLASSES:
        count = files_copied[view_class]
        if count > 0:
            print(f"    {view_class}/ ({count} files)")
    print()
    print("Files are named: cohort_mrn_studyid_seriesid.png")
    print()

if __name__ == "__main__":
    main()
