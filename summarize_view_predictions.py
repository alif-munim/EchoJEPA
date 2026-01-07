#!/usr/bin/env python3
"""
summarize_view_predictions.py - Summarize view predictions per study and overall

Reads view_predictions_cohorts1-9_unmasked.csv and creates:
1. Per-study summary: For each study, count of each predicted view
2. Overall summary: Total counts across all studies
3. Per-cohort summary: Counts by cohort
"""

import csv
from pathlib import Path
from collections import defaultdict, Counter

# Configuration
RESULTS_DIR = Path("/gpfs/data/whitney-lab/echo-FM/RESULTS/classification")
INPUT_CSV = RESULTS_DIR / "view_predictions_cohorts1-9_unmasked.csv"
STUDY_SUMMARY_CSV = RESULTS_DIR / "view_predictions_by_study.csv"
OVERALL_SUMMARY_CSV = RESULTS_DIR / "view_predictions_overall_summary.csv"

VIEW_LABELS = ["a2c", "a3c", "a4c", "a5c", "plax", "plax-d", "tee", "subcostal", "exclude",
               "psax-av", "psax-mv", "psax-ap", "psax-pm"]

def main():
    print("=" * 70)
    print("View Prediction Summary Generator")
    print("=" * 70)
    print()
    
    if not INPUT_CSV.exists():
        print(f"ERROR: Input file not found: {INPUT_CSV}")
        return 1
    
    print(f"Reading: {INPUT_CSV}")
    print()
    
    # Data structures
    study_predictions = defaultdict(lambda: defaultdict(int))  # {(cohort, mrn, study_id): {view: count}}
    cohort_totals = defaultdict(lambda: defaultdict(int))      # {cohort: {view: count}}
    overall_totals = defaultdict(int)                          # {view: count}
    
    # Read CSV and aggregate
    total_images = 0
    with open(INPUT_CSV, 'r') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            cohort = row['cohort']
            mrn = row['mrn']
            study_id = row['study_id']
            predicted_view = row['predicted_view']
            
            # Skip errors
            if predicted_view == 'error':
                continue
            
            total_images += 1
            
            # Aggregate per study
            study_key = (cohort, mrn, study_id)
            study_predictions[study_key][predicted_view] += 1
            
            # Aggregate per cohort
            cohort_totals[cohort][predicted_view] += 1
            
            # Aggregate overall
            overall_totals[predicted_view] += 1
    
    print(f"✓ Processed {total_images} images")
    print(f"✓ Found {len(study_predictions)} unique studies")
    print()
    
    # Write per-study summary
    print(f"Writing per-study summary: {STUDY_SUMMARY_CSV}")
    
    # Create header with all view columns
    view_columns = sorted(VIEW_LABELS)
    header = ['cohort', 'mrn', 'study_id', 'total_images'] + view_columns + ['most_common_view', 'most_common_count']
    
    with open(STUDY_SUMMARY_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        
        for (cohort, mrn, study_id), view_counts in sorted(study_predictions.items()):
            # Count total images in this study
            total_in_study = sum(view_counts.values())
            
            # Find most common view
            most_common = max(view_counts.items(), key=lambda x: x[1])
            most_common_view, most_common_count = most_common
            
            # Build row
            row = {
                'cohort': cohort,
                'mrn': mrn,
                'study_id': study_id,
                'total_images': total_in_study,
                'most_common_view': most_common_view,
                'most_common_count': most_common_count
            }
            
            # Add counts for each view
            for view in view_columns:
                row[view] = view_counts.get(view, 0)
            
            writer.writerow(row)
    
    print(f"✓ Per-study summary written")
    print()
    
    # Write overall summary
    print(f"Writing overall summary: {OVERALL_SUMMARY_CSV}")
    
    with open(OVERALL_SUMMARY_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Summary Type', 'Category', 'View', 'Count', 'Percentage'])
        
        # Overall totals
        writer.writerow(['=== OVERALL TOTALS ===', '', '', '', ''])
        for view in sorted(VIEW_LABELS):
            count = overall_totals.get(view, 0)
            percentage = (count / total_images * 100) if total_images > 0 else 0
            writer.writerow(['Overall', 'All Cohorts', view, count, f'{percentage:.2f}%'])
        
        writer.writerow(['Overall', 'All Cohorts', 'TOTAL', total_images, '100.00%'])
        writer.writerow([])
        
        # Per-cohort totals
        writer.writerow(['=== PER-COHORT BREAKDOWN ===', '', '', '', ''])
        
        for cohort in sorted(cohort_totals.keys()):
            cohort_total = sum(cohort_totals[cohort].values())
            
            for view in sorted(VIEW_LABELS):
                count = cohort_totals[cohort].get(view, 0)
                percentage = (count / cohort_total * 100) if cohort_total > 0 else 0
                writer.writerow(['Per-Cohort', cohort, view, count, f'{percentage:.2f}%'])
            
            writer.writerow(['Per-Cohort', cohort, 'TOTAL', cohort_total, '100.00%'])
            writer.writerow([])
    
    print(f"✓ Overall summary written")
    print()
    
    # Print summary to console
    print("=" * 70)
    print("OVERALL VIEW DISTRIBUTION")
    print("=" * 70)
    print()
    
    for view in sorted(VIEW_LABELS):
        count = overall_totals.get(view, 0)
        percentage = (count / total_images * 100) if total_images > 0 else 0
        print(f"  {view:15s}: {count:7d} images ({percentage:5.2f}%)")
    
    print(f"  {'TOTAL':15s}: {total_images:7d} images (100.00%)")
    print()
    
    print("=" * 70)
    print("PER-COHORT BREAKDOWN")
    print("=" * 70)
    print()
    
    for cohort in sorted(cohort_totals.keys()):
        cohort_total = sum(cohort_totals[cohort].values())
        print(f"{cohort}:")
        print(f"  Total images: {cohort_total}")
        print(f"  Top 3 views:")
        
        # Get top 3 views for this cohort
        top_views = sorted(cohort_totals[cohort].items(), key=lambda x: x[1], reverse=True)[:3]
        for view, count in top_views:
            percentage = (count / cohort_total * 100) if cohort_total > 0 else 0
            print(f"    {view:15s}: {count:6d} ({percentage:5.2f}%)")
        print()
    
    print("=" * 70)
    print("STUDY-LEVEL STATISTICS")
    print("=" * 70)
    print()
    
    # Calculate study-level stats
    images_per_study = [sum(counts.values()) for counts in study_predictions.values()]
    
    if images_per_study:
        print(f"  Total studies: {len(study_predictions)}")
        print(f"  Images per study:")
        print(f"    Min:    {min(images_per_study)}")
        print(f"    Max:    {max(images_per_study)}")
        print(f"    Mean:   {sum(images_per_study) / len(images_per_study):.1f}")
        print(f"    Median: {sorted(images_per_study)[len(images_per_study)//2]}")
    
    print()
    
    # Find studies with most diverse views
    print("  Studies with most view diversity (top 10):")
    study_diversity = [(key, len(counts)) for key, counts in study_predictions.items()]
    study_diversity.sort(key=lambda x: x[1], reverse=True)
    
    for i, ((cohort, mrn, study_id), num_views) in enumerate(study_diversity[:10], 1):
        total_imgs = sum(study_predictions[(cohort, mrn, study_id)].values())
        print(f"    {i:2d}. {cohort}/{mrn}/{study_id[:20]}... : {num_views} different views, {total_imgs} images")
    
    print()
    print("=" * 70)
    print("Summary complete!")
    print("=" * 70)
    print()
    print(f"Files created:")
    print(f"  1. {STUDY_SUMMARY_CSV}")
    print(f"     - One row per study with view counts")
    print(f"  2. {OVERALL_SUMMARY_CSV}")
    print(f"     - Overall and per-cohort totals")
    print()

if __name__ == "__main__":
    exit(main() or 0)
