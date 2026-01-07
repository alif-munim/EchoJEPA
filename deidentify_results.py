#!/usr/bin/env python3
"""
deidentify_results.py - Add deidentified IDs to classification results

Reads view_predictions_cohorts123.csv and creates a deidentified version with:
- deid_mrn: Sequential patient ID (P0001, P0002, ...)
- deid_study: Sequential study ID per patient (S0001, S0002, ...)
- deid_series: Sequential series ID per study (SER0001, SER0002, ...)

Also creates a mapping file (KEEP SECURE!) that maps real IDs to deidentified IDs.
"""

import csv
from pathlib import Path
from collections import defaultdict

# Configuration
RESULTS_DIR = Path("/gpfs/data/whitney-lab/echo-FM/RESULTS/classification")
INPUT_CSV = RESULTS_DIR / "view_predictions_cohorts123.csv"
OUTPUT_CSV = RESULTS_DIR / "view_predictions_cohorts123_deidentified.csv"
MAPPING_CSV = RESULTS_DIR / "deidentification_mapping_CONFIDENTIAL.csv"

def main():
    print("=" * 70)
    print("Deidentifying Classification Results")
    print("=" * 70)
    print()
    
    if not INPUT_CSV.exists():
        print(f"ERROR: Input file not found: {INPUT_CSV}")
        return 1
    
    print(f"Input:  {INPUT_CSV}")
    print(f"Output: {OUTPUT_CSV}")
    print(f"Mapping: {MAPPING_CSV}")
    print()
    
    # Dictionaries to track mappings
    mrn_map = {}           # real_mrn -> deid_mrn
    study_map = {}         # (mrn, study_id) -> deid_study
    series_map = {}        # (mrn, study_id, series_id) -> deid_series
    
    # Counters
    mrn_counter = 1
    study_counters = defaultdict(int)    # per MRN
    series_counters = defaultdict(int)   # per study
    
    # Read input and create mappings
    print("Reading input and creating deidentified IDs...")
    
    rows = []
    with open(INPUT_CSV, 'r') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            cohort = row['cohort']
            mrn = row['mrn']
            study_id = row['study_id']
            series_id = row['series_id']
            image_path = row['image_path']
            predicted_view = row['predicted_view']
            
            # Create MRN mapping if new
            if mrn not in mrn_map:
                mrn_map[mrn] = f"P{mrn_counter:04d}"
                mrn_counter += 1
            
            deid_mrn = mrn_map[mrn]
            
            # Create study mapping if new
            study_key = (mrn, study_id)
            if study_key not in study_map:
                study_counters[mrn] += 1
                study_map[study_key] = f"S{study_counters[mrn]:04d}"
            
            deid_study = study_map[study_key]
            
            # Create series mapping if new
            series_key = (mrn, study_id, series_id)
            if series_key not in series_map:
                series_counters[study_key] += 1
                series_map[series_key] = f"SER{series_counters[study_key]:04d}"
            
            deid_series = series_map[series_key]
            
            # Store row with deidentified IDs
            rows.append({
                'cohort': cohort,
                'deid_mrn': deid_mrn,
                'deid_study': deid_study,
                'deid_series': deid_series,
                'predicted_view': predicted_view,
                # Keep original IDs for mapping file only
                '_real_mrn': mrn,
                '_real_study': study_id,
                '_real_series': series_id,
                '_image_path': image_path
            })
    
    print(f"✓ Processed {len(rows)} images")
    print(f"  Unique MRNs: {len(mrn_map)}")
    print(f"  Unique Studies: {len(study_map)}")
    print(f"  Unique Series: {len(series_map)}")
    print()
    
    # Write deidentified CSV (public-safe)
    print("Writing deidentified results...")
    with open(OUTPUT_CSV, 'w', newline='') as f:
        fieldnames = ['cohort', 'deid_mrn', 'deid_study', 'deid_series', 'predicted_view']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in rows:
            writer.writerow({
                'cohort': row['cohort'],
                'deid_mrn': row['deid_mrn'],
                'deid_study': row['deid_study'],
                'deid_series': row['deid_series'],
                'predicted_view': row['predicted_view']
            })
    
    print(f"✓ Deidentified CSV written: {OUTPUT_CSV}")
    print()
    
    # Write mapping file (CONFIDENTIAL - keep secure!)
    print("Writing mapping file (CONFIDENTIAL)...")
    with open(MAPPING_CSV, 'w', newline='') as f:
        fieldnames = ['cohort', 'deid_mrn', 'real_mrn', 'deid_study', 'real_study_id', 
                      'deid_series', 'real_series_id', 'real_image_path', 'predicted_view']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in rows:
            writer.writerow({
                'cohort': row['cohort'],
                'deid_mrn': row['deid_mrn'],
                'real_mrn': row['_real_mrn'],
                'deid_study': row['deid_study'],
                'real_study_id': row['_real_study'],
                'deid_series': row['deid_series'],
                'real_series_id': row['_real_series'],
                'real_image_path': row['_image_path'],
                'predicted_view': row['predicted_view']
            })
    
    print(f"✓ Mapping file written: {MAPPING_CSV}")
    print()
    
    # Show sample deidentified data
    print("=" * 70)
    print("Sample Deidentified Data (first 10 rows)")
    print("=" * 70)
    print()
    
    with open(OUTPUT_CSV, 'r') as f:
        for i, line in enumerate(f):
            if i < 11:  # Header + 10 data rows
                print(line.rstrip())
            else:
                break
    
    print()
    print("=" * 70)
    print("View Distribution (Deidentified Data)")
    print("=" * 70)
    print()
    
    # Count views
    from collections import Counter
    view_counts = Counter(row['predicted_view'] for row in rows)
    
    for view, count in view_counts.most_common():
        percentage = count * 100 / len(rows)
        print(f"  {view:15s}: {count:6d} ({percentage:5.1f}%)")
    
    print()
    print("=" * 70)
    print("IMPORTANT SECURITY NOTES")
    print("=" * 70)
    print()
    print(f"✓ PUBLIC FILE (safe to share):")
    print(f"    {OUTPUT_CSV}")
    print()
    print(f"⚠ CONFIDENTIAL FILE (keep secure!):")
    print(f"    {MAPPING_CSV}")
    print()
    print("  The mapping file contains PHI and must be:")
    print("  - Stored in a secure location")
    print("  - Never shared or published")
    print("  - Backed up securely")
    print("  - Only accessible to authorized personnel")
    print()
    print("=" * 70)
    print("Deidentification complete!")
    print("=" * 70)
    print()

if __name__ == "__main__":
    exit(main() or 0)
