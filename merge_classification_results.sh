#!/bin/bash
# merge_classification_results.sh - Combine individual MRN CSV files into final results

set -euo pipefail

echo "========================================="
echo "Merging Classification Results"
echo "========================================="
echo ""

RESULTS_DIR="/gpfs/data/whitney-lab/echo-FM/RESULTS/classification"
FINAL_CSV="$RESULTS_DIR/view_predictions_cohorts123.csv"

cd "$RESULTS_DIR"

# Count CSV files
CSV_COUNT=$(find . -name "*_predictions.csv" 2>/dev/null | wc -l)
echo "Found $CSV_COUNT MRN CSV files"
echo ""

if [[ "$CSV_COUNT" -eq 0 ]]; then
    echo "ERROR: No MRN CSV files found in $RESULTS_DIR"
    exit 1
fi

# Create final CSV with header from first file
echo "Creating merged CSV: $FINAL_CSV"

FIRST_CSV=$(find . -name "*_predictions.csv" 2>/dev/null | head -n 1)

if [[ -z "$FIRST_CSV" ]]; then
    echo "ERROR: Could not find any CSV files"
    exit 1
fi

# Write header
echo "Writing header..."
head -n 1 "$FIRST_CSV" > "$FINAL_CSV"
echo "✓ Header written"

# Append all data (skip headers) - faster approach
echo "Merging $CSV_COUNT CSV files..."
echo "This may take 2-3 minutes for ~1000 files..."

# Use a temporary file list for better performance
TMPFILE=$(mktemp)
find . -name "*_predictions.csv" 2>/dev/null | sort > "$TMPFILE"

COUNT=0
while IFS= read -r csv_file; do
    tail -n +2 "$csv_file" >> "$FINAL_CSV"
    COUNT=$((COUNT + 1))
    if (( COUNT % 200 == 0 )); then
        echo "  Processed: $COUNT / $CSV_COUNT files"
    fi
done < "$TMPFILE"

rm -f "$TMPFILE"

echo "✓ All $COUNT files merged"
echo ""

# Show statistics
LINE_COUNT=$(wc -l < "$FINAL_CSV")
DATA_ROWS=$((LINE_COUNT - 1))

echo "========================================="
echo "Results Summary"
echo "========================================="
echo ""
echo "Output file: $FINAL_CSV"
echo "Total rows: $DATA_ROWS images (+ 1 header)"
echo ""

echo "View Distribution:"
tail -n +2 "$FINAL_CSV" | cut -d, -f6 | sort | uniq -c | sort -rn
echo ""

echo "Per-Cohort Breakdown:"
tail -n +2 "$FINAL_CSV" | cut -d, -f1 | sort | uniq -c
echo ""

echo "========================================="
echo "Merge complete!"
echo "========================================="
