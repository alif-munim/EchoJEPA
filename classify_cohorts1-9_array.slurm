#!/bin/bash
#SBATCH --job-name=classify_c1-9
#SBATCH --partition=tier1q
#SBATCH --array=0-8999%100
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --output=slurm/classify_c1-9_%A_%a.out
#SBATCH --error=slurm/classify_c1-9_%A_%a.err
#SBATCH --export=ALL,CUDA_VISIBLE_DEVICES=""

# =============================================================================
# classify_cohorts1-9_array.slurm - Array job to classify cohorts 1-9 in parallel
# Each array task processes ONE MRN (all studies within it)
# Uses unmasked PNGs from unmasked/png/ directories
# Outputs probabilities for all 13 view classes
# %100 = max 100 jobs running at once
# =============================================================================

set -euo pipefail

echo "========================================="
echo "Classification Array Job - Task $SLURM_ARRAY_TASK_ID"
echo "========================================="
echo "Job ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo ""

# ---------- Environment setup ----------
module purge
module load gcc/12.1.0
module load python/3.12.3

# Use CPU-only PyTorch virtual environment
SCRIPT_DIR="/gpfs/data/whitney-lab/echo-FM/CODE/adapted_cri"
VENV_DIR="$SCRIPT_DIR/torch_cpu_venv"

if [[ ! -d "$VENV_DIR" ]]; then
    echo "ERROR: CPU-only PyTorch virtual environment not found!"
    echo "Please run: bash setup_cpu_pytorch.sh"
    exit 1
fi

echo "Activating CPU-only PyTorch virtual environment..."
source "$VENV_DIR/bin/activate"

# Force CPU-only mode (redundant safety)
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Verify environment
echo "Python: $(which python3)"
echo "CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES'"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo ""

# ---------- Paths ----------
CLASSIFY_SCRIPT="$SCRIPT_DIR/classify_cohorts1-9.py"
CHECKPOINT_PATH="$SCRIPT_DIR/p1_best.pth"
BASE_CONVERTED_DIR="/gpfs/data/whitney-lab/echo-FM/CONVERTED_DATA"
RESULTS_DIR="/gpfs/data/whitney-lab/echo-FM/RESULTS/classification"

mkdir -p "$RESULTS_DIR"

# Validate files
if [[ ! -f "$CLASSIFY_SCRIPT" ]]; then
    echo "ERROR: Classification script not found: $CLASSIFY_SCRIPT"
    exit 1
fi

if [[ ! -f "$CHECKPOINT_PATH" ]]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT_PATH"
    exit 1
fi

# ---------- Build unified MRN list from cohorts 1-9 ----------
UNIFIED_MRN_LIST="$RESULTS_DIR/unified_mrn_list_cohorts1-9_classification.txt"
COHORTS=("cohort1_1000" "cohort2_1000" "cohort3_1000" "cohort4_1000" "cohort5_1000" "cohort6_1000" "cohort7_1000" "cohort8_1000" "cohort9_1000")

# Create unified list if it doesn't exist (only first task creates it)
if [[ ! -f "$UNIFIED_MRN_LIST" ]]; then
    echo "Building unified MRN list from cohorts 1-9..."
    > "$UNIFIED_MRN_LIST"  # Clear file
    
    for cohort in "${COHORTS[@]}"; do
        cohort_dir="$BASE_CONVERTED_DIR/$cohort"
        
        if [[ ! -d "$cohort_dir" ]]; then
            echo "  WARNING: Cohort directory not found: $cohort_dir"
            continue
        fi
        
        echo "  Scanning $cohort..."
        
        # Add MRNs with cohort prefix (format: cohortX_1000/MRN_ID)
        find "$cohort_dir" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | while read mrn_dir; do
            mrn_name=$(basename "$mrn_dir")
            echo "$cohort/$mrn_name" >> "$UNIFIED_MRN_LIST"
        done
    done
    
    # Sort the list
    sort -o "$UNIFIED_MRN_LIST" "$UNIFIED_MRN_LIST"
    
    total=$(wc -l < "$UNIFIED_MRN_LIST")
    echo "  Total MRNs across all cohorts: $total"
fi

# Wait for list to be created
while [[ ! -f "$UNIFIED_MRN_LIST" ]]; do
    sleep 1
done

TOTAL_MRNS=$(wc -l < "$UNIFIED_MRN_LIST")
echo "Total MRNs across cohorts 1-9: $TOTAL_MRNS"

# Exit if array index is out of range
if [[ "$SLURM_ARRAY_TASK_ID" -ge "$TOTAL_MRNS" ]]; then
    echo "Task ID $SLURM_ARRAY_TASK_ID >= $TOTAL_MRNS MRNs, nothing to do"
    exit 0
fi

# ---------- Get MRN for this task ----------
# Format: cohortX_1000/MRN_ID
MRN_ENTRY=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$UNIFIED_MRN_LIST")
COHORT=$(dirname "$MRN_ENTRY")
MRN_NAME=$(basename "$MRN_ENTRY")

echo "Processing: $COHORT / $MRN_NAME"
echo "Input: $BASE_CONVERTED_DIR/$MRN_ENTRY"
echo ""

# Count PNG files in unmasked/png/ for this MRN
PNG_COUNT=$(find "$BASE_CONVERTED_DIR/$MRN_ENTRY" -path "*/unmasked/png/*.png" 2>/dev/null | wc -l)
echo "PNG files in this MRN: $PNG_COUNT"

if [[ "$PNG_COUNT" -eq 0 ]]; then
    echo "No unmasked PNG files found, skipping"
    exit 0
fi

echo ""

# ---------- Set up output ----------
MRN_OUTPUT_DIR="$RESULTS_DIR/$COHORT"
OUTPUT_CSV="$MRN_OUTPUT_DIR/${MRN_NAME}_predictions.csv"
DONE_FLAG="$MRN_OUTPUT_DIR/${MRN_NAME}.done"

mkdir -p "$MRN_OUTPUT_DIR"

# Skip if already processed
if [[ -f "$DONE_FLAG" ]]; then
    echo "✓ MRN already processed (found .done flag)"
    exit 0
fi

# ---------- Run classification ----------
echo "============================================================"
echo "Echo View Classification - Cohorts 1-9"
echo "============================================================"
echo ""
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Base directory: $BASE_CONVERTED_DIR"
echo "Results directory: $RESULTS_DIR"
echo "Processing single MRN: $MRN_ENTRY"
echo "Output CSV: $OUTPUT_CSV"
echo ""

# Set environment variables for the Python script
export BASE_CONVERTED_DIR="$BASE_CONVERTED_DIR"
export CHECKPOINT_PATH="$CHECKPOINT_PATH"
export RESULTS_DIR="$RESULTS_DIR"
export PROCESS_MRN="$MRN_ENTRY"
export OUTPUT_CSV="$OUTPUT_CSV"

# Debug: verify environment variables are set
echo "DEBUG: Environment variables:"
echo "  BASE_CONVERTED_DIR=$BASE_CONVERTED_DIR"
echo "  CHECKPOINT_PATH=$CHECKPOINT_PATH"
echo "  RESULTS_DIR=$RESULTS_DIR"
echo "  PROCESS_MRN=$PROCESS_MRN"
echo "  OUTPUT_CSV=$OUTPUT_CSV"
echo ""

# Run classification
echo "Running classification..."
python3 "$CLASSIFY_SCRIPT"

# Mark as done if successful
if [[ -f "$OUTPUT_CSV" ]]; then
    touch "$DONE_FLAG"
    echo ""
    echo "✓ Created .done flag: $DONE_FLAG"
    
    # Count classified images
    IMG_COUNT=$(($(wc -l < "$OUTPUT_CSV") - 1))  # Subtract header
    echo "✓ Classified $IMG_COUNT images"
fi

echo ""
echo "========================================="
echo "Task complete!"
echo "========================================="
