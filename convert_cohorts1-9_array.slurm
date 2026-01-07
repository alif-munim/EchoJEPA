#!/bin/bash
#SBATCH --job-name=cohorts1-9_array
#SBATCH --partition=tier1q
#SBATCH --array=0-8999%100
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=2G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/cohorts1-9_array_%A_%a.out
#SBATCH --error=slurm/cohorts1-9_array_%A_%a.err

# =============================================================================
# convert_cohorts1-9_array.slurm - Array job to process cohorts 1-9 in parallel
# Each array task processes ONE MRN from any cohort
# %100 = max 100 jobs running at once (adjust based on cluster policy)
# =============================================================================

module purge
module load gcc/12.1.0
module load python/3.12.3
module load openjpeg/2.5.0
module load parallel/20240622
source /gpfs/data/whitney-lab/echo-FM/CODE/echofm_venv/bin/activate

echo "========================================="
echo "Cohorts 1-9 Array Job - Task $SLURM_ARRAY_TASK_ID"
echo "========================================="
echo "Job ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo ""

# ---------- Paths ----------
export FFMPEG_BIN="/gpfs/data/whitney-lab/echo-FM/CODE/echofm_venv/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
BASE_DATA_DIR="/gpfs/data/whitney-lab/echo-FM/DATA"
BASE_OUTPUT_DIR="/gpfs/data/whitney-lab/echo-FM/CONVERTED_DATA"
SCRIPT_DIR="/gpfs/data/whitney-lab/echo-FM/CODE/adapted_cri"

# ---------- Configuration ----------
export JOBS_PER_STUDY=12
export FORCE_GRAY=1
export PYTHON_THREADS=2
export OPENJPEG_NUM_THREADS=2

# Silence GNU Parallel citation
mkdir -p ~/.parallel
touch ~/.parallel/will-cite

# ---------- Build unified MRN list from cohorts 1-9 ----------
UNIFIED_MRN_LIST="$BASE_OUTPUT_DIR/unified_mrn_list_cohorts1-9.txt"
COHORTS=("cohort1_1000" "cohort2_1000" "cohort3_1000" "cohort4_1000" "cohort5_1000" "cohort6_1000" "cohort7_1000" "cohort8_1000" "cohort9_1000")

# Create unified list if it doesn't exist (only first task creates it)
if [[ ! -f "$UNIFIED_MRN_LIST" ]]; then
    mkdir -p "$BASE_OUTPUT_DIR"
    
    echo "Building unified MRN list from cohorts 1-9..."
    > "$UNIFIED_MRN_LIST"  # Clear file
    
    for cohort in "${COHORTS[@]}"; do
        cohort_input="$BASE_DATA_DIR/$cohort"
        
        if [[ ! -d "$cohort_input" ]]; then
            echo "  WARNING: Cohort directory not found: $cohort_input"
            continue
        fi
        
        echo "  Scanning $cohort..."
        
        # Add MRNs with cohort prefix (format: cohortX_1000/MRN_ID)
        find "$cohort_input" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | while read mrn_dir; do
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

MRN_DIR="$BASE_DATA_DIR/$COHORT/$MRN_NAME"
MRN_OUTPUT="$BASE_OUTPUT_DIR/$COHORT/$MRN_NAME"

echo "Processing: $COHORT / $MRN_NAME"
echo "Input: $MRN_DIR"
echo "Output: $MRN_OUTPUT"
echo ""

if [[ ! -d "$MRN_DIR" ]]; then
    echo "ERROR: MRN directory not found: $MRN_DIR"
    exit 1
fi

# Count studies in this MRN
STUDY_COUNT=$(find "$MRN_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
echo "Studies in this MRN: $STUDY_COUNT"
echo ""

if [[ "$STUDY_COUNT" -eq 0 ]]; then
    echo "No studies found, exiting"
    exit 0
fi

mkdir -p "$MRN_OUTPUT"

# ---------- Process each study in this MRN ----------
START_TIME=$(date +%s)

find "$MRN_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | while read -r STUDY_DIR; do
    STUDY_NAME=$(basename "$STUDY_DIR")
    
    # Check if already processed
    DONE_FLAG="$MRN_OUTPUT/$STUDY_NAME/.done"
    if [[ -f "$DONE_FLAG" ]]; then
        echo "  ✓ $STUDY_NAME already processed, skipping"
        continue
    fi
    
    # Count files
    FILE_COUNT=$(find "$STUDY_DIR" -type f 2>/dev/null | wc -l)
    if [[ "$FILE_COUNT" -eq 0 ]]; then
        echo "  ✗ $STUDY_NAME has no files, skipping"
        continue
    fi
    
    echo "$(date +%T) Processing study: $STUDY_NAME ($FILE_COUNT files)"
    
    # Run conversion for this study
    export BASE_INPUT_DIR="$MRN_DIR"
    export OUTPUT_ROOT="$MRN_OUTPUT"
    export SINGLE_STUDY="$STUDY_NAME"
    
    bash "$SCRIPT_DIR/dicom_batch_convert.sh" 0
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "========================================="
echo "$COHORT / $MRN_NAME Complete!"
echo "========================================="
echo "Time: ${ELAPSED}s ($((ELAPSED / 60))m)"
echo ""

# ---------- Summary for this MRN ----------
DONE_COUNT=$(find "$MRN_OUTPUT" -name ".done" 2>/dev/null | wc -l)
MP4_COUNT=$(find "$MRN_OUTPUT" -type f -name "*.mp4" ! -name "mask_visualization.mp4" ! -path "*/unmasked/*" 2>/dev/null | wc -l)
PNG_COUNT=$(find "$MRN_OUTPUT" -path "*/png/*.png" 2>/dev/null | wc -l)

echo "Results for $COHORT / $MRN_NAME:"
echo "  Studies completed: $DONE_COUNT"
echo "  MP4 files: $MP4_COUNT"
echo "  PNG files: $PNG_COUNT"
echo ""
echo "Task complete!"
