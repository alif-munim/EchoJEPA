#!/bin/bash
set -euo pipefail

# ==============================================================================
# DATA EFFICIENCY EXPERIMENTS - Direct Execution Script
# ==============================================================================
# Usage: bash run_efficiency_experiments.sh
# ==============================================================================

# --- CONFIGURATION ---
REPO_DIR="/home/sagemaker-user/user-default-efs/vjepa2"
# UPDATED: Points to the ViT-L view config
BASE_CFG="${REPO_DIR}/configs/eval/vitl/view.yaml" 
TRAIN_FULL="${REPO_DIR}/data/csv/uhn_views_22k_train.csv"
OUTPUT_DIR="${REPO_DIR}/results/efficiency_experiments"
LOG_DIR="${OUTPUT_DIR}/logs"

# Matches the device list in your command
DEVICES="cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7"

# --- SETUP ---
cd "$REPO_DIR"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

echo "[INFO] Starting efficiency experiments at $(date)"
echo "[INFO] Base config: $BASE_CFG"
echo "[INFO] Training data: $TRAIN_FULL"

# ==============================================================================
# HELPER: Create Stratified Subset
# ==============================================================================
create_stratified_subset() {
    local INPUT=$1
    local OUTPUT=$2
    local PERCENT=$3
    local SEED=$4
    local MIN_PER_CLASS=${5:-3}

    python3 - "$INPUT" "$OUTPUT" "$PERCENT" "$SEED" "$MIN_PER_CLASS" << 'PYTHON'
import sys, random, math
from collections import defaultdict

input_path, output_path, percent, seed, min_per_class = sys.argv[1:6]
percent, seed, min_per_class = float(percent), int(seed), int(min_per_class)

with open(input_path, 'r') as f:
    lines = f.readlines()

label_to_lines = defaultdict(list)
for line in lines:
    parts = line.strip().split()
    if len(parts) >= 2:
        label_to_lines[parts[1]].append(line)

rng = random.Random(seed)
subset = []

for label, class_lines in label_to_lines.items():
    total = len(class_lines)
    target = math.ceil(total * (percent / 100.0))
    count = max(target, min(min_per_class, total))
    rng.shuffle(class_lines)
    subset.extend(class_lines[:count])

rng.shuffle(subset)
with open(output_path, 'w') as f:
    f.writelines(subset)

print(f"Created {output_path}: {len(subset)} samples ({percent}% of {len(lines)}, seed={seed})")
PYTHON
}

# ==============================================================================
# HELPER: Generate Run Config
# ==============================================================================
generate_config() {
    local BASE=$1
    local OUT=$2
    local TAG=$3
    local TRAIN_CSV=$4

    python3 - "$BASE" "$OUT" "$TAG" "$TRAIN_CSV" << 'PYTHON'
import yaml, sys, os

base_path, out_path, tag, train_csv = sys.argv[1:5]

with open(base_path) as f:
    cfg = yaml.safe_load(f)

cfg['tag'] = tag
cfg['experiment']['data']['dataset_train'] = train_csv

# Ensure results go to the efficiency folder structure
if 'folder' in cfg:
    cfg['folder'] = os.path.join(os.path.dirname(out_path), "..", "results", "classifier")

os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, 'w') as f:
    yaml.dump(cfg, f)

print(f"Generated config: {out_path} (tag={tag})")
PYTHON
}

# ==============================================================================
# HELPER: Run Single Training
# ==============================================================================
run_training() {
    local TAG=$1
    local TRAIN_CSV=$2
    
    echo ""
    echo "=================================================================="
    echo "STARTING: $TAG"
    echo "Dataset: $TRAIN_CSV"
    echo "Time: $(date)"
    echo "=================================================================="

    local RUN_CFG="${OUTPUT_DIR}/configs/${TAG}.yaml"
    local LOG_FILE="${LOG_DIR}/${TAG}.log"

    # Generate config
    generate_config "$BASE_CFG" "$RUN_CFG" "$TAG" "$TRAIN_CSV"

    # Run training
    python -m evals.main \
        --fname "$RUN_CFG" \
        --devices $DEVICES \
        2>&1 | tee "$LOG_FILE"

    echo "[INFO] Completed: $TAG at $(date)"
}

# ==============================================================================
# MAIN EXPERIMENT LOOP
# ==============================================================================

mkdir -p "${OUTPUT_DIR}/configs"
mkdir -p "${OUTPUT_DIR}/subsets"

# --- 1. Full Data (100%) ---
# This runs first as requested
run_training "echojepa_l_view_100pct" "$TRAIN_FULL"

# --- 2. 10% Data (3 seeds) ---
for SEED in 1 2 3; do
    SUBSET_CSV="${OUTPUT_DIR}/subsets/train_10pct_s${SEED}.csv"
    create_stratified_subset "$TRAIN_FULL" "$SUBSET_CSV" 10.0 "$SEED"
    run_training "echojepa_l_view_10pct_s${SEED}" "$SUBSET_CSV"
done

# --- 3. 1% Data (3 seeds) ---
for SEED in 1 2 3; do
     SUBSET_CSV="${OUTPUT_DIR}/subsets/train_1pct_s${SEED}.csv"
     create_stratified_subset "$TRAIN_FULL" "$SUBSET_CSV" 1.0 "$SEED"
     run_training "echojepa_l_view_1pct_s${SEED}" "$SUBSET_CSV"
done

# ==============================================================================
# DONE
# ==============================================================================
echo ""
echo "=================================================================="
echo "ALL EXPERIMENTS COMPLETED at $(date)"
echo "Results: $OUTPUT_DIR"
echo "Logs: $LOG_DIR"
echo "=================================================================="