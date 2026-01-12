#!/bin/bash

# =============================================================================
# DEFAULTS
# =============================================================================
# Base path for all experiments
BASE_ROOT="/home/sagemaker-user/user-default-efs/vjepa2/evals/vitg-384"

# Default subdirectory (e.g., 'classifier' or 'rvsf_a4c')
EXP_DIR="classifier" 
EXP_TAG="uhn22k-classifier-fs2-ns2-nvs1-echojepa-224px-fcl-a1g2-ep50-test"
CKPT_FILENAME="latest.pt"

# Default Data Paths
TRAIN_DATA="/home/sagemaker-user/user-default-efs/vjepa2/data/csv/uhn_views_22k_train.csv"
VAL_DATA="/home/sagemaker-user/user-default-efs/vjepa2/data/csv/uhn_views_22k_test.csv"

EPOCHS=30
NUM_HEADS=16
NUM_BLOCKS=4
VAL_ONLY=true
RES=336
FOCAL_LOSS=false
DEVICE="cuda:0"
BATCH_SIZE=6
NUM_CLASSES=13  # Default to 13 classes, override with -C for regression (1)

# =============================================================================
# USAGE / HELP
# =============================================================================
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  -E <dir>      Experiment Directory (e.g., 'rvsf_a4c' or 'classifier') [Default: classifier]"
    echo "  -t <tag>      Experiment Tag (folder name)"
    echo "  -c <file>     Checkpoint Filename (default: latest.pt)"
    echo "  -C <int>      Num Classes (Use 1 for regression, 13 for classification) [Default: 13]"
    echo "  -e <int>      Num Epochs (default: 30)"
    echo "  -H <int>      Num Heads (default: 16)"
    echo "  -b <int>      Num Blocks (default: 4)"
    echo "  -r <int>      Resolution (default: 336)"
    echo "  -B <int>      Batch Size (default: 6)"
    echo "  -D <path>     Train Data CSV Path"
    echo "  -V <path>     Validation Data CSV Path"
    echo "  -f            Enable Focal Loss (flag toggles to true)"
    echo "  -T            Enable Training Mode (flag toggles val_only to false)"
    echo "  -d <device>   Device (default: cuda:0)"
    echo "  -h            Show this help message"
    exit 1
}

# =============================================================================
# PARSE ARGUMENTS
# =============================================================================
while getopts ":E:t:c:C:e:H:b:r:B:D:V:d:fTh" opt; do
  case ${opt} in
    E) EXP_DIR=$OPTARG ;;
    t) EXP_TAG=$OPTARG ;;
    c) CKPT_FILENAME=$OPTARG ;;
    C) NUM_CLASSES=$OPTARG ;;
    e) EPOCHS=$OPTARG ;;
    H) NUM_HEADS=$OPTARG ;;
    b) NUM_BLOCKS=$OPTARG ;;
    r) RES=$OPTARG ;;
    B) BATCH_SIZE=$OPTARG ;;
    D) TRAIN_DATA=$OPTARG ;;
    V) VAL_DATA=$OPTARG ;;
    d) DEVICE=$OPTARG ;;
    f) FOCAL_LOSS=true ;;
    T) VAL_ONLY=false ;;
    h) usage ;;
    \?) echo "Invalid option: -$OPTARG" >&2; usage ;;
  esac
done

# =============================================================================
# 1. PATH SETUP & CONDITIONAL CHECKPOINT FIX
# =============================================================================
# Construct the full evaluation root dynamically
EVAL_ROOT="${BASE_ROOT}/${EXP_DIR}/video_classification_frozen"
CKPT_INPUT="${EVAL_ROOT}/${EXP_TAG}/${CKPT_FILENAME}"

if [ ! -f "$CKPT_INPUT" ]; then
    echo "Error: Input checkpoint not found at: $CKPT_INPUT"
    echo "Check your -E (Experiment Dir) and -t (Tag) arguments."
    exit 1
fi

# Regex check: If DEVICE contains a space " " or a comma ",", treat as multi-GPU
if [[ "$DEVICE" =~ [,\ ] ]]; then
    echo ">> Multi-GPU detected ($DEVICE). Skipping checkpoint fix..."
    # Use the original file path
    CKPT_PATH="$CKPT_INPUT"
else
    echo ">> Single-GPU detected. Fixing Checkpoint..."
    # Define fixed path
    CKPT_PATH="${EVAL_ROOT}/${EXP_TAG}/${CKPT_FILENAME%.*}_fixed.pt"
    
    python classifier/fix_inference_checkpoint.py \
      --input "$CKPT_INPUT" \
      --output "$CKPT_PATH"

    if [ $? -ne 0 ]; then
        echo "Error: Checkpoint fix failed."
        exit 1
    fi
fi

# =============================================================================
# 2. GENERATE RUN ID
# =============================================================================
# basename will capture either "latest.pt" or "latest_fixed.pt" depending on logic above
CKPT_NAME_USED=$(basename "${CKPT_PATH%.*}")

RUN_ID="${EXP_TAG}_${CKPT_NAME_USED}_ep${EPOCHS}_h${NUM_HEADS}_b${NUM_BLOCKS}_bs${BATCH_SIZE}_val${VAL_ONLY}_res${RES}_focal${FOCAL_LOSS}_cls${NUM_CLASSES}"

echo "----------------------------------------------------------------"
echo "RUN ID:      $RUN_ID"
echo "Checkpoint:  $CKPT_PATH"
echo "Data:        $VAL_DATA"
echo "Classes:     $NUM_CLASSES"
echo "----------------------------------------------------------------"

# =============================================================================
# 3. RUN INFERENCE
# =============================================================================
OVERRIDE_TAG="$RUN_ID" \
OVERRIDE_VAL_ONLY=$VAL_ONLY \
OVERRIDE_PRED_PATH="/home/sagemaker-user/user-default-efs/vjepa2/predictions/${RUN_ID}.csv" \
OVERRIDE_CKPT="$CKPT_PATH" \
OVERRIDE_NUM_HEADS=$NUM_HEADS \
OVERRIDE_NUM_BLOCKS=$NUM_BLOCKS \
OVERRIDE_TRAIN_DATA="$TRAIN_DATA" \
OVERRIDE_VAL_DATA="$VAL_DATA" \
OVERRIDE_NUM_CLASSES=$NUM_CLASSES \
OVERRIDE_RES=$RES \
OVERRIDE_EPOCHS=$EPOCHS \
OVERRIDE_FOCAL_LOSS=$FOCAL_LOSS \
OVERRIDE_BATCH=$BATCH_SIZE \
python -m evals.main \
  --fname /home/sagemaker-user/user-default-efs/vjepa2/configs/inference/vitg-384/view/uhn_22k_336px.yaml \
  --devices $DEVICE \
  2>&1 | tee "inf_${RUN_ID}.log"