#!/bin/bash
# Run zero-shot anomaly detection on UHN disease tasks via JEPA prediction error.
#
# Usage:
#   bash scripts/run_anomaly_detection.sh [--max_samples N] [--device cuda:X]
#
# Default: run all 8 disease tasks on test sets with 10K subsample.

set -euo pipefail

BASE="/mnt/custom-file-systems/efs/fs-0049217cdf69186d7_fsap-0fa7145b64eaa046b/vjepa2"
CHECKPOINT="${BASE}/checkpoints/vitg-384.pt"
CSV_DIR="${BASE}/experiments/nature_medicine/uhn/probe_csvs"
OUTPUT_DIR="${BASE}/results/anomaly_detection"
MAX_SAMPLES="${MAX_SAMPLES:-10000}"
DEVICE="${DEVICE:-cuda:0}"
NUM_MASKS=10
BATCH_SIZE=4

mkdir -p "${OUTPUT_DIR}"

TASKS=(
    disease_takotsubo
    disease_stemi
    disease_rheumatic_mv
    disease_amyloidosis
    disease_dcm
    disease_hcm
    disease_myxomatous_mv
    disease_bicuspid_av
)

echo "=== JEPA Anomaly Detection (zero-shot) ==="
echo "Checkpoint: ${CHECKPOINT}"
echo "Device: ${DEVICE}"
echo "Max samples: ${MAX_SAMPLES}"
echo "Num masks per clip: ${NUM_MASKS}"
echo ""

for task in "${TASKS[@]}"; do
    csv="${CSV_DIR}/${task}/test.csv"
    output="${OUTPUT_DIR}/${task}.csv"

    if [ ! -f "${csv}" ]; then
        echo "SKIP ${task}: CSV not found at ${csv}"
        continue
    fi

    echo "--- ${task} ---"
    python -m evals.forward_prediction.eval \
        --checkpoint "${CHECKPOINT}" \
        --csv "${csv}" \
        --output "${output}" \
        --num_masks ${NUM_MASKS} \
        --batch_size ${BATCH_SIZE} \
        --device "${DEVICE}" \
        --max_samples ${MAX_SAMPLES}
    echo ""
done

echo "=== Forward Prediction ==="
# Run forward prediction on a subset of tasks (smallest first)
FWD_OUTPUT_DIR="${BASE}/results/forward_prediction"
mkdir -p "${FWD_OUTPUT_DIR}"

for task in disease_takotsubo disease_stemi disease_amyloidosis; do
    csv="${CSV_DIR}/${task}/test.csv"
    output="${FWD_OUTPUT_DIR}/${task}.csv"

    if [ ! -f "${csv}" ]; then
        echo "SKIP ${task}: CSV not found"
        continue
    fi

    echo "--- Forward prediction: ${task} ---"
    python -m evals.forward_prediction.forward_predict \
        --checkpoint "${CHECKPOINT}" \
        --csv "${csv}" \
        --output "${output}" \
        --batch_size ${BATCH_SIZE} \
        --device "${DEVICE}" \
        --max_samples ${MAX_SAMPLES}
    echo ""
done

echo "Done! Results in ${OUTPUT_DIR} and ${FWD_OUTPUT_DIR}"
