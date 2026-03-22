#!/bin/bash
# Chain UHN prediction averaging for 3 disease tasks on GPUs 4-7.
# Order: DCM, STEMI, Rheumatic MV
# Uses existing UHN-trained probes, runs pred avg inference on UHN test data.

set -euo pipefail

export DEVICES="cuda:4 cuda:5 cuda:6 cuda:7"
export MASTER_PORT=29502

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== UHN pred avg chain: DCM -> STEMI -> Rheumatic MV ==="
echo "=== GPUs 4-7, 4 models each ==="
echo "=== Started: $(date) ==="

for TASK in disease_dcm disease_stemi disease_rheumatic_mv; do
    echo ""
    echo "========================================"
    echo ">>> Starting: ${TASK} — $(date)"
    echo "========================================"
    bash "${SCRIPT_DIR}/run_pred_avg.sh" --models "echojepa-g echojepa-l-k echoprime panecho" "$TASK"
done

echo ""
echo "=== UHN pred avg chain: ALL DONE — $(date) ==="
