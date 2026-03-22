#!/bin/bash
# Chain MIMIC cross-institution transfer for 4 diseases on GPUs 0-3.
# Order: HCM, DCM, STEMI, Amyloidosis
# Uses UHN-trained probes (4 models: G, L-K, EP, Pan) on MIMIC test data.

set -euo pipefail

export DEVICES="cuda:0 cuda:1 cuda:2 cuda:3"
export MASTER_PORT=29500

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== MIMIC cross-institution chain: HCM -> DCM -> STEMI -> Amyloidosis ==="
echo "=== GPUs 0-3, 4 models each ==="
echo "=== Started: $(date) ==="

for TASK in disease_hcm disease_dcm disease_stemi disease_amyloidosis; do
    echo ""
    echo "========================================"
    echo ">>> Starting: ${TASK} — $(date)"
    echo "========================================"
    bash "${SCRIPT_DIR}/run_mimic_xfer.sh" "$TASK"
done

echo ""
echo "=== MIMIC cross-institution chain: ALL DONE — $(date) ==="
