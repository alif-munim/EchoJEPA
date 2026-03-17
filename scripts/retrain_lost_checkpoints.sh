#!/bin/bash
# Retrain all lost checkpoints from Bug 007.
# Runs sequentially on GPUs 4-7 (port 29501) while TR severity runs on GPUs 0-3.
# Total: 23 runs across 5 tasks (~15 hours).
#
# Usage: nohup bash scripts/retrain_lost_checkpoints.sh > logs/retrain_all_$(date +%Y%m%d_%H%M%S).log 2>&1 &
set -euo pipefail

export DEVICES="cuda:4 cuda:5 cuda:6 cuda:7"
export MASTER_PORT=29501

REPO=/mnt/custom-file-systems/efs/fs-0049217cdf69186d7_fsap-0fa7145b64eaa046b/vjepa2
cd "$REPO"

echo "=== Retrain Lost Checkpoints (Bug 007) ==="
echo "Started: $(date)"
echo "GPUs: $DEVICES | Port: $MASTER_PORT"
echo ""

# 1. LVEF — 5 models (already running, will be skipped if complete)
echo ">>> [1/5] LVEF — 5 models"
bash scripts/run_uhn_probe.sh lvef
echo ">>> LVEF done: $(date)"
echo ""

# 2. TAPSE — 5 models
echo ">>> [2/5] TAPSE — 5 models"
bash scripts/run_uhn_probe.sh tapse
echo ">>> TAPSE done: $(date)"
echo ""

# 3. MR severity — 5 models
echo ">>> [3/5] MR severity — 5 models"
bash scripts/run_uhn_probe.sh mr_severity
echo ">>> MR severity done: $(date)"
echo ""

# 4. AS severity — 5 models
echo ">>> [4/5] AS severity — 5 models"
bash scripts/run_uhn_probe.sh as_severity
echo ">>> AS severity done: $(date)"
echo ""

# 5. AV Vmax — G, L, L-K only (EchoPrime + PanEcho survived)
echo ">>> [5/5] AV Vmax — 3 models (G, L, L-K)"
bash scripts/run_uhn_probe.sh --models "echojepa-g echojepa-l echojepa-l-k" aov_vmax
echo ">>> AV Vmax done: $(date)"
echo ""

echo "=== All retraining complete: $(date) ==="
