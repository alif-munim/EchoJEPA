#!/bin/bash
# Chain: TR severity pred avg (all 5 models) → AS severity training (all 5 models)
# Runs on GPUs 4-7 with MASTER_PORT=29502
set -euo pipefail

export DEVICES="cuda:4 cuda:5 cuda:6 cuda:7"
export MASTER_PORT=29502

REPO=/mnt/custom-file-systems/efs/fs-0049217cdf69186d7_fsap-0fa7145b64eaa046b/vjepa2
cd "$REPO"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "=== Phase 1: TR severity prediction averaging (all 5 models) ==="
bash scripts/run_pred_avg.sh tr_severity

log "=== Phase 2: AS severity probe training (all 5 models) ==="
FRESH=true bash scripts/run_uhn_probe.sh as_severity

log "=== ALL DONE ==="
