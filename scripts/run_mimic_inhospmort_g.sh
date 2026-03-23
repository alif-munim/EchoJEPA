#!/bin/bash
# Standalone: Train + pred avg in_hospital_mortality for EchoJEPA-G.
# Waits for the outcome chain (PID 352661) to finish first, then runs.
#
# Usage:
#   nohup bash scripts/run_mimic_inhospmort_g.sh > logs/mimic_inhospmort_g.log 2>&1 &

set -euo pipefail

REPO=/mnt/custom-file-systems/efs/fs-0049217cdf69186d7_fsap-0fa7145b64eaa046b/vjepa2
cd "$REPO"

CHAIN_PID=352661
TASK=in_hospital_mortality
MODEL=echojepa-g
PROBE_DIR="${REPO}/checkpoints/probes/mimic"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# --- Already done? ---
if [ -f "${PROBE_DIR}/${TASK}/${MODEL}/best.pt" ]; then
    log "best.pt already exists for ${TASK}/${MODEL} — skipping training."
else
    # --- Wait for chain to finish ---
    if kill -0 "$CHAIN_PID" 2>/dev/null; then
        log "Waiting for outcome chain (PID ${CHAIN_PID}) to finish..."
        while kill -0 "$CHAIN_PID" 2>/dev/null; do
            sleep 60
        done
        log "Chain finished. Waiting 30s for cleanup..."
        sleep 30
    fi

    # --- Kill orphaned DDP workers ---
    ps -eo pid,ppid,args 2>/dev/null | grep "multiprocessing.spawn" | grep -v grep | awk '$2 == 1 {print $1}' | xargs -r kill 2>/dev/null || true
    ps -eo pid,ppid,args 2>/dev/null | grep "multiprocessing.resource_tracker" | grep -v grep | awk '$2 == 1 {print $1}' | xargs -r kill 2>/dev/null || true
    sleep 5

    # --- Train ---
    log "=== Training ${TASK} / ${MODEL} ==="
    DEVICES="cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7" MASTER_PORT=29500 \
        bash scripts/run_mimic_probe.sh --models "$MODEL" "$TASK"
    log "=== Training done ==="
    sleep 10
fi

# --- Pred avg ---
if [ -f "${PROBE_DIR}/${TASK}/${MODEL}/best.pt" ]; then
    log "=== Prediction averaging ${TASK} / ${MODEL} ==="
    # Kill orphans before pred avg
    ps -eo pid,ppid,args 2>/dev/null | grep "multiprocessing.spawn" | grep -v grep | awk '$2 == 1 {print $1}' | xargs -r kill 2>/dev/null || true
    sleep 5
    DEVICES="cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7" MASTER_PORT=29500 \
        bash scripts/run_mimic_pred_avg.sh --models "$MODEL" "$TASK"
    log "=== Pred avg done ==="
else
    log "ERROR: No best.pt after training — ${TASK}/${MODEL} failed."
    exit 1
fi

log "=== ALL DONE: ${TASK} / ${MODEL} ==="
