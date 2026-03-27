#!/bin/bash
# Retry missing MIMIC pred avgs: finds all task×model with best.pt but no study_predictions.csv.
# Runs 2 at a time on split GPUs. Safe to run while chain is NOT running.
#
# Usage:
#   nohup bash scripts/run_mimic_predavg_retry.sh > logs/mimic_predavg_retry.log 2>&1 &

set -euo pipefail

REPO=/mnt/custom-file-systems/efs/fs-0049217cdf69186d7_fsap-0fa7145b64eaa046b/vjepa2
cd "$REPO"

LOG=logs/mimic_predavg_retry.log
PROBE_DIR="${REPO}/checkpoints/probes/mimic"
OUT_DIR="${REPO}/evals/vitg-384/nature_medicine/mimic"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

ALL_TASKS=(
    mortality_1yr mortality_90d mortality_30d in_hospital_mortality
    readmission_30d discharge_destination los_remaining
    troponin_t nt_probnp creatinine lactate
)
ALL_MODELS=(echojepa-g echojepa-l-k echoprime panecho)

# --- Collect missing pred avgs ---
MISSING=()
for model in "${ALL_MODELS[@]}"; do
    for task in "${ALL_TASKS[@]}"; do
        bp="${PROBE_DIR}/${task}/${model}/best.pt"
        pa="${OUT_DIR}/video_classification_frozen/${task}-predavg-${model}/study_predictions.csv"
        if [ -f "$bp" ] && [ ! -f "$pa" ]; then
            MISSING+=("${task}:${model}")
        fi
    done
done

if [ ${#MISSING[@]} -eq 0 ]; then
    log "All pred avgs complete — nothing to retry."
    exit 0
fi

log "==========================================="
log "MIMIC Pred Avg Retry — ${#MISSING[@]} missing:"
for m in "${MISSING[@]}"; do log "  $m"; done
log "==========================================="

# --- Run pairs on split GPUs ---
for (( i=0; i<${#MISSING[@]}; i+=2 )); do
    entry_a="${MISSING[$i]}"
    task_a="${entry_a%%:*}"
    model_a="${entry_a##*:}"

    entry_b=""
    task_b=""
    model_b=""
    if (( i+1 < ${#MISSING[@]} )); then
        entry_b="${MISSING[$((i+1))]}"
        task_b="${entry_b%%:*}"
        model_b="${entry_b##*:}"
    fi

    log "--- Pair: ${task_a}/${model_a} + ${task_b:-[none]}/${model_b:-} ---"

    pid_a="" pid_b=""

    DEVICES="cuda:0 cuda:1 cuda:2 cuda:3" MASTER_PORT=29500 \
        bash scripts/run_mimic_pred_avg.sh --models "$model_a" "$task_a" 2>&1 | tee -a "$LOG" &
    pid_a=$!

    if [ -n "$task_b" ]; then
        DEVICES="cuda:4 cuda:5 cuda:6 cuda:7" MASTER_PORT=29501 \
            bash scripts/run_mimic_pred_avg.sh --models "$model_b" "$task_b" 2>&1 | tee -a "$LOG" &
        pid_b=$!
    fi

    wait "$pid_a" || log "WARNING: ${task_a}/${model_a} pred avg failed"
    [ -n "$pid_b" ] && { wait "$pid_b" || log "WARNING: ${task_b}/${model_b} pred avg failed"; }

    sleep 10
done

log "==========================================="
log "MIMIC Pred Avg Retry — DONE"
log "==========================================="
