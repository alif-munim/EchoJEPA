#!/bin/bash
# Nature Medicine Phase 1: Wendy's pillars + standard benchmarks + disease detection
# 18 UHN tasks × 5 models = 90 runs
#
# Usage:
#   # Run all Phase 1 tasks (full run, ~90 × 5h = 450h GPU-time, sequentially)
#   nohup bash scripts/run_phase1.sh 2>&1 | tee logs/phase1_$(date +%Y%m%d_%H%M%S).log &
#
#   # Run a specific group only
#   bash scripts/run_phase1.sh --group rv
#   bash scripts/run_phase1.sh --group hemodynamics
#   bash scripts/run_phase1.sh --group standard
#   bash scripts/run_phase1.sh --group disease
#
#   # Run with subset of models (e.g., only available models)
#   bash scripts/run_phase1.sh --models "echojepa-g echojepa-l"
set -euo pipefail

GROUP="all"
MODELS=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --group) GROUP="$2"; shift 2 ;;
        --models) MODELS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

SCRIPT=scripts/run_uhn_probe.sh
MODEL_ARG=""
[ -n "$MODELS" ] && MODEL_ARG="--models \"$MODELS\""

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"; }

run_task() {
    local task="$1"
    log "========== Starting: $task =========="
    if [ -n "$MODELS" ]; then
        bash "$SCRIPT" --models "$MODELS" "$task"
    else
        bash "$SCRIPT" "$task"
    fi
    log "========== Finished: $task =========="
    echo ""
}

# --- RV Mechanics (Pillar 1) ---
if [ "$GROUP" = "all" ] || [ "$GROUP" = "rv" ]; then
    log "===== RV MECHANICS (Pillar 1): 5 tasks ====="
    run_task tapse
    run_task rv_sp
    run_task rv_fac
    run_task rv_function
    run_task rv_size
fi

# --- Hemodynamics (Pillar 2, B-mode only) ---
if [ "$GROUP" = "all" ] || [ "$GROUP" = "hemodynamics" ]; then
    log "===== HEMODYNAMICS (Pillar 2): 5 tasks ====="
    run_task mr_severity
    run_task as_severity
    run_task tr_severity
    run_task aov_vmax
    run_task aov_mean_grad
fi

# --- Standard Benchmarks ---
if [ "$GROUP" = "all" ] || [ "$GROUP" = "standard" ]; then
    log "===== STANDARD BENCHMARKS: 6 tasks ====="
    run_task lvef
    run_task rvsp
    run_task lv_mass
    run_task ivsd
    run_task lv_systolic_function
    run_task rwma
fi

# --- Disease Detection ---
if [ "$GROUP" = "all" ] || [ "$GROUP" = "disease" ]; then
    log "===== DISEASE DETECTION: 2 tasks ====="
    run_task disease_hcm
    run_task disease_amyloidosis
fi

log "===== PHASE 1 COMPLETE ====="
