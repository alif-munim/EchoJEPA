#!/bin/bash
# Chain: Train + pred avg ALL MIMIC outcome/biomarker tasks.
# Phase 1: EchoJEPA-G only (fills manuscript EchoJEPA-G column first)
# Phase 2: L-K, EchoPrime, PanEcho for all tasks
#
# Runs 2 tasks in parallel on split GPUs (0-3 and 4-7).
# For each task×model: skip if best.pt exists, else train, then pred avg.
#
# Usage:
#   nohup bash scripts/run_mimic_outcome_chain.sh > logs/mimic_outcome_chain.log 2>&1 &
#
# NOTE: end-of-life CSVs not yet built — excluded from task list.

set -euo pipefail

REPO=/mnt/custom-file-systems/efs/fs-0049217cdf69186d7_fsap-0fa7145b64eaa046b/vjepa2
cd "$REPO"

LOG=logs/mimic_outcome_chain.log
PROBE_DIR="${REPO}/checkpoints/probes/mimic"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

# --- Task list (manuscript table order) ---
# 8 outcome tasks + 4 biomarker tasks = 12 total (end-of-life excluded — no CSVs yet)
ALL_TASKS=(
    mortality_1yr
    mortality_90d
    mortality_30d
    in_hospital_mortality
    readmission_30d
    discharge_destination
    los_remaining
    troponin_t
    nt_probnp
    creatinine
    lactate
)

# --- Model order: G first (manuscript priority), then others ---
ALL_MODELS=(echojepa-g echojepa-l-k echoprime panecho)

# --- Wait for currently running training jobs ---
wait_for_training() {
    local pids
    pids=$(ps -eo pid,args 2>/dev/null | grep 'python -m evals.main.*nm_mimic' | grep -v grep | awk '{print $1}' || true)
    if [ -n "$pids" ]; then
        log "Waiting for running MIMIC training jobs to finish (PIDs: $(echo $pids | tr '\n' ' '))..."
        for pid in $pids; do
            while kill -0 "$pid" 2>/dev/null; do
                sleep 30
            done
        done
        log "All prior jobs finished."
        sleep 10
    fi
}

# --- Check if model×task has best.pt ---
has_checkpoint() {
    local task="$1" model="$2"
    [ -f "${PROBE_DIR}/${task}/${model}/best.pt" ]
}

# --- Check if model×task has completed pred avg (study_predictions.csv exists) ---
has_pred_avg() {
    local task="$1" model="$2"
    local OUT_DIR="${REPO}/evals/vitg-384/nature_medicine/mimic"
    [ -f "${OUT_DIR}/video_classification_frozen/${task}-predavg-${model}/study_predictions.csv" ]
}

# --- Kill orphaned DDP workers (ppid=1 only — safe for concurrent jobs) ---
cleanup_orphans() {
    ps -eo pid,ppid,args 2>/dev/null | grep "multiprocessing.spawn" | grep -v grep | awk '$2 == 1 {print $1}' | xargs -r kill 2>/dev/null || true
    ps -eo pid,ppid,args 2>/dev/null | grep "multiprocessing.resource_tracker" | grep -v grep | awk '$2 == 1 {print $1}' | xargs -r kill 2>/dev/null || true
}

# --- Wait for a DDP port to be free (prevents port collision between pairs) ---
wait_for_port_free() {
    local port="$1"
    local timeout="${2:-600}"
    local waited=0
    while python3 -c "import socket,sys; s=socket.socket(); s.settimeout(1); sys.exit(0 if s.connect_ex(('localhost',$port))==0 else 1)" 2>/dev/null; do
        if [ "$waited" -ge "$timeout" ]; then
            log "  WARNING: Port ${port} still in use after ${timeout}s — force-killing holders"
            # Find and kill processes holding this port
            local holders
            holders=$(lsof -ti :${port} 2>/dev/null || true)
            if [ -n "$holders" ]; then
                echo "$holders" | xargs -r kill 2>/dev/null || true
                sleep 5
            fi
            return 0
        fi
        [ "$waited" -eq 0 ] && log "  Waiting for port ${port} to free..."
        sleep 10
        waited=$((waited + 10))
    done
}

# --- Train a single task×model on specified GPUs ---
train_one() {
    local task="$1" model="$2" devices="$3" port="$4"
    log "  TRAIN: ${task} / ${model} [${devices}]"
    DEVICES="$devices" MASTER_PORT="$port" \
        bash scripts/run_mimic_probe.sh --models "$model" "$task" 2>&1 | tee -a "$LOG" || {
        log "  TRAIN FAILED: ${task} / ${model}"
        return 1
    }
    log "  TRAIN DONE: ${task} / ${model}"
}

# --- Pred avg a single task×model on specified GPUs ---
pred_avg_one() {
    local task="$1" model="$2" devices="$3" port="$4"
    if ! has_checkpoint "$task" "$model"; then
        log "  SKIP pred avg: ${task} / ${model} (no best.pt)"
        return 1
    fi
    log "  PRED AVG: ${task} / ${model} [${devices}]"
    DEVICES="$devices" MASTER_PORT="$port" \
        bash scripts/run_mimic_pred_avg.sh --models "$model" "$task" 2>&1 | tee -a "$LOG" || {
        log "  PRED AVG FAILED: ${task} / ${model}"
        return 1
    }
    log "  PRED AVG DONE: ${task} / ${model}"
}

# --- Process a task×model: train if needed, then pred avg ---
process_one() {
    local task="$1" model="$2" devices="$3" port="$4"
    if ! has_checkpoint "$task" "$model"; then
        train_one "$task" "$model" "$devices" "$port" || return 1
    else
        log "  SKIP train: ${task} / ${model} (best.pt exists)"
    fi
    pred_avg_one "$task" "$model" "$devices" "$port"
}

# --- Run two tasks in parallel on split GPUs ---
run_pair() {
    local task_a="$1" task_b="$2" model="$3"
    local pid_a pid_b rc_a=0 rc_b=0

    if [ -n "$task_a" ]; then
        process_one "$task_a" "$model" "cuda:0 cuda:1 cuda:2 cuda:3" 29500 &
        pid_a=$!
    fi
    if [ -n "$task_b" ]; then
        process_one "$task_b" "$model" "cuda:4 cuda:5 cuda:6 cuda:7" 29501 &
        pid_b=$!
    fi

    [ -n "${pid_a:-}" ] && wait "$pid_a" || rc_a=$?
    [ -n "${pid_b:-}" ] && wait "$pid_b" || rc_b=$?

    [ "$rc_a" -ne 0 ] && log "WARNING: ${task_a} / ${model} had errors"
    [ "$rc_b" -ne 0 ] && [ -n "$task_b" ] && log "WARNING: ${task_b} / ${model} had errors"

    # Clean up between pairs: kill orphans, wait for ports to free
    cleanup_orphans
    sleep 5
    wait_for_port_free 29500 600
    wait_for_port_free 29501 600
    sleep 10
}

# =============================================================================
log "=========================================="
log "MIMIC Outcome Chain — START"
log "Tasks: ${ALL_TASKS[*]}"
log "Models: ${ALL_MODELS[*]}"
log "=========================================="

# --- Phase 0: Wait for any running jobs ---
wait_for_training

# --- Run all models, G first ---
for MODEL in "${ALL_MODELS[@]}"; do
    log ""
    log "===== MODEL: ${MODEL} ====="
    log ""

    # Process tasks in pairs
    NUM_TASKS=${#ALL_TASKS[@]}
    for (( i=0; i<NUM_TASKS; i+=2 )); do
        task_a="${ALL_TASKS[$i]}"
        task_b=""
        if (( i+1 < NUM_TASKS )); then
            task_b="${ALL_TASKS[$((i+1))]}"
        fi
        log "--- Pair: ${task_a} + ${task_b:-[none]} ---"
        run_pair "$task_a" "$task_b" "$MODEL"
    done

    # --- Retry any tasks that failed (missing checkpoint OR missing pred avg) ---
    RETRY_TRAIN=()
    RETRY_PREDAVG=()
    for task in "${ALL_TASKS[@]}"; do
        if ! has_checkpoint "$task" "$MODEL"; then
            RETRY_TRAIN+=("$task")
        elif ! has_pred_avg "$task" "$MODEL"; then
            RETRY_PREDAVG+=("$task")
        fi
    done

    # Retry failed training (full process_one: train + pred avg)
    if [ ${#RETRY_TRAIN[@]} -gt 0 ]; then
        log ""
        log "--- RETRY TRAIN: ${#RETRY_TRAIN[@]} tasks for ${MODEL}: ${RETRY_TRAIN[*]} ---"
        cleanup_orphans
        wait_for_port_free 29500 600
        wait_for_port_free 29501 600
        for (( r=0; r<${#RETRY_TRAIN[@]}; r+=2 )); do
            rt_a="${RETRY_TRAIN[$r]}"
            rt_b=""
            if (( r+1 < ${#RETRY_TRAIN[@]} )); then
                rt_b="${RETRY_TRAIN[$((r+1))]}"
            fi
            log "--- Retry train pair: ${rt_a} + ${rt_b:-[none]} ---"
            run_pair "$rt_a" "$rt_b" "$MODEL"
        done
    fi

    # Retry failed pred avg only (checkpoint exists but pred avg missing)
    if [ ${#RETRY_PREDAVG[@]} -gt 0 ]; then
        log ""
        log "--- RETRY PRED AVG: ${#RETRY_PREDAVG[@]} tasks for ${MODEL}: ${RETRY_PREDAVG[*]} ---"
        cleanup_orphans
        wait_for_port_free 29500 600
        wait_for_port_free 29501 600
        for (( r=0; r<${#RETRY_PREDAVG[@]}; r+=2 )); do
            rt_a="${RETRY_PREDAVG[$r]}"
            rt_b=""
            if (( r+1 < ${#RETRY_PREDAVG[@]} )); then
                rt_b="${RETRY_PREDAVG[$((r+1))]}"
            fi
            log "--- Retry pred avg pair: ${rt_a} + ${rt_b:-[none]} ---"
            pid_a="" pid_b=""
            if [ -n "$rt_a" ]; then
                pred_avg_one "$rt_a" "$MODEL" "cuda:0 cuda:1 cuda:2 cuda:3" 29500 &
                pid_a=$!
            fi
            if [ -n "$rt_b" ]; then
                pred_avg_one "$rt_b" "$MODEL" "cuda:4 cuda:5 cuda:6 cuda:7" 29501 &
                pid_b=$!
            fi
            [ -n "$pid_a" ] && wait "$pid_a" || log "WARNING: retry pred avg ${rt_a} / ${MODEL} failed"
            [ -n "$pid_b" ] && [ -n "$rt_b" ] && wait "$pid_b" || log "WARNING: retry pred avg ${rt_b:-} / ${MODEL} failed"
            cleanup_orphans
            sleep 5
            wait_for_port_free 29500 600
            wait_for_port_free 29501 600
            sleep 10
        done
    fi

    log "===== MODEL ${MODEL}: ALL TASKS DONE ====="
done

log ""
log "=========================================="
log "MIMIC Outcome Chain — ALL COMPLETE"
log "=========================================="
