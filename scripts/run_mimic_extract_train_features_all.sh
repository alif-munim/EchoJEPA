#!/bin/bash
# Extract attentive pooler features from TRAINING set for all 10 MIMIC outcome tasks.
# Runs 2 tasks in parallel on split GPUs (0-3 and 4-7).
#
# Usage:
#   nohup bash scripts/run_mimic_extract_train_features_all.sh > logs/mimic_trainfeat_g.log 2>&1 &

set -euo pipefail

REPO=/mnt/custom-file-systems/efs/fs-0049217cdf69186d7_fsap-0fa7145b64eaa046b/vjepa2
cd "$REPO"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

TASKS=(
    mortality_1yr
    mortality_90d
    mortality_30d
    readmission_30d
    discharge_destination
    los_remaining
    troponin_t
    nt_probnp
    creatinine
    lactate
)

MODEL=echojepa-g

cleanup_orphans() {
    ps -eo pid,ppid,args 2>/dev/null | grep "multiprocessing.spawn" | grep -v grep | awk '$2 == 1 {print $1}' | xargs -r kill 2>/dev/null || true
    ps -eo pid,ppid,args 2>/dev/null | grep "multiprocessing.resource_tracker" | grep -v grep | awk '$2 == 1 {print $1}' | xargs -r kill 2>/dev/null || true
}

wait_for_port_free() {
    local port="$1"
    local waited=0
    while python3 -c "import socket,sys; s=socket.socket(); s.settimeout(1); sys.exit(0 if s.connect_ex(('localhost',$port))==0 else 1)" 2>/dev/null; do
        if [ "$waited" -ge 300 ]; then
            log "  WARNING: Port ${port} still in use after 300s — force-killing"
            lsof -ti :${port} 2>/dev/null | xargs -r kill 2>/dev/null || true
            sleep 5
            return 0
        fi
        [ "$waited" -eq 0 ] && log "  Waiting for port ${port} to free..."
        sleep 5
        waited=$((waited + 5))
    done
}

run_one() {
    local task="$1" devices="$2" port="$3"
    log "  EXTRACT: ${task} / ${MODEL} [${devices}]"
    DEVICES="$devices" MASTER_PORT="$port" \
        bash scripts/run_mimic_extract_train_features.sh --models "$MODEL" "$task" 2>&1 || {
        log "  EXTRACT FAILED: ${task} / ${MODEL}"
        return 1
    }
    log "  EXTRACT DONE: ${task} / ${MODEL}"
}

run_pair() {
    local task_a="$1" task_b="${2:-}"
    local pid_a pid_b rc_a=0 rc_b=0

    run_one "$task_a" "cuda:0 cuda:1 cuda:2 cuda:3" 29500 &
    pid_a=$!

    if [ -n "$task_b" ]; then
        run_one "$task_b" "cuda:4 cuda:5 cuda:6 cuda:7" 29501 &
        pid_b=$!
    fi

    wait "$pid_a" || rc_a=$?
    [ -n "${pid_b:-}" ] && { wait "$pid_b" || rc_b=$?; }

    [ "$rc_a" -ne 0 ] && log "WARNING: ${task_a} had errors"
    [ -n "$task_b" ] && [ "$rc_b" -ne 0 ] && log "WARNING: ${task_b} had errors"

    cleanup_orphans
    sleep 5
    wait_for_port_free 29500
    wait_for_port_free 29501
    sleep 5
}

log "=========================================="
log "MIMIC Train Feature Extraction (G) — START"
log "Tasks: ${TASKS[*]}"
log "=========================================="

NUM_TASKS=${#TASKS[@]}
for (( i=0; i<NUM_TASKS; i+=2 )); do
    task_a="${TASKS[$i]}"
    task_b=""
    if (( i+1 < NUM_TASKS )); then
        task_b="${TASKS[$((i+1))]}"
    fi
    log "--- Pair: ${task_a} + ${task_b:-[none]} ---"
    run_pair "$task_a" "$task_b"
done

log ""
log "=========================================="
log "MIMIC Train Feature Extraction (G) — ALL DONE"
log "=========================================="
