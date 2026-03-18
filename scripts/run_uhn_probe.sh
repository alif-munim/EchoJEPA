#!/bin/bash
# Generic Nature Medicine UHN probe runner.
# Auto-detects task type (regression/classification), view filtering, z-score params.
# Runs 5 models sequentially with HP grid search (12 heads × 15 epochs, BS2).
# Automatically skips completed models and resumes interrupted ones.
# After each model, archives checkpoints locally and pushes to S3.
#
# Usage:
#   bash scripts/run_uhn_probe.sh tapse
#   bash scripts/run_uhn_probe.sh mr_severity
#   bash scripts/run_uhn_probe.sh --models "echojepa-g echojepa-l" tapse   # subset of models
#   bash scripts/run_uhn_probe.sh --epochs 10 tapse                        # override epochs
#
# Concurrency (2 jobs on split GPUs):
#   DEVICES="cuda:0 cuda:1 cuda:2 cuda:3" MASTER_PORT=29500 bash scripts/run_uhn_probe.sh tapse
#   DEVICES="cuda:4 cuda:5 cuda:6 cuda:7" MASTER_PORT=29501 bash scripts/run_uhn_probe.sh lvef
#
# Environment variables:
#   DEVICES       — GPU list (default: all 8 GPUs)
#   MASTER_PORT   — DDP port (default: 37129). MUST differ between concurrent jobs.
#   RESUME        — resume from latest.pt checkpoint (default: true)
#   FRESH         — set to "true" to ignore existing checkpoints and start from scratch
#   NO_S3         — set to "true" to skip S3 upload (local archive still saved)
set -euo pipefail

# --- Parse args ---
MODELS="echojepa-g echojepa-l echojepa-l-k echoprime panecho"
EPOCHS=15
BALANCE=""
while [[ $# -gt 1 ]]; do
    case "$1" in
        --models) MODELS="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --balance) BALANCE="$2"; shift 2 ;;
        --no-balance) BALANCE="0"; shift ;;
        *) break ;;
    esac
done
TASK="${1:?Usage: run_uhn_probe.sh [--models '...'] [--epochs N] <task_name>}"

# --- Paths ---
REPO=/mnt/custom-file-systems/efs/fs-0049217cdf69186d7_fsap-0fa7145b64eaa046b/vjepa2
cd "$REPO"
export LD_LIBRARY_PATH=/opt/conda/lib:${LD_LIBRARY_PATH:-}

CSV_DIR=experiments/nature_medicine/uhn/probe_csvs/${TASK}
OUT_DIR=evals/vitg-384/nature_medicine/uhn
ARCHIVE_DIR=checkpoints/probes
S3_PREFIX=s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts/checkpoints/probes
DEVICES="${DEVICES:-cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7}"
SLEEP=30
PORT_TIMEOUT=300  # max seconds to wait for port to become free

# --- Export MASTER_PORT for DDP (if set) ---
if [ -n "${MASTER_PORT:-}" ]; then
    export MASTER_PORT
fi

# --- Resume / fresh support ---
RESUME_FLAG="${RESUME:-true}"
if [ "${FRESH:-false}" = "true" ]; then
    RESUME_FLAG=false
fi

# --- Auto-detect task type ---
if [ -f "${CSV_DIR}/zscore_params.json" ]; then
    TASK_TYPE=regression
    NUM_TARGETS=1
    NUM_CLASSES=1
else
    TASK_TYPE=classification
    NUM_CLASSES=$(awk '{print $NF}' "${CSV_DIR}/train.csv" | sort -u | wc -l)
    NUM_TARGETS=$NUM_CLASSES
fi

# --- Auto-detect view filtering ---
if [ -f "${CSV_DIR}/train_vf.csv" ]; then
    TRAIN_CSV="${REPO}/${CSV_DIR}/train_vf.csv"
    VAL_CSV="${REPO}/${CSV_DIR}/val_vf.csv"
    VF_TAG="(view-filtered)"
else
    TRAIN_CSV="${REPO}/${CSV_DIR}/train.csv"
    VAL_CSV="${REPO}/${CSV_DIR}/val.csv"
    VF_TAG="(all views)"
fi

# --- Auto-detect study sampling ---
# All tasks use study sampling (1 random clip per study per epoch)
STUDY_SAMPLING=true

# --- Default class balancing for classification + study_sampling ---
if [ -z "$BALANCE" ] && [ "$TASK_TYPE" = "classification" ] && [ "$STUDY_SAMPLING" = "true" ]; then
    BALANCE=3
fi

# --- Logging ---
START=$(date +%s)
log() { echo "[$(date '+%H:%M:%S')] $1"; }

log "=== ${TASK} ${TASK_TYPE} ${VF_TAG}, study_sampling=${STUDY_SAMPLING} ==="
log "  Train: ${TRAIN_CSV}"
log "  Val:   ${VAL_CSV}"
if [ "$TASK_TYPE" = "regression" ]; then
    log "  Z-score params: auto-loaded from zscore_params.json"
else
    log "  Classes: ${NUM_CLASSES}"
fi
log "  Models: ${MODELS}"
log "  Epochs: ${EPOCHS}"
log "  Devices: ${DEVICES}"
log "  MASTER_PORT: ${MASTER_PORT:-37129 (default)}"
log "  Resume: ${RESUME_FLAG}"
if [ -n "$BALANCE" ] && [ "$BALANCE" != "0" ]; then
    log "  Class balance ratio: ${BALANCE}x minority"
elif [ "$BALANCE" = "0" ]; then
    log "  Class balance: disabled"
fi
echo ""

# --- Generate config ---
generate_config() {
    local MODEL_TAG="$1"
    local MODULE_NAME="$2"
    local CHECKPOINT="$3"
    local PRETRAIN_KWARGS="$4"
    local WRAPPER_KWARGS="$5"
    local VAL_BS="${6:-64}"
    local OUTFILE="/tmp/nm_${TASK}_${MODEL_TAG}.yaml"

    cat > "$OUTFILE" <<YAML
app: vjepa
cpus_per_task: 32
folder: ${OUT_DIR}
mem_per_gpu: 80G
nodes: 1
tasks_per_node: 8
num_workers: 4

eval_name: video_classification_frozen
resume_checkpoint: ${RESUME_FLAG}
tag: ${TASK}-${MODEL_TAG}

experiment:
  classifier:
    task_type: ${TASK_TYPE}
    num_heads: 16
    num_probe_blocks: 1
    num_targets: ${NUM_TARGETS}

  data:
    dataset_type: VideoDataset
    dataset_train: ${TRAIN_CSV}
    dataset_val: ${VAL_CSV}
    num_classes: ${NUM_CLASSES}
    resolution: 224
    frames_per_clip: 16
    frame_step: 2
    num_segments: 2
    num_views_per_segment: 1
    study_sampling: ${STUDY_SAMPLING}$(if [ -n "$BALANCE" ] && [ "$BALANCE" != "0" ]; then echo "
    class_balance_ratio: ${BALANCE}"; fi)

  optimization:
    batch_size: 2
    val_batch_size: ${VAL_BS}
    num_epochs: ${EPOCHS}
    use_bfloat16: true
    use_pos_embed: false
    multihead_kwargs:
    # 12-head grid: 4 LRs x 3 WDs (dropped LR=1e-3, WD=0.4 — always worst)
    # LR=5e-4
    - {lr: 0.0005, start_lr: 0.0, warmup: 2.0, final_lr: 0.0, weight_decay: 0.001, final_weight_decay: 0.001}
    - {lr: 0.0005, start_lr: 0.0, warmup: 2.0, final_lr: 0.0, weight_decay: 0.01,  final_weight_decay: 0.01}
    - {lr: 0.0005, start_lr: 0.0, warmup: 2.0, final_lr: 0.0, weight_decay: 0.1,   final_weight_decay: 0.1}
    # LR=1e-4
    - {lr: 0.0001, start_lr: 0.0, warmup: 2.0, final_lr: 0.0, weight_decay: 0.001, final_weight_decay: 0.001}
    - {lr: 0.0001, start_lr: 0.0, warmup: 2.0, final_lr: 0.0, weight_decay: 0.01,  final_weight_decay: 0.01}
    - {lr: 0.0001, start_lr: 0.0, warmup: 2.0, final_lr: 0.0, weight_decay: 0.1,   final_weight_decay: 0.1}
    # LR=5e-5
    - {lr: 0.00005, start_lr: 0.0, warmup: 2.0, final_lr: 0.0, weight_decay: 0.001, final_weight_decay: 0.001}
    - {lr: 0.00005, start_lr: 0.0, warmup: 2.0, final_lr: 0.0, weight_decay: 0.01,  final_weight_decay: 0.01}
    - {lr: 0.00005, start_lr: 0.0, warmup: 2.0, final_lr: 0.0, weight_decay: 0.1,   final_weight_decay: 0.1}
    # LR=1e-5
    - {lr: 0.00001, start_lr: 0.0, warmup: 2.0, final_lr: 0.0, weight_decay: 0.001, final_weight_decay: 0.001}
    - {lr: 0.00001, start_lr: 0.0, warmup: 2.0, final_lr: 0.0, weight_decay: 0.01,  final_weight_decay: 0.01}
    - {lr: 0.00001, start_lr: 0.0, warmup: 2.0, final_lr: 0.0, weight_decay: 0.1,   final_weight_decay: 0.1}

model_kwargs:
  checkpoint: ${CHECKPOINT}
  module_name: ${MODULE_NAME}
  pretrain_kwargs:
${PRETRAIN_KWARGS}
  wrapper_kwargs:
${WRAPPER_KWARGS}
YAML
    echo "$OUTFILE"
}

# --- Check if model already complete ---
is_complete() {
    local model_tag="$1"
    # FRESH mode always re-runs
    if [ "${FRESH:-false}" = "true" ]; then
        return 1
    fi
    local latest="${OUT_DIR}/video_classification_frozen/${TASK}-${model_tag}/latest.pt"
    if [ ! -f "$latest" ]; then
        return 1
    fi
    local epoch
    epoch=$(python3 -c "import torch; c=torch.load('$latest', map_location='cpu'); print(c.get('epoch',0))" 2>/dev/null)
    [ "$epoch" -ge "$EPOCHS" ] 2>/dev/null
}

# --- Wait for DDP port to be free ---
port_in_use() {
    python3 -c "
import socket, sys
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(1)
r = s.connect_ex(('localhost', int(sys.argv[1])))
s.close()
sys.exit(0 if r == 0 else 1)
" "$1" 2>/dev/null
}

wait_for_port() {
    local port="${MASTER_PORT:-37129}"
    local waited=0
    while port_in_use "$port"; do
        if [ "$waited" -ge "$PORT_TIMEOUT" ]; then
            log "ERROR: Port ${port} still in use after ${PORT_TIMEOUT}s"
            return 1
        fi
        if [ "$waited" -eq 0 ]; then
            log "    Waiting for port ${port} to become free..."
        fi
        sleep 5
        waited=$((waited + 5))
    done
    if [ "$waited" -gt 0 ]; then
        log "    Port ${port} free after ${waited}s"
    fi
}

# --- Archive checkpoints locally and push to S3 ---
archive_model() {
    local model_tag="$1"
    local src_dir="${OUT_DIR}/video_classification_frozen/${TASK}-${model_tag}"
    local local_archive="${ARCHIVE_DIR}/${TASK}/${model_tag}"
    local s3_dest="${S3_PREFIX}/${TASK}/${model_tag}"

    mkdir -p "$local_archive"

    # Copy best.pt, log_r0.csv, latest.pt to local archive
    for f in best.pt log_r0.csv latest.pt; do
        if [ -f "${src_dir}/${f}" ]; then
            cp "${src_dir}/${f}" "${local_archive}/${f}"
        fi
    done
    log "    Archived locally: ${local_archive}/"

    # Push to S3 (skip if NO_S3=true)
    if [ "${NO_S3:-false}" != "true" ]; then
        for f in best.pt log_r0.csv; do
            if [ -f "${local_archive}/${f}" ]; then
                aws s3 cp "${local_archive}/${f}" "${s3_dest}/${f}" --quiet 2>/dev/null && \
                    log "    Uploaded: ${s3_dest}/${f}" || \
                    log "    WARNING: S3 upload failed for ${f} (continuing)"
            fi
        done
    fi
}

# --- Run function ---
run() {
    local tag="$1"
    local config="$2"
    local model_tag="$3"

    # Skip if already complete
    if is_complete "$model_tag"; then
        log ">>> SKIP ${tag} (already complete, ${EPOCHS} epochs)"
        # Still archive in case previous run didn't (idempotent)
        archive_model "$model_tag"
        return 0
    fi

    # Wait for DDP port to be free (prevents TCPStore bind failures)
    wait_for_port

    # Log resume status
    local latest="${OUT_DIR}/video_classification_frozen/${TASK}-${model_tag}/latest.pt"
    if [ -f "$latest" ] && [ "$RESUME_FLAG" = "true" ]; then
        local ep
        ep=$(python3 -c "import torch; c=torch.load('$latest', map_location='cpu'); print(c.get('epoch',0))" 2>/dev/null || echo "?")
        log ">>> ${tag} (resuming from epoch ${ep})..."
    else
        log ">>> ${tag} (starting fresh)..."
    fi

    # Kill only ORPHANED multiprocessing workers (ppid=1) — safe for concurrent jobs
    # NOTE: Do NOT delete /dev/shm/torch_* files here — kills concurrent jobs (Bug 011)
    ps -eo pid,ppid,args | grep "multiprocessing.spawn" | grep -v grep | awk '$2 == 1 {print $1}' | xargs -r kill 2>/dev/null || true
    ps -eo pid,ppid,args | grep "multiprocessing.resource_tracker" | grep -v grep | awk '$2 == 1 {print $1}' | xargs -r kill 2>/dev/null || true
    sleep 2

    local rc=0
    export CHECKPOINT_ARCHIVE_PATH="${ARCHIVE_DIR}/${TASK}/${model_tag}"
    PYTHONUNBUFFERED=1 python -m evals.main --fname "$config" --devices $DEVICES || rc=$?
    unset CHECKPOINT_ARCHIVE_PATH
    if [ "$rc" -ne 0 ]; then
        log ">>> FAILED: ${tag} (exit code ${rc})"
        # Kill only ORPHANED workers (ppid=1) — do NOT use unfiltered pkill (Bug 010) or rm shm files (Bug 011)
        ps -eo pid,ppid,args | grep "multiprocessing.spawn" | grep -v grep | awk '$2 == 1 {print $1}' | xargs -r kill 2>/dev/null || true
        ps -eo pid,ppid,args | grep "multiprocessing.resource_tracker" | grep -v grep | awk '$2 == 1 {print $1}' | xargs -r kill 2>/dev/null || true
        return "$rc"
    fi

    # Verify training actually completed (catch silent crashes)
    local log_csv="${OUT_DIR}/video_classification_frozen/${TASK}-${model_tag}/log_r0.csv"
    local n_epochs
    n_epochs=$(grep -c "^[0-9]" "$log_csv" 2>/dev/null || echo "0")
    if [ "$n_epochs" -lt "$EPOCHS" ]; then
        log ">>> FAILED: ${tag} (only ${n_epochs}/${EPOCHS} epochs in log — likely crashed)"
        # Kill only ORPHANED workers (ppid=1) — do NOT use unfiltered pkill (Bug 010) or rm shm files (Bug 011)
        ps -eo pid,ppid,args | grep "multiprocessing.spawn" | grep -v grep | awk '$2 == 1 {print $1}' | xargs -r kill 2>/dev/null || true
        ps -eo pid,ppid,args | grep "multiprocessing.resource_tracker" | grep -v grep | awk '$2 == 1 {print $1}' | xargs -r kill 2>/dev/null || true
        return 1
    fi

    # Verify best.pt was saved
    local best_pt="${OUT_DIR}/video_classification_frozen/${TASK}-${model_tag}/best.pt"
    if [ ! -f "$best_pt" ]; then
        log ">>> WARNING: ${tag} completed ${n_epochs} epochs but best.pt is missing!"
    fi

    # Archive checkpoints and push to S3
    archive_model "$model_tag"

    log ">>> DONE: ${tag}"
    echo ""
}

# --- Model definitions ---
MODEL_IDX=0
MODEL_COUNT=$(echo $MODELS | wc -w)

for MODEL in $MODELS; do
    MODEL_IDX=$((MODEL_IDX + 1))

    case "$MODEL" in
        echojepa-g)
            CFG=$(generate_config "echojepa-g" \
                "evals.video_classification_frozen.modelcustom.vit_encoder_multiclip" \
                "/mnt/custom-file-systems/efs/fs-0049217cdf69186d7_fsap-0fa7145b64eaa046b/vjepa2/checkpoints/anneal/keep/pt-280-an81.pt" \
                "    encoder:
      checkpoint_key: target_encoder
      img_temporal_dim_size: null
      model_name: vit_giant_xformers
      patch_size: 16
      tubelet_size: 2
      uniform_power: true
      use_rope: true" \
                "    max_frames: 128
    use_pos_embed: false")
            ;;
        echojepa-l)
            CFG=$(generate_config "echojepa-l" \
                "evals.video_classification_frozen.modelcustom.vit_encoder_multiclip" \
                "/home/sagemaker-user/user-default-efs/vjepa2/checkpoints/anneal/keep/vitl-pt-210-an25.pt" \
                "    encoder:
      checkpoint_key: target_encoder
      img_temporal_dim_size: null
      model_name: vit_large
      patch_size: 16
      tubelet_size: 2
      uniform_power: true
      use_rope: true" \
                "    max_frames: 128
    use_pos_embed: false")
            ;;
        echojepa-l-k)
            CFG=$(generate_config "echojepa-l-k" \
                "evals.video_classification_frozen.modelcustom.vit_encoder_multiclip" \
                "/home/sagemaker-user/user-default-efs/vjepa2/checkpoints/anneal/keep/vitl-kinetics-pt220-an55.pt" \
                "    encoder:
      checkpoint_key: target_encoder
      img_temporal_dim_size: null
      model_name: vit_large
      patch_size: 16
      tubelet_size: 2
      uniform_power: true
      use_rope: true" \
                "    max_frames: 128
    use_pos_embed: false")
            ;;
        echoprime)
            CFG=$(generate_config "echoprime" \
                "evals.video_classification_frozen.modelcustom.echo_prime_encoder" \
                "null" \
                "    {}" \
                "    echo_prime_root: /home/sagemaker-user/user-default-efs/vjepa2/evals/video_classification_frozen/modelcustom/EchoPrime
    force_fp32: true
    bin_size: 50")
            ;;
        panecho)
            CFG=$(generate_config "panecho" \
                "evals.video_classification_frozen.modelcustom.panecho_encoder" \
                "null" \
                "    {}" \
                "    {}")
            ;;
        *)
            log "ERROR: Unknown model '$MODEL'"
            exit 1
            ;;
    esac

    run "[${MODEL_IDX}/${MODEL_COUNT}] ${MODEL}" "$CFG" "$MODEL"

    # Brief cooldown between models (port availability checked in run())
    if [ "$MODEL_IDX" -lt "$MODEL_COUNT" ]; then
        sleep $SLEEP
    fi
done

END=$(date +%s)
log "=== ${TASK}: ALL DONE in $(( (END - START) / 60 )) minutes ==="
