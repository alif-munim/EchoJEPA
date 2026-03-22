#!/bin/bash
# Generic Nature Medicine MIMIC probe runner.
# Auto-detects task type (regression/classification).
# Runs 4 manuscript models sequentially with HP grid search (15 heads x 35 epochs, BS2).
# Automatically skips completed models and resumes interrupted ones.
# After each model, archives checkpoints locally and pushes to S3.
#
# Usage:
#   bash scripts/run_mimic_probe.sh mortality_1yr
#   bash scripts/run_mimic_probe.sh --models "echojepa-g echoprime" mortality_1yr
#   bash scripts/run_mimic_probe.sh --epochs 20 creatinine
#
# Concurrency (2 jobs on split GPUs):
#   DEVICES="cuda:0 cuda:1 cuda:2 cuda:3" MASTER_PORT=29500 bash scripts/run_mimic_probe.sh mortality_1yr
#   DEVICES="cuda:4 cuda:5 cuda:6 cuda:7" MASTER_PORT=29501 bash scripts/run_mimic_probe.sh mortality_90d
#
# Environment variables:
#   DEVICES       — GPU list (default: cuda:0 cuda:1 cuda:2 cuda:3)
#   MASTER_PORT   — DDP port (default: 37129). MUST differ between concurrent jobs.
#   RESUME        — resume from latest.pt checkpoint (default: true)
#   FRESH         — set to "true" to ignore existing checkpoints and start from scratch
#   NO_S3         — set to "true" to skip S3 upload
set -euo pipefail

# --- Parse args ---
MODELS="echojepa-g echojepa-l-k echoprime panecho"
EPOCHS=35
BALANCE=""
while [[ $# -gt 1 ]]; do
    case "$1" in
        --models) MODELS="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --balance) BALANCE="$2"; shift 2 ;;
        *) break ;;
    esac
done
TASK="${1:?Usage: run_mimic_probe.sh [--models '...'] [--epochs N] <task_name>}"

# --- Paths ---
REPO=/mnt/custom-file-systems/efs/fs-0049217cdf69186d7_fsap-0fa7145b64eaa046b/vjepa2
cd "$REPO"
export LD_LIBRARY_PATH=/opt/conda/lib:${LD_LIBRARY_PATH:-}

CSV_DIR=experiments/nature_medicine/mimic/probe_csvs/${TASK}
OUT_DIR=evals/vitg-384/nature_medicine/mimic
ARCHIVE_DIR=checkpoints/probes/mimic
S3_PREFIX=s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts/checkpoints/probes/mimic
DEVICES="${DEVICES:-cuda:0 cuda:1 cuda:2 cuda:3}"
SLEEP=30
PORT_TIMEOUT=300

# --- Export MASTER_PORT for DDP (if set) ---
if [ -n "${MASTER_PORT:-}" ]; then
    export MASTER_PORT
fi

# --- Resume / fresh support ---
RESUME_FLAG="${RESUME:-true}"
if [ "${FRESH:-false}" = "true" ]; then
    RESUME_FLAG=false
fi

# --- Validate CSV dir exists ---
if [ ! -d "${CSV_DIR}" ]; then
    echo "ERROR: CSV directory not found: ${CSV_DIR}"
    echo "Available tasks:"
    ls experiments/nature_medicine/mimic/probe_csvs/
    exit 1
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

# --- MIMIC: all tasks use study sampling, no view filtering ---
TRAIN_CSV="${REPO}/${CSV_DIR}/train.csv"
VAL_CSV="${REPO}/${CSV_DIR}/val.csv"
STUDY_SAMPLING=true

# --- Class balancing (off by default for MIMIC — imbalance is clinical) ---
# Override with --balance N if needed
BALANCE_LINE=""
if [ -n "$BALANCE" ] && [ "$BALANCE" != "0" ]; then
    BALANCE_LINE="
    class_balance_ratio: ${BALANCE}"
fi

# --- Logging ---
START=$(date +%s)
log() { echo "[$(date '+%H:%M:%S')] $1"; }

log "=== MIMIC ${TASK} ${TASK_TYPE}, study_sampling=${STUDY_SAMPLING} ==="
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
else
    log "  Class balance: off (default for MIMIC)"
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
    local OUTFILE="/tmp/nm_mimic_${TASK}_${MODEL_TAG}.yaml"

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
    study_sampling: ${STUDY_SAMPLING}${BALANCE_LINE}

  optimization:
    batch_size: 2
    val_batch_size: ${VAL_BS}
    num_epochs: ${EPOCHS}
    use_bfloat16: true
    use_pos_embed: false
    multihead_kwargs:
    # 15-head grid: 5 LRs x 3 WDs (wider LR range for smaller MIMIC datasets)
    # LR=1e-3
    - {lr: 0.001,   start_lr: 0.0, warmup: 3.0, final_lr: 0.0, weight_decay: 0.001, final_weight_decay: 0.001}
    - {lr: 0.001,   start_lr: 0.0, warmup: 3.0, final_lr: 0.0, weight_decay: 0.01,  final_weight_decay: 0.01}
    - {lr: 0.001,   start_lr: 0.0, warmup: 3.0, final_lr: 0.0, weight_decay: 0.1,   final_weight_decay: 0.1}
    # LR=5e-4
    - {lr: 0.0005,  start_lr: 0.0, warmup: 3.0, final_lr: 0.0, weight_decay: 0.001, final_weight_decay: 0.001}
    - {lr: 0.0005,  start_lr: 0.0, warmup: 3.0, final_lr: 0.0, weight_decay: 0.01,  final_weight_decay: 0.01}
    - {lr: 0.0005,  start_lr: 0.0, warmup: 3.0, final_lr: 0.0, weight_decay: 0.1,   final_weight_decay: 0.1}
    # LR=1e-4
    - {lr: 0.0001,  start_lr: 0.0, warmup: 3.0, final_lr: 0.0, weight_decay: 0.001, final_weight_decay: 0.001}
    - {lr: 0.0001,  start_lr: 0.0, warmup: 3.0, final_lr: 0.0, weight_decay: 0.01,  final_weight_decay: 0.01}
    - {lr: 0.0001,  start_lr: 0.0, warmup: 3.0, final_lr: 0.0, weight_decay: 0.1,   final_weight_decay: 0.1}
    # LR=5e-5
    - {lr: 0.00005, start_lr: 0.0, warmup: 3.0, final_lr: 0.0, weight_decay: 0.001, final_weight_decay: 0.001}
    - {lr: 0.00005, start_lr: 0.0, warmup: 3.0, final_lr: 0.0, weight_decay: 0.01,  final_weight_decay: 0.01}
    - {lr: 0.00005, start_lr: 0.0, warmup: 3.0, final_lr: 0.0, weight_decay: 0.1,   final_weight_decay: 0.1}
    # LR=1e-5
    - {lr: 0.00001, start_lr: 0.0, warmup: 3.0, final_lr: 0.0, weight_decay: 0.001, final_weight_decay: 0.001}
    - {lr: 0.00001, start_lr: 0.0, warmup: 3.0, final_lr: 0.0, weight_decay: 0.01,  final_weight_decay: 0.01}
    - {lr: 0.00001, start_lr: 0.0, warmup: 3.0, final_lr: 0.0, weight_decay: 0.1,   final_weight_decay: 0.1}

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

    for f in best.pt log_r0.csv latest.pt; do
        if [ -f "${src_dir}/${f}" ]; then
            cp "${src_dir}/${f}" "${local_archive}/${f}"
        fi
    done
    log "    Archived locally: ${local_archive}/"

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

    if is_complete "$model_tag"; then
        log ">>> SKIP ${tag} (already complete, ${EPOCHS} epochs)"
        archive_model "$model_tag"
        return 0
    fi

    wait_for_port

    local latest="${OUT_DIR}/video_classification_frozen/${TASK}-${model_tag}/latest.pt"
    if [ -f "$latest" ] && [ "$RESUME_FLAG" = "true" ]; then
        local ep
        ep=$(python3 -c "import torch; c=torch.load('$latest', map_location='cpu'); print(c.get('epoch',0))" 2>/dev/null || echo "?")
        log ">>> ${tag} (resuming from epoch ${ep})..."
    else
        log ">>> ${tag} (starting fresh)..."
    fi

    # Kill only ORPHANED multiprocessing workers (ppid=1)
    ps -eo pid,ppid,args | grep "multiprocessing.spawn" | grep -v grep | awk '$2 == 1 {print $1}' | xargs -r kill 2>/dev/null || true
    ps -eo pid,ppid,args | grep "multiprocessing.resource_tracker" | grep -v grep | awk '$2 == 1 {print $1}' | xargs -r kill 2>/dev/null || true
    sleep 2

    local rc=0
    export CHECKPOINT_ARCHIVE_PATH="${ARCHIVE_DIR}/${TASK}/${model_tag}"
    PYTHONUNBUFFERED=1 python -m evals.main --fname "$config" --devices $DEVICES || rc=$?
    unset CHECKPOINT_ARCHIVE_PATH
    if [ "$rc" -ne 0 ]; then
        log ">>> FAILED: ${tag} (exit code ${rc})"
        ps -eo pid,ppid,args | grep "multiprocessing.spawn" | grep -v grep | awk '$2 == 1 {print $1}' | xargs -r kill 2>/dev/null || true
        ps -eo pid,ppid,args | grep "multiprocessing.resource_tracker" | grep -v grep | awk '$2 == 1 {print $1}' | xargs -r kill 2>/dev/null || true
        return "$rc"
    fi

    local log_csv="${OUT_DIR}/video_classification_frozen/${TASK}-${model_tag}/log_r0.csv"
    local n_epochs
    n_epochs=$(grep -c "^[0-9]" "$log_csv" 2>/dev/null || echo "0")
    if [ "$n_epochs" -lt "$EPOCHS" ]; then
        log ">>> FAILED: ${tag} (only ${n_epochs}/${EPOCHS} epochs in log — likely crashed)"
        ps -eo pid,ppid,args | grep "multiprocessing.spawn" | grep -v grep | awk '$2 == 1 {print $1}' | xargs -r kill 2>/dev/null || true
        ps -eo pid,ppid,args | grep "multiprocessing.resource_tracker" | grep -v grep | awk '$2 == 1 {print $1}' | xargs -r kill 2>/dev/null || true
        return 1
    fi

    local best_pt="${OUT_DIR}/video_classification_frozen/${TASK}-${model_tag}/best.pt"
    if [ ! -f "$best_pt" ]; then
        log ">>> WARNING: ${tag} completed ${n_epochs} epochs but best.pt is missing!"
    fi

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
    use_pos_embed: false" \
                "256")
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
    use_pos_embed: false" \
                "256")
            ;;
        *)
            log "ERROR: Unknown model '$MODEL'"
            exit 1
            ;;
    esac

    run "[${MODEL_IDX}/${MODEL_COUNT}] ${MODEL}" "$CFG" "$MODEL"

    if [ "$MODEL_IDX" -lt "$MODEL_COUNT" ]; then
        sleep $SLEEP
    fi
done

END=$(date +%s)
log "=== MIMIC ${TASK}: ALL DONE in $(( (END - START) / 60 )) minutes ==="
