#!/bin/bash
# Pred avg for MIMIC LVEF structured + MIMIC MR severity (echojepa-g only).
# Downloads MR checkpoint from S3 (job 415) once available, then runs pred avg
# on test sets for both tasks.
#
# Usage:
#   bash scripts/run_mimic_lvef_mr_pred_avg.sh
#   DEVICES="cuda:0 cuda:1 cuda:2 cuda:3" bash scripts/run_mimic_lvef_mr_pred_avg.sh

set -euo pipefail

REPO=/home/sagemaker-user/user-default-efs/vjepa2
cd "$REPO"
export LD_LIBRARY_PATH=/opt/conda/lib:${LD_LIBRARY_PATH:-}

EFS="/mnt/custom-file-systems/efs/fs-0049217cdf69186d7_fsap-0fa7145b64eaa046b/vjepa2"
DEVICES="${DEVICES:-cuda:0 cuda:1 cuda:2 cuda:3}"
export MASTER_PORT="${MASTER_PORT:-29500}"
PROBE_DIR="${EFS}/checkpoints/probes/mimic"
CSV_DIR="${EFS}/experiments/nature_medicine/mimic/probe_csvs"
OUT_DIR="${EFS}/evals/vitg-384/nature_medicine/mimic"

# S3 paths for HyperPod job outputs
S3_BASE="s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts/runs"
MR_S3="${S3_BASE}/echojepa_g_mr_415/training_folder/video_classification_frozen/echojepa-g-mitral-regurg/best.pt"

log() { echo "[$(date +%H:%M:%S)] $*"; }

# --- Shared encoder config for EchoJEPA-G ---
ENCODER_CKPT="${EFS}/checkpoints/anneal/keep/pt-280-an81.pt"
MODULE="evals.video_classification_frozen.modelcustom.vit_encoder_multiclip"
PRETRAIN_KWARGS="    encoder:
      checkpoint_key: target_encoder
      img_temporal_dim_size: null
      model_name: vit_giant_xformers
      patch_size: 16
      tubelet_size: 2
      uniform_power: true
      use_rope: true"
WRAPPER_KWARGS="    max_frames: 128
    use_pos_embed: false"

generate_config() {
    local TASK="$1"
    local TASK_TYPE="$2"
    local NUM_CLASSES="$3"
    local PROBE_CKPT="$4"
    local OUTFILE="/tmp/mimic_predavg_${TASK}_echojepa-g.yaml"

    local TEST_CSV="${CSV_DIR}/${TASK}/test.csv"
    local TRAIN_CSV="${CSV_DIR}/${TASK}/train.csv"

    # HP grid must match training grid (20 heads: 5 LR x 4 WD)
    cat > "$OUTFILE" <<YAML
app: vjepa
cpus_per_task: 32
folder: ${OUT_DIR}
mem_per_gpu: 80G
nodes: 1
tasks_per_node: 8
num_workers: 2

eval_name: video_classification_frozen
val_only: true
resume_checkpoint: true
probe_checkpoint: ${PROBE_CKPT}
tag: ${TASK}-predavg-echojepa-g

experiment:
  classifier:
    task_type: ${TASK_TYPE}
    num_heads: 16
    num_probe_blocks: 1
    num_targets: ${NUM_CLASSES}

  data:
    dataset_type: VideoDataset
    dataset_train: ${TRAIN_CSV}
    dataset_val: ${TEST_CSV}
    num_classes: ${NUM_CLASSES}
    resolution: 224
    frames_per_clip: 16
    frame_step: 2
    num_segments: 2
    num_views_per_segment: 1
    study_sampling: true

  optimization:
    batch_size: 2
    val_batch_size: 128
    num_epochs: 1
    use_bfloat16: true
    use_pos_embed: false
    multihead_kwargs:
    # 20 heads: 5 LR x 4 WD (matches HyperPod training grid)
    - {lr: 0.001,   start_lr: 0.0, warmup: 2.0, final_lr: 0.0, weight_decay: 0.001, final_weight_decay: 0.001}
    - {lr: 0.001,   start_lr: 0.0, warmup: 2.0, final_lr: 0.0, weight_decay: 0.01,  final_weight_decay: 0.01}
    - {lr: 0.001,   start_lr: 0.0, warmup: 2.0, final_lr: 0.0, weight_decay: 0.1,   final_weight_decay: 0.1}
    - {lr: 0.001,   start_lr: 0.0, warmup: 2.0, final_lr: 0.0, weight_decay: 0.4,   final_weight_decay: 0.4}
    - {lr: 0.0005,  start_lr: 0.0, warmup: 2.0, final_lr: 0.0, weight_decay: 0.001, final_weight_decay: 0.001}
    - {lr: 0.0005,  start_lr: 0.0, warmup: 2.0, final_lr: 0.0, weight_decay: 0.01,  final_weight_decay: 0.01}
    - {lr: 0.0005,  start_lr: 0.0, warmup: 2.0, final_lr: 0.0, weight_decay: 0.1,   final_weight_decay: 0.1}
    - {lr: 0.0005,  start_lr: 0.0, warmup: 2.0, final_lr: 0.0, weight_decay: 0.4,   final_weight_decay: 0.4}
    - {lr: 0.0001,  start_lr: 0.0, warmup: 2.0, final_lr: 0.0, weight_decay: 0.001, final_weight_decay: 0.001}
    - {lr: 0.0001,  start_lr: 0.0, warmup: 2.0, final_lr: 0.0, weight_decay: 0.01,  final_weight_decay: 0.01}
    - {lr: 0.0001,  start_lr: 0.0, warmup: 2.0, final_lr: 0.0, weight_decay: 0.1,   final_weight_decay: 0.1}
    - {lr: 0.0001,  start_lr: 0.0, warmup: 2.0, final_lr: 0.0, weight_decay: 0.4,   final_weight_decay: 0.4}
    - {lr: 0.00005, start_lr: 0.0, warmup: 2.0, final_lr: 0.0, weight_decay: 0.001, final_weight_decay: 0.001}
    - {lr: 0.00005, start_lr: 0.0, warmup: 2.0, final_lr: 0.0, weight_decay: 0.01,  final_weight_decay: 0.01}
    - {lr: 0.00005, start_lr: 0.0, warmup: 2.0, final_lr: 0.0, weight_decay: 0.1,   final_weight_decay: 0.1}
    - {lr: 0.00005, start_lr: 0.0, warmup: 2.0, final_lr: 0.0, weight_decay: 0.4,   final_weight_decay: 0.4}
    - {lr: 0.00001, start_lr: 0.0, warmup: 2.0, final_lr: 0.0, weight_decay: 0.001, final_weight_decay: 0.001}
    - {lr: 0.00001, start_lr: 0.0, warmup: 2.0, final_lr: 0.0, weight_decay: 0.01,  final_weight_decay: 0.01}
    - {lr: 0.00001, start_lr: 0.0, warmup: 2.0, final_lr: 0.0, weight_decay: 0.1,   final_weight_decay: 0.1}
    - {lr: 0.00001, start_lr: 0.0, warmup: 2.0, final_lr: 0.0, weight_decay: 0.4,   final_weight_decay: 0.4}

model_kwargs:
  checkpoint: ${ENCODER_CKPT}
  module_name: ${MODULE}
  pretrain_kwargs:
${PRETRAIN_KWARGS}
  wrapper_kwargs:
${WRAPPER_KWARGS}
YAML
    echo "$OUTFILE"
}

run_pred_avg() {
    local TASK="$1"
    local CFG="$2"
    local OUT_TAG_DIR="${OUT_DIR}/video_classification_frozen/${TASK}-predavg-echojepa-g"

    log ">>> ${TASK} — pred avg"
    rm -rf "$OUT_TAG_DIR"
    ps -eo pid,ppid,args | grep "multiprocessing.spawn" | grep -v grep | awk '$2 == 1 {print $1}' | xargs -r kill 2>/dev/null || true
    sleep 2

    rc=0
    PYTHONUNBUFFERED=1 python -m evals.main --fname "$CFG" --devices $DEVICES || rc=$?
    if [ "$rc" -ne 0 ]; then
        log ">>> FAILED: ${TASK} (exit code ${rc})"
    else
        log ">>> DONE: ${TASK}"
        PRED_CSV="${OUT_TAG_DIR}/study_predictions.csv"
        if [ -f "$PRED_CSV" ]; then
            log "  Studies: $(wc -l < "$PRED_CSV") (incl. header)"
        fi
    fi
    export MASTER_PORT=$((MASTER_PORT + 1))
}

# ============================================================
# 1. LVEF structured (checkpoint already on EFS)
# ============================================================
LVEF_CKPT="${PROBE_DIR}/lvef_structured/echojepa-g/best.pt"
if [ ! -f "$LVEF_CKPT" ]; then
    log "ERROR: LVEF checkpoint not found at ${LVEF_CKPT}"
    log "  Run: aws s3 cp s3://sagemaker-hyperpod.../best.pt ${LVEF_CKPT}"
    exit 1
fi

log "=== MIMIC LVEF structured pred avg ==="
CFG=$(generate_config "lvef_structured" "regression" "1" "$LVEF_CKPT")
run_pred_avg "lvef_structured" "$CFG"

# ============================================================
# 2. Mitral regurgitation (wait for job 415 checkpoint on S3)
# ============================================================
MR_CKPT="${PROBE_DIR}/mitral_regurg/echojepa-g/best.pt"
mkdir -p "$(dirname "$MR_CKPT")"

if [ ! -f "$MR_CKPT" ]; then
    log "=== Waiting for MIMIC MR checkpoint (job 415) on S3 ==="
    log "  Polling: ${MR_S3}"
    for i in $(seq 1 120); do
        if aws s3 ls "$MR_S3" &>/dev/null; then
            log "  Found checkpoint — downloading..."
            aws s3 cp "$MR_S3" "$MR_CKPT" --no-progress
            log "  Downloaded to ${MR_CKPT}"
            break
        fi
        log "  Not found yet (attempt ${i}/120) — sleeping 60s..."
        sleep 60
    done
fi

if [ ! -f "$MR_CKPT" ]; then
    log "ERROR: MR checkpoint never appeared on S3 after 2h. Exiting."
    exit 1
fi

log "=== MIMIC MR severity pred avg ==="
CFG=$(generate_config "mitral_regurg" "classification" "4" "$MR_CKPT")
run_pred_avg "mitral_regurg" "$CFG"

log "=== All done ==="
log "Results:"
for TASK in lvef_structured mitral_regurg; do
    PRED_CSV="${OUT_DIR}/video_classification_frozen/${TASK}-predavg-echojepa-g/study_predictions.csv"
    LOG_CSV="${OUT_DIR}/video_classification_frozen/${TASK}-predavg-echojepa-g/log_r0.csv"
    if [ -f "$PRED_CSV" ]; then
        log "  ${TASK}: $(wc -l < "$PRED_CSV") studies"
    fi
    if [ -f "$LOG_CSV" ]; then
        log "  ${TASK} metrics: $(tail -1 "$LOG_CSV")"
    fi
done
