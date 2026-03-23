#!/bin/bash
# Extract attentive pooler features from TRAINING set clips.
# Same as run_mimic_pred_avg.sh but dataset_val points to train.csv
# to extract features for sklearn experiments.
#
# Usage:
#   bash scripts/run_mimic_extract_train_features.sh creatinine
#   DEVICES="cuda:4 cuda:5 cuda:6 cuda:7" MASTER_PORT=29501 bash scripts/run_mimic_extract_train_features.sh troponin_t

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

log() { echo "[$(date +%H:%M:%S)] $*"; }

# --- Parse args ---
MODELS="echojepa-g"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --models) MODELS="$2"; shift 2 ;;
        *) TASK="$1"; shift ;;
    esac
done

if [ -z "${TASK:-}" ]; then
    echo "Usage: bash scripts/run_mimic_extract_train_features.sh [--models 'model1 model2'] <task>"
    exit 1
fi

if [ ! -d "${CSV_DIR}/${TASK}" ]; then
    echo "ERROR: CSV directory not found: ${CSV_DIR}/${TASK}"
    exit 1
fi

# --- Detect task type ---
if [ -f "${CSV_DIR}/${TASK}/zscore_params.json" ]; then
    TASK_TYPE="regression"
    NUM_CLASSES=1
    NUM_TARGETS=1
else
    TASK_TYPE="classification"
    NUM_CLASSES=$(awk '{print $NF}' "${CSV_DIR}/${TASK}/train.csv" | sort -u | wc -l)
    NUM_TARGETS=$NUM_CLASSES
fi

# KEY DIFFERENCE: val set = train.csv (to extract train features)
TRAIN_CSV="${CSV_DIR}/${TASK}/train.csv"

log "=== MIMIC train feature extraction: ${TASK} ==="
log "  Task type: ${TASK_TYPE}"
log "  Train CSV (as val): ${TRAIN_CSV} ($(wc -l < "$TRAIN_CSV") clips)"

generate_inference_config() {
    local MODEL_TAG="$1"
    local MODULE_NAME="$2"
    local ENCODER_CHECKPOINT="$3"
    local PRETRAIN_KWARGS="$4"
    local WRAPPER_KWARGS="$5"
    local VAL_BS="${6:-64}"
    local PROBE_CKPT="${PROBE_DIR}/${TASK}/${MODEL_TAG}/latest.pt"
    local OUTFILE="/tmp/nm_mimic_trainfeat_${TASK}_${MODEL_TAG}.yaml"

    if [ ! -f "$PROBE_CKPT" ]; then
        log "ERROR: Probe checkpoint not found: ${PROBE_CKPT}"
        return 1
    fi

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
tag: ${TASK}-trainfeat-${MODEL_TAG}

experiment:
  classifier:
    task_type: ${TASK_TYPE}
    num_heads: 16
    num_probe_blocks: 1
    num_targets: ${NUM_TARGETS}

  data:
    dataset_type: VideoDataset
    dataset_train: ${TRAIN_CSV}
    dataset_val: ${TRAIN_CSV}
    num_classes: ${NUM_CLASSES}
    resolution: 224
    frames_per_clip: 16
    frame_step: 2
    num_segments: 2
    num_views_per_segment: 1
    study_sampling: true

  optimization:
    batch_size: 2
    val_batch_size: ${VAL_BS}
    num_epochs: 1
    use_bfloat16: true
    use_pos_embed: false
    multihead_kwargs:
    - {lr: 0.001,   start_lr: 0.0, warmup: 3.0, final_lr: 0.0, weight_decay: 0.001, final_weight_decay: 0.001}
    - {lr: 0.001,   start_lr: 0.0, warmup: 3.0, final_lr: 0.0, weight_decay: 0.01,  final_weight_decay: 0.01}
    - {lr: 0.001,   start_lr: 0.0, warmup: 3.0, final_lr: 0.0, weight_decay: 0.1,   final_weight_decay: 0.1}
    - {lr: 0.0005,  start_lr: 0.0, warmup: 3.0, final_lr: 0.0, weight_decay: 0.001, final_weight_decay: 0.001}
    - {lr: 0.0005,  start_lr: 0.0, warmup: 3.0, final_lr: 0.0, weight_decay: 0.01,  final_weight_decay: 0.01}
    - {lr: 0.0005,  start_lr: 0.0, warmup: 3.0, final_lr: 0.0, weight_decay: 0.1,   final_weight_decay: 0.1}
    - {lr: 0.0001,  start_lr: 0.0, warmup: 3.0, final_lr: 0.0, weight_decay: 0.001, final_weight_decay: 0.001}
    - {lr: 0.0001,  start_lr: 0.0, warmup: 3.0, final_lr: 0.0, weight_decay: 0.01,  final_weight_decay: 0.01}
    - {lr: 0.0001,  start_lr: 0.0, warmup: 3.0, final_lr: 0.0, weight_decay: 0.1,   final_weight_decay: 0.1}
    - {lr: 0.00005, start_lr: 0.0, warmup: 3.0, final_lr: 0.0, weight_decay: 0.001, final_weight_decay: 0.001}
    - {lr: 0.00005, start_lr: 0.0, warmup: 3.0, final_lr: 0.0, weight_decay: 0.01,  final_weight_decay: 0.01}
    - {lr: 0.00005, start_lr: 0.0, warmup: 3.0, final_lr: 0.0, weight_decay: 0.1,   final_weight_decay: 0.1}
    - {lr: 0.00001, start_lr: 0.0, warmup: 3.0, final_lr: 0.0, weight_decay: 0.001, final_weight_decay: 0.001}
    - {lr: 0.00001, start_lr: 0.0, warmup: 3.0, final_lr: 0.0, weight_decay: 0.01,  final_weight_decay: 0.01}
    - {lr: 0.00001, start_lr: 0.0, warmup: 3.0, final_lr: 0.0, weight_decay: 0.1,   final_weight_decay: 0.1}

model_kwargs:
  checkpoint: ${ENCODER_CHECKPOINT}
  module_name: ${MODULE_NAME}
  pretrain_kwargs:
${PRETRAIN_KWARGS}
  wrapper_kwargs:
${WRAPPER_KWARGS}
YAML
    echo "$OUTFILE"
}

for MODEL in $MODELS; do
    case "$MODEL" in
        echojepa-g)
            CFG=$(generate_inference_config "echojepa-g" \
                "evals.video_classification_frozen.modelcustom.vit_encoder_multiclip" \
                "${EFS}/checkpoints/anneal/keep/pt-280-an81.pt" \
                "    encoder:
      checkpoint_key: target_encoder
      img_temporal_dim_size: null
      model_name: vit_giant_xformers
      patch_size: 16
      tubelet_size: 2
      uniform_power: true
      use_rope: true" \
                "    max_frames: 128
    use_pos_embed: false" \
                "128")
            ;;
        *)
            log "ERROR: Unknown model '$MODEL'"
            continue
            ;;
    esac

    log ">>> ${MODEL} — train feature extraction for ${TASK}"

    OUT_TAG_DIR="${OUT_DIR}/video_classification_frozen/${TASK}-trainfeat-${MODEL}"
    if [ -d "$OUT_TAG_DIR" ]; then
        log ">>> Clearing stale output dir: ${OUT_TAG_DIR}"
        rm -rf "$OUT_TAG_DIR"
    fi

    ps -eo pid,ppid,args | grep "multiprocessing.spawn" | grep -v grep | awk '$2 == 1 {print $1}' | xargs -r kill 2>/dev/null || true
    sleep 2

    rc=0
    PYTHONUNBUFFERED=1 python -m evals.main --fname "$CFG" --devices $DEVICES || rc=$?
    if [ "$rc" -ne 0 ]; then
        log ">>> FAILED: ${MODEL} ${TASK} (exit code ${rc})"
    else
        log ">>> DONE: ${MODEL} ${TASK}"
    fi
done

log "=== Train feature extraction: ${TASK} DONE ==="
