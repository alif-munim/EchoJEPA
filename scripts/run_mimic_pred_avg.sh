#!/bin/bash
# Prediction averaging (inference) for MIMIC tasks.
# Loads latest.pt probe checkpoint from MIMIC training, scores ALL clips per study
# on the test set, and averages predictions per study for study-level metrics.
# Auto-detects task type (regression/classification).
#
# Usage:
#   bash scripts/run_mimic_pred_avg.sh creatinine
#   bash scripts/run_mimic_pred_avg.sh --models "echojepa-g echoprime" creatinine
#   DEVICES="cuda:4 cuda:5 cuda:6 cuda:7" MASTER_PORT=29501 bash scripts/run_mimic_pred_avg.sh troponin_t
#
# Concurrency (2 jobs on split GPUs):
#   DEVICES="cuda:0 cuda:1 cuda:2 cuda:3" MASTER_PORT=29500 bash scripts/run_mimic_pred_avg.sh creatinine
#   DEVICES="cuda:4 cuda:5 cuda:6 cuda:7" MASTER_PORT=29501 bash scripts/run_mimic_pred_avg.sh troponin_t

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
MODELS="echojepa-g echojepa-l-k echoprime panecho"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --models) MODELS="$2"; shift 2 ;;
        *) TASK="$1"; shift ;;
    esac
done

if [ -z "${TASK:-}" ]; then
    echo "Usage: bash scripts/run_mimic_pred_avg.sh [--models 'model1 model2'] <task>"
    exit 1
fi

# --- Validate CSV dir ---
if [ ! -d "${CSV_DIR}/${TASK}" ]; then
    echo "ERROR: CSV directory not found: ${CSV_DIR}/${TASK}"
    echo "Available tasks:"
    ls "${CSV_DIR}/"
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

TEST_CSV="${CSV_DIR}/${TASK}/test.csv"
TRAIN_CSV="${CSV_DIR}/${TASK}/train.csv"

log "=== MIMIC prediction averaging: ${TASK} ==="
log "  Task type: ${TASK_TYPE}"
log "  Test CSV: ${TEST_CSV} ($(wc -l < "$TEST_CSV") clips)"
log "  Devices: ${DEVICES}"
log "  Models: ${MODELS}"

# --- Generate inference config ---
generate_inference_config() {
    local MODEL_TAG="$1"
    local MODULE_NAME="$2"
    local ENCODER_CHECKPOINT="$3"
    local PRETRAIN_KWARGS="$4"
    local WRAPPER_KWARGS="$5"
    local VAL_BS="${6:-64}"
    local PROBE_CKPT="${PROBE_DIR}/${TASK}/${MODEL_TAG}/latest.pt"
    local OUTFILE="/tmp/nm_mimic_predavg_${TASK}_${MODEL_TAG}.yaml"

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
tag: ${TASK}-predavg-${MODEL_TAG}

experiment:
  classifier:
    task_type: ${TASK_TYPE}
    num_heads: 16
    num_probe_blocks: 1
    num_targets: ${NUM_TARGETS}

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
    val_batch_size: ${VAL_BS}
    num_epochs: 1
    use_bfloat16: true
    use_pos_embed: false
    multihead_kwargs:
    # Must match training grid (15 heads: 5 LR x 3 WD) so checkpoint classifier indices align
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
  checkpoint: ${ENCODER_CHECKPOINT}
  module_name: ${MODULE_NAME}
  pretrain_kwargs:
${PRETRAIN_KWARGS}
  wrapper_kwargs:
${WRAPPER_KWARGS}
YAML
    echo "$OUTFILE"
}

# --- Run inference for each model ---
MODEL_COUNT=$(echo $MODELS | wc -w)
MODEL_IDX=0
START=$(date +%s)

for MODEL in $MODELS; do
    MODEL_IDX=$((MODEL_IDX + 1))

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
        echojepa-l-k)
            CFG=$(generate_inference_config "echojepa-l-k" \
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
            CFG=$(generate_inference_config "echoprime" \
                "evals.video_classification_frozen.modelcustom.echo_prime_encoder" \
                "null" \
                "    {}" \
                "    echo_prime_root: /home/sagemaker-user/user-default-efs/vjepa2/evals/video_classification_frozen/modelcustom/EchoPrime
    force_fp32: true
    bin_size: 50" \
                "16")
            ;;
        panecho)
            CFG=$(generate_inference_config "panecho" \
                "evals.video_classification_frozen.modelcustom.panecho_encoder" \
                "null" \
                "    {}" \
                "    {}" \
                "64")
            ;;
        echojepa-b)
            CFG=$(generate_inference_config "echojepa-b" \
                "evals.video_classification_frozen.modelcustom.vit_encoder_multiclip_v21" \
                "/mnt/custom-file-systems/efs/fs-0049217cdf69186d7_fsap-0fa7145b64eaa046b/vjepa2/checkpoints/vjepa2_1_vitb_mimic_p169_c60.pt" \
                "    encoder:
      checkpoint_key: target_encoder
      model_name: vit_base
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
            continue
            ;;
    esac

    log ">>> [${MODEL_IDX}/${MODEL_COUNT}] ${MODEL} — prediction averaging"

    # Clear stale output dir to prevent resume logic from skipping inference (Bug 012)
    OUT_TAG_DIR="${OUT_DIR}/video_classification_frozen/${TASK}-predavg-${MODEL}"
    if [ -d "$OUT_TAG_DIR" ]; then
        log ">>> Clearing stale output dir: ${OUT_TAG_DIR}"
        rm -rf "$OUT_TAG_DIR"
    fi

    # Kill only ORPHANED multiprocessing workers (ppid=1) — safe for concurrent jobs
    ps -eo pid,ppid,args | grep "multiprocessing.spawn" | grep -v grep | awk '$2 == 1 {print $1}' | xargs -r kill 2>/dev/null || true
    ps -eo pid,ppid,args | grep "multiprocessing.resource_tracker" | grep -v grep | awk '$2 == 1 {print $1}' | xargs -r kill 2>/dev/null || true
    sleep 2

    rc=0
    PYTHONUNBUFFERED=1 python -m evals.main --fname "$CFG" --devices $DEVICES || rc=$?
    if [ "$rc" -ne 0 ]; then
        log ">>> FAILED: [${MODEL_IDX}/${MODEL_COUNT}] ${MODEL} (exit code ${rc})"
    else
        log ">>> DONE: [${MODEL_IDX}/${MODEL_COUNT}] ${MODEL}"
    fi

    sleep 10
done

END=$(date +%s)
log "=== MIMIC ${TASK} prediction averaging: ALL DONE in $(( (END - START) / 60 )) minutes ==="
