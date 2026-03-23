#!/bin/bash
# Cross-institution transfer: UHN-trained probes → MIMIC structured measurements.
# Loads UHN best.pt, scores ALL MIMIC clips per study, averages predictions per study.
#
# Supported tasks:
#   ef_note_extracted  — UHN LVEF probe → MIMIC note-extracted EF (regression)
#   mitral_regurg      — UHN MR severity probe → MIMIC MR (5→4 class, post-hoc merge)
#   tricuspid_regurg   — UHN TR severity probe → MIMIC TR (5→4 class, post-hoc merge)
#
# Usage:
#   bash scripts/run_mimic_xfer_structured.sh ef_note_extracted
#   bash scripts/run_mimic_xfer_structured.sh mitral_regurg
#   DEVICES="cuda:4 cuda:5 cuda:6 cuda:7" MASTER_PORT=29501 bash scripts/run_mimic_xfer_structured.sh tricuspid_regurg

set -euo pipefail

REPO=/home/sagemaker-user/user-default-efs/vjepa2
cd "$REPO"
export LD_LIBRARY_PATH=/opt/conda/lib:${LD_LIBRARY_PATH:-}

DEVICES="${DEVICES:-cuda:0 cuda:1 cuda:2 cuda:3}"
export MASTER_PORT="${MASTER_PORT:-29500}"
EFS="/mnt/custom-file-systems/efs/fs-0049217cdf69186d7_fsap-0fa7145b64eaa046b/vjepa2"
PROBE_DIR="${EFS}/checkpoints/probes"
CSV_DIR="${EFS}/experiments/nature_medicine/mimic/probe_csvs"
UHN_CSV_DIR="${EFS}/experiments/nature_medicine/uhn/probe_csvs"
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
    echo "Usage: bash scripts/run_mimic_xfer_structured.sh [--models 'model1 model2'] <task>"
    echo "Tasks: ef_note_extracted, mitral_regurg, tricuspid_regurg"
    exit 1
fi

# --- Task-specific config ---
case "$TASK" in
    ef_note_extracted)
        UHN_TASK="lvef"
        TASK_TYPE="regression"
        NUM_CLASSES=1
        NUM_TARGETS=1
        # Use UHN train CSV for zscore params (probe outputs in UHN z-score space)
        TRAIN_CSV="${UHN_CSV_DIR}/lvef/train.csv"
        TEST_CSV="${CSV_DIR}/ef_note_extracted/test.csv"
        ;;
    mitral_regurg)
        UHN_TASK="mr_severity"
        TASK_TYPE="classification"
        NUM_CLASSES=5  # UHN 5-class: none/trace/mild/moderate/severe
        NUM_TARGETS=5
        # Use remapped CSVs (MIMIC labels +1 → UHN 5-class space)
        TRAIN_CSV="${CSV_DIR}/mitral_regurg_xfer5/train.csv"
        TEST_CSV="${CSV_DIR}/mitral_regurg_xfer5/test.csv"
        ;;
    tricuspid_regurg)
        UHN_TASK="tr_severity"
        TASK_TYPE="classification"
        NUM_CLASSES=5  # UHN 5-class: none/trace/mild/moderate/severe
        NUM_TARGETS=5
        TRAIN_CSV="${CSV_DIR}/tricuspid_regurg_xfer5/train.csv"
        TEST_CSV="${CSV_DIR}/tricuspid_regurg_xfer5/test.csv"
        ;;
    *)
        echo "ERROR: Unknown task '$TASK'. Supported: ef_note_extracted, mitral_regurg, tricuspid_regurg"
        exit 1
        ;;
esac

if [ ! -f "$TEST_CSV" ]; then
    log "ERROR: Test CSV not found: ${TEST_CSV}"
    exit 1
fi

log "=== Cross-institution structured transfer: ${TASK} (UHN ${UHN_TASK} → MIMIC) ==="
log "  Task type: ${TASK_TYPE}, num_classes: ${NUM_CLASSES}"
log "  Train CSV (zscore source): ${TRAIN_CSV}"
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
    local PROBE_CKPT="${PROBE_DIR}/${UHN_TASK}/${MODEL_TAG}/best.pt"
    local OUTFILE="/tmp/nm_xfer_struct_${TASK}_${MODEL_TAG}.yaml"

    if [ ! -f "$PROBE_CKPT" ]; then
        log "ERROR: UHN probe checkpoint not found: ${PROBE_CKPT}"
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
tag: ${TASK}-xfer-${MODEL_TAG}

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
    # Must match UHN training grid (12 heads: 4 LR x 3 WD)
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
        *)
            log "ERROR: Unknown model '$MODEL'"
            continue
            ;;
    esac

    log ">>> [${MODEL_IDX}/${MODEL_COUNT}] ${MODEL} — ${TASK} cross-institution transfer"

    # Clear stale output dir
    OUT_TAG_DIR="${OUT_DIR}/video_classification_frozen/${TASK}-xfer-${MODEL}"
    if [ -d "$OUT_TAG_DIR" ]; then
        log ">>> Clearing stale output dir: ${OUT_TAG_DIR}"
        rm -rf "$OUT_TAG_DIR"
    fi

    # Kill only ORPHANED multiprocessing workers (ppid=1)
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
log "=== ${TASK} cross-institution transfer: ALL DONE in $(( (END - START) / 60 )) minutes ==="
