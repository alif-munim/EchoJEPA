#!/bin/bash
# Single-clip test inference for RVSP — all 5 manuscript models.
# Scores each clip in the test set independently (no study-level averaging).
# Produces clip-level R²/Pearson to compare against pred avg study-level metrics.
#
# Usage:
#   bash scripts/run_rvsp_single_clip.sh
#   DEVICES="cuda:4 cuda:5 cuda:6 cuda:7" bash scripts/run_rvsp_single_clip.sh

set -euo pipefail

REPO=/home/sagemaker-user/user-default-efs/vjepa2
cd "$REPO"
export LD_LIBRARY_PATH=/opt/conda/lib:${LD_LIBRARY_PATH:-}

DEVICES="${DEVICES:-cuda:0 cuda:1 cuda:2 cuda:3}"
export MASTER_PORT="${MASTER_PORT:-29501}"
EFS="/mnt/custom-file-systems/efs/fs-0049217cdf69186d7_fsap-0fa7145b64eaa046b/vjepa2"
PROBE_DIR="${EFS}/checkpoints/probes"
OUT_DIR="${EFS}/evals/vitg-384/nature_medicine/uhn"

TASK="rvsp"
TEST_CSV="${EFS}/data/csv/rvsp_test.csv"
TRAIN_CSV="${EFS}/data/csv/rvsp_train.csv"
MODELS="echojepa-g echojepa-l-k echojepa-l echoprime panecho"

log() { echo "[$(date +%H:%M:%S)] $*"; }

log "=== RVSP single-clip test inference ==="
log "  Test CSV: ${TEST_CSV} ($(wc -l < "$TEST_CSV") clips)"
log "  Devices: ${DEVICES}"
log "  Models: ${MODELS}"

generate_inference_config() {
    local MODEL_TAG="$1"
    local MODULE_NAME="$2"
    local ENCODER_CHECKPOINT="$3"
    local PRETRAIN_KWARGS="$4"
    local WRAPPER_KWARGS="$5"
    local VAL_BS="${6:-64}"
    local PROBE_CKPT="${PROBE_DIR}/${TASK}/${MODEL_TAG}/best.pt"
    local OUTFILE="/tmp/rvsp_singleclip_${MODEL_TAG}.yaml"

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
num_workers: 4

eval_name: video_classification_frozen
val_only: true
resume_checkpoint: true
probe_checkpoint: ${PROBE_CKPT}
tag: ${TASK}-singleclip-${MODEL_TAG}

experiment:
  classifier:
    task_type: regression
    num_heads: 16
    num_probe_blocks: 1
    num_targets: 1

  data:
    dataset_type: VideoDataset
    dataset_train: ${TRAIN_CSV}
    dataset_val: ${TEST_CSV}
    num_classes: 1
    resolution: 224
    frames_per_clip: 16
    frame_step: 2
    num_segments: 2
    num_views_per_segment: 1
    study_sampling: false

  optimization:
    batch_size: 2
    val_batch_size: ${VAL_BS}
    num_epochs: 1
    use_bfloat16: true
    use_pos_embed: false
    multihead_kwargs:
    - {lr: 0.0005, start_lr: 0.0, warmup: 2.0, final_lr: 0.0, weight_decay: 0.001, final_weight_decay: 0.001}
    - {lr: 0.0005, start_lr: 0.0, warmup: 2.0, final_lr: 0.0, weight_decay: 0.01,  final_weight_decay: 0.01}
    - {lr: 0.0005, start_lr: 0.0, warmup: 2.0, final_lr: 0.0, weight_decay: 0.1,   final_weight_decay: 0.1}
    - {lr: 0.0001, start_lr: 0.0, warmup: 2.0, final_lr: 0.0, weight_decay: 0.001, final_weight_decay: 0.001}
    - {lr: 0.0001, start_lr: 0.0, warmup: 2.0, final_lr: 0.0, weight_decay: 0.01,  final_weight_decay: 0.01}
    - {lr: 0.0001, start_lr: 0.0, warmup: 2.0, final_lr: 0.0, weight_decay: 0.1,   final_weight_decay: 0.1}
    - {lr: 0.00005, start_lr: 0.0, warmup: 2.0, final_lr: 0.0, weight_decay: 0.001, final_weight_decay: 0.001}
    - {lr: 0.00005, start_lr: 0.0, warmup: 2.0, final_lr: 0.0, weight_decay: 0.01,  final_weight_decay: 0.01}
    - {lr: 0.00005, start_lr: 0.0, warmup: 2.0, final_lr: 0.0, weight_decay: 0.1,   final_weight_decay: 0.1}
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

MODELS_LIST=($MODELS)
TOTAL=${#MODELS_LIST[@]}
IDX=0

for MODEL in "${MODELS_LIST[@]}"; do
    IDX=$((IDX + 1))

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
        echojepa-l)
            CFG=$(generate_inference_config "echojepa-l" \
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
        echoprime)
            CFG=$(generate_inference_config "echoprime" \
                "evals.video_classification_frozen.modelcustom.echo_prime_encoder" \
                "null" \
                "    {}" \
                "    echo_prime_root: /home/sagemaker-user/user-default-efs/vjepa2/evals/video_classification_frozen/modelcustom/EchoPrime
    force_fp32: true
    bin_size: 50" \
                "64")
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

    log ">>> [${IDX}/${TOTAL}] ${MODEL} — single-clip inference"

    OUT_TAG_DIR="${OUT_DIR}/video_classification_frozen/${TASK}-singleclip-${MODEL}"
    rm -rf "$OUT_TAG_DIR"

    ps -eo pid,ppid,args | grep "multiprocessing.spawn" | grep -v grep | awk '$2 == 1 {print $1}' | xargs -r kill 2>/dev/null || true
    sleep 2

    rc=0
    PYTHONUNBUFFERED=1 python -m evals.main --fname "$CFG" --devices $DEVICES || rc=$?
    if [ "$rc" -ne 0 ]; then
        log ">>> FAILED: [${IDX}/${TOTAL}] ${MODEL} (exit code ${rc})"
    else
        log ">>> DONE: [${IDX}/${TOTAL}] ${MODEL}"
        # Print clip-level R² from log
        LOGFILE="${OUT_TAG_DIR}/log_r0.csv"
        if [ -f "$LOGFILE" ]; then
            log "  Results: $(tail -1 "$LOGFILE")"
        fi
    fi

    export MASTER_PORT=$((MASTER_PORT + 1))
done

log "=== All done. Results in ${OUT_DIR}/video_classification_frozen/rvsp-singleclip-* ==="
log ""
log "Clip-level R² per model:"
for MODEL in "${MODELS_LIST[@]}"; do
    LOGFILE="${OUT_DIR}/video_classification_frozen/${TASK}-singleclip-${MODEL}/log_r0.csv"
    if [ -f "$LOGFILE" ]; then
        echo "  ${MODEL}: $(tail -1 "$LOGFILE")"
    fi
done
