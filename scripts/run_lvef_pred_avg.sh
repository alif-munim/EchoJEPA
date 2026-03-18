#!/bin/bash
# Run prediction averaging (inference) for LVEF.
# Loads trained probe checkpoints, scores ALL test clips, averages per study.
#
# Usage:
#   bash scripts/run_lvef_pred_avg.sh                                    # all 5 models
#   MODELS="echojepa-l echojepa-l-k echoprime panecho" bash scripts/run_lvef_pred_avg.sh  # subset
#   DEVICES="cuda:0 cuda:1 cuda:2 cuda:3" bash scripts/run_lvef_pred_avg.sh
set -euo pipefail

REPO=/home/sagemaker-user/user-default-efs/vjepa2
cd "$REPO"
export LD_LIBRARY_PATH=/opt/conda/lib:${LD_LIBRARY_PATH:-}

TASK=lvef
CSV_DIR=experiments/nature_medicine/uhn/probe_csvs/${TASK}
TRAIN_CSV="${REPO}/${CSV_DIR}/train_vf.csv"
TEST_CSV="${REPO}/${CSV_DIR}/test_vf.csv"
PROBE_DIR="${REPO}/checkpoints/probes/${TASK}"
DEVICES="${DEVICES:-cuda:0 cuda:1 cuda:2 cuda:3}"
MODELS="${MODELS:-echojepa-g echojepa-l echojepa-l-k echoprime panecho}"
export MASTER_PORT="${MASTER_PORT:-29500}"

log() { echo "[$(date '+%H:%M:%S')] $1"; }

log "=== LVEF prediction averaging (test set) ==="
log "  Test CSV: ${TEST_CSV} ($(wc -l < "$TEST_CSV") clips)"
log "  Models: ${MODELS}"
log "  Devices: ${DEVICES}"
echo ""

generate_inference_config() {
    local MODEL_TAG="$1"
    local MODULE_NAME="$2"
    local ENCODER_CHECKPOINT="$3"
    local PRETRAIN_KWARGS="$4"
    local WRAPPER_KWARGS="$5"
    local VAL_BS="${6:-192}"
    local OUTFILE="/tmp/nm_${TASK}_predavg_${MODEL_TAG}.yaml"

    cat > "$OUTFILE" <<YAML
app: vjepa
cpus_per_task: 32
folder: evals/vitg-384/nature_medicine/uhn
mem_per_gpu: 80G
nodes: 1
tasks_per_node: 8
num_workers: 4

eval_name: video_classification_frozen
val_only: true
resume_checkpoint: true
probe_checkpoint: ${PROBE_DIR}/${MODEL_TAG}/best.pt
tag: ${TASK}-predavg-${MODEL_TAG}

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
    study_sampling: true

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

run_inference() {
    local model_tag="$1"
    local config="$2"

    log ">>> [${model_tag}] Running prediction averaging..."

    if [ ! -f "${PROBE_DIR}/${model_tag}/best.pt" ]; then
        log ">>> ERROR: ${PROBE_DIR}/${model_tag}/best.pt not found!"
        return 1
    fi

    rm -f /dev/shm/torch_* /dev/shm/__KMP_REGISTERED_LIB_* /dev/shm/sem.loky-* /dev/shm/sem.mp-* 2>/dev/null
    # Kill only ORPHANED multiprocessing workers (ppid=1) — safe for concurrent jobs
    ps -eo pid,ppid,args | grep "multiprocessing.spawn" | grep -v grep | awk '$2 == 1 {print $1}' | xargs -r kill 2>/dev/null || true
    ps -eo pid,ppid,args | grep "multiprocessing.resource_tracker" | grep -v grep | awk '$2 == 1 {print $1}' | xargs -r kill 2>/dev/null || true
    sleep 2

    local rc=0
    PYTHONUNBUFFERED=1 python -m evals.main --fname "$config" --devices $DEVICES || rc=$?
    if [ "$rc" -ne 0 ]; then
        log ">>> FAILED: ${model_tag} (exit code ${rc})"
        return "$rc"
    fi
    log ">>> DONE: ${model_tag}"
    echo ""
}

for MODEL in $MODELS; do
    case "$MODEL" in
        echojepa-g)
            CFG=$(generate_inference_config "echojepa-g" \
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
    use_pos_embed: false" \
                128)
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
                256)
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
                256)
            ;;
        echoprime)
            CFG=$(generate_inference_config "echoprime" \
                "evals.video_classification_frozen.modelcustom.echo_prime_encoder" \
                "null" \
                "    {}" \
                "    echo_prime_root: /home/sagemaker-user/user-default-efs/vjepa2/evals/video_classification_frozen/modelcustom/EchoPrime
    force_fp32: true
    bin_size: 50" \
                256)
            ;;
        panecho)
            CFG=$(generate_inference_config "panecho" \
                "evals.video_classification_frozen.modelcustom.panecho_encoder" \
                "null" \
                "    {}" \
                "    {}" \
                256)
            ;;
        *)
            log "ERROR: Unknown model '$MODEL'"
            exit 1
            ;;
    esac

    run_inference "$MODEL" "$CFG"
    sleep 10
done

log "=== LVEF prediction averaging: ALL DONE ==="
