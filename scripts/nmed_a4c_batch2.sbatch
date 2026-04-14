#!/bin/bash
#SBATCH --job-name=nmed-a4c-b2
#SBATCH --partition=ml-p5-48xlarge
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --output=/tmp/nmed_a4c_b2-%j.out
#SBATCH --error=/tmp/nmed_a4c_b2-%j.err
#SBATCH --time=0-03:00:00
#SBATCH --nodelist=ip-10-0-50-7

# Nature Medicine: A4C-only single-view ablation — Batch 2
# Bicuspid AV (binary) + AoV Mean Gradient (regression)
# 3 models (EchoJEPA-G, EchoPrime, PanEcho) × 2 tasks = 6 runs

set -eo pipefail

export PATH="/opt/vjepa2-312/bin:$PATH"
source /opt/vjepa2-312/bin/activate
set -u
cd /opt/vjepa2

# Fix cuBLAS 12.9 bf16 bug on H100
export LD_LIBRARY_PATH="/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cublas/lib:/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cudnn/lib:/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/nccl/lib:${LD_LIBRARY_PATH:-}"
export TORCH_HOME="/opt/dlami/nvme/torch_cache"

rm -rf /dev/shm/* 2>/dev/null || true

S3_BASE="s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts"
LOCAL="/opt/dlami/nvme/nmed_a4c_b2"
OUT_DIR="/opt/dlami/nvme/nmed_a4c_b2/output"
S3_OUT="${S3_BASE}/nmed_a4c/results"

log() { echo "[$(date +%H:%M:%S)] $*"; }

# ============================================================
# Step 1: Download assets from S3
# ============================================================
log "=== Downloading assets from S3 ==="

# Create directories
mkdir -p ${LOCAL}/{encoder,csvs/disease_bicuspid_av_a4c,csvs/aov_mean_grad_a4c}
for TASK in disease_bicuspid_av aov_mean_grad; do
    for MODEL in echojepa-g echoprime panecho; do
        mkdir -p "${LOCAL}/probes/${TASK}/${MODEL}"
    done
done

# EchoJEPA-G encoder
if [ ! -f "${LOCAL}/encoder/pt-280-an81.pt" ]; then
    log "  Downloading G encoder..."
    aws s3 cp "${S3_BASE}/checkpoints/anneal/keep/pt-280-an81.pt" "${LOCAL}/encoder/pt-280-an81.pt"
fi

# Probe checkpoints (3 models × 2 tasks)
for TASK in disease_bicuspid_av aov_mean_grad; do
    for MODEL in echojepa-g echoprime panecho; do
        DEST="${LOCAL}/probes/${TASK}/${MODEL}"
        if [ ! -f "${DEST}/best.pt" ]; then
            log "  Downloading probe: ${TASK}/${MODEL}"
            aws s3 cp "${S3_BASE}/checkpoints/probes/${TASK}/${MODEL}/best.pt" "${DEST}/best.pt"
        fi
    done
done

# EchoPrime encoder weights
EP_WEIGHTS="/opt/vjepa2/evals/video_classification_frozen/modelcustom/EchoPrime/model_data/weights/echo_prime_encoder.pt"
if [ ! -f "$EP_WEIGHTS" ]; then
    log "  Downloading EchoPrime encoder weights..."
    sudo mkdir -p "$(dirname $EP_WEIGHTS)"
    sudo aws s3 cp "${S3_BASE}/setup/echo_prime_encoder.pt" "$EP_WEIGHTS"
fi

# PanEcho weights
PE_CACHE="${TORCH_HOME}/hub/checkpoints"
mkdir -p "$PE_CACHE"
if [ ! -f "${PE_CACHE}/panecho.pt" ]; then
    log "  Downloading PanEcho weights to torch hub cache..."
    aws s3 cp "${S3_BASE}/setup/panecho.pt" "${PE_CACHE}/panecho.pt"
fi

# A4C-only probe CSVs
for TASK in disease_bicuspid_av_a4c aov_mean_grad_a4c; do
    for f in train_vf.csv test_vf.csv; do
        if [ ! -f "${LOCAL}/csvs/${TASK}/${f}" ]; then
            aws s3 cp "${S3_BASE}/nmed_a4c/probe_csvs/${TASK}/${f}" "${LOCAL}/csvs/${TASK}/${f}"
        fi
    done
    aws s3 cp "${S3_BASE}/nmed_a4c/probe_csvs/${TASK}/zscore_params.json" "${LOCAL}/csvs/${TASK}/zscore_params.json" 2>/dev/null || true
done

log "=== All assets downloaded ==="

# ============================================================
# Step 2: Config generation
# ============================================================

# --- EchoJEPA-G configs ---
generate_config_g() {
    local TASK_A4C="$1"
    local TASK_BASE="$2"
    local TASK_TYPE="$3"
    local NUM_CLASSES="$4"
    local VAL_BS="$5"

    local PROBE_CKPT="${LOCAL}/probes/${TASK_BASE}/echojepa-g/best.pt"
    local TRAIN_CSV="${LOCAL}/csvs/${TASK_A4C}/train_vf.csv"
    local TEST_CSV="${LOCAL}/csvs/${TASK_A4C}/test_vf.csv"
    local OUTFILE="/opt/dlami/nvme/nmed_a4c_b2_cfg_${TASK_A4C}_g.yaml"

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
tag: ${TASK_A4C}-predavg-echojepa-g

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
  checkpoint: ${LOCAL}/encoder/pt-280-an81.pt
  module_name: evals.video_classification_frozen.modelcustom.vit_encoder_multiclip
  pretrain_kwargs:
    encoder:
      checkpoint_key: target_encoder
      img_temporal_dim_size: null
      model_name: vit_giant_xformers
      patch_size: 16
      tubelet_size: 2
      uniform_power: true
      use_rope: true
  wrapper_kwargs:
    max_frames: 128
    use_pos_embed: false
YAML
    echo "$OUTFILE"
}

# --- EchoPrime configs ---
generate_config_echoprime() {
    local TASK_A4C="$1"
    local TASK_BASE="$2"
    local TASK_TYPE="$3"
    local NUM_CLASSES="$4"
    local VAL_BS="$5"

    local PROBE_CKPT="${LOCAL}/probes/${TASK_BASE}/echoprime/best.pt"
    local TRAIN_CSV="${LOCAL}/csvs/${TASK_A4C}/train_vf.csv"
    local TEST_CSV="${LOCAL}/csvs/${TASK_A4C}/test_vf.csv"
    local OUTFILE="/opt/dlami/nvme/nmed_a4c_b2_cfg_${TASK_A4C}_echoprime.yaml"

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
tag: ${TASK_A4C}-predavg-echoprime

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
    normalization:
      - [0.0, 0.0, 0.0]
      - [1.0, 1.0, 1.0]

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
  checkpoint: null
  module_name: evals.video_classification_frozen.modelcustom.echo_prime_encoder
  pretrain_kwargs: {}
  wrapper_kwargs:
    echo_prime_root: /opt/vjepa2/evals/video_classification_frozen/modelcustom/EchoPrime
    force_fp32: true
    bin_size: 50
YAML
    echo "$OUTFILE"
}

# --- PanEcho configs ---
generate_config_panecho() {
    local TASK_A4C="$1"
    local TASK_BASE="$2"
    local TASK_TYPE="$3"
    local NUM_CLASSES="$4"
    local VAL_BS="$5"

    local PROBE_CKPT="${LOCAL}/probes/${TASK_BASE}/panecho/best.pt"
    local TRAIN_CSV="${LOCAL}/csvs/${TASK_A4C}/train_vf.csv"
    local TEST_CSV="${LOCAL}/csvs/${TASK_A4C}/test_vf.csv"
    local OUTFILE="/opt/dlami/nvme/nmed_a4c_b2_cfg_${TASK_A4C}_panecho.yaml"

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
tag: ${TASK_A4C}-predavg-panecho

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
    normalization:
      - [0.0, 0.0, 0.0]
      - [1.0, 1.0, 1.0]

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
  checkpoint: null
  module_name: evals.video_classification_frozen.modelcustom.panecho_encoder
  pretrain_kwargs: {}
  wrapper_kwargs: {}
YAML
    echo "$OUTFILE"
}

# ============================================================
# Step 3: Run function
# ============================================================
run_single() {
    local CFG="$1"
    local TAG="$2"
    local DEVICES="$3"
    local PORT="$4"

    export MASTER_PORT="$PORT"
    local START=$(date +%s)

    ps -eo pid,ppid,args | grep "multiprocessing.spawn" | grep -v grep | awk '$2 == 1 {print $1}' | xargs -r kill 2>/dev/null || true
    sleep 2

    log "[${TAG}] Starting on ${DEVICES} (port ${PORT})"

    local OUT_TAG_DIR="${OUT_DIR}/video_classification_frozen/${TAG##*:}"
    rm -rf "$OUT_TAG_DIR" 2>/dev/null || true

    local rc=0
    PYTHONUNBUFFERED=1 python -m evals.main --fname "$CFG" --devices $DEVICES || rc=$?

    local END=$(date +%s)
    if [ "$rc" -ne 0 ]; then
        log "[${TAG}] FAILED (exit ${rc}) after $(( (END - START) / 60 )) min"
    else
        log "[${TAG}] DONE in $(( (END - START) / 60 )) min"
    fi
    return $rc
}

# ============================================================
# Step 4: Run inference — 6 runs (3 models × 2 tasks)
# ============================================================
ALL_DEVICES="cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7"
FAILURES=0
TOTAL=6

log "=== A4C-only Batch 2: Bicuspid AV + AoV Mean Gradient (3 models each) ==="

# --- Bicuspid AV (binary classification) ---
BICUSPID_CLASSES=2
log "Bicuspid AV: ${BICUSPID_CLASSES}-class binary"

CFG=$(generate_config_g "disease_bicuspid_av_a4c" "disease_bicuspid_av" "classification" "${BICUSPID_CLASSES}" "128")
run_single "$CFG" "predavg:disease_bicuspid_av_a4c-predavg-echojepa-g" "$ALL_DEVICES" "29500" || FAILURES=$((FAILURES + 1))

CFG=$(generate_config_echoprime "disease_bicuspid_av_a4c" "disease_bicuspid_av" "classification" "${BICUSPID_CLASSES}" "128")
run_single "$CFG" "predavg:disease_bicuspid_av_a4c-predavg-echoprime" "$ALL_DEVICES" "29500" || FAILURES=$((FAILURES + 1))

CFG=$(generate_config_panecho "disease_bicuspid_av_a4c" "disease_bicuspid_av" "classification" "${BICUSPID_CLASSES}" "128")
run_single "$CFG" "predavg:disease_bicuspid_av_a4c-predavg-panecho" "$ALL_DEVICES" "29500" || FAILURES=$((FAILURES + 1))

# --- AoV Mean Gradient (regression) ---
CFG=$(generate_config_g "aov_mean_grad_a4c" "aov_mean_grad" "regression" "1" "128")
run_single "$CFG" "predavg:aov_mean_grad_a4c-predavg-echojepa-g" "$ALL_DEVICES" "29500" || FAILURES=$((FAILURES + 1))

CFG=$(generate_config_echoprime "aov_mean_grad_a4c" "aov_mean_grad" "regression" "1" "128")
run_single "$CFG" "predavg:aov_mean_grad_a4c-predavg-echoprime" "$ALL_DEVICES" "29500" || FAILURES=$((FAILURES + 1))

CFG=$(generate_config_panecho "aov_mean_grad_a4c" "aov_mean_grad" "regression" "1" "128")
run_single "$CFG" "predavg:aov_mean_grad_a4c-predavg-panecho" "$ALL_DEVICES" "29500" || FAILURES=$((FAILURES + 1))

log "=== Results: $((TOTAL - FAILURES))/${TOTAL} succeeded ==="

# ============================================================
# Step 5: Upload results to S3
# ============================================================
log "=== Uploading results to S3 ==="
for TASK in disease_bicuspid_av_a4c aov_mean_grad_a4c; do
    for MODEL in echojepa-g echoprime panecho; do
        DIR="${OUT_DIR}/video_classification_frozen/${TASK}-predavg-${MODEL}"
        if [ -d "$DIR" ]; then
            aws s3 sync "$DIR" "${S3_OUT}/uhn/${TASK}-predavg-${MODEL}/"
            log "  Uploaded: ${TASK}-predavg-${MODEL}"
        fi
    done
done

# ============================================================
# Step 6: Cleanup
# ============================================================
log "=== Cleaning up ==="
rm -rf "${LOCAL}" "${TORCH_HOME}" 2>/dev/null || true
log "=== Job complete ==="
