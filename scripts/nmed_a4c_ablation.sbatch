#!/bin/bash
#SBATCH --job-name=nmed-a4c
#SBATCH --partition=ml-p5-48xlarge
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --output=/tmp/nmed_a4c-%j.out
#SBATCH --error=/tmp/nmed_a4c-%j.err
#SBATCH --time=0-03:00:00
#SBATCH --nodelist=ip-10-0-50-241

# Nature Medicine: A4C-only single-view ablation (inference only)
# Reuses existing multi-view probes, runs inference on A4C-only B-mode test clips.
# Same probe, restricted input — isolates the view restriction effect.
#
# 3 tasks × 1 model (EchoJEPA-G) = 3 inference runs
#   AS severity (4-class): ~45 min (154K clips × 1B params)
#   MR severity (5-class): ~40 min (144K clips × 1B params)
#   AoV Vmax (regression): ~30 min (105K clips × 1B params)
#
# Estimated total: ~2h
#
# Prerequisites:
#   1. A4C-only view-filtered CSVs uploaded to S3
#   2. Multi-view probe checkpoints uploaded to S3
#   3. Deploy code: cd ~/EchoJEPA-repo && git pull && ~/deploy.sh
#   4. Submit: sbatch scripts/nmed_a4c_ablation.sbatch

set -eo pipefail

export PATH="/opt/vjepa2-312/bin:$PATH"
source /opt/vjepa2-312/bin/activate
set -u
cd /opt/vjepa2

# Fix cuBLAS 12.9 bf16 bug on H100
export LD_LIBRARY_PATH="/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cublas/lib:/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cudnn/lib:/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/nccl/lib:${LD_LIBRARY_PATH:-}"
export TORCH_HOME="/tmp/torch_cache"

rm -rf /dev/shm/* 2>/dev/null || true

S3_BASE="s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts"
LOCAL="/tmp/nmed_a4c"
OUT_DIR="/opt/vjepa2/evals/vitg-384/nature_medicine/uhn"
S3_OUT="${S3_BASE}/nmed_a4c/results"

log() { echo "[$(date +%H:%M:%S)] $*"; }

# ============================================================
# Step 1: Download assets from S3
# ============================================================
log "=== Downloading assets from S3 ==="
mkdir -p ${LOCAL}/{encoder,probes/as_severity,probes/mr_severity,probes/aov_vmax,csvs/as_severity_a4c,csvs/mr_severity_a4c,csvs/aov_vmax_a4c}

# EchoJEPA-G encoder (~16.5GB)
if [ ! -f "${LOCAL}/encoder/pt-280-an81.pt" ]; then
    log "  Downloading G encoder..."
    aws s3 cp "${S3_BASE}/checkpoints/anneal/keep/pt-280-an81.pt" "${LOCAL}/encoder/pt-280-an81.pt"
fi

# Existing multi-view probe checkpoints (reuse, not retrain)
for TASK in as_severity mr_severity aov_vmax; do
    DEST="${LOCAL}/probes/${TASK}/echojepa-g"
    mkdir -p "$DEST"
    if [ ! -f "${DEST}/best.pt" ]; then
        log "  Downloading probe: ${TASK}/echojepa-g"
        aws s3 cp "${S3_BASE}/checkpoints/probes/${TASK}/echojepa-g/best.pt" "${DEST}/best.pt"
    fi
done

# A4C-only probe CSVs (test set + train for dummy reference)
for TASK in as_severity_a4c mr_severity_a4c aov_vmax_a4c; do
    for f in train_vf.csv test_vf.csv; do
        if [ ! -f "${LOCAL}/csvs/${TASK}/${f}" ]; then
            aws s3 cp "${S3_BASE}/nmed_a4c/probe_csvs/${TASK}/${f}" "${LOCAL}/csvs/${TASK}/${f}"
        fi
    done
    aws s3 cp "${S3_BASE}/nmed_a4c/probe_csvs/${TASK}/zscore_params.json" "${LOCAL}/csvs/${TASK}/zscore_params.json" 2>/dev/null || true
done

log "=== All assets downloaded ==="

# ============================================================
# Step 2: Config generation (inference only)
# ============================================================
generate_config() {
    local TASK_A4C="$1"      # e.g. as_severity_a4c
    local TASK_BASE="$2"     # e.g. as_severity (for probe checkpoint)
    local TASK_TYPE="$3"
    local NUM_CLASSES="$4"
    local VAL_BS="$5"

    local PROBE_CKPT="${LOCAL}/probes/${TASK_BASE}/echojepa-g/best.pt"
    local TRAIN_CSV="${LOCAL}/csvs/${TASK_A4C}/train_vf.csv"
    local TEST_CSV="${LOCAL}/csvs/${TASK_A4C}/test_vf.csv"
    local OUTFILE="/tmp/nmed_a4c_cfg_${TASK_A4C}.yaml"

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
    # 12-head grid (4 LR x 3 WD) — must match training grid
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

    # Kill orphaned workers
    ps -eo pid,ppid,args | grep "multiprocessing.spawn" | grep -v grep | awk '$2 == 1 {print $1}' | xargs -r kill 2>/dev/null || true
    sleep 2

    log "[${TAG}] Starting on ${DEVICES} (port ${PORT})"

    # Clear stale output dir
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
# Step 4: Run inference (existing probes, A4C-only test data)
# ============================================================
ALL_DEVICES="cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7"
FAILURES=0
TOTAL=3

log "=== A4C-only inference with existing multi-view probes ==="

# AS severity (4-class)
AS_CLASSES=$(awk '{print $NF}' "${LOCAL}/csvs/as_severity_a4c/train_vf.csv" | sort -u | wc -l)
log "AS severity: ${AS_CLASSES} classes"
CFG=$(generate_config "as_severity_a4c" "as_severity" "classification" "${AS_CLASSES}" "128")
run_single "$CFG" "predavg:as_severity_a4c-predavg-echojepa-g" "$ALL_DEVICES" "29500" || FAILURES=$((FAILURES + 1))

# MR severity (5-class)
MR_CLASSES=$(awk '{print $NF}' "${LOCAL}/csvs/mr_severity_a4c/train_vf.csv" | sort -u | wc -l)
log "MR severity: ${MR_CLASSES} classes"
CFG=$(generate_config "mr_severity_a4c" "mr_severity" "classification" "${MR_CLASSES}" "128")
run_single "$CFG" "predavg:mr_severity_a4c-predavg-echojepa-g" "$ALL_DEVICES" "29500" || FAILURES=$((FAILURES + 1))

# AoV Vmax (regression)
CFG=$(generate_config "aov_vmax_a4c" "aov_vmax" "regression" "1" "128")
run_single "$CFG" "predavg:aov_vmax_a4c-predavg-echojepa-g" "$ALL_DEVICES" "29500" || FAILURES=$((FAILURES + 1))

log "=== Results: $((TOTAL - FAILURES))/${TOTAL} succeeded ==="

# ============================================================
# Step 5: Upload results to S3
# ============================================================
log "=== Uploading results to S3 ==="
for TASK in as_severity_a4c mr_severity_a4c aov_vmax_a4c; do
    DIR="${OUT_DIR}/video_classification_frozen/${TASK}-predavg-echojepa-g"
    if [ -d "$DIR" ]; then
        aws s3 sync "$DIR" "${S3_OUT}/uhn/${TASK}-predavg-echojepa-g/"
        log "  Uploaded: ${TASK}-predavg-echojepa-g"
    fi
done

log "=== Job complete ==="
