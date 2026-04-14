#!/bin/bash
#SBATCH --job-name=nmed-a4c-pe
#SBATCH --partition=ml-p5-48xlarge
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --output=/tmp/nmed_a4c_pe-%j.out
#SBATCH --error=/tmp/nmed_a4c_pe-%j.err
#SBATCH --time=0-01:00:00
#SBATCH --nodelist=ip-10-0-50-7

# Nature Medicine: A4C-only single-view ablation -- PanEcho baselines only
# Retry after tasks.pkl deployment fix.

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
LOCAL="/opt/dlami/nvme/nmed_a4c_pe"
OUT_DIR="/opt/dlami/nvme/nmed_a4c_pe/output"
S3_OUT="${S3_BASE}/nmed_a4c/results"

log() { echo "[$(date +%H:%M:%S)] $*"; }

# ============================================================
# Step 1: Download assets from S3
# ============================================================
log "=== Downloading assets from S3 ==="
mkdir -p ${LOCAL}/{probes/as_severity/panecho,probes/aov_vmax/panecho,csvs/as_severity_a4c,csvs/aov_vmax_a4c}

# Probe checkpoints
for TASK in as_severity aov_vmax; do
    DEST="${LOCAL}/probes/${TASK}/panecho"
    if [ ! -f "${DEST}/best.pt" ]; then
        log "  Downloading probe: ${TASK}/panecho"
        aws s3 cp "${S3_BASE}/checkpoints/probes/${TASK}/panecho/best.pt" "${DEST}/best.pt"
    fi
done

# PanEcho weights (torch.hub cache)
PE_CACHE="${TORCH_HOME}/hub/checkpoints"
mkdir -p "$PE_CACHE"
if [ ! -f "${PE_CACHE}/panecho.pt" ]; then
    log "  Downloading PanEcho weights to torch hub cache..."
    aws s3 cp "${S3_BASE}/setup/panecho.pt" "${PE_CACHE}/panecho.pt"
fi

# A4C-only probe CSVs
for TASK in as_severity_a4c aov_vmax_a4c; do
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
generate_config_panecho() {
    local TASK_A4C="$1"
    local TASK_BASE="$2"
    local TASK_TYPE="$3"
    local NUM_CLASSES="$4"
    local VAL_BS="$5"

    local PROBE_CKPT="${LOCAL}/probes/${TASK_BASE}/panecho/best.pt"
    local TRAIN_CSV="${LOCAL}/csvs/${TASK_A4C}/train_vf.csv"
    local TEST_CSV="${LOCAL}/csvs/${TASK_A4C}/test_vf.csv"
    local OUTFILE="/opt/dlami/nvme/nmed_a4c_pe_cfg_${TASK_A4C}_panecho.yaml"

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
# Step 4: Run inference
# ============================================================
ALL_DEVICES="cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7"
FAILURES=0
TOTAL=2

log "=== A4C-only inference: PanEcho baselines ==="

# AS severity (4-class)
AS_CLASSES=$(awk '{print $NF}' "${LOCAL}/csvs/as_severity_a4c/train_vf.csv" | sort -u | wc -l)
log "AS severity (PanEcho): ${AS_CLASSES} classes"
CFG=$(generate_config_panecho "as_severity_a4c" "as_severity" "classification" "${AS_CLASSES}" "128")
run_single "$CFG" "predavg:as_severity_a4c-predavg-panecho" "$ALL_DEVICES" "29500" || FAILURES=$((FAILURES + 1))

# AoV Vmax (regression)
CFG=$(generate_config_panecho "aov_vmax_a4c" "aov_vmax" "regression" "1" "128")
run_single "$CFG" "predavg:aov_vmax_a4c-predavg-panecho" "$ALL_DEVICES" "29500" || FAILURES=$((FAILURES + 1))

log "=== Results: $((TOTAL - FAILURES))/${TOTAL} succeeded ==="

# ============================================================
# Step 5: Upload results to S3
# ============================================================
log "=== Uploading results to S3 ==="
for TASK in as_severity_a4c aov_vmax_a4c; do
    DIR="${OUT_DIR}/video_classification_frozen/${TASK}-predavg-panecho"
    if [ -d "$DIR" ]; then
        aws s3 sync "$DIR" "${S3_OUT}/uhn/${TASK}-predavg-panecho/"
        log "  Uploaded: ${TASK}-predavg-panecho"
    fi
done

# ============================================================
# Step 6: Cleanup
# ============================================================
log "=== Cleaning up ==="
rm -rf "${LOCAL}" "${TORCH_HOME}" 2>/dev/null || true
log "=== Job complete ==="
