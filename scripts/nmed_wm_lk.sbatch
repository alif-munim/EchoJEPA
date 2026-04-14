#!/bin/bash
#SBATCH --job-name=nmed-wm-lk
#SBATCH --partition=ml-p5-48xlarge
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --output=/tmp/nmed_wm_lk-%j.out
#SBATCH --error=/tmp/nmed_wm_lk-%j.err
#SBATCH --time=0-02:00:00
#SBATCH --nodelist=ip-10-0-50-7

# Nature Medicine: EchoJEPA-L-K inference on all world model tasks
# Inference only — reuses existing multi-view L-K probes on view-restricted test clips.
#
# A4C ablation (5 tasks):
#   AS severity (4-class), Bicuspid AV (binary), AoV Vmax (regression),
#   AoV Mean Grad (regression), MR severity (5-class, control)
#
# PLAX ablation (3 tasks):
#   LVEF (regression), ESV (regression), EDV (regression)
#
# L-K is ~300M params → ~5-10 min per task on 8x H100
# Estimated total: ~1h

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
LOCAL="/opt/dlami/nvme/nmed_wm_lk"
OUT_DIR="/opt/dlami/nvme/nmed_wm_lk/output"
S3_OUT="${S3_BASE}/nmed_a4c/results"

log() { echo "[$(date +%H:%M:%S)] $*"; }

# ============================================================
# Step 1: Download assets from S3
# ============================================================
log "=== Downloading assets from S3 ==="

# Create directories
mkdir -p ${LOCAL}/{encoder,csvs}
for TASK in as_severity disease_bicuspid_av aov_vmax aov_mean_grad mr_severity lvef esv edv; do
    mkdir -p "${LOCAL}/probes/${TASK}/echojepa-l-k"
done
for TASK in as_severity_a4c disease_bicuspid_av_a4c aov_vmax_a4c aov_mean_grad_a4c mr_severity_a4c lvef_plax esv_plax edv_plax; do
    mkdir -p "${LOCAL}/csvs/${TASK}"
done

# EchoJEPA-L-K encoder (~5.1GB)
if [ ! -f "${LOCAL}/encoder/vitl-kinetics-pt220-an55.pt" ]; then
    log "  Downloading L-K encoder..."
    aws s3 cp "${S3_BASE}/checkpoints/anneal/keep/vitl-kinetics-pt220-an55.pt" "${LOCAL}/encoder/vitl-kinetics-pt220-an55.pt"
fi

# L-K probe checkpoints (8 tasks)
for TASK in as_severity disease_bicuspid_av aov_vmax aov_mean_grad mr_severity lvef esv edv; do
    DEST="${LOCAL}/probes/${TASK}/echojepa-l-k"
    if [ ! -f "${DEST}/best.pt" ]; then
        log "  Downloading probe: ${TASK}/echojepa-l-k"
        aws s3 cp "${S3_BASE}/checkpoints/probes/${TASK}/echojepa-l-k/best.pt" "${DEST}/best.pt"
    fi
done

# View-filtered CSVs
for TASK in as_severity_a4c disease_bicuspid_av_a4c aov_vmax_a4c aov_mean_grad_a4c mr_severity_a4c lvef_plax esv_plax edv_plax; do
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

# 12-head HP grid (must match training grid)
HP_GRID='    - {lr: 0.0005, start_lr: 0.0, warmup: 2.0, final_lr: 0.0, weight_decay: 0.001, final_weight_decay: 0.001}
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
    - {lr: 0.00001, start_lr: 0.0, warmup: 2.0, final_lr: 0.0, weight_decay: 0.1,   final_weight_decay: 0.1}'

generate_config_lk() {
    local TASK_VF="$1"      # e.g., as_severity_a4c or lvef_plax
    local TASK_BASE="$2"    # e.g., as_severity or lvef (for probe checkpoint)
    local TASK_TYPE="$3"    # classification or regression
    local NUM_CLASSES="$4"  # number of classes (or 1 for regression)
    local VAL_BS="$5"

    local PROBE_CKPT="${LOCAL}/probes/${TASK_BASE}/echojepa-l-k/best.pt"
    local TRAIN_CSV="${LOCAL}/csvs/${TASK_VF}/train_vf.csv"
    local TEST_CSV="${LOCAL}/csvs/${TASK_VF}/test_vf.csv"
    local OUTFILE="/opt/dlami/nvme/nmed_wm_lk_cfg_${TASK_VF}.yaml"

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
tag: ${TASK_VF}-predavg-echojepa-l-k

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
${HP_GRID}

model_kwargs:
  checkpoint: ${LOCAL}/encoder/vitl-kinetics-pt220-an55.pt
  module_name: evals.video_classification_frozen.modelcustom.vit_encoder_multiclip
  pretrain_kwargs:
    encoder:
      checkpoint_key: target_encoder
      img_temporal_dim_size: null
      model_name: vit_large
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
# Step 4: Run inference — 8 tasks, all L-K
# ============================================================
ALL_DEVICES="cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7"
FAILURES=0
TOTAL=8

log "=== L-K World Model Inference (8 tasks) ==="

# --- A4C Ablation (5 tasks) ---
log "--- A4C Ablation ---"

# AS severity (4-class)
AS_CLASSES=$(awk '{print $NF}' "${LOCAL}/csvs/as_severity_a4c/train_vf.csv" | sort -u | wc -l)
log "AS severity: ${AS_CLASSES} classes"
CFG=$(generate_config_lk "as_severity_a4c" "as_severity" "classification" "${AS_CLASSES}" "128")
run_single "$CFG" "predavg:as_severity_a4c-predavg-echojepa-l-k" "$ALL_DEVICES" "29500" || FAILURES=$((FAILURES + 1))

# Bicuspid AV (binary)
BAV_CLASSES=$(awk '{print $NF}' "${LOCAL}/csvs/disease_bicuspid_av_a4c/train_vf.csv" | sort -u | wc -l)
log "Bicuspid AV: ${BAV_CLASSES} classes"
CFG=$(generate_config_lk "disease_bicuspid_av_a4c" "disease_bicuspid_av" "classification" "${BAV_CLASSES}" "128")
run_single "$CFG" "predavg:disease_bicuspid_av_a4c-predavg-echojepa-l-k" "$ALL_DEVICES" "29500" || FAILURES=$((FAILURES + 1))

# AoV Vmax (regression)
CFG=$(generate_config_lk "aov_vmax_a4c" "aov_vmax" "regression" "1" "128")
run_single "$CFG" "predavg:aov_vmax_a4c-predavg-echojepa-l-k" "$ALL_DEVICES" "29500" || FAILURES=$((FAILURES + 1))

# AoV Mean Grad (regression)
CFG=$(generate_config_lk "aov_mean_grad_a4c" "aov_mean_grad" "regression" "1" "128")
run_single "$CFG" "predavg:aov_mean_grad_a4c-predavg-echojepa-l-k" "$ALL_DEVICES" "29500" || FAILURES=$((FAILURES + 1))

# MR severity (5-class, control)
MR_CLASSES=$(awk '{print $NF}' "${LOCAL}/csvs/mr_severity_a4c/train_vf.csv" | sort -u | wc -l)
log "MR severity: ${MR_CLASSES} classes"
CFG=$(generate_config_lk "mr_severity_a4c" "mr_severity" "classification" "${MR_CLASSES}" "128")
run_single "$CFG" "predavg:mr_severity_a4c-predavg-echojepa-l-k" "$ALL_DEVICES" "29500" || FAILURES=$((FAILURES + 1))

# --- PLAX Ablation (3 tasks) ---
log "--- PLAX Ablation ---"

# LVEF (regression)
CFG=$(generate_config_lk "lvef_plax" "lvef" "regression" "1" "128")
run_single "$CFG" "predavg:lvef_plax-predavg-echojepa-l-k" "$ALL_DEVICES" "29500" || FAILURES=$((FAILURES + 1))

# ESV (regression)
CFG=$(generate_config_lk "esv_plax" "esv" "regression" "1" "128")
run_single "$CFG" "predavg:esv_plax-predavg-echojepa-l-k" "$ALL_DEVICES" "29500" || FAILURES=$((FAILURES + 1))

# EDV (regression)
CFG=$(generate_config_lk "edv_plax" "edv" "regression" "1" "128")
run_single "$CFG" "predavg:edv_plax-predavg-echojepa-l-k" "$ALL_DEVICES" "29500" || FAILURES=$((FAILURES + 1))

log "=== Results: $((TOTAL - FAILURES))/${TOTAL} succeeded ==="

# ============================================================
# Step 5: Upload results to S3
# ============================================================
log "=== Uploading results to S3 ==="
for TASK in as_severity_a4c disease_bicuspid_av_a4c aov_vmax_a4c aov_mean_grad_a4c mr_severity_a4c lvef_plax esv_plax edv_plax; do
    DIR="${OUT_DIR}/video_classification_frozen/${TASK}-predavg-echojepa-l-k"
    if [ -d "$DIR" ]; then
        aws s3 sync "$DIR" "${S3_OUT}/uhn/${TASK}-predavg-echojepa-l-k/"
        log "  Uploaded: ${TASK}-predavg-echojepa-l-k"
    fi
done

# ============================================================
# Step 6: Cleanup
# ============================================================
log "=== Cleaning up ==="
rm -rf "${LOCAL}/encoder" "${LOCAL}/probes" "${LOCAL}/csvs" "${TORCH_HOME}" 2>/dev/null || true
log "=== Job complete. Failures: ${FAILURES}/${TOTAL} ==="
