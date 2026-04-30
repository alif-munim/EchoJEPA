#!/bin/bash
#SBATCH -p dev
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 2-12:00:00
#SBATCH -J phase_mask
#SBATCH -o /opt/dlami/nvme/logs/phase_mask_%j.out
#SBATCH -e /opt/dlami/nvme/logs/phase_mask_%j.err
#SBATCH --chdir=/tmp

# Mask-phi only pretraining run (Run 4 in the phase ablation matrix).
# Predictor phase-conditioning OFF; phase-aware target masking ON;
# shuffled_hr OFF. Reads per-clip (hr_bpm, frame_time_ms) from the full-corpus
# MIMIC phase CSV (preprocessed, with is_irregular column).

set -euxo pipefail

export PATH="/opt/vjepa2-312/bin:$PATH"

# =========================
# AWS CONFIG
# =========================
export AWS_DEFAULT_REGION="us-west-2"
export AWS_REGION="us-west-2"
export AWS_SDK_LOAD_CONFIG=1
export HOME="/home/ubuntu"
export AWS_SHARED_CREDENTIALS_FILE="${HOME}/.aws/credentials"
export AWS_CONFIG_FILE="${HOME}/.aws/config"
unset AWS_EC2_METADATA_DISABLED 2>/dev/null || true
export AWS_METADATA_SERVICE_TIMEOUT=5
export AWS_METADATA_SERVICE_NUM_ATTEMPTS=3

# Force PyTorch's bundled CUDA libs (cuBLAS bf16 fix for H100).
export LD_LIBRARY_PATH="/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cublas/lib:/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"

export TORCH_NCCL_TIMEOUT=3600

# =========================
# S3 CONFIGURATION
# =========================
ART_BUCKET="sagemaker-hyperpod-lifecycle-495467399120-usw2"
S3_BASE="s3://${ART_BUCKET}/vjepa2-artifacts"

IN21K_CKPT_S3="${S3_BASE}/checkpoints/vitl_raw.pth"
DATA_CSV_S3="${S3_BASE}/data/csv/mimic_annotations_s3.csv"
PHASE_CSV_S3="${S3_BASE}/phase/mimic_clip_phase_metadata.csv"

CFG_REL="${CFG_REL:-configs/train/vitl16/pretrain-mimic-224px-16f-in21k-phase-mask.yaml}"

RUN_S3="${S3_BASE}/runs/phase_mask_${SLURM_JOB_ID}"
CKPT_DIR="/opt/dlami/nvme/checkpoints/pretrain/mimic/vjepa2_vitl_224px_16f_in21k_phase_mask"

# =========================
# LOGGING / SYNC
# =========================
sync_ckpts() {
  if [ -d "$CKPT_DIR" ]; then
    aws s3 sync "$CKPT_DIR" "${RUN_S3}/checkpoints/" \
      --exclude "*" \
      --include "e*.pt" --include "latest.pt" --include "*.csv" --include "*.yaml" \
      --only-show-errors || true
  fi
}

upload_logs() {
  aws s3 cp "/opt/dlami/nvme/logs/phase_mask_${SLURM_JOB_ID}.out" "${RUN_S3}/logs/job.out" --only-show-errors 2>/dev/null || true
  aws s3 cp "/opt/dlami/nvme/logs/phase_mask_${SLURM_JOB_ID}.err" "${RUN_S3}/logs/job.err" --only-show-errors 2>/dev/null || true
}

on_exit() {
  rc=$?
  sync_ckpts
  upload_logs
  exit $rc
}
trap on_exit EXIT

echo "[INFO] host=$(hostname) job=${SLURM_JOB_ID}"
nvidia-smi -L || true
rm -rf /dev/shm/* 2>/dev/null || true

# =========================
# CODE
# =========================
REPO_DIR="/opt/vjepa2"
cd "$REPO_DIR"

if [ ! -f "$CFG_REL" ]; then
    echo "[FATAL] Config $CFG_REL not found in $REPO_DIR"
    exit 1
fi

pip install -e . --no-deps --no-build-isolation 2>/dev/null || true
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# =========================
# STAGE DATA + CHECKPOINT
# =========================
echo "[INFO] Downloading MIMIC annotations CSV..."
mkdir -p /opt/dlami/nvme/data/csv
aws s3 cp "$DATA_CSV_S3" /opt/dlami/nvme/data/csv/mimic_annotations_s3.csv

echo "[INFO] Downloading full-corpus phase metadata CSV..."
aws s3 cp "$PHASE_CSV_S3" /opt/dlami/nvme/data/csv/mimic_clip_phase_metadata.csv
ls -lh /opt/dlami/nvme/data/csv/mimic_clip_phase_metadata.csv

echo "[INFO] Downloading ImageNet-21K ViT-L checkpoint..."
mkdir -p /opt/dlami/nvme/checkpoints
aws s3 cp "$IN21K_CKPT_S3" /opt/dlami/nvme/checkpoints/vitl_in21k.pt
ls -lh /opt/dlami/nvme/checkpoints/vitl_in21k.pt

mkdir -p "$CKPT_DIR"

# =========================
# PERIODIC S3 SYNC
# =========================
(
  while true; do
    sleep 900
    sync_ckpts
    upload_logs
  done
) &
SYNC_PID=$!

# =========================
# RUN
# =========================
echo "[INFO] Launching Mask-phi pretraining with config: $CFG_REL"
python -u -m app.main \
  --fname "$CFG_REL" \
  --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
  2>&1 | tee "/opt/dlami/nvme/logs/phase_mask_${SLURM_JOB_ID}.log"
RC=$?

kill $SYNC_PID 2>/dev/null || true
wait $SYNC_PID 2>/dev/null || true
sync_ckpts
upload_logs
exit $RC
