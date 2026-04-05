#!/bin/bash
#SBATCH -p ml-p5-48xlarge
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 7-00:00:00
#SBATCH -J salt_s2
#SBATCH -o /tmp/salt_s2-%j.out
#SBATCH -e /tmp/salt_s2-%j.err
#SBATCH --chdir=/tmp

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

# Force PyTorch's bundled CUDA libs (cuBLAS bf16 fix for H100)
export LD_LIBRARY_PATH="/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cublas/lib:/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"

# =========================
# S3 CONFIGURATION
# =========================
ART_BUCKET="sagemaker-hyperpod-lifecycle-495467399120-usw2"
S3_BASE="s3://${ART_BUCKET}/vjepa2-artifacts"

DATA_CSV_S3="${S3_BASE}/data/csv/mimic_annotations_s3.csv"

# =====================================================================
# SALT S1 TEACHER CHECKPOINT
# Update this path to the actual S3 location of the S1 latest.pt
# after S1 training completes. The sbatch syncs it to NVMe before launch.
# =====================================================================
SALT_S1_RUN_S3="${S3_BASE}/runs/FILL_IN_S1_RUN_ID/checkpoints/latest.pt"

CFG_REL="configs/train/vitl16/pretrain-salt-s2-mimic-224px-16f-hp.yaml"

RUN_S3="${S3_BASE}/runs/salt_s2_pretrain_${SLURM_JOB_ID}"
CKPT_DIR="/opt/dlami/nvme/checkpoints/pretrain/mimic/salt_s2_vitl_224px_16f"

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
  aws s3 cp "/tmp/salt_s2-${SLURM_JOB_ID}.out" "${RUN_S3}/logs/job.out" --only-show-errors 2>/dev/null || true
  aws s3 cp "/tmp/salt_s2-${SLURM_JOB_ID}.err" "${RUN_S3}/logs/job.err" --only-show-errors 2>/dev/null || true
}

on_exit() {
  rc=$?
  sync_ckpts
  upload_logs
  exit $rc
}
trap on_exit EXIT

# =========================
# HOST SETUP
# =========================
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
# STAGE DATA + TEACHER CHECKPOINT
# =========================
echo "[INFO] Downloading data CSV..."
mkdir -p /opt/dlami/nvme/data/csv
aws s3 cp "$DATA_CSV_S3" /opt/dlami/nvme/data/csv/mimic_annotations_s3.csv

echo "[INFO] Downloading SALT S1 teacher checkpoint..."
S1_CKPT_DIR="/opt/dlami/nvme/checkpoints/pretrain/mimic/salt_s1_vitl_224px_16f"
mkdir -p "$S1_CKPT_DIR"
aws s3 cp "$SALT_S1_RUN_S3" "$S1_CKPT_DIR/latest.pt"

echo "[INFO] Teacher checkpoint staged:"
ls -lh "$S1_CKPT_DIR/latest.pt"

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
# RUN SALT STAGE 2
# =========================
echo "[INFO] Launching SALT Stage 2 (frozen-teacher student) pretraining..."
python -m app.main \
  --fname "$CFG_REL" \
  --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
  2>&1 | tee "salt_s2_${SLURM_JOB_ID}.log"
RC=$?

kill $SYNC_PID 2>/dev/null || true
wait $SYNC_PID 2>/dev/null || true
sync_ckpts
upload_logs
exit $RC
