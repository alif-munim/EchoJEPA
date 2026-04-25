#!/bin/bash
#SBATCH -p dev
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 7-00:00:00
#SBATCH -J phi_jepa_D
#SBATCH -o /opt/dlami/nvme/logs/phi_jepa_D_%j.out
#SBATCH -e /opt/dlami/nvme/logs/phi_jepa_D_%j.err
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

# NCCL watchdog: 60 min. A straggler rank waiting on a slow S3 fetch + decode
# retry can exceed the 30-min default; 60 min is a safety buffer while the
# dataloader's decord retry path does the real work.
export TORCH_NCCL_TIMEOUT=3600

# =========================
# S3 CONFIGURATION
# =========================
ART_BUCKET="sagemaker-hyperpod-lifecycle-495467399120-usw2"
S3_BASE="s3://${ART_BUCKET}/vjepa2-artifacts"

IN21K_CKPT_S3="${S3_BASE}/checkpoints/vitl_raw.pth"
DATA_CSV_S3="${S3_BASE}/data/csv/mimic_annotations_s3.csv"
PHASE_CSV_S3="${S3_BASE}/phase/mimic_clip_phase_metadata.csv"

# Config path (pilot10 for the debug pilot; full for Run D).
# Override via: CFG_REL=... sbatch pretrain_phi_jepa_vitl.sbatch
CFG_REL="${CFG_REL:-configs/train/vitl16/pretrain-phi-jepa-mimic-224px-16f.yaml}"

RUN_S3="${S3_BASE}/runs/phi_jepa_D_${SLURM_JOB_ID}"
# The folder value in the YAML selects the on-disk checkpoint dir; sync both.
CKPT_DIR_PILOT="/opt/dlami/nvme/checkpoints/pretrain/mimic/vjepa2_vitl_224px_16f_phi_jepa_pilot10"
CKPT_DIR_FULL="/opt/dlami/nvme/checkpoints/pretrain/mimic/vjepa2_vitl_224px_16f_phi_jepa"

# =========================
# LOGGING / SYNC
# =========================
sync_ckpts() {
  for d in "$CKPT_DIR_PILOT" "$CKPT_DIR_FULL"; do
    if [ -d "$d" ]; then
      aws s3 sync "$d" "${RUN_S3}/checkpoints/" \
        --exclude "*" \
        --include "e*.pt" --include "latest.pt" --include "*.csv" --include "*.yaml" \
        --only-show-errors || true
    fi
  done
}

upload_logs() {
  aws s3 cp "/opt/dlami/nvme/logs/phi_jepa_D_${SLURM_JOB_ID}.out" "${RUN_S3}/logs/job.out" --only-show-errors 2>/dev/null || true
  aws s3 cp "/opt/dlami/nvme/logs/phi_jepa_D_${SLURM_JOB_ID}.err" "${RUN_S3}/logs/job.err" --only-show-errors 2>/dev/null || true
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
# STAGE DATA + CHECKPOINT
# =========================
echo "[INFO] Downloading data CSV..."
mkdir -p /opt/dlami/nvme/data/csv
aws s3 cp "$DATA_CSV_S3" /opt/dlami/nvme/data/csv/mimic_annotations_s3.csv

echo "[INFO] Downloading phi-JEPA per-clip phase metadata CSV..."
aws s3 cp "$PHASE_CSV_S3" /opt/dlami/nvme/data/csv/mimic_clip_phase_metadata.csv
ls -lh /opt/dlami/nvme/data/csv/mimic_clip_phase_metadata.csv

echo "[INFO] Downloading ImageNet-21K ViT-L checkpoint..."
mkdir -p /opt/dlami/nvme/checkpoints
aws s3 cp "$IN21K_CKPT_S3" /opt/dlami/nvme/checkpoints/vitl_in21k.pt
ls -lh /opt/dlami/nvme/checkpoints/vitl_in21k.pt

mkdir -p "$CKPT_DIR_PILOT" "$CKPT_DIR_FULL"

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
# RUN phi-JEPA PRETRAINING
# =========================
echo "[INFO] Launching phi-JEPA pretraining with config: $CFG_REL"
python -u -m app.main \
  --fname "$CFG_REL" \
  --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
  2>&1 | tee "/opt/dlami/nvme/logs/phi_jepa_D_${SLURM_JOB_ID}.log"
RC=$?

kill $SYNC_PID 2>/dev/null || true
wait $SYNC_PID 2>/dev/null || true
sync_ckpts
upload_logs
exit $RC
