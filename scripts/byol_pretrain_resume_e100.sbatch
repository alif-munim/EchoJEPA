#!/bin/bash
#SBATCH -p ml-p5-48xlarge
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 7-00:00:00
#SBATCH -J byol_pt
#SBATCH -o /tmp/byol_pt-%j.out
#SBATCH -e /tmp/byol_pt-%j.err
#SBATCH --chdir=/tmp
#SBATCH --nodelist=ip-10-0-50-83

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

ENV_S3="${S3_BASE}/env/vjepa2-312-pack-env.tar"

# BYOL e100 checkpoint from previous run (job 342)
CKPT_S3="${S3_BASE}/runs/byol_pretrain_resume_342/checkpoints/e100.pt"
CKPT_OPT_S3="${S3_BASE}/runs/byol_pretrain_resume_342/checkpoints/e100_opt.pt"

# Data CSV (MIMIC 525K with S3 paths)
DATA_CSV_S3="${S3_BASE}/data/csv/mimic_annotations_s3.csv"

CFG_REL="configs/train/vitl16/pretrain-byol-mimic-224px-16f-resume-e100-hp.yaml"

RUN_S3="${S3_BASE}/runs/byol_pretrain_resume_e100_${SLURM_JOB_ID}"

CKPT_DIR="/opt/dlami/nvme/checkpoints/pretrain/mimic/byol_vitl_224px_16f_imagenet_v2_resume"

# =========================
# LOGGING / SYNC
# =========================
sync_ckpts() {
  if [ -d "$CKPT_DIR" ]; then
    aws s3 sync "$CKPT_DIR" "${RUN_S3}/checkpoints/" \
      --exclude "*" \
      --include "e*.pt" --include "e*_opt.pt" --include "latest.pt" --include "latest_opt.pt" --include "*.csv" \
      --only-show-errors || true
  fi
}

upload_logs() {
  aws s3 cp "/tmp/byol_pt-${SLURM_JOB_ID}.out" "${RUN_S3}/logs/job.out" --only-show-errors 2>/dev/null || true
  aws s3 cp "/tmp/byol_pt-${SLURM_JOB_ID}.err" "${RUN_S3}/logs/job.err" --only-show-errors 2>/dev/null || true
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

# Clear shared memory
rm -rf /dev/shm/* 2>/dev/null || true

SCRATCH="/opt/dlami/nvme"
WORKDIR="${SCRATCH}/byol_pt_${SLURM_JOB_ID}"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

# =========================
# ENVIRONMENT
# =========================
echo "[INFO] Setting up environment..."
if [ ! -d "env/bin" ]; then
    aws s3 cp "$ENV_S3" env.tar
    mkdir -p env && tar -xf env.tar -C env
    set +u && source env/bin/activate && set -u
    env/bin/conda-unpack || true
    pip install "numpy<2.0" boto3 --no-cache-dir
    rm -f env.tar
else
    set +u && source env/bin/activate && set -u
fi

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
# STAGE DATA CSV + CHECKPOINT
# =========================
echo "[INFO] Downloading data CSV..."
mkdir -p /opt/dlami/nvme/data/csv
aws s3 cp "$DATA_CSV_S3" /opt/dlami/nvme/data/csv/mimic_annotations_s3.csv

echo "[INFO] Staging e100 checkpoint as latest.pt for resume..."
mkdir -p "$CKPT_DIR"
aws s3 cp "$CKPT_S3" "$CKPT_DIR/latest.pt"
aws s3 cp "$CKPT_OPT_S3" "$CKPT_DIR/latest_opt.pt"

echo "[INFO] Checkpoint staged:"
ls -la "$CKPT_DIR/"

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
# RUN PRETRAINING
# =========================
echo "[INFO] Launching BYOL pretraining (resume from e100 → e200)..."
python -m app.main \
  --fname "$CFG_REL" \
  --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
  2>&1 | tee "byol_pretrain_${SLURM_JOB_ID}.log"
RC=$?

kill $SYNC_PID 2>/dev/null || true
wait $SYNC_PID 2>/dev/null || true
sync_ckpts
upload_logs
exit $RC
