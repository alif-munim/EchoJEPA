#!/bin/bash
#SBATCH -p ml-p5-48xlarge
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH -t 7-00:00:00
#SBATCH -J jepa_e200
#SBATCH -o /tmp/jepa_e200_%j.out
#SBATCH -e /tmp/jepa_e200_%j.err
#SBATCH --signal=B:USR1@300
#SBATCH --dependency=afterany:197

set -euxo pipefail

# =============================================================================
# JEPA IN21K Resume: e100 → e200
# =============================================================================
# Extends the canonical JEPA run (job 376) from epoch 100 to 200.
# Purpose: test whether JEPA's temporal encoding persists under extended
# training (consolidation vs eventual spatial convergence).
#
# Queued with --dependency=afterany:159 (MAE e200 run).
# =============================================================================

export AWS_DEFAULT_REGION="us-west-2"
export AWS_REGION="us-west-2"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export MASTER_PORT=$((12000 + $RANDOM % 20000))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

export AWS_SDK_LOAD_CONFIG=1
export HOME="/home/ubuntu"
export AWS_SHARED_CREDENTIALS_FILE="${HOME}/.aws/credentials"
export AWS_CONFIG_FILE="${HOME}/.aws/config"
unset AWS_EC2_METADATA_DISABLED 2>/dev/null || true
export AWS_METADATA_SERVICE_TIMEOUT=5
export AWS_METADATA_SERVICE_NUM_ATTEMPTS=3

export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# =========================
# PREFLIGHT
# =========================
echo "[PREFLIGHT] Checking AWS credentials..."
aws sts get-caller-identity || { echo "[FATAL] AWS CLI credentials check failed"; exit 1; }
echo "[PREFLIGHT] OK"

# =========================
# S3 CONFIG
# =========================
ART_BUCKET="sagemaker-hyperpod-lifecycle-495467399120-usw2"
S3_BASE="s3://${ART_BUCKET}/vjepa2-artifacts"
CSV_S3="${S3_BASE}/data/csv/mimic_annotations_s3.csv"
ENV_S3="${S3_BASE}/env/vjepa2-312-pack-env.tar"
CKPT_S3="${S3_BASE}/runs/jepa_in21k_pretrain_376/checkpoints/latest.pt"
RUN_S3="${S3_BASE}/runs/jepa_in21k_e200_${SLURM_JOB_ID}"

# =========================
# LOGGING / SYNC
# =========================
sync_ckpts() {
  local outdir="/opt/dlami/nvme/jepa_e200"
  if [ -d "${outdir}" ]; then
    aws s3 sync "${outdir}" "${RUN_S3}/training_folder/" \
      --exclude "*" \
      --include "*.pt" --include "*.pth" --include "*.yaml" --include "*.json" --include "*.csv" \
      --only-show-errors || true
  fi
}

upload_logs() {
  local out="/tmp/jepa_e200_${SLURM_JOB_ID}.out"
  local err="/tmp/jepa_e200_${SLURM_JOB_ID}.err"
  [ -f "${out}" ] && aws s3 cp "${out}" "${RUN_S3}/logs/job.out" --only-show-errors || true
  [ -f "${err}" ] && aws s3 cp "${err}" "${RUN_S3}/logs/job.err" --only-show-errors || true
}

on_exit() {
  rc=$?
  upload_logs
  sync_ckpts
  exit $rc
}

trap on_exit EXIT
trap 'sync_ckpts; upload_logs' USR1

# =========================
# SETUP
# =========================
nvidia-smi -L || true
SCRATCH="/opt/dlami/nvme"

# Env
aws s3 cp "$ENV_S3" "${SCRATCH}/env.tar"
mkdir -p "${SCRATCH}/env" && tar -xf "${SCRATCH}/env.tar" -C "${SCRATCH}/env"
set +u && source "${SCRATCH}/env/bin/activate" && set -u
"${SCRATCH}/env/bin/conda-unpack" || true

# Code — deploy from S3 tarball to scratch
CODE_DIR="${SCRATCH}/code"
mkdir -p "$CODE_DIR"
aws s3 cp "${S3_BASE}/setup/vjepa2-src.tar.gz" "${SCRATCH}/vjepa2-src.tar.gz" --quiet
tar xzf "${SCRATCH}/vjepa2-src.tar.gz" -C "$CODE_DIR"
cd "$CODE_DIR"
pip install -e . --no-deps --no-build-isolation 2>/dev/null || true

# Data
mkdir -p "${SCRATCH}/data/csv"
aws s3 cp "$CSV_S3" "${SCRATCH}/data/csv/mimic_annotations_s3.csv"

# Checkpoint — download e100 (latest.pt from job 376)
CKPT_DIR="${SCRATCH}/jepa_e200"
mkdir -p "$CKPT_DIR"
echo "[INFO] Downloading JEPA e100 checkpoint..."
aws s3 cp "$CKPT_S3" "${CKPT_DIR}/latest.pt"
echo "[INFO] Checkpoint: $(ls -lh ${CKPT_DIR}/latest.pt)"

# Config
CONFIG="${CODE_DIR}/configs/train/vitl16/pretrain-mimic-224px-16f-resume-e200.yaml"

# Start periodic uploads
(
  while true; do
    sleep 900
    sync_ckpts
    upload_logs
  done
) &
SYNC_PID=$!

echo "[INFO] JEPA IN21K Resume e100 → e200"
echo "       Config: ${CONFIG}"
echo "       Checkpoint: ${CKPT_DIR}/latest.pt"
echo "       Output: ${CKPT_DIR}"

# =========================
# LAUNCH
# =========================
srun --kill-on-bad-exit=1 \
  python3 -m app.main \
    --fname "${CONFIG}" \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7
RC=$?

kill $SYNC_PID 2>/dev/null || true
wait $SYNC_PID 2>/dev/null || true
upload_logs
sync_ckpts
exit $RC
