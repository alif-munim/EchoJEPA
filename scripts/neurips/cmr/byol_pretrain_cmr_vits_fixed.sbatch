#!/bin/bash
#SBATCH -p dev
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH -t 14:00:00
#SBATCH -J byol_cmr_s
#SBATCH -o /opt/dlami/nvme/logs/byol_cmr_s_%j.out
#SBATCH -e /opt/dlami/nvme/logs/byol_cmr_s_%j.err
#SBATCH --signal=B:USR1@300

set -euxo pipefail

# =============================================================================
# BYOL-Video ViT-S CMR SAX pretraining — config-fixed variant.
# ema = [0.99925, 0.99925], 800 epochs (matches CMR MAE/JEPA comparators).
# =============================================================================

export AWS_DEFAULT_REGION="us-west-2"
export AWS_REGION="us-west-2"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export MASTER_PORT=$((12000 + $RANDOM % 20000))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

export HOME="/home/ubuntu"
unset AWS_SHARED_CREDENTIALS_FILE 2>/dev/null || true
unset AWS_CONFIG_FILE 2>/dev/null || true
unset AWS_EC2_METADATA_DISABLED 2>/dev/null || true
export AWS_METADATA_SERVICE_TIMEOUT=10
export AWS_METADATA_SERVICE_NUM_ATTEMPTS=5

export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

echo "[PREFLIGHT] Checking AWS credentials..."
aws sts get-caller-identity || { echo "[FATAL] AWS CLI credentials check failed"; exit 1; }
echo "[PREFLIGHT] OK"

# =========================
# S3 CONFIG
# =========================
ART_BUCKET="sagemaker-hyperpod-lifecycle-495467399120-usw2"
S3_BASE="s3://${ART_BUCKET}/vjepa2-artifacts"
CSV_S3="${S3_BASE}/data/csv/cmr_pretrain.csv"
ENV_S3="${S3_BASE}/env/vjepa2-312-pack-env.tar"
CKPT_S3="${S3_BASE}/checkpoints/vits_in21k.pt"
RUN_S3="${S3_BASE}/runs/byol_cmr_vits_fixed_${SLURM_JOB_ID}"

# =========================
# LOGGING / SYNC
# =========================
BYOL_OUTPUT="/opt/dlami/nvme/checkpoints/pretrain/cmr/byol_vits_224px_16f_in21k_fixed"

sync_ckpts() {
  if [ -d "${BYOL_OUTPUT}" ]; then
    aws s3 sync "${BYOL_OUTPUT}" "${RUN_S3}/training_folder/" \
      --exclude "*" \
      --include "*.pt" --include "*.pth" --include "*.yaml" --include "*.json" --include "*.csv" \
      --only-show-errors || true
  fi
}

mkdir -p /opt/dlami/nvme/logs

upload_logs() {
  local out="/opt/dlami/nvme/logs/byol_cmr_s_${SLURM_JOB_ID}.out"
  local err="/opt/dlami/nvme/logs/byol_cmr_s_${SLURM_JOB_ID}.err"
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

pkill -9 -f "app.main" 2>/dev/null || true
pkill -9 -f "app.byol" 2>/dev/null || true
sleep 2

SCRATCH="/opt/dlami/nvme"

# Env
aws s3 cp "$ENV_S3" "${SCRATCH}/env.tar"
mkdir -p "${SCRATCH}/env" && tar -xf "${SCRATCH}/env.tar" -C "${SCRATCH}/env"
set +u && source "${SCRATCH}/env/bin/activate" && set -u
"${SCRATCH}/env/bin/conda-unpack" || true
export TMPDIR="${SCRATCH}/tmp_cmr_byol_fixed" && mkdir -p "$TMPDIR"
pip install boto3 --no-cache-dir --quiet

# Code
CODE_DIR="${SCRATCH}/code"
mkdir -p "$CODE_DIR"
aws s3 cp "${S3_BASE}/setup/vjepa2-src.tar.gz" "${SCRATCH}/vjepa2-src.tar.gz" --quiet
tar xzf "${SCRATCH}/vjepa2-src.tar.gz" -C "$CODE_DIR"
cd "$CODE_DIR"
pip install -e . --no-deps --no-build-isolation 2>/dev/null || true

# Data
mkdir -p "${SCRATCH}/data/csv"
aws s3 cp "$CSV_S3" "${SCRATCH}/data/csv/cmr_pretrain.csv"

# Checkpoint
mkdir -p "${SCRATCH}/checkpoints"
echo "[INFO] Downloading ViT-S IN21K checkpoint..."
aws s3 cp "$CKPT_S3" "${SCRATCH}/checkpoints/vits_in21k.pt"
echo "[INFO] Checkpoint: $(ls -lh ${SCRATCH}/checkpoints/vits_in21k.pt)"

CONFIG="${CODE_DIR}/configs/train/vits16/pretrain-byol-cmr-224px-16f-in21k-fixed.yaml"

# Periodic upload
(
  while true; do
    sleep 900
    sync_ckpts
    upload_logs
  done
) &
SYNC_PID=$!

echo "[INFO] BYOL-Video ViT-S CMR config-fixed pretrain"
echo "       Config: ${CONFIG}"
echo "       Output: ${BYOL_OUTPUT}"
echo "       S3:     ${RUN_S3}"

srun --kill-on-bad-exit=1 \
  python3 -m app.main_srun \
    --fname "${CONFIG}"
RC=$?

kill $SYNC_PID 2>/dev/null || true
wait $SYNC_PID 2>/dev/null || true
upload_logs
sync_ckpts
exit $RC
