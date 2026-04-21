#!/bin/bash
#SBATCH -p ml-p5-48xlarge
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 12:00:00
#SBATCH -J mae200_lvef
#SBATCH -o /tmp/mae200_lvef_%j.out
#SBATCH -e /tmp/mae200_lvef_%j.err
#SBATCH --signal=B:USR1@300
# MAE extended to e194 (e197 crash reproducible; e194 ckpt is final)

set -euxo pipefail

# =============================================================================
# EchoMAE-L e200 → EchoNet-Dynamic LVEF probe (d=4 attentive)
# =============================================================================
# Trains a d=4 attentive probe on frozen EchoMAE-L (200 epoch) encoder.
# 6 HP heads: lr={1e-4, 5e-5} × wd={0.01, 0.1, 0.4}
# 20 epochs of probe training on EchoNet-Dynamic (7465 train / 1288 val).
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

echo "[PREFLIGHT] Checking AWS credentials..."
aws sts get-caller-identity || { echo "[FATAL] AWS CLI credentials check failed"; exit 1; }
echo "[PREFLIGHT] OK"

# =========================
# S3 CONFIG
# =========================
ART_BUCKET="sagemaker-hyperpod-lifecycle-495467399120-usw2"
S3_BASE="s3://${ART_BUCKET}/vjepa2-artifacts"
ENV_S3="${S3_BASE}/env/vjepa2-312-pack-env.tar"
RUN_S3="${S3_BASE}/runs/mae200_lvef_${SLURM_JOB_ID}"

# Checkpoint from the extended MAE run (job 179)
# VideoMAE saves checkpoint-{epoch} where epoch is 0-indexed
# e200 training with save_ckpt_freq=5: last save at e199 (epoch 199) or e200 if it saves at completion
CKPT_S3="${S3_BASE}/runs/videomae_e200b_179/training_folder"

# =========================
# LOGGING / SYNC
# =========================
EVAL_OUTPUT="/opt/dlami/nvme/evals/neurips/echomae_e200_end_lvef"

sync_results() {
  if [ -d "${EVAL_OUTPUT}" ]; then
    aws s3 sync "${EVAL_OUTPUT}" "${RUN_S3}/eval_output/" \
      --exclude "*" \
      --include "*.pt" --include "*.pth" --include "*.yaml" --include "*.json" --include "*.csv" --include "*.npz" \
      --only-show-errors || true
  fi
}

upload_logs() {
  local out="/tmp/mae200_lvef_${SLURM_JOB_ID}.out"
  local err="/tmp/mae200_lvef_${SLURM_JOB_ID}.err"
  [ -f "${out}" ] && aws s3 cp "${out}" "${RUN_S3}/logs/job.out" --only-show-errors || true
  [ -f "${err}" ] && aws s3 cp "${err}" "${RUN_S3}/logs/job.err" --only-show-errors || true
}

on_exit() {
  rc=$?
  upload_logs
  sync_results
  exit $rc
}

trap on_exit EXIT
trap 'sync_results; upload_logs' USR1

# =========================
# SETUP
# =========================
nvidia-smi -L || true

# Kill any stale python processes from previous failed jobs
pkill -9 -f "evals.main" 2>/dev/null || true
pkill -9 -f "run_mae_pretraining" 2>/dev/null || true
sleep 2

SCRATCH="/opt/dlami/nvme"

# Env
aws s3 cp "$ENV_S3" "${SCRATCH}/env.tar"
mkdir -p "${SCRATCH}/env" && tar -xf "${SCRATCH}/env.tar" -C "${SCRATCH}/env"
set +u && source "${SCRATCH}/env/bin/activate" && set -u
"${SCRATCH}/env/bin/conda-unpack" || true
pip install boto3 --no-cache-dir --quiet

# Code
CODE_DIR="${SCRATCH}/code"
mkdir -p "$CODE_DIR"
aws s3 cp "${S3_BASE}/setup/vjepa2-src.tar.gz" "${SCRATCH}/vjepa2-src.tar.gz" --quiet
tar xzf "${SCRATCH}/vjepa2-src.tar.gz" -C "$CODE_DIR"
cd "$CODE_DIR"
pip install -e . --no-deps --no-build-isolation 2>/dev/null || true

# Data CSVs
mkdir -p "${SCRATCH}/data/csv"
aws s3 cp "${S3_BASE}/data/csv/echonet_dynamic_train_s3_raw.csv" "${SCRATCH}/data/csv/"
aws s3 cp "${S3_BASE}/data/csv/echonet_dynamic_val_s3_raw.csv" "${SCRATCH}/data/csv/"

# Checkpoint — find the latest one from the e200 run
echo "[INFO] Finding latest MAE e200 checkpoint..."
LATEST_CKPT=$(aws s3 ls "${CKPT_S3}/" --region us-west-2 | grep "checkpoint-" | sort -k4 | tail -1 | awk '{print $4}')
echo "[INFO] Latest checkpoint: ${LATEST_CKPT}"
mkdir -p "${SCRATCH}/checkpoints"
aws s3 cp "${CKPT_S3}/${LATEST_CKPT}" "${SCRATCH}/checkpoints/echomae_l_mimic_ep199.pth"
echo "[INFO] Checkpoint: $(ls -lh ${SCRATCH}/checkpoints/echomae_l_mimic_ep199.pth)"

# Config
CONFIG="${CODE_DIR}/configs/eval/vitl/neurips/echomae_l_e200_end_lvef_d4.yaml"

echo "[INFO] EchoMAE-L e200 → EchoNet-Dynamic LVEF"
echo "       Config: ${CONFIG}"
echo "       Checkpoint: ${LATEST_CKPT}"
echo "       Output: ${EVAL_OUTPUT}"

# =========================
# LAUNCH
# =========================
# evals.main uses mp.Process internally — do NOT use srun (which would launch
# 8 independent copies, each spawning 8 workers = 64 processes on 8 GPUs).
# Run directly and let evals.main manage its own multi-GPU workers.
python3 -m evals.main \
  --fname "${CONFIG}" \
  --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7
RC=$?

upload_logs
sync_results
exit $RC
