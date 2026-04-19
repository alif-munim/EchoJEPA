#!/bin/bash
#SBATCH -p ml-p5-48xlarge
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH -t 1-00:00:00
#SBATCH -J mae_traj
#SBATCH -o /tmp/mae_traj_%j.out
#SBATCH -e /tmp/mae_traj_%j.err
#SBATCH --signal=B:USR1@300
#SBATCH --dependency=afterany:196

set -euxo pipefail

# =============================================================================
# EchoMAE-L training trajectory → EchoNet-Dynamic LVEF probes
# =============================================================================
# Runs d=4 attentive probes at e124, e149, e174 to track temporal encoding
# strength across extended MAE training. Combined with e200 (job 190), this
# gives 4 probe points for the training dynamics figure.
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
RUN_S3="${S3_BASE}/runs/mae_traj_lvef_${SLURM_JOB_ID}"

# Checkpoints from original run (159) and resume run (179)
CKPT_RUN_159="${S3_BASE}/runs/videomae_e200_159/training_folder"
CKPT_RUN_179="${S3_BASE}/runs/videomae_e200b_179/training_folder"

EPOCHS_TO_PROBE="124 149 174"

# =========================
# LOGGING / SYNC
# =========================
upload_logs() {
  local out="/tmp/mae_traj_${SLURM_JOB_ID}.out"
  local err="/tmp/mae_traj_${SLURM_JOB_ID}.err"
  [ -f "${out}" ] && aws s3 cp "${out}" "${RUN_S3}/logs/job.out" --only-show-errors || true
  [ -f "${err}" ] && aws s3 cp "${err}" "${RUN_S3}/logs/job.err" --only-show-errors || true
}

sync_results() {
  for ep in ${EPOCHS_TO_PROBE}; do
    local outdir="/opt/dlami/nvme/evals/neurips/echomae_e${ep}_end_lvef"
    if [ -d "${outdir}" ]; then
      aws s3 sync "${outdir}" "${RUN_S3}/echomae_e${ep}_end_lvef/" \
        --exclude "*" \
        --include "*.pt" --include "*.pth" --include "*.yaml" --include "*.json" --include "*.csv" --include "*.npz" \
        --only-show-errors || true
    fi
  done
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
# SETUP (once for all probes)
# =========================
nvidia-smi -L || true
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

mkdir -p "${SCRATCH}/checkpoints"

# =========================
# RUN PROBES SEQUENTIALLY
# =========================
for EP in ${EPOCHS_TO_PROBE}; do
  echo "============================================================"
  echo "[INFO] Probing EchoMAE-L e${EP}"
  echo "============================================================"

  # Download checkpoint (e124-e184 from run 159, e189+ from run 179)
  if [ "${EP}" -le 184 ]; then
    CKPT_SRC="${CKPT_RUN_159}/checkpoint-${EP}.pth"
  else
    CKPT_SRC="${CKPT_RUN_179}/checkpoint-${EP}.pth"
  fi

  CKPT_LOCAL="${SCRATCH}/checkpoints/echomae_l_mimic_ep${EP}.pth"
  echo "[INFO] Downloading ${CKPT_SRC}..."
  aws s3 cp "${CKPT_SRC}" "${CKPT_LOCAL}"
  echo "[INFO] Checkpoint: $(ls -lh ${CKPT_LOCAL})"

  CONFIG="${CODE_DIR}/configs/eval/vitl/neurips/echomae_l_e${EP}_end_lvef_d4.yaml"

  # Need a fresh MASTER_PORT for each run
  export MASTER_PORT=$((12000 + $RANDOM % 20000))

  srun --kill-on-bad-exit=1 \
    python3 -m evals.main \
      --fname "${CONFIG}" \
      --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7
  PROBE_RC=$?

  echo "[INFO] e${EP} probe finished with rc=${PROBE_RC}"

  # Sync after each probe
  sync_results
  upload_logs

  # Clean up checkpoint to save disk
  rm -f "${CKPT_LOCAL}"
done

echo "[INFO] All trajectory probes complete"
