#!/bin/bash
#SBATCH -p dev
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 1-00:00:00
#SBATCH -J cmr_probe
#SBATCH -o /tmp/cmr_probe_%j.out
#SBATCH -e /tmp/cmr_probe_%j.err
#SBATCH --signal=B:USR1@300

set -euxo pipefail

# =============================================================================
# VideoMAE ViT-S CMR trajectory → ACDC LVEF probes
# =============================================================================
# Trains d=4 attentive probes at e30, e50, e100, e200, e300, e400, e600, e800
# on the ACDC holdout set (951 train / 538 test, EF labels).
# Tests whether MAE temporal shortcut manifests in CMR domain.
# =============================================================================

export AWS_DEFAULT_REGION="us-west-2"
export AWS_REGION="us-west-2"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

export AWS_SDK_LOAD_CONFIG=1
export HOME="/home/ubuntu"
export AWS_SHARED_CREDENTIALS_FILE="${HOME}/.aws/credentials"
export AWS_CONFIG_FILE="${HOME}/.aws/config"
unset AWS_EC2_METADATA_DISABLED 2>/dev/null || true
export AWS_METADATA_SERVICE_TIMEOUT=5
export AWS_METADATA_SERVICE_NUM_ATTEMPTS=3

echo "[PREFLIGHT] Checking AWS credentials..."
aws sts get-caller-identity || { echo "[FATAL] AWS CLI credentials check failed"; exit 1; }
echo "[PREFLIGHT] OK"

# =========================
# S3 CONFIG
# =========================
ART_BUCKET="sagemaker-hyperpod-lifecycle-495467399120-usw2"
S3_BASE="s3://${ART_BUCKET}/vjepa2-artifacts"
ENV_S3="${S3_BASE}/env/vjepa2-312-pack-env.tar"
CKPT_S3="${S3_BASE}/runs/mae_cmr_vits_183/training_folder"
RUN_S3="${S3_BASE}/runs/cmr_probe_traj_${SLURM_JOB_ID}"

# checkpoint-N is 0-indexed: e30=ckpt-29, e50=ckpt-49, etc.
EPOCHS="30 50 100 200 300 400 600 800"

# =========================
# LOGGING / SYNC
# =========================
upload_logs() {
  local out="/tmp/cmr_probe_${SLURM_JOB_ID}.out"
  local err="/tmp/cmr_probe_${SLURM_JOB_ID}.err"
  [ -f "${out}" ] && aws s3 cp "${out}" "${RUN_S3}/logs/job.out" --only-show-errors || true
  [ -f "${err}" ] && aws s3 cp "${err}" "${RUN_S3}/logs/job.err" --only-show-errors || true
}

sync_results() {
  for ep in ${EPOCHS}; do
    local outdir="/opt/dlami/nvme/evals/cmr/mae_e${ep}_acdc_lvef"
    if [ -d "${outdir}" ]; then
      aws s3 sync "${outdir}" "${RUN_S3}/mae_e${ep}_acdc_lvef/" \
        --exclude "*" \
        --include "*.csv" --include "*.json" --include "*.pt" --include "*.npz" \
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
# SETUP
# =========================
nvidia-smi -L || true

pkill -9 -f "evals.main" 2>/dev/null || true
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

# Safety: patch torch.load for VideoMAE checkpoints
find . -name "*.py" -path "*/modelcustom/*" -exec grep -l "torch.load(" {} \; | while read f; do
    sed -i 's/torch\.load(\([^)]*\))/torch.load(\1, weights_only=False)/g' "$f"
    sed -i 's/weights_only=False, weights_only=False/weights_only=False/g' "$f"
done

# Data CSVs
mkdir -p "${SCRATCH}/data/csv"
aws s3 cp "${S3_BASE}/data/csv/acdc_train_s3.csv" "${SCRATCH}/data/csv/"
aws s3 cp "${S3_BASE}/data/csv/acdc_test_s3.csv" "${SCRATCH}/data/csv/"

mkdir -p "${SCRATCH}/checkpoints"

# =========================
# RUN PROBES SEQUENTIALLY
# =========================
for EP in ${EPOCHS}; do
  CKPT_IDX=$((EP - 1))  # 0-indexed
  echo "============================================================"
  echo "[INFO] Probing CMR MAE ViT-S e${EP} (checkpoint-${CKPT_IDX})"
  echo "============================================================"

  CKPT_LOCAL="${SCRATCH}/checkpoints/mae_cmr_vits_ep${CKPT_IDX}.pth"
  echo "[INFO] Downloading checkpoint-${CKPT_IDX}..."
  aws s3 cp "${CKPT_S3}/checkpoint-${CKPT_IDX}.pth" "${CKPT_LOCAL}" --quiet
  echo "[INFO] Checkpoint: $(ls -lh ${CKPT_LOCAL})"

  CONFIG="${CODE_DIR}/configs/eval/vits16/cmr/mae_cmr_e${EP}_acdc_lvef_d4.yaml"

  # evals.main uses mp.Process — no srun
  python3 -m evals.main \
    --fname "${CONFIG}" \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7
  PROBE_RC=$?

  echo "[INFO] e${EP} probe finished with rc=${PROBE_RC}"

  sync_results
  upload_logs

  # Clean up checkpoint
  rm -f "${CKPT_LOCAL}"
done

echo "[INFO] All CMR MAE trajectory probes complete"
