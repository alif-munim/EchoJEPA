#!/bin/bash
#SBATCH -p dev
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 6:00:00
#SBATCH -J mae_shuf
#SBATCH -o /tmp/mae_shuf_%j.out
#SBATCH -e /tmp/mae_shuf_%j.err
#SBATCH --signal=B:USR1@300

set -euxo pipefail

# =============================================================================
# EchoMAE-L trajectory → EchoNet-Dynamic LVEF test: clean + frame-shuffled
# =============================================================================
# For each checkpoint (e124, e149, e174, e194): run clean test, then shuffled test.
# Frame shuffle uses FRAME_SHUFFLE env var (seed=100, type=frame).
# Results go to separate output dirs for clean vs shuffled.
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
RUN_S3="${S3_BASE}/runs/mae_shuf_test_${SLURM_JOB_ID}"

# Probe checkpoints
PROBE_E194_S3="${S3_BASE}/runs/mae200_lvef_201/eval_output/video_classification_frozen/neurips-echomae-l-e200-end-lvef-d4/best.pt"
PROBE_E124_S3="${S3_BASE}/runs/mae_traj_lvef_202/echomae_e124_end_lvef/video_classification_frozen/neurips-echomae-l-e124-end-lvef-d4/best.pt"
PROBE_E149_S3="${S3_BASE}/runs/mae_traj_lvef_202/echomae_e149_end_lvef/video_classification_frozen/neurips-echomae-l-e149-end-lvef-d4/best.pt"
PROBE_E174_S3="${S3_BASE}/runs/mae_traj_lvef_202/echomae_e174_end_lvef/video_classification_frozen/neurips-echomae-l-e174-end-lvef-d4/best.pt"

# MAE encoder checkpoints
CKPT_RUN_159="${S3_BASE}/runs/videomae_e200_159/training_folder"
CKPT_RUN_179="${S3_BASE}/runs/videomae_e200b_179/training_folder"

EPOCHS="124 149 174 194"

# =========================
# LOGGING / SYNC
# =========================
upload_logs() {
  local out="/tmp/mae_shuf_${SLURM_JOB_ID}.out"
  local err="/tmp/mae_shuf_${SLURM_JOB_ID}.err"
  [ -f "${out}" ] && aws s3 cp "${out}" "${RUN_S3}/logs/job.out" --only-show-errors || true
  [ -f "${err}" ] && aws s3 cp "${err}" "${RUN_S3}/logs/job.err" --only-show-errors || true
}

sync_results() {
  for ep in ${EPOCHS}; do
    for cond in clean shuffled; do
      local outdir="/opt/dlami/nvme/evals/neurips/echomae_e${ep}_end_${cond}"
      if [ -d "${outdir}" ]; then
        aws s3 sync "${outdir}" "${RUN_S3}/echomae_e${ep}_end_${cond}/" \
          --exclude "*" \
          --include "*.csv" --include "*.json" --include "*.pt" --include "*.npz" \
          --only-show-errors || true
      fi
    done
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

# Data CSVs
mkdir -p "${SCRATCH}/data/csv"
aws s3 cp "${S3_BASE}/data/csv/echonet_dynamic_train_s3_raw.csv" "${SCRATCH}/data/csv/"
aws s3 cp "${S3_BASE}/data/csv/echonet_dynamic_test_s3_raw.csv" "${SCRATCH}/data/csv/"

mkdir -p "${SCRATCH}/checkpoints" "${SCRATCH}/probes"

# Download all probe checkpoints
echo "[INFO] Downloading probe checkpoints..."
for ep in ${EPOCHS}; do
  PROBE_VAR="PROBE_E${ep}_S3"
  PROBE_SRC="${!PROBE_VAR}"
  mkdir -p "${SCRATCH}/probes/echomae_e${ep}_end"
  aws s3 cp "${PROBE_SRC}" "${SCRATCH}/probes/echomae_e${ep}_end/best.pt" --quiet
done

# =========================
# RUN CLEAN + SHUFFLED FOR EACH CHECKPOINT
# =========================
for EP in ${EPOCHS}; do
  # Download encoder checkpoint
  if [ "${EP}" -le 184 ]; then
    CKPT_SRC="${CKPT_RUN_159}/checkpoint-${EP}.pth"
  else
    CKPT_SRC="${CKPT_RUN_179}/checkpoint-${EP}.pth"
  fi
  CKPT_LOCAL="${SCRATCH}/checkpoints/echomae_l_mimic_ep${EP}.pth"
  echo "[INFO] Downloading encoder e${EP}..."
  aws s3 cp "${CKPT_SRC}" "${CKPT_LOCAL}" --quiet

  # --- CLEAN ---
  echo "============================================================"
  echo "[INFO] CLEAN test: EchoMAE-L e${EP}"
  echo "============================================================"

  # Override output folder and tag via env vars
  OVERRIDE_TAG="neurips-echomae-l-e${EP}-end-clean" \
  OVERRIDE_PRED_PATH="/opt/dlami/nvme/evals/neurips/echomae_e${EP}_end_clean/predictions.csv" \
  python3 -m evals.main \
    --fname "${CODE_DIR}/configs/inference/vitl/neurips/echonet-dynamic/echomae_e${EP}_end_test.yaml" \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7
  echo "[INFO] e${EP} clean done (rc=$?)"

  # --- SHUFFLED ---
  echo "============================================================"
  echo "[INFO] SHUFFLED test: EchoMAE-L e${EP}"
  echo "============================================================"

  FRAME_SHUFFLE=100 \
  FRAME_SHUFFLE_TYPE=frame \
  OVERRIDE_TAG="neurips-echomae-l-e${EP}-end-shuffled" \
  OVERRIDE_PRED_PATH="/opt/dlami/nvme/evals/neurips/echomae_e${EP}_end_shuffled/predictions.csv" \
  python3 -m evals.main \
    --fname "${CODE_DIR}/configs/inference/vitl/neurips/echonet-dynamic/echomae_e${EP}_end_test.yaml" \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7
  echo "[INFO] e${EP} shuffled done (rc=$?)"

  sync_results
  upload_logs

  # Clean up encoder checkpoint
  rm -f "${CKPT_LOCAL}"
done

echo "[INFO] All clean + shuffled test inference complete"

# =========================
# PRINT SUMMARY
# =========================
echo ""
echo "============================================================"
echo "SUMMARY: Clean vs Shuffled R²"
echo "============================================================"
for EP in ${EPOCHS}; do
  for cond in clean shuffled; do
    LOGDIR="/opt/dlami/nvme/evals/neurips/echomae_e${EP}_end_${cond}"
    CSV=$(find "${LOGDIR}" -name "log_r0.csv" 2>/dev/null | head -1)
    if [ -n "${CSV}" ]; then
      R2=$(tail -1 "${CSV}" | cut -d',' -f4)
      echo "  e${EP} ${cond}: R²=${R2}"
    fi
  done
done
