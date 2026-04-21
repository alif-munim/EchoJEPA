#!/bin/bash
#SBATCH -p dev
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 6:00:00
#SBATCH -J all_mf
#SBATCH -o /tmp/all_mf_%j.out
#SBATCH -e /tmp/all_mf_%j.err
#SBATCH --signal=B:USR1@300

set -euxo pipefail

# =============================================================================
# All 4 models × 4 epochs: clean + matched_frame on EchoNet-Dynamic LVEF test
# =============================================================================
# JEPA (e25/50/75/100), BYOL (e24/50/75/100), SALT (e4/29/54/79)
# MAE already done in job 216 — skip here.
# evals.main, num_segments=2, matched_frame with RoPE remap.
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

aws sts get-caller-identity || exit 1

ART_BUCKET="sagemaker-hyperpod-lifecycle-495467399120-usw2"
S3_BASE="s3://${ART_BUCKET}/vjepa2-artifacts"
ENV_S3="${S3_BASE}/env/vjepa2-312-pack-env.tar"
RUN_S3="${S3_BASE}/runs/all_mf_traj_${SLURM_JOB_ID}"

upload_logs() {
  [ -f "/tmp/all_mf_${SLURM_JOB_ID}.out" ] && aws s3 cp "/tmp/all_mf_${SLURM_JOB_ID}.out" "${RUN_S3}/logs/job.out" --only-show-errors || true
  [ -f "/tmp/all_mf_${SLURM_JOB_ID}.err" ] && aws s3 cp "/tmp/all_mf_${SLURM_JOB_ID}.err" "${RUN_S3}/logs/job.err" --only-show-errors || true
}
trap 'upload_logs' EXIT

# =========================
# SETUP
# =========================
nvidia-smi -L || true
pkill -9 -f "evals.main" 2>/dev/null || true; sleep 2

SCRATCH="/opt/dlami/nvme"

aws s3 cp "$ENV_S3" "${SCRATCH}/env.tar"
mkdir -p "${SCRATCH}/env" && tar -xf "${SCRATCH}/env.tar" -C "${SCRATCH}/env"
set +u && source "${SCRATCH}/env/bin/activate" && set -u
"${SCRATCH}/env/bin/conda-unpack" || true
pip install boto3 --no-cache-dir --quiet

CODE_DIR="${SCRATCH}/code"
mkdir -p "$CODE_DIR"
aws s3 cp "${S3_BASE}/setup/vjepa2-src.tar.gz" "${SCRATCH}/vjepa2-src.tar.gz" --quiet
tar xzf "${SCRATCH}/vjepa2-src.tar.gz" -C "$CODE_DIR"
cd "$CODE_DIR"
pip install -e . --no-deps --no-build-isolation 2>/dev/null || true

mkdir -p "${SCRATCH}/data/csv" "${SCRATCH}/checkpoints" "${SCRATCH}/probes"
aws s3 cp "${S3_BASE}/data/csv/echonet_dynamic_train_s3_raw.csv" "${SCRATCH}/data/csv/"
aws s3 cp "${S3_BASE}/data/csv/echonet_dynamic_test_s3_raw.csv" "${SCRATCH}/data/csv/"

# =========================
# HELPER FUNCTION
# =========================
run_clean_and_mf() {
  local MODEL_NAME="$1"
  local EP="$2"
  local CKPT_S3="$3"
  local PROBE_S3="$4"
  local CONFIG="$5"
  local CKPT_LOCAL="$6"
  local PROBE_DIR="$7"

  echo "============================================================"
  echo "[INFO] ${MODEL_NAME} e${EP}"
  echo "============================================================"

  aws s3 cp "${CKPT_S3}" "${CKPT_LOCAL}" --quiet
  mkdir -p "${PROBE_DIR}"
  aws s3 cp "${PROBE_S3}" "${PROBE_DIR}/best.pt" --quiet

  # CLEAN
  echo "[INFO] CLEAN: ${MODEL_NAME} e${EP}"
  python3 -m evals.main --fname "${CONFIG}" \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7
  echo "[INFO] ${MODEL_NAME} e${EP} clean done"

  # MATCHED_FRAME
  echo "[INFO] MATCHED_FRAME: ${MODEL_NAME} e${EP}"
  FRAME_SHUFFLE=100 FRAME_SHUFFLE_TYPE=matched_frame \
  python3 -m evals.main --fname "${CONFIG}" \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7
  echo "[INFO] ${MODEL_NAME} e${EP} matched_frame done"

  upload_logs
  rm -f "${CKPT_LOCAL}"
}

# =========================
# JEPA IN21K (e25, e50, e75, e100)
# =========================
echo ">>> JEPA IN21K"
for EP in 25 50 75 100; do
  run_clean_and_mf "JEPA" "${EP}" \
    "${S3_BASE}/checkpoints/jepa_in21k/jepa_in21k_vitl_e${EP}.pt" \
    "${S3_BASE}/probes/jepa_in21k_e${EP}_end_lvef/best.pt" \
    "${CODE_DIR}/configs/inference/vitl/neurips/echonet-dynamic/jepa_in21k_e${EP}_end_test.yaml" \
    "${SCRATCH}/checkpoints/jepa_in21k_vitl_e${EP}.pt" \
    "${SCRATCH}/probes/jepa_in21k_e${EP}_end"
done

# =========================
# BYOL (e24, e50, e75, e100)
# =========================
echo ">>> BYOL"
for EP in 24 50 75 100; do
  run_clean_and_mf "BYOL" "${EP}" \
    "${S3_BASE}/checkpoints/byol/byol_vitl_imagenet_v2_e${EP}.pt" \
    "${S3_BASE}/probes/byol_e${EP}_end_lvef/best.pt" \
    "${CODE_DIR}/configs/inference/vitl/neurips/echonet-dynamic/byol_e${EP}_end_test.yaml" \
    "${SCRATCH}/checkpoints/byol_vitl_imagenet_v2_e${EP}.pt" \
    "${SCRATCH}/probes/byol_e${EP}_end"
done

# =========================
# SALT S2v1 (e4, e29, e54, e79)
# =========================
echo ">>> SALT"
for EP in 4 29 54 79; do
  run_clean_and_mf "SALT" "${EP}" \
    "${S3_BASE}/checkpoints/salt_v1/salt_s2v1_e${EP}.pt" \
    "${S3_BASE}/probes/salt_e${EP}_end_lvef/best.pt" \
    "${CODE_DIR}/configs/inference/vitl/neurips/echonet-dynamic/salt_s2v1_e${EP}_end_test.yaml" \
    "${SCRATCH}/checkpoints/salt_s2v1_e${EP}.pt" \
    "${SCRATCH}/probes/salt_e${EP}_end"
done

echo ""
echo "============================================================"
echo "[INFO] All 12 model-epoch combos × 2 conditions complete"
echo "============================================================"
