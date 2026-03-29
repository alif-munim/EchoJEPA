#!/bin/bash
#SBATCH -p ml-p5-48xlarge
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 0-04:00:00
#SBATCH -J mae_test
#SBATCH -o /tmp/mae_test-%j.out
#SBATCH -e /tmp/mae_test-%j.err
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

# =========================
# S3 CONFIGURATION
# =========================
ART_BUCKET="sagemaker-hyperpod-lifecycle-495467399120-usw2"
S3_BASE="s3://${ART_BUCKET}/vjepa2-artifacts"

ENV_S3="${S3_BASE}/env/vjepa2-312-pack-env.tar"

# Config (relative to repo root)
CFG_REL="configs/inference/vitl/icml/echomae_l_pt50_lvef_test.yaml"

# Test CSV (53K clips)
TEST_CSV_S3="${S3_BASE}/data/csv/a4c_b_lvef_test_224px.csv"

# Backbone checkpoint
ENC_CKPT_S3="${S3_BASE}/runs/videomae_matched_2n_245/training_folder/checkpoint-49.pth"
ENC_CKPT_LOCAL="/opt/dlami/nvme/checkpoints/echomae_l_pt50.pth"

# Probe checkpoint (best from training job 247)
PROBE_CKPT_S3="${S3_BASE}/runs/echomae_pt50_lvef_247/training_folder/video_classification_frozen/icml-echomae-l-pt50-lvef-d4/best.pt"
PROBE_CKPT_LOCAL="/opt/dlami/nvme/evals/icml/echomae_pt50_lvef/video_classification_frozen/icml-echomae-l-pt50-lvef-d4/best.pt"

# Run output
RUN_S3="${S3_BASE}/runs/echomae_pt50_lvef_test_${SLURM_JOB_ID}"

# =========================
# LOGGING
# =========================
upload_logs() {
  aws s3 cp "/tmp/mae_test-${SLURM_JOB_ID}.out" "${RUN_S3}/logs/job.out" --only-show-errors 2>/dev/null || true
  aws s3 cp "/tmp/mae_test-${SLURM_JOB_ID}.err" "${RUN_S3}/logs/job.err" --only-show-errors 2>/dev/null || true
}
trap 'rc=$?; upload_logs; exit $rc' EXIT

# =========================
# HOST SETUP
# =========================
echo "[INFO] host=$(hostname) job=${SLURM_JOB_ID}"
nvidia-smi -L || true

SCRATCH="/opt/dlami/nvme"
[ ! -d "$SCRATCH" ] && SCRATCH="/tmp"
WORKDIR="${SCRATCH}/mae_test_${SLURM_JOB_ID}"
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
# CODE (use deploy.sh-deployed code at /opt/vjepa2)
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
# DOWNLOAD DATA & CHECKPOINTS
# =========================
echo "[INFO] Downloading test CSV..."
mkdir -p /opt/dlami/nvme/data/csv
aws s3 cp "$TEST_CSV_S3" /opt/dlami/nvme/data/csv/a4c_b_lvef_test_224px.csv

echo "[INFO] Downloading backbone checkpoint..."
mkdir -p "$(dirname "$ENC_CKPT_LOCAL")"
aws s3 cp "$ENC_CKPT_S3" "$ENC_CKPT_LOCAL"

echo "[INFO] Downloading probe checkpoint..."
mkdir -p "$(dirname "$PROBE_CKPT_LOCAL")"
aws s3 cp "$PROBE_CKPT_S3" "$PROBE_CKPT_LOCAL"

# =========================
# OUTPUT FOLDER
# =========================
CFG_FOLDER="/opt/dlami/nvme/evals/icml/echomae_l_pt50_lvef_test"
mkdir -p "$CFG_FOLDER"
mkdir -p /opt/dlami/nvme/predictions/icml

# =========================
# RUN TEST INFERENCE
# =========================
GPU_COUNT=$(nvidia-smi -L | wc -l)
DEVICES=()
for ((i=0; i<GPU_COUNT; i++)); do DEVICES+=("cuda:${i}"); done

echo "[INFO] Launching test inference on ${GPU_COUNT} GPUs..."

set +e
python -m evals.main \
  --fname "$CFG_REL" \
  --devices "${DEVICES[@]}" \
  2>&1 | tee "test_inference_${SLURM_JOB_ID}.log"
RC=${PIPESTATUS[0]}
set -e

# Final upload
echo "[INFO] Uploading results..."
aws s3 sync "$CFG_FOLDER" "${RUN_S3}/training_folder/" \
  --include "*.csv" --include "*.log" --include "*.json" --include "*.pt" \
  --only-show-errors || true

# Upload predictions CSV
aws s3 cp /opt/dlami/nvme/predictions/icml/echomae_l_pt50_lvef_test.csv \
  "${RUN_S3}/predictions/echomae_l_pt50_lvef_test.csv" --only-show-errors 2>/dev/null || true

aws s3 cp "test_inference_${SLURM_JOB_ID}.log" "${RUN_S3}/logs/test_inference.log" --only-show-errors || true

if [ "$RC" -ne 0 ]; then
  echo "[ERROR] Test inference failed with exit code ${RC}"
  exit "$RC"
fi

echo "[INFO] DONE. Results at: ${RUN_S3}"
