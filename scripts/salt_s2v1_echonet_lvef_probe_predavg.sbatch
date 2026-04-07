#!/bin/bash
#SBATCH -p ml-p5-48xlarge
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 1-00:00:00
#SBATCH -J salt1_lvef
#SBATCH -o /tmp/salt1_lvef-%j.out
#SBATCH -e /tmp/salt1_lvef-%j.err
#SBATCH --chdir=/tmp
#SBATCH --nodelist=ip-10-0-50-184

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

RUN_S3="${S3_BASE}/runs/salt_s2v1_echonet_lvef_${SLURM_JOB_ID}"

# =========================
# LOGGING
# =========================
upload_logs() {
  aws s3 cp "/tmp/salt1_lvef-${SLURM_JOB_ID}.out" "${RUN_S3}/logs/job.out" --only-show-errors 2>/dev/null || true
  aws s3 cp "/tmp/salt1_lvef-${SLURM_JOB_ID}.err" "${RUN_S3}/logs/job.err" --only-show-errors 2>/dev/null || true
}

on_exit() {
  rc=$?
  # Upload probe checkpoint
  PROBE_BEST="/opt/vjepa2/evals/vitl/neurips/salt_s2v1_echonet_lvef_d4/video_classification_frozen/neurips-salt-s2v1-echonet-lvef-d4/best.pt"
  if [ -f "$PROBE_BEST" ]; then
    aws s3 cp "$PROBE_BEST" "${RUN_S3}/probe/best.pt" --only-show-errors 2>/dev/null || true
  fi
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

pip install -e . --no-deps --no-build-isolation 2>/dev/null || true
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# =========================
# STAGE DATA
# =========================
echo "[INFO] Downloading EchoNet-Dynamic CSVs..."
mkdir -p "$REPO_DIR/data/csv"
aws s3 cp "${S3_BASE}/data/csv/echonet_dynamic_train_s3_raw.csv" "$REPO_DIR/data/csv/echonet_dynamic_train_s3_raw.csv"
aws s3 cp "${S3_BASE}/data/csv/echonet_dynamic_val_s3_raw.csv" "$REPO_DIR/data/csv/echonet_dynamic_val_s3_raw.csv"
aws s3 cp "${S3_BASE}/data/csv/echonet_dynamic_test_s3_raw.csv" "$REPO_DIR/data/csv/echonet_dynamic_test_s3_raw.csv"

# =========================
# SALT S2 v1 ENCODER (job 388, e79)
# =========================
S2V1_CKPT="$REPO_DIR/checkpoints/pretrain/mimic/salt_s2v1_e79.pt"
if [ ! -f "$S2V1_CKPT" ]; then
  echo "[INFO] Downloading SALT S2 v1 e79 checkpoint..."
  mkdir -p "$(dirname "$S2V1_CKPT")"
  aws s3 cp "${S3_BASE}/runs/salt_s2_pretrain_388/checkpoints/e79.pt" "$S2V1_CKPT"
fi
echo "[INFO] SALT S2 v1 checkpoint:"
ls -lh "$S2V1_CKPT"

# =========================
# PHASE 1: Train probe (20 epochs on val set)
# =========================
echo "[INFO] === PHASE 1: Training SALT S2 v1 probe ==="
python -m evals.main \
  --fname configs/eval/vitl/neurips/salt_s2v1_echonet_lvef_d4.yaml \
  --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
  2>&1 | tee "salt1_probe_${SLURM_JOB_ID}.log"

# Copy probe best.pt to expected location for phase 2
PROBE_BEST="$REPO_DIR/evals/vitl/neurips/salt_s2v1_echonet_lvef_d4/video_classification_frozen/neurips-salt-s2v1-echonet-lvef-d4/best.pt"
PROBE_DIR="$REPO_DIR/checkpoints/probes/salt-s2v1-echonet-lvef"
mkdir -p "$PROBE_DIR"
cp "$PROBE_BEST" "$PROBE_DIR/best.pt"

# Upload probe to S3
aws s3 cp "$PROBE_BEST" "${RUN_S3}/probe/best.pt" --only-show-errors || true

upload_logs

# =========================
# PHASE 2: Pred-avg on test set
# =========================
echo "[INFO] === PHASE 2: Pred-avg on EchoNet-Dynamic test set ==="

# Increment MASTER_PORT to avoid conflict
export MASTER_PORT=$((29500 + 1))

python -m evals.main \
  --fname configs/eval/vitl/neurips/salt_s2v1_echonet_lvef_d4_predavg.yaml \
  --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
  2>&1 | tee "salt1_predavg_${SLURM_JOB_ID}.log"

upload_logs
