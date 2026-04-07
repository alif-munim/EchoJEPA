#!/bin/bash
#SBATCH -p ml-p5-48xlarge
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 0-04:00:00
#SBATCH -J salt_pavg
#SBATCH -o /tmp/salt_pavg-%j.out
#SBATCH -e /tmp/salt_pavg-%j.err
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

CFG_REL="configs/eval/vitl/neurips/salt_s2v3_echonet_lvef_d4_predavg.yaml"
RUN_S3="${S3_BASE}/runs/salt_s2v3_echonet_lvef_predavg_${SLURM_JOB_ID}"

# =========================
# LOGGING
# =========================
upload_logs() {
  aws s3 cp "/tmp/salt_pavg-${SLURM_JOB_ID}.out" "${RUN_S3}/logs/job.out" --only-show-errors 2>/dev/null || true
  aws s3 cp "/tmp/salt_pavg-${SLURM_JOB_ID}.err" "${RUN_S3}/logs/job.err" --only-show-errors 2>/dev/null || true
}

on_exit() {
  rc=$?
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

if [ ! -f "$CFG_REL" ]; then
    echo "[FATAL] Config $CFG_REL not found in $REPO_DIR"
    exit 1
fi

pip install -e . --no-deps --no-build-isolation 2>/dev/null || true
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# =========================
# STAGE DATA
# =========================
echo "[INFO] Downloading EchoNet-Dynamic test CSV..."
mkdir -p "$REPO_DIR/data/csv"
aws s3 cp "${S3_BASE}/data/csv/echonet_dynamic_train_s3_raw.csv" "$REPO_DIR/data/csv/echonet_dynamic_train_s3_raw.csv"
aws s3 cp "${S3_BASE}/data/csv/echonet_dynamic_test_s3_raw.csv" "$REPO_DIR/data/csv/echonet_dynamic_test_s3_raw.csv"

# =========================
# PROBE CHECKPOINT (from job 454)
# =========================
PROBE_DIR="$REPO_DIR/checkpoints/probes/salt-s2v3-echonet-lvef"
mkdir -p "$PROBE_DIR"
PROBE_S3="${S3_BASE}/runs/salt_s2v3_echonet_lvef_454/probe/best.pt"
if [ ! -f "$PROBE_DIR/best.pt" ]; then
  echo "[INFO] Downloading probe checkpoint from S3..."
  aws s3 cp "$PROBE_S3" "$PROBE_DIR/best.pt"
fi
echo "[INFO] Probe checkpoint:"
ls -lh "$PROBE_DIR/best.pt"

# =========================
# ENCODER CHECKPOINT
# The S2 checkpoint should still be on NVMe from the training run.
# =========================
S2_CKPT="/opt/dlami/nvme/checkpoints/pretrain/mimic/salt_s2_vitl_224px_16f/latest.pt"
if [ ! -f "$S2_CKPT" ]; then
  echo "[INFO] S2 v3 checkpoint not on NVMe, fetching from S3..."
  S2_S3=$(aws s3 ls "${S3_BASE}/runs/" --region us-west-2 | grep salt_s2v2_pretrain | sort | tail -1 | awk '{print $NF}')
  if [ -n "$S2_S3" ]; then
    mkdir -p "$(dirname "$S2_CKPT")"
    aws s3 cp "${S3_BASE}/runs/${S2_S3}checkpoints/latest.pt" "$S2_CKPT"
  else
    echo "[FATAL] No SALT S2 v3 checkpoint found"
    exit 1
  fi
fi
mkdir -p "$REPO_DIR/checkpoints/pretrain/mimic/salt_s2_vitl_224px_16f"
ln -sf "$S2_CKPT" "$REPO_DIR/checkpoints/pretrain/mimic/salt_s2_vitl_224px_16f/latest.pt"

# =========================
# RUN PRED-AVG ON TEST SET
# =========================
echo "[INFO] Launching SALT S2 v3 EchoNet-Dynamic LVEF pred-avg (test set)..."
python -m evals.main \
  --fname "$CFG_REL" \
  --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
  2>&1 | tee "salt_pavg_${SLURM_JOB_ID}.log"
RC=$?

upload_logs
exit $RC
