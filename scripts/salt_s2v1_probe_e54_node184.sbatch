#!/bin/bash
#SBATCH -p ml-p5-48xlarge
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 1-00:00:00
#SBATCH -J salt1_e54
#SBATCH -o /tmp/salt1_e54-%j.out
#SBATCH -e /tmp/salt1_e54-%j.err
#SBATCH --chdir=/tmp
#SBATCH -w ip-10-0-50-184

set -euxo pipefail

# SALT S2 v1 probe: e54 on node 184 (with NVMe eval output to avoid disk full)

export PATH="/opt/vjepa2-312/bin:$PATH"

export AWS_DEFAULT_REGION="us-west-2"
export AWS_REGION="us-west-2"
export AWS_SDK_LOAD_CONFIG=1
export HOME="/home/ubuntu"
export AWS_SHARED_CREDENTIALS_FILE="${HOME}/.aws/credentials"
export AWS_CONFIG_FILE="${HOME}/.aws/config"
unset AWS_EC2_METADATA_DISABLED 2>/dev/null || true
export AWS_METADATA_SERVICE_TIMEOUT=5
export AWS_METADATA_SERVICE_NUM_ATTEMPTS=3

export LD_LIBRARY_PATH="/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cublas/lib:/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"

ART_BUCKET="sagemaker-hyperpod-lifecycle-495467399120-usw2"
S3_BASE="s3://${ART_BUCKET}/vjepa2-artifacts"
S3_CKPTS="${S3_BASE}/runs/salt_s2_pretrain_388/checkpoints"
RUN_S3="${S3_BASE}/runs/salt_s2v1_probe_e54_${SLURM_JOB_ID}"

upload_logs() {
  aws s3 cp "/tmp/salt1_e54-${SLURM_JOB_ID}.out" "${RUN_S3}/logs/job.out" --only-show-errors 2>/dev/null || true
  aws s3 cp "/tmp/salt1_e54-${SLURM_JOB_ID}.err" "${RUN_S3}/logs/job.err" --only-show-errors 2>/dev/null || true
}
on_exit() {
  rc=$?
  PROBE_DIR="/opt/dlami/nvme/salt_evals/salt_s2v1_e54_echonet_lvef_d4"
  [ -d "$PROBE_DIR" ] && aws s3 sync "$PROBE_DIR" "${RUN_S3}/probes/e54/" \
    --exclude "*" --include "*.pt" --include "*.csv" --include "*.log" \
    --only-show-errors 2>/dev/null || true
  upload_logs
  exit $rc
}
trap on_exit EXIT

nvidia-smi -L || true
rm -rf /dev/shm/* 2>/dev/null || true

REPO_DIR="/opt/vjepa2"
cd "$REPO_DIR"
pip install -e . --no-deps --no-build-isolation 2>/dev/null || true
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

mkdir -p "$REPO_DIR/data/csv"
aws s3 cp "${S3_BASE}/data/csv/echonet_dynamic_train_s3_raw.csv" "$REPO_DIR/data/csv/echonet_dynamic_train_s3_raw.csv"
aws s3 cp "${S3_BASE}/data/csv/echonet_dynamic_val_s3_raw.csv" "$REPO_DIR/data/csv/echonet_dynamic_val_s3_raw.csv"

# Download checkpoint to NVMe
SCRATCH="/opt/dlami/nvme"
CKPT_DIR="${SCRATCH}/salt_s2v1_dynamics"
LINK_DIR="$REPO_DIR/checkpoints/pretrain/mimic"
mkdir -p "$CKPT_DIR" "$LINK_DIR"

DEST="${CKPT_DIR}/salt_s2v1_e54.pt"
LINK="${LINK_DIR}/salt_s2v1_e54.pt"
[ ! -f "$DEST" ] && aws s3 cp "${S3_CKPTS}/e54.pt" "$DEST"
ln -sf "$DEST" "$LINK"

# CRITICAL: Symlink eval output dir to NVMe to avoid disk full
EVAL_NVME="${SCRATCH}/salt_evals/salt_s2v1_e54_echonet_lvef_d4"
EVAL_REPO="$REPO_DIR/evals/vitl/neurips/salt_s2v1_e54_echonet_lvef_d4"
mkdir -p "$EVAL_NVME"
rm -rf "$EVAL_REPO" 2>/dev/null || true
mkdir -p "$(dirname "$EVAL_REPO")"
ln -sf "$EVAL_NVME" "$EVAL_REPO"
echo "[INFO] Eval output symlinked: $EVAL_REPO -> $EVAL_NVME"

echo "================================================================"
echo "[INFO] === SALT S2 v1 e54 (total=74) ==="
echo "================================================================"
export MASTER_PORT=29500
python -m evals.main \
  --fname configs/eval/vitl/neurips/salt_s2v1_e54_echonet_lvef_d4.yaml \
  --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
  2>&1 | tee "/tmp/salt1_e54_probe_${SLURM_JOB_ID}.log"
upload_logs

echo "[INFO] Node 184 done: e54 probe complete."
