#!/bin/bash
#SBATCH -p ml-p5-48xlarge
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 1-00:00:00
#SBATCH -J byol_e100_probe
#SBATCH -o /tmp/byol_e100_probe-%j.out
#SBATCH -e /tmp/byol_e100_probe-%j.err
#SBATCH --chdir=/tmp

set -euxo pipefail

export PATH="/opt/vjepa2-312/bin:$PATH"

# AWS CONFIG
export AWS_DEFAULT_REGION="us-west-2"
export AWS_REGION="us-west-2"
export AWS_SDK_LOAD_CONFIG=1
export HOME="/home/ubuntu"
export AWS_SHARED_CREDENTIALS_FILE="${HOME}/.aws/credentials"
export AWS_CONFIG_FILE="${HOME}/.aws/config"
unset AWS_EC2_METADATA_DISABLED 2>/dev/null || true

export LD_LIBRARY_PATH="/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cublas/lib:/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"

ART_BUCKET="sagemaker-hyperpod-lifecycle-495467399120-usw2"
S3_BASE="s3://${ART_BUCKET}/vjepa2-artifacts"
RUN_S3="${S3_BASE}/runs/neurips_byol_e100_end_lvef_${SLURM_JOB_ID}"

# S3 sync
EVAL_DIR="/opt/dlami/nvme/evals/neurips/echobyol_e100_end_lvef"
sync_results() {
  [ -d "$EVAL_DIR" ] && aws s3 sync "$EVAL_DIR" "${RUN_S3}/eval/" --only-show-errors || true
}
upload_logs() {
  aws s3 cp "/tmp/byol_e100_probe-${SLURM_JOB_ID}.out" "${RUN_S3}/logs/job.out" --only-show-errors 2>/dev/null || true
  aws s3 cp "/tmp/byol_e100_probe-${SLURM_JOB_ID}.err" "${RUN_S3}/logs/job.err" --only-show-errors 2>/dev/null || true
}
trap 'sync_results; upload_logs' EXIT

# HOST SETUP
echo "[INFO] host=$(hostname) job=${SLURM_JOB_ID}"
nvidia-smi -L || true
rm -rf /dev/shm/* 2>/dev/null || true

# CODE
cd /opt/vjepa2
pip install -e . --no-deps --no-build-isolation 2>/dev/null || true
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# STAGE DATA + CHECKPOINT
mkdir -p /opt/dlami/nvme/data/csv /opt/dlami/nvme/checkpoints
aws s3 cp "${S3_BASE}/data/csv/echonet_dynamic_train_s3_raw.csv" /opt/dlami/nvme/data/csv/
aws s3 cp "${S3_BASE}/data/csv/echonet_dynamic_val_s3_raw.csv" /opt/dlami/nvme/data/csv/

echo "[INFO] Downloading BYOL e100 checkpoint..."
aws s3 cp "${S3_BASE}/runs/byol_pretrain_resume_342/checkpoints/e100.pt" /opt/dlami/nvme/checkpoints/echobyol_l_e100.pt

mkdir -p "$EVAL_DIR"

# RUN PROBE TRAINING
echo "[INFO] Training EchoNet-Dynamic LVEF probe on BYOL e100..."
python -m evals.main \
  --fname configs/eval/vitl/neurips/echobyol_l_e100_end_lvef_d4.yaml \
  --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7

sync_results
upload_logs
