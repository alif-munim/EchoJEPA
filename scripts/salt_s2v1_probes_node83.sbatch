#!/bin/bash
#SBATCH -p ml-p5-48xlarge
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 1-00:00:00
#SBATCH -J salt1_ab
#SBATCH -o /tmp/salt1_ab-%j.out
#SBATCH -e /tmp/salt1_ab-%j.err
#SBATCH --chdir=/tmp
#SBATCH -w ip-10-0-50-83

set -euxo pipefail

# SALT S2 v1 probes: e4 + e29 on node 83

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
RUN_S3="${S3_BASE}/runs/salt_s2v1_probes_83_${SLURM_JOB_ID}"

upload_logs() {
  aws s3 cp "/tmp/salt1_ab-${SLURM_JOB_ID}.out" "${RUN_S3}/logs/job.out" --only-show-errors 2>/dev/null || true
  aws s3 cp "/tmp/salt1_ab-${SLURM_JOB_ID}.err" "${RUN_S3}/logs/job.err" --only-show-errors 2>/dev/null || true
}
on_exit() {
  rc=$?
  for ep in e4 e29; do
    PROBE_DIR="/opt/vjepa2/evals/vitl/neurips/salt_s2v1_${ep}_echonet_lvef_d4"
    [ -d "$PROBE_DIR" ] && aws s3 sync "$PROBE_DIR" "${RUN_S3}/probes/${ep}/" \
      --exclude "*" --include "*.pt" --include "*.csv" --include "*.log" \
      --only-show-errors 2>/dev/null || true
  done
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

SCRATCH="/opt/dlami/nvme"
CKPT_DIR="${SCRATCH}/salt_s2v1_dynamics"
LINK_DIR="$REPO_DIR/checkpoints/pretrain/mimic"
mkdir -p "$CKPT_DIR" "$LINK_DIR"

for ep in e4 e29; do
  DEST="${CKPT_DIR}/salt_s2v1_${ep}.pt"
  LINK="${LINK_DIR}/salt_s2v1_${ep}.pt"
  [ ! -f "$DEST" ] && aws s3 cp "${S3_CKPTS}/${ep}.pt" "$DEST"
  ln -sf "$DEST" "$LINK"
  echo "[INFO] ${ep}: $(ls -lh "$LINK")"
done

echo "================================================================"
echo "[INFO] === PHASE 1/2: SALT S2 v1 e4 (total=24) ==="
echo "================================================================"
export MASTER_PORT=29500
python -m evals.main \
  --fname configs/eval/vitl/neurips/salt_s2v1_e4_echonet_lvef_d4.yaml \
  --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
  2>&1 | tee "/tmp/salt1_e4_probe_${SLURM_JOB_ID}.log"
upload_logs

echo "================================================================"
echo "[INFO] === PHASE 2/2: SALT S2 v1 e29 (total=49) ==="
echo "================================================================"
export MASTER_PORT=29501
python -m evals.main \
  --fname configs/eval/vitl/neurips/salt_s2v1_e29_echonet_lvef_d4.yaml \
  --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
  2>&1 | tee "/tmp/salt1_e29_probe_${SLURM_JOB_ID}.log"
upload_logs

echo "[INFO] Node 83 done: e4 + e29 probes complete."
