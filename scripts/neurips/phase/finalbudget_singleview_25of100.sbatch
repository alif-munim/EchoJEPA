#!/bin/bash
# Single-view JEPA finalbudget +25 of 100. Compute-matched control.
#SBATCH -p ml-p5-48xlarge
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 0-16:00:00
#SBATCH -J final_sv_25
#SBATCH -o /opt/dlami/nvme/logs/final_sv_25-%j.out
#SBATCH -e /opt/dlami/nvme/logs/final_sv_25-%j.err
#SBATCH --chdir=/opt/dlami/nvme

set -euxo pipefail

export PATH="/opt/vjepa2-312/bin:$PATH"
export TMPDIR="/opt/dlami/nvme/tmp"
mkdir -p "$TMPDIR" /opt/dlami/nvme/logs

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
E100_CKPT_S3="${S3_BASE}/checkpoints/jepa_in21k/jepa_in21k_vitl_e100.pt"
CSV_S3="${S3_BASE}/data/csv/mimic_annotations_s3.csv"
SRC_TARBALL_S3="${S3_BASE}/setup/vjepa2-src.tar.gz"

REPO_DIR="/opt/dlami/nvme/src/vjepa2"
CFG_REL="configs/train/vitl16/pretrain-singleview-highmedium-finalbudget-25of100.yaml"
RUN_S3="${S3_BASE}/runs/final_sv_25_pairs24_${SLURM_JOB_ID}"
CKPT_DIR="/opt/dlami/nvme/checkpoints/finalbudget_pairs24/singleview_25of100"
CSV_DIR="/opt/dlami/nvme/data/csv"
LOG_OUT="/opt/dlami/nvme/logs/final_sv_25-${SLURM_JOB_ID}.out"
LOG_ERR="/opt/dlami/nvme/logs/final_sv_25-${SLURM_JOB_ID}.err"
JOB_LOG="/opt/dlami/nvme/logs/final_sv_25_${SLURM_JOB_ID}.log"

sync_ckpts() {
  [ -d "$CKPT_DIR" ] && aws s3 sync "$CKPT_DIR" "${RUN_S3}/checkpoints/" \
    --exclude "*" --include "e*.pt" --include "latest.pt" --include "*.csv" --include "*.yaml" \
    --only-show-errors || true
}
upload_logs() {
  [ -f "$LOG_OUT" ] && aws s3 cp "$LOG_OUT" "${RUN_S3}/logs/job.out" --only-show-errors 2>/dev/null || true
  [ -f "$LOG_ERR" ] && aws s3 cp "$LOG_ERR" "${RUN_S3}/logs/job.err" --only-show-errors 2>/dev/null || true
  [ -f "$JOB_LOG" ] && aws s3 cp "$JOB_LOG" "${RUN_S3}/logs/job.log" --only-show-errors 2>/dev/null || true
}
on_exit() { rc=$?; sync_ckpts; upload_logs; exit $rc; }
trap on_exit EXIT

echo "[INFO] host=$(hostname) job=${SLURM_JOB_ID}"
nvidia-smi -L || true

mkdir -p "$REPO_DIR"
rm -rf "$REPO_DIR"/*
aws s3 cp "$SRC_TARBALL_S3" /opt/dlami/nvme/tmp/vjepa2-src.tar.gz --only-show-errors
tar xzf /opt/dlami/nvme/tmp/vjepa2-src.tar.gz -C "$REPO_DIR"
cd "$REPO_DIR"
[ -f "$CFG_REL" ] || { echo "[FATAL] $CFG_REL missing"; exit 1; }
export PYTHONPATH="$REPO_DIR:/opt/dlami/nvme/pyarrow_site:${PYTHONPATH:-}"

mkdir -p "$CSV_DIR" /opt/dlami/nvme/checkpoints
[ ! -f "$CSV_DIR/mimic_annotations_s3.csv" ] && aws s3 cp "$CSV_S3" "$CSV_DIR/mimic_annotations_s3.csv" --only-show-errors
[ ! -f /opt/dlami/nvme/checkpoints/jepa_in21k_vitl_e100.pt ] && aws s3 cp "$E100_CKPT_S3" /opt/dlami/nvme/checkpoints/jepa_in21k_vitl_e100.pt --only-show-errors
ls -lh /opt/dlami/nvme/checkpoints/jepa_in21k_vitl_e100.pt "$CSV_DIR/mimic_annotations_s3.csv"
df -h /opt/dlami/nvme

mkdir -p "$CKPT_DIR"

(
  while true; do
    sleep 600
    sync_ckpts; upload_logs
  done
) &
SYNC_PID=$!

echo "[INFO] Launching final_sv_25 (single-view)..."
python -m app.main \
  --fname "$CFG_REL" \
  --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
  2>&1 | tee "$JOB_LOG"
RC=$?

kill $SYNC_PID 2>/dev/null || true
wait $SYNC_PID 2>/dev/null || true
sync_ckpts; upload_logs
exit $RC
