#!/bin/bash
# 8-GPU phase-matched view-distance curriculum +25e. NVMe-only.
# Stages source + parquet + IN21K e100 ckpt + view-labels CSV.
#SBATCH -p ml-p5-48xlarge
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 0-16:00:00
#SBATCH -J mv_curric_25e
#SBATCH -o /opt/dlami/nvme/logs/mv_curric_25e-%j.out
#SBATCH -e /opt/dlami/nvme/logs/mv_curric_25e-%j.err
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
PARQUET_S3="${S3_BASE}/data/phase_annotations/phase_annotations.parquet"
VIEW_LABELS_S3="${S3_BASE}/data/view_labels/mimic_view_predictions.csv"
SRC_TARBALL_S3="${S3_BASE}/setup/vjepa2-src.tar.gz"

REPO_DIR="/opt/dlami/nvme/src/vjepa2"
CFG_REL="configs/train/vitl16/pretrain-multiview-phase-curriculum-high-25e.yaml"
RUN_S3="${S3_BASE}/runs/mv_curric_25e_${SLURM_JOB_ID}"
CKPT_DIR="/opt/dlami/nvme/checkpoints/multiview/phase_curriculum_high_25e"
PROBE_DIR="/opt/dlami/nvme/probe"
VIEW_LABELS_LOCAL="/opt/dlami/nvme/data/view_labels/mimic_view_predictions.csv"
LOG_OUT="/opt/dlami/nvme/logs/mv_curric_25e-${SLURM_JOB_ID}.out"
LOG_ERR="/opt/dlami/nvme/logs/mv_curric_25e-${SLURM_JOB_ID}.err"
JOB_LOG="/opt/dlami/nvme/logs/mv_curric_25e_${SLURM_JOB_ID}.log"

sync_ckpts() {
  if [ -d "$CKPT_DIR" ]; then
    aws s3 sync "$CKPT_DIR" "${RUN_S3}/checkpoints/" \
      --exclude "*" \
      --include "e*.pt" --include "latest.pt" --include "*.csv" --include "*.yaml" \
      --only-show-errors || true
  fi
}

upload_logs() {
  [ -f "$LOG_OUT" ] && aws s3 cp "$LOG_OUT" "${RUN_S3}/logs/job.out" --only-show-errors 2>/dev/null || true
  [ -f "$LOG_ERR" ] && aws s3 cp "$LOG_ERR" "${RUN_S3}/logs/job.err" --only-show-errors 2>/dev/null || true
  [ -f "$JOB_LOG" ] && aws s3 cp "$JOB_LOG" "${RUN_S3}/logs/job.log" --only-show-errors 2>/dev/null || true
}

on_exit() {
  rc=$?
  sync_ckpts
  upload_logs
  exit $rc
}
trap on_exit EXIT

echo "[INFO] host=$(hostname) job=${SLURM_JOB_ID}"
nvidia-smi -L || true

# Deploy source
mkdir -p "$REPO_DIR"
rm -rf "$REPO_DIR"/*
aws s3 cp "$SRC_TARBALL_S3" /opt/dlami/nvme/tmp/vjepa2-src.tar.gz --only-show-errors
tar xzf /opt/dlami/nvme/tmp/vjepa2-src.tar.gz -C "$REPO_DIR"
cd "$REPO_DIR"
if [ ! -f "$CFG_REL" ]; then
    echo "[FATAL] Config $CFG_REL not found"; exit 1
fi
export PYTHONPATH="$REPO_DIR:/opt/dlami/nvme/pyarrow_site:${PYTHONPATH:-}"

# Stage data + ckpt + view-labels (cached)
mkdir -p "$PROBE_DIR" /opt/dlami/nvme/checkpoints "$(dirname "$VIEW_LABELS_LOCAL")"
if [ ! -f "$PROBE_DIR/phase_annotations.parquet" ]; then
    aws s3 cp "$PARQUET_S3" "$PROBE_DIR/phase_annotations.parquet" --only-show-errors
fi
if [ ! -f /opt/dlami/nvme/checkpoints/jepa_in21k_vitl_e100.pt ]; then
    aws s3 cp "$E100_CKPT_S3" /opt/dlami/nvme/checkpoints/jepa_in21k_vitl_e100.pt --only-show-errors
fi
if [ ! -f "$VIEW_LABELS_LOCAL" ]; then
    aws s3 cp "$VIEW_LABELS_S3" "$VIEW_LABELS_LOCAL" --only-show-errors
fi
ls -lh "$VIEW_LABELS_LOCAL" /opt/dlami/nvme/checkpoints/jepa_in21k_vitl_e100.pt "$PROBE_DIR/phase_annotations.parquet"
df -h /opt/dlami/nvme

mkdir -p "$CKPT_DIR"

(
  while true; do
    sleep 600
    sync_ckpts
    upload_logs
  done
) &
SYNC_PID=$!

echo "[INFO] Launching phase-curriculum +25e pilot..."
python -m app.main \
  --fname "$CFG_REL" \
  --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
  2>&1 | tee "$JOB_LOG"
RC=$?

kill $SYNC_PID 2>/dev/null || true
wait $SYNC_PID 2>/dev/null || true
sync_ckpts
upload_logs
exit $RC
