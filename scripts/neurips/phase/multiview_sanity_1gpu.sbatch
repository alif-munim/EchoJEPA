#!/bin/bash
# Single-GPU multiview sanity. Runs entirely off /opt/dlami/nvme because
# the compute-node root FS is 100% full (holding deleted files). No writes
# to /tmp or /opt/vjepa2.
#SBATCH -p ml-p5-48xlarge
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH -t 0-01:00:00
#SBATCH -J mv_sanity_1gpu
#SBATCH -o /opt/dlami/nvme/logs/mv_sanity_1gpu-%j.out
#SBATCH -e /opt/dlami/nvme/logs/mv_sanity_1gpu-%j.err
#SBATCH --chdir=/opt/dlami/nvme

set -euxo pipefail

# Prepend python env
export PATH="/opt/vjepa2-312/bin:$PATH"

# Force tempfiles to NVMe (root FS is full)
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

# Force PyTorch's bundled CUDA libs (H100 bf16 fix)
export LD_LIBRARY_PATH="/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cublas/lib:/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"

ART_BUCKET="sagemaker-hyperpod-lifecycle-495467399120-usw2"
S3_BASE="s3://${ART_BUCKET}/vjepa2-artifacts"

E100_CKPT_S3="${S3_BASE}/checkpoints/jepa_in21k/jepa_in21k_vitl_e100.pt"
PARQUET_S3="${S3_BASE}/data/phase_annotations/phase_annotations.parquet"
SRC_TARBALL_S3="${S3_BASE}/setup/vjepa2-src.tar.gz"

# Everything lands on NVMe
REPO_DIR="/opt/dlami/nvme/src/vjepa2"
CFG_REL="configs/train/vitl16/debug-multiview-phase-high-gpu.yaml"
RUN_S3="${S3_BASE}/runs/multiview_sanity_1gpu_${SLURM_JOB_ID}"
CKPT_DIR="/opt/dlami/nvme/checkpoints/sanity/multiview_phase_high_gpu"
PROBE_DIR="/opt/dlami/nvme/probe"
LOG_OUT="/opt/dlami/nvme/logs/mv_sanity_1gpu-${SLURM_JOB_ID}.out"
LOG_ERR="/opt/dlami/nvme/logs/mv_sanity_1gpu-${SLURM_JOB_ID}.err"
JOB_LOG="/opt/dlami/nvme/logs/mv_sanity_1gpu_${SLURM_JOB_ID}.log"

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

# 1. Deploy source onto NVMe (DO NOT write to /opt/vjepa2 or /tmp)
mkdir -p "$REPO_DIR"
echo "[INFO] Downloading source tarball to NVMe..."
aws s3 cp "$SRC_TARBALL_S3" /opt/dlami/nvme/tmp/vjepa2-src.tar.gz --only-show-errors
tar xzf /opt/dlami/nvme/tmp/vjepa2-src.tar.gz -C "$REPO_DIR"
ls -la "$REPO_DIR/app/vjepa_multiview/train.py"

cd "$REPO_DIR"
if [ ! -f "$CFG_REL" ]; then
    echo "[FATAL] Config $CFG_REL not found in $REPO_DIR"
    exit 1
fi

# Put repo + pyarrow site (installed to NVMe during probe) on sys.path.
# Skip `pip install -e .` because it writes to /opt/vjepa2-312 on the root FS.
export PYTHONPATH="$REPO_DIR:/opt/dlami/nvme/pyarrow_site:${PYTHONPATH:-}"

# 2. Stage parquet
echo "[INFO] Staging parquet..."
mkdir -p "$PROBE_DIR"
if [ ! -f "$PROBE_DIR/phase_annotations.parquet" ]; then
    aws s3 cp "$PARQUET_S3" "$PROBE_DIR/phase_annotations.parquet" --only-show-errors
fi

# 3. Stage checkpoint
echo "[INFO] Staging IN21K e100 single-view JEPA checkpoint..."
mkdir -p /opt/dlami/nvme/checkpoints
if [ ! -f /opt/dlami/nvme/checkpoints/jepa_in21k_vitl_e100.pt ]; then
    aws s3 cp "$E100_CKPT_S3" /opt/dlami/nvme/checkpoints/jepa_in21k_vitl_e100.pt --only-show-errors
fi

ls -lh /opt/dlami/nvme/checkpoints/jepa_in21k_vitl_e100.pt "$PROBE_DIR/phase_annotations.parquet"
df -h /opt/dlami/nvme

mkdir -p "$CKPT_DIR"

# Periodic upload (every 5 min during sanity run)
(
  while true; do
    sleep 300
    sync_ckpts
    upload_logs
  done
) &
SYNC_PID=$!

echo "[INFO] Launching 1-GPU multiview sanity (max_steps=100)..."
python -m app.main \
  --fname "$CFG_REL" \
  --devices cuda:0 \
  2>&1 | tee "$JOB_LOG"
RC=$?

kill $SYNC_PID 2>/dev/null || true
wait $SYNC_PID 2>/dev/null || true
sync_ckpts
upload_logs
exit $RC
