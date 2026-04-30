#!/bin/bash
# Finalbudget batch-size sweep (bs=16, 32, 64 per GPU) on 8 H100s.
# Uses high+medium, pairs_per_study=16, strict RR, 100 steps each.
# NVMe-only deployment.
#SBATCH -p ml-p5-48xlarge
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 0-03:00:00
#SBATCH -J fb_sweep_bs
#SBATCH -o /opt/dlami/nvme/logs/fb_sweep_bs-%j.out
#SBATCH -e /opt/dlami/nvme/logs/fb_sweep_bs-%j.err
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
SRC_TARBALL_S3="${S3_BASE}/setup/vjepa2-src.tar.gz"

REPO_DIR="/opt/dlami/nvme/src/vjepa2"
RUN_S3="${S3_BASE}/runs/fb_sweep_bs_${SLURM_JOB_ID}"
PROBE_DIR="/opt/dlami/nvme/probe"

sync_logs() {
  aws s3 cp /opt/dlami/nvme/logs/fb_sweep_bs-${SLURM_JOB_ID}.out "${RUN_S3}/logs/sweep.out" --only-show-errors 2>/dev/null || true
}
on_exit() {
  rc=$?
  for bs in 16 32 64; do
    d=/opt/dlami/nvme/checkpoints/finalbudget/sweep_bs${bs}
    [ -d "$d" ] && aws s3 sync "$d" "${RUN_S3}/bs${bs}/" --only-show-errors || true
  done
  sync_logs
  exit $rc
}
trap on_exit EXIT

echo "[INFO] host=$(hostname) job=${SLURM_JOB_ID}"
nvidia-smi -L || true

mkdir -p "$REPO_DIR"
rm -rf "$REPO_DIR"/*
aws s3 cp "$SRC_TARBALL_S3" /opt/dlami/nvme/tmp/vjepa2-src.tar.gz --only-show-errors
tar xzf /opt/dlami/nvme/tmp/vjepa2-src.tar.gz -C "$REPO_DIR"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:/opt/dlami/nvme/pyarrow_site:${PYTHONPATH:-}"

mkdir -p "$PROBE_DIR" /opt/dlami/nvme/checkpoints
[ ! -f "$PROBE_DIR/phase_annotations.parquet" ] && aws s3 cp "$PARQUET_S3" "$PROBE_DIR/phase_annotations.parquet" --only-show-errors
[ ! -f /opt/dlami/nvme/checkpoints/jepa_in21k_vitl_e100.pt ] && aws s3 cp "$E100_CKPT_S3" /opt/dlami/nvme/checkpoints/jepa_in21k_vitl_e100.pt --only-show-errors

df -h /opt/dlami/nvme

for bs in 16 32 64; do
  echo "========================================================================"
  echo "[SWEEP] Starting bs=${bs} @ $(date -Iseconds)"
  echo "========================================================================"
  CFG="configs/train/vitl16/finalbudget-sweep-bs${bs}.yaml"
  if [ ! -f "$CFG" ]; then echo "[FATAL] $CFG not found"; exit 1; fi
  # Capture peak GPU memory at end of each run.
  python -m app.main --fname "$CFG" \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
    2>&1 | sed -u "s|^|[bs=${bs}] |" | tee -a "/opt/dlami/nvme/logs/fb_sweep_bs_${SLURM_JOB_ID}_bs${bs}.log" || true
  # Dump nvidia-smi after the run for peak-mem reference.
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | sed "s|^|[bs=${bs}] after-run-gpu-mem: |"
  echo "[SWEEP] Finished bs=${bs} @ $(date -Iseconds)"
done

echo "[INFO] sweep complete"
