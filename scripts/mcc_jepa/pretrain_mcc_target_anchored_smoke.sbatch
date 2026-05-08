#!/bin/bash
# MCC-JEPA matched-compute control: vanilla V-JEPA +25 of 100 from e100.
# Mirrors scripts/neurips/phase/final_phase_rel_hardneg25_paper.sbatch;
# only CFG_REL and folder paths differ.
#SBATCH -p ml-p5-48xlarge
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 0-01:30:00
#SBATCH -J mcc_anch_sm
#SBATCH -o /opt/dlami/nvme/logs/mcc_anch_sm-%j.out
#SBATCH -e /opt/dlami/nvme/logs/mcc_anch_sm-%j.err
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
VIEW_CSV_S3="${S3_BASE}/data/view_labels/mimic_view_predictions.csv"
SRC_TARBALL_S3="${S3_BASE}/setup/vjepa2-src.tar.gz"

REPO_DIR="/opt/dlami/nvme/src/vjepa2"
CFG_REL="configs/train/vitl16/pretrain-mcc-jepa-target-anchored-smoke.yaml"
RUN_S3="${S3_BASE}/runs/mcc_target_anchored_smoke_${SLURM_JOB_ID}"
CKPT_DIR="/opt/dlami/nvme/checkpoints/mcc_jepa/target_anchored_smoke"
PROBE_DIR="/opt/dlami/nvme/probe"
VIEW_DIR="/opt/dlami/nvme/data/view_labels"
LOG_OUT="/opt/dlami/nvme/logs/mcc_anch_sm-${SLURM_JOB_ID}.out"
LOG_ERR="/opt/dlami/nvme/logs/mcc_anch_sm-${SLURM_JOB_ID}.err"
JOB_LOG="/opt/dlami/nvme/logs/mcc_anch_sm_${SLURM_JOB_ID}.log"

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

mkdir -p "$PROBE_DIR" "$VIEW_DIR" /opt/dlami/nvme/checkpoints
[ ! -f "$PROBE_DIR/phase_annotations.parquet" ] && aws s3 cp "$PARQUET_S3" "$PROBE_DIR/phase_annotations.parquet" --only-show-errors
[ ! -f "$VIEW_DIR/mimic_view_predictions.csv" ] && aws s3 cp "$VIEW_CSV_S3" "$VIEW_DIR/mimic_view_predictions.csv" --only-show-errors
[ ! -f /opt/dlami/nvme/checkpoints/jepa_in21k_vitl_e100.pt ] && aws s3 cp "$E100_CKPT_S3" /opt/dlami/nvme/checkpoints/jepa_in21k_vitl_e100.pt --only-show-errors
ls -lh /opt/dlami/nvme/checkpoints/jepa_in21k_vitl_e100.pt "$PROBE_DIR/phase_annotations.parquet" "$VIEW_DIR/mimic_view_predictions.csv"
df -h /opt/dlami/nvme

echo "[INFO] Running init-checkpoint preflight..."
python scripts/neurips/phase/verify_init_checkpoint.py \
  --ckpt /opt/dlami/nvme/checkpoints/jepa_in21k_vitl_e100.pt \
  --expected-tag mimic_standard_jepa_e100 \
  --strict

if [ -d "$CKPT_DIR" ]; then
  echo "[INFO] Clearing stale CKPT_DIR=$CKPT_DIR before launch"
  rm -rf "$CKPT_DIR"/*
fi
mkdir -p "$CKPT_DIR"

(
  while true; do
    sleep 600
    sync_ckpts; upload_logs
  done
) &
SYNC_PID=$!

echo "[INFO] Launching MCC-JEPA target-anchored smoke..."
python -m app.main \
  --fname "$CFG_REL" \
  --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
  2>&1 | tee "$JOB_LOG"
RC=$?

kill $SYNC_PID 2>/dev/null || true
wait $SYNC_PID 2>/dev/null || true
sync_ckpts; upload_logs
exit $RC
