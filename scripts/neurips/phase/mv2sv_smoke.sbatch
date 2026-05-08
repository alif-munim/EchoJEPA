#!/bin/bash
# MV2SV v2 smoke — parity then Stage B engineering ablation.
# Two sequential 30-step runs (~5-10 min each, 8xH100). Not a
# scientific run. Verifies dispatch wiring, DDP, EMA update path,
# and pair_shared_loss trajectory on real data.
#SBATCH -p dev
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 0-01:00:00
#SBATCH -J mv2sv_smoke
#SBATCH -o /opt/dlami/nvme/logs/mv2sv_smoke-%j.out
#SBATCH -e /opt/dlami/nvme/logs/mv2sv_smoke-%j.err
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
RUN_S3="${S3_BASE}/runs/mv2sv_smoke_${SLURM_JOB_ID}"
PROBE_DIR="/opt/dlami/nvme/probe"
VIEW_DIR="/opt/dlami/nvme/data/view_labels"
LOG_OUT="/opt/dlami/nvme/logs/mv2sv_smoke-${SLURM_JOB_ID}.out"
LOG_ERR="/opt/dlami/nvme/logs/mv2sv_smoke-${SLURM_JOB_ID}.err"

CFG_PARITY="configs/train/vitl16/smoke/mv2sv-smoke-parity.yaml"
CFG_STAGEB="configs/train/vitl16/smoke/mv2sv-smoke-stageB.yaml"
CKPT_PARITY="/opt/dlami/nvme/checkpoints/mv2sv_smoke_parity"
CKPT_STAGEB="/opt/dlami/nvme/checkpoints/mv2sv_smoke_stageB"
JOB_LOG_PARITY="/opt/dlami/nvme/logs/mv2sv_smoke_parity_${SLURM_JOB_ID}.log"
JOB_LOG_STAGEB="/opt/dlami/nvme/logs/mv2sv_smoke_stageB_${SLURM_JOB_ID}.log"

upload_artifacts() {
  for d in "$CKPT_PARITY" "$CKPT_STAGEB"; do
    [ -d "$d" ] && aws s3 sync "$d" "${RUN_S3}/$(basename "$d")/" \
      --exclude "*" --include "log_r*.csv" --include "*.yaml" --include "*.log" \
      --only-show-errors || true
  done
  [ -f "$LOG_OUT" ] && aws s3 cp "$LOG_OUT" "${RUN_S3}/logs/job.out" --only-show-errors 2>/dev/null || true
  [ -f "$LOG_ERR" ] && aws s3 cp "$LOG_ERR" "${RUN_S3}/logs/job.err" --only-show-errors 2>/dev/null || true
  [ -f "$JOB_LOG_PARITY" ] && aws s3 cp "$JOB_LOG_PARITY" "${RUN_S3}/logs/parity.log" --only-show-errors 2>/dev/null || true
  [ -f "$JOB_LOG_STAGEB" ] && aws s3 cp "$JOB_LOG_STAGEB" "${RUN_S3}/logs/stageB.log" --only-show-errors 2>/dev/null || true
}
on_exit() { rc=$?; upload_artifacts; exit $rc; }
trap on_exit EXIT

echo "[INFO] host=$(hostname) job=${SLURM_JOB_ID}"
nvidia-smi -L || true

mkdir -p "$REPO_DIR"
rm -rf "$REPO_DIR"/*
aws s3 cp "$SRC_TARBALL_S3" /opt/dlami/nvme/tmp/vjepa2-src.tar.gz --only-show-errors
tar xzf /opt/dlami/nvme/tmp/vjepa2-src.tar.gz -C "$REPO_DIR"
cd "$REPO_DIR"
[ -f "$CFG_PARITY" ] || { echo "[FATAL] $CFG_PARITY missing"; exit 1; }
[ -f "$CFG_STAGEB" ] || { echo "[FATAL] $CFG_STAGEB missing"; exit 1; }
export PYTHONPATH="$REPO_DIR:/opt/dlami/nvme/pyarrow_site:${PYTHONPATH:-}"

mkdir -p "$PROBE_DIR" "$VIEW_DIR" /opt/dlami/nvme/checkpoints
[ ! -f "$PROBE_DIR/phase_annotations.parquet" ] && aws s3 cp "$PARQUET_S3" "$PROBE_DIR/phase_annotations.parquet" --only-show-errors
[ ! -f "$VIEW_DIR/mimic_view_predictions.csv" ] && aws s3 cp "$VIEW_CSV_S3" "$VIEW_DIR/mimic_view_predictions.csv" --only-show-errors
[ ! -f /opt/dlami/nvme/checkpoints/jepa_in21k_vitl_e100.pt ] && aws s3 cp "$E100_CKPT_S3" /opt/dlami/nvme/checkpoints/jepa_in21k_vitl_e100.pt --only-show-errors

# Init-checkpoint preflight (same as method/control runs).
python scripts/neurips/phase/verify_init_checkpoint.py \
  --ckpt /opt/dlami/nvme/checkpoints/jepa_in21k_vitl_e100.pt \
  --expected-tag mimic_standard_jepa_e100 \
  --strict

# Fresh output dirs.
for d in "$CKPT_PARITY" "$CKPT_STAGEB"; do
  if [ -d "$d" ]; then
    echo "[INFO] Clearing stale $d"
    rm -rf "$d"/*
  fi
  mkdir -p "$d"
done

# --- PARITY RUN ---------------------------------------------------------- #
echo "[INFO] ========== mv2sv smoke PARITY =========="
set +e
python -m app.main \
  --fname "$CFG_PARITY" \
  --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
  2>&1 | tee "$JOB_LOG_PARITY"
RC_PARITY=$?
set -e
echo "[INFO] PARITY exit code: $RC_PARITY"

if [ $RC_PARITY -ne 0 ]; then
  echo "[FATAL] PARITY run failed — skipping STAGE B and exiting with parity rc"
  exit $RC_PARITY
fi

# --- STAGE B RUN --------------------------------------------------------- #
echo "[INFO] ========== mv2sv smoke STAGE B =========="
set +e
python -m app.main \
  --fname "$CFG_STAGEB" \
  --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
  2>&1 | tee "$JOB_LOG_STAGEB"
RC_STAGEB=$?
set -e
echo "[INFO] STAGE B exit code: $RC_STAGEB"

upload_artifacts
exit $RC_STAGEB
