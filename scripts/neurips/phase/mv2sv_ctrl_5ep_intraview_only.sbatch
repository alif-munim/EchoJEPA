#!/bin/bash
# MV2SV v5 — 5-epoch intraview-only control, fused OFF, all MV2SV lambdas = 0.
#SBATCH -p dev
#SBATCH --nodelist=ip-10-0-50-56
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 0-08:00:00
#SBATCH -J mv2sv_ctrl_5ep
#SBATCH -o /opt/dlami/nvme/logs/mv2sv_ctrl_5ep-%j.out
#SBATCH -e /opt/dlami/nvme/logs/mv2sv_ctrl_5ep-%j.err
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
SRC_TARBALL_S3="${S3_BASE}/setup/vjepa2-src-mv2sv-ctrl.tar.gz"

REPO_DIR="/opt/dlami/nvme/src/vjepa2"
RUN_S3="${S3_BASE}/runs/mv2sv_ctrl_5ep_${SLURM_JOB_ID}"
CKPT_DIR="/opt/dlami/nvme/checkpoints/mv2sv_ctrl_5ep_intraview_only"
JOB_LOG="/opt/dlami/nvme/logs/mv2sv_ctrl_5ep_${SLURM_JOB_ID}.log"
CFG_REL="configs/train/vitl16/mv2sv-ctrl-5ep-intraview-only.yaml"

upload_artifacts() {
  [ -d "$CKPT_DIR" ] && aws s3 sync "$CKPT_DIR" "${RUN_S3}/ckpt/" \
    --exclude "*" --include "log_r*.csv" --include "*.log" --include "*.pt" \
    --include "params-pretrain.yaml" --only-show-errors 2>/dev/null || true
  [ -f "$JOB_LOG" ] && aws s3 cp "$JOB_LOG" "${RUN_S3}/logs/job.log" --only-show-errors 2>/dev/null || true
}
trap 'rc=$?; upload_artifacts; exit $rc' EXIT

echo "[INFO] host=$(hostname) job=${SLURM_JOB_ID}"
nvidia-smi -L || true

mkdir -p "$REPO_DIR"
rm -rf "$REPO_DIR"/*
aws s3 cp "$SRC_TARBALL_S3" /opt/dlami/nvme/tmp/vjepa2-src-mv2sv-ctrl.tar.gz --only-show-errors
tar xzf /opt/dlami/nvme/tmp/vjepa2-src-mv2sv-ctrl.tar.gz -C "$REPO_DIR"
cd "$REPO_DIR"
[ -f "$CFG_REL" ] || { echo "[FATAL] $CFG_REL missing"; exit 1; }
export PYTHONPATH="$REPO_DIR:/opt/dlami/nvme/pyarrow_site:${PYTHONPATH:-}"

# Ensure pyarrow is importable. Pilot node 146 has /opt/dlami/nvme/pyarrow_site
# pre-populated; node 56 does not. Install pyarrow into a node-local site if the
# import fails.
if ! python -c "import pyarrow" 2>/dev/null; then
  echo "[INFO] pyarrow missing; installing to /opt/dlami/nvme/pyarrow_site"
  mkdir -p /opt/dlami/nvme/pyarrow_site
  pip install --target=/opt/dlami/nvme/pyarrow_site --no-cache-dir pyarrow 2>&1 | tail -5
  python -c "import pyarrow; print('pyarrow', pyarrow.__version__)"
fi

mkdir -p /opt/dlami/nvme/probe /opt/dlami/nvme/data/view_labels /opt/dlami/nvme/checkpoints
[ ! -f /opt/dlami/nvme/probe/phase_annotations.parquet ] && aws s3 cp "$PARQUET_S3" /opt/dlami/nvme/probe/phase_annotations.parquet --only-show-errors
[ ! -f /opt/dlami/nvme/data/view_labels/mimic_view_predictions.csv ] && aws s3 cp "$VIEW_CSV_S3" /opt/dlami/nvme/data/view_labels/mimic_view_predictions.csv --only-show-errors
[ ! -f /opt/dlami/nvme/checkpoints/jepa_in21k_vitl_e100.pt ] && aws s3 cp "$E100_CKPT_S3" /opt/dlami/nvme/checkpoints/jepa_in21k_vitl_e100.pt --only-show-errors

python scripts/neurips/phase/verify_init_checkpoint.py \
  --ckpt /opt/dlami/nvme/checkpoints/jepa_in21k_vitl_e100.pt \
  --expected-tag mimic_standard_jepa_e100 --strict

if [ -d "$CKPT_DIR" ]; then rm -rf "$CKPT_DIR"/*; fi
mkdir -p "$CKPT_DIR"

echo "[INFO] === Launching MV2SV v5 5-epoch INTRAVIEW-ONLY control ==="
set +e
python -m app.main \
  --fname "$CFG_REL" \
  --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
  2>&1 | tee "$JOB_LOG"
RC_TRAIN=$?
set -e
echo "[INFO] train exit code: $RC_TRAIN"

if [ $RC_TRAIN -ne 0 ]; then
  echo "[FATAL] training failed"
  exit $RC_TRAIN
fi

echo "[INFO] === Running MV2SV v5 gate check (ctrl) on control log ==="
LOG_CSV="$CKPT_DIR/log_r0.csv"
set +e
python scripts/neurips/phase/mv2sv_gate_check.py ctrl --csv "$LOG_CSV" --stdout "$JOB_LOG"
RC_GATE=$?
set -e
echo "[INFO] gate exit code: $RC_GATE"
exit $RC_GATE
