#!/bin/bash
#SBATCH -p ml-p5-48xlarge
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 4-00:00:00
#SBATCH -J ejg_mr_uhn
#SBATCH -o /tmp/ejg_mr_uhn-%j.out
#SBATCH -e /tmp/ejg_mr_uhn-%j.err
#SBATCH --signal=B:USR1@300
#SBATCH --nodelist=ip-10-0-50-83

set -euxo pipefail

# =========================
# GLOBAL CONFIG
# =========================
export AWS_DEFAULT_REGION="us-west-2"
export AWS_REGION="us-west-2"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export MASTER_PORT=$((12000 + $RANDOM % 20000))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

export AWS_SDK_LOAD_CONFIG=1
export HOME="/home/ubuntu"
export AWS_SHARED_CREDENTIALS_FILE="${HOME}/.aws/credentials"
export AWS_CONFIG_FILE="${HOME}/.aws/config"
unset AWS_EC2_METADATA_DISABLED 2>/dev/null || true
export AWS_METADATA_SERVICE_TIMEOUT=5
export AWS_METADATA_SERVICE_NUM_ATTEMPTS=3

export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export FI_EFA_USE_DEVICE_RDMA=1
export FI_PROVIDER=efa
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# cuBLAS fix for H100 bf16
export LD_LIBRARY_PATH=/opt/amazon/efa/lib:/opt/aws-ofi-nccl/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

# S3 paths
ART_BUCKET="sagemaker-hyperpod-lifecycle-495467399120-usw2"
ART_PREFIX="vjepa2-artifacts"
S3_BASE="s3://${ART_BUCKET}/${ART_PREFIX}"
RUN_S3="${S3_BASE}/runs/echojepa_g_mr_severity_uhn_${SLURM_JOB_ID}"

# =========================
# HOST SETUP
# =========================
echo "[INFO] host=$(hostname) job=${SLURM_JOB_ID}"
nvidia-smi -L || true

SCRATCH="/opt/dlami/nvme"
WORKDIR="${SCRATCH}/ejg_mr_uhn_${SLURM_JOB_ID}"
mkdir -p "$WORKDIR"
RUNLOG="${WORKDIR}/probe_train_${SLURM_JOB_ID}.log"

# =========================
# TRAPS
# =========================
CFG_FOLDER=""
upload_logs() {
  echo "[INFO] Uploading logs..."
  if [ -f "$RUNLOG" ]; then
    aws s3 cp "$RUNLOG" "${RUN_S3}/logs/job.out" --region us-west-2 2>/dev/null || true
  fi
  if [ -n "$CFG_FOLDER" ] && [ -d "$CFG_FOLDER" ]; then
    aws s3 sync "$CFG_FOLDER" "${RUN_S3}/training_folder/" \
      --include "*.pt" --include "*.csv" --include "*.log" --include "*.json" \
      --exclude "*.pth" --region us-west-2 --only-show-errors 2>/dev/null || true
  fi
}
trap 'rc=$?; upload_logs; exit $rc' EXIT

# =========================
# ENVIRONMENT
# =========================
ENV_S3="${S3_BASE}/env/vjepa2-312-pack-env.tar"
BOTO3_WHL_S3="${S3_BASE}/wheels/boto3-wheelhouse.tar"

aws s3 cp "$ENV_S3" "${WORKDIR}/env.tar" --region us-west-2
mkdir -p "${WORKDIR}/env"
tar -xf "${WORKDIR}/env.tar" -C "${WORKDIR}/env"
set +u; source "${WORKDIR}/env/bin/activate"; set -u
"${WORKDIR}/env/bin/conda-unpack" 2>/dev/null || true

aws s3 cp "$BOTO3_WHL_S3" "${WORKDIR}/boto3-wheelhouse.tar" --region us-west-2
mkdir -p "${WORKDIR}/wheelhouse"
tar -xf "${WORKDIR}/boto3-wheelhouse.tar" -C "${WORKDIR}/wheelhouse"
python -m pip install --no-index --find-links "${WORKDIR}/wheelhouse" boto3

# =========================
# CODE
# =========================
REPO_DIR="/opt/vjepa2"
cd "$REPO_DIR"
pip install -e . --no-deps --no-build-isolation 2>/dev/null || true
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# =========================
# DOWNLOAD INPUTS
# =========================
CFG_REL="configs/eval/vitg-384/nature_medicine/echojepa_g_mr_severity_uhn_hp.yaml"

echo "[INFO] Downloading CSVs (large — 1.2 GB train, 150 MB val)..."
mkdir -p data/csv
aws s3 cp s3://echodata25/nature_medicine/uhn/probe_csvs/mr_severity_4class/train.csv data/csv/uhn_mr_severity_4class_train.csv --region us-west-2
aws s3 cp s3://echodata25/nature_medicine/uhn/probe_csvs/mr_severity_4class/val.csv data/csv/uhn_mr_severity_4class_val.csv --region us-west-2

echo "[INFO] Downloading EchoJEPA-G checkpoint (16 GB) to NVMe..."
CKPT_DIR="/opt/dlami/nvme/checkpoints/anneal/keep"
mkdir -p "$CKPT_DIR"
if [ ! -f "${CKPT_DIR}/pt-280-an81.pt" ]; then
  aws s3 cp "${S3_BASE}/checkpoints/anneal/keep/pt-280-an81.pt" "${CKPT_DIR}/pt-280-an81.pt" --region us-west-2
fi
# Symlink so config path resolves
mkdir -p checkpoints/anneal/keep
ln -sf "${CKPT_DIR}/pt-280-an81.pt" checkpoints/anneal/keep/pt-280-an81.pt

echo "[INFO] Downloading resume checkpoint from job 437 (epoch 1)..."
RESUME_DIR="/opt/dlami/nvme/evals/vitg-384/nature_medicine/video_classification_frozen/echojepa-g-mr-severity-uhn"
mkdir -p "$RESUME_DIR"
PREV_S3="${S3_BASE}/runs/echojepa_g_mr_severity_uhn_437/training_folder/video_classification_frozen/echojepa-g-mr-severity-uhn"
aws s3 cp "${PREV_S3}/latest.pt" "${RESUME_DIR}/latest.pt" --region us-west-2 || true
aws s3 cp "${PREV_S3}/best.pt" "${RESUME_DIR}/best.pt" --region us-west-2 || true
aws s3 cp "${PREV_S3}/log_r0.csv" "${RESUME_DIR}/log_r0.csv" --region us-west-2 || true

# =========================
# PARSE CONFIG
# =========================
export CFG_REL
eval $(python - <<'PY'
import re, os
from pathlib import Path
cfg_rel = os.environ["CFG_REL"]
folder, tag = "", ""
if os.path.exists(cfg_rel):
    t = Path(cfg_rel).read_text()
    m = re.search(r'^\s*folder:\s*(.+)\s*$', t, re.M)
    if m: folder = m.group(1).strip().strip('"').strip("'")
    m = re.search(r'^\s*tag:\s*(.+)\s*$', t, re.M)
    if m: tag = m.group(1).strip().strip('"').strip("'")
print(f"export CFG_FOLDER='{folder}'")
print(f"export CFG_TAG='{tag}'")
PY
)
mkdir -p "$CFG_FOLDER"
echo "[INFO] folder=$CFG_FOLDER tag=$CFG_TAG"

# =========================
# CREDENTIALS
# =========================
# Do NOT export static AWS_ACCESS_KEY_ID/SECRET/TOKEN from IMDS — they expire after ~1h.
# Instead, let boto3 use the default credential chain which auto-refreshes from IMDS.
unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN 2>/dev/null || true

# =========================
# BACKGROUND SYNC
# =========================
(
  while true; do
    sleep 600
    aws s3 sync "$CFG_FOLDER" "${RUN_S3}/training_folder/" \
      --include "*.pt" --include "*.csv" --include "*.log" --include "*.json" \
      --exclude "*.pth" --region us-west-2 --only-show-errors 2>/dev/null || true
    [ -f "$RUNLOG" ] && aws s3 cp "$RUNLOG" "${RUN_S3}/logs/job_latest.log" --region us-west-2 2>/dev/null || true
  done
) &
SYNC_PID=$!

# =========================
# RUN PROBE TRAINING
# =========================
GPU_COUNT=$(nvidia-smi -L | wc -l)
DEVICES=()
for ((i=0; i<GPU_COUNT; i++)); do DEVICES+=("cuda:${i}"); done
echo "[INFO] ${GPU_COUNT} GPUs: ${DEVICES[*]}"

set +e
python -m evals.main \
  --fname "$CFG_REL" \
  --devices "${DEVICES[@]}" \
  2>&1 | tee "$RUNLOG"
RC=${PIPESTATUS[0]}
set -e

kill $SYNC_PID 2>/dev/null || true; wait $SYNC_PID 2>/dev/null || true

echo "[INFO] Uploading final artifacts..."
aws s3 cp "$RUNLOG" "${RUN_S3}/logs/job_final.log" --region us-west-2 || true
aws s3 cp "$CFG_REL" "${RUN_S3}/config/run_config.yaml" --region us-west-2 || true
aws s3 sync "$CFG_FOLDER" "${RUN_S3}/training_folder/" \
  --include "*.pt" --include "*.csv" --include "*.log" --include "*.json" \
  --exclude "*.pth" --region us-west-2 --only-show-errors || true

[ "$RC" -ne 0 ] && echo "[ERROR] Training failed with exit code ${RC}" && exit "$RC"
echo "[INFO] DONE. Results: ${RUN_S3}"
