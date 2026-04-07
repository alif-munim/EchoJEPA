#!/bin/bash
#SBATCH -p ml-p5-48xlarge
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 0-04:00:00
#SBATCH -J ejg_mr_test
#SBATCH -o /tmp/ejg_mr_test-%j.out
#SBATCH -e /tmp/ejg_mr_test-%j.err
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
RUN_S3="${S3_BASE}/runs/echojepa_g_mr_test_${SLURM_JOB_ID}"
TRAIN_RUN_S3="${S3_BASE}/runs/echojepa_g_mitral_regurg_436"

# =========================
# HOST SETUP
# =========================
echo "[INFO] host=$(hostname) job=${SLURM_JOB_ID}"
nvidia-smi -L || true

SCRATCH="/opt/dlami/nvme"
WORKDIR="${SCRATCH}/ejg_mr_test_${SLURM_JOB_ID}"
mkdir -p "$WORKDIR"

# =========================
# TRAPS
# =========================
CFG_FOLDER=""
upload_logs() {
  echo "[INFO] Uploading logs..."
  for f in "$WORKDIR"/*.log; do
    [ -f "$f" ] && aws s3 cp "$f" "${RUN_S3}/logs/$(basename $f)" --region us-west-2 2>/dev/null || true
  done
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

# Install scikit-learn
echo "[INFO] Installing scikit-learn..."
pip install scikit-learn 2>&1 | tail -3

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
echo "[INFO] Downloading CSVs (train + test)..."
mkdir -p data/csv
aws s3 cp s3://echodata25/nature_medicine/mimic/probe_csvs/mitral_regurg/train.csv data/csv/mimic_mitral_regurg_train.csv --region us-west-2
aws s3 cp s3://echodata25/nature_medicine/mimic/probe_csvs/mitral_regurg/test.csv data/csv/mimic_mitral_regurg_test.csv --region us-west-2

echo "[INFO] Downloading EchoJEPA-G checkpoint (16 GB) to NVMe..."
CKPT_DIR="/opt/dlami/nvme/checkpoints/anneal/keep"
mkdir -p "$CKPT_DIR"
if [ ! -f "${CKPT_DIR}/pt-280-an81.pt" ]; then
  aws s3 cp "${S3_BASE}/checkpoints/anneal/keep/pt-280-an81.pt" "${CKPT_DIR}/pt-280-an81.pt" --region us-west-2
fi
mkdir -p checkpoints/anneal/keep
ln -sf "${CKPT_DIR}/pt-280-an81.pt" checkpoints/anneal/keep/pt-280-an81.pt

echo "[INFO] Downloading MR probe best.pt from job 436..."
mkdir -p checkpoints/probes/echojepa-g-mitral-regurg
aws s3 cp "${TRAIN_RUN_S3}/training_folder/video_classification_frozen/echojepa-g-mitral-regurg/best.pt" \
  checkpoints/probes/echojepa-g-mitral-regurg/best.pt --region us-west-2

# =========================
# CREDENTIALS
# =========================
unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN 2>/dev/null || true

export CFG_FOLDER="/opt/dlami/nvme/evals/vitg-384/nature_medicine"
mkdir -p "$CFG_FOLDER"

# =========================
# GPU SETUP
# =========================
GPU_COUNT=$(nvidia-smi -L | wc -l)
DEVICES=()
for ((i=0; i<GPU_COUNT; i++)); do DEVICES+=("cuda:${i}"); done
echo "[INFO] ${GPU_COUNT} GPUs: ${DEVICES[*]}"

# =========================
# PHASE 1: SINGLE-CLIP TEST (study_sampling: false)
# =========================
echo "[INFO] === PHASE 1: Single-clip test evaluation ==="
SC_LOG="${WORKDIR}/singleclip_test.log"

set +e
python -m evals.main \
  --fname configs/eval/vitg-384/nature_medicine/echojepa_g_mitral_regurg_singleclip_test.yaml \
  --devices "${DEVICES[@]}" \
  2>&1 | tee "$SC_LOG"
SC_RC=${PIPESTATUS[0]}
set -e
echo "[INFO] Single-clip test exit code: ${SC_RC}"

# =========================
# PHASE 2: PRED-AVG TEST (study_sampling: true)
# =========================
echo "[INFO] === PHASE 2: Pred-avg test evaluation ==="
PA_LOG="${WORKDIR}/predavg_test.log"

export MASTER_PORT=$((MASTER_PORT + 1))
set +e
python -m evals.main \
  --fname configs/eval/vitg-384/nature_medicine/echojepa_g_mitral_regurg_predavg.yaml \
  --devices "${DEVICES[@]}" \
  2>&1 | tee "$PA_LOG"
PA_RC=${PIPESTATUS[0]}
set -e
echo "[INFO] Pred-avg test exit code: ${PA_RC}"

# =========================
# FINAL UPLOAD
# =========================
echo "[INFO] Uploading all artifacts..."
aws s3 sync "/opt/dlami/nvme/evals/vitg-384/nature_medicine/video_classification_frozen/" "${RUN_S3}/training_folder/video_classification_frozen/" \
  --include "*.csv" --include "*.log" --include "*.json" \
  --exclude "*.pt" --exclude "*.pth" --region us-west-2 --only-show-errors || true

echo "[INFO] DONE. SingleClip RC=${SC_RC}, PredAvg RC=${PA_RC}"
