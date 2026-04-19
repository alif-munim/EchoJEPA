#!/bin/bash
#SBATCH -p ml-p5-48xlarge
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH -t 7-00:00:00
#SBATCH -J mae_e200
#SBATCH -o /tmp/mae_e200_%j.out
#SBATCH -e /tmp/mae_e200_%j.err
#SBATCH --signal=B:USR1@300

set -euxo pipefail

# =============================================================================
# VideoMAE-L Resume: e116 → e200
# =============================================================================
# Extends the original MAE run (seed 0, IN21K init) from epoch 116 to 200.
# Purpose: test whether temporal features recover under extended training,
# strengthening the "does not recover within the training budget" claim.
# Key checkpoints: e124, e149, e174, e199 (for frame shuffling comparison).
# =============================================================================

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
export NCCL_BLOCKING_WAIT=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# =========================
# PREFLIGHT
# =========================
echo "[PREFLIGHT] Checking AWS credentials..."
aws sts get-caller-identity || { echo "[FATAL] AWS CLI credentials check failed"; exit 1; }
echo "[PREFLIGHT] AWS credentials verified"

# =========================
# S3 CONFIG
# =========================
ART_BUCKET="sagemaker-hyperpod-lifecycle-495467399120-usw2"
S3_BASE="s3://${ART_BUCKET}/vjepa2-artifacts"
CSV_S3="${S3_BASE}/data/csv/mimic_annotations_s3.csv"
ENV_S3="${S3_BASE}/env/vjepa2-312-pack-env.tar"

# Resume checkpoint from the previous run
PREV_RUN_S3="${S3_BASE}/runs/videomae_resume_e54_354/training_folder"
RUN_S3="${S3_BASE}/runs/videomae_e200_${SLURM_JOB_ID}"

# =========================
# COMPUTE CONFIG
# =========================
TARGET_EFF_BS=1024
WORLD_SIZE=8
BATCH_PER_GPU=32
MICRO_GLOBAL_BS=$((BATCH_PER_GPU * WORLD_SIZE))  # 256
ACCUM_ITER=$((TARGET_EFF_BS / MICRO_GLOBAL_BS))   # 4

TARGET_FPS=8
NUM_FRAMES=16
LR_VAL="1.5e-4"
MIN_LR_VAL="1e-5"

# =========================
# LOGGING / SYNC
# =========================
sync_ckpts() {
  local outdir="${OUTPUT_DIR:-output}"
  if [ -d "${outdir}" ]; then
    aws s3 sync "${outdir}" "${RUN_S3}/training_folder/" \
      --exclude "*" \
      --include "*.pth" --include "*.pt" --include "*.yaml" --include "*.json" --include "*.csv" \
      --only-show-errors || true
  fi
}

upload_logs() {
  local out="/tmp/mae_e200_${SLURM_JOB_ID}.out"
  local err="/tmp/mae_e200_${SLURM_JOB_ID}.err"
  [ -f "${out}" ] && aws s3 cp "${out}" "${RUN_S3}/logs/job.out" --only-show-errors || true
  [ -f "${err}" ] && aws s3 cp "${err}" "${RUN_S3}/logs/job.err" --only-show-errors || true
}

on_exit() {
  rc=$?
  upload_logs
  sync_ckpts
  exit $rc
}

trap on_exit EXIT
trap 'sync_ckpts; upload_logs' USR1

# =========================
# SETUP
# =========================
nvidia-smi -L || true
SCRATCH="/opt/dlami/nvme"
[ ! -d "$SCRATCH" ] && SCRATCH="/tmp"
export WORKDIR="${SCRATCH}/mae_e200_${SLURM_JOB_ID}"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

export VIDEOMAE_TMP="${WORKDIR}/tmp_videos"
mkdir -p "${VIDEOMAE_TMP}"

# Env
aws s3 cp "$ENV_S3" env.tar
mkdir -p env && tar -xf env.tar -C env
set +u && source env/bin/activate && set -u
env/bin/conda-unpack || true
pip install "numpy<2.0" timm==0.4.12 boto3 tensorboardX decord --no-cache-dir

# =========================
# CODE — deploy from S3 tarball to scratch
# =========================
CODE_DIR="${WORKDIR}/code"
mkdir -p "$CODE_DIR"
aws s3 cp "${S3_BASE}/setup/vjepa2-src.tar.gz" "${WORKDIR}/vjepa2-src.tar.gz" --quiet
tar xzf "${WORKDIR}/vjepa2-src.tar.gz" -C "$CODE_DIR"
cd "$CODE_DIR"

pip install -e . --no-deps --no-build-isolation 2>/dev/null || true
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# Data
DATA_CSV="${WORKDIR}/data/train.csv"
mkdir -p "${WORKDIR}/data"
aws s3 cp "$CSV_S3" "$DATA_CSV"

# Safety Patch for Legacy PyTorch
if [ -f "utils.py" ] && grep -q "torch._six" utils.py; then
    sed -i 's/from torch._six import inf/from torch import inf/g' utils.py
fi

# =========================
# STAGE CHECKPOINT FOR RESUME
# =========================
OUTPUT_DIR="${WORKDIR}/output"
mkdir -p "$OUTPUT_DIR"

echo "[INFO] Downloading checkpoint-116 from previous run (job 354)..."
aws s3 cp "${PREV_RUN_S3}/checkpoint-116.pth" "${OUTPUT_DIR}/checkpoint-116.pth"
echo "[INFO] Checkpoint staged: $(ls -lh ${OUTPUT_DIR}/checkpoint-116.pth)"

# =========================
# COMPUTE EPOCHS
# =========================
NUM_SAMPLES=$(python3 - "${DATA_CSV}" <<'PY'
import sys; p=sys.argv[1]
n=0
with open(p,"r") as f:
    for line in f:
        line=line.strip()
        if not line or line.startswith("#"):
            continue
        if "," in line:
            first = line.split(",")[0].strip()
        else:
            first = line.split()[0].strip()
        if first.startswith("s3://"):
            n+=1
print(n)
PY
)
echo "[INFO] NUM_SAMPLES=${NUM_SAMPLES}"

STEPS_PER_EPOCH=$((NUM_SAMPLES / MICRO_GLOBAL_BS))
UPDATES_PER_EPOCH=$((STEPS_PER_EPOCH / ACCUM_ITER))

# Target: 200 VideoMAE epochs (~102K updates)
EPOCHS=200
# Warmup already done in original run; set to 0 for resume
WARMUP_EPOCHS=0

echo "[INFO] RESUME CONFIG:"
echo "       Resuming from epoch 116 (seed 0, original run)"
echo "       Target epochs: ${EPOCHS}"
echo "       Updates per epoch: ${UPDATES_PER_EPOCH}"
echo "       Remaining: $((EPOCHS - 116)) epochs = $(( (EPOCHS - 116) * UPDATES_PER_EPOCH )) updates"
echo "       Key checkpoints: e124, e149, e174, e199"

# =========================
# LAUNCH (RESUME)
# =========================
(
  while true; do
    sleep 900
    sync_ckpts
    upload_logs
  done
) &
SYNC_PID=$!

srun --kill-on-bad-exit=1 \
  python3 -u run_mae_pretraining.py \
    --data_path "${DATA_CSV}" \
    --mask_type tube --mask_ratio 0.9 \
    --model pretrain_videomae_large_patch16_224 \
    --resume "${OUTPUT_DIR}/checkpoint-116.pth" \
    --decoder_depth 4 \
    --batch_size "${BATCH_PER_GPU}" \
    --accum_iter "${ACCUM_ITER}" \
    --num_frames "${NUM_FRAMES}" \
    --sampling_rate "${TARGET_FPS}" \
    --opt adamw \
    --opt_betas 0.9 0.95 \
    --opt_eps 1e-6 \
    --lr "${LR_VAL}" \
    --min_lr "${MIN_LR_VAL}" \
    --weight_decay 0.04 \
    --warmup_epochs "${WARMUP_EPOCHS}" \
    --epochs "${EPOCHS}" \
    --log_dir "${OUTPUT_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_workers 6 \
    --clip_grad 0.5 \
    --save_ckpt_freq 5
RC=$?

kill $SYNC_PID 2>/dev/null || true
wait $SYNC_PID 2>/dev/null || true
upload_logs
sync_ckpts
exit $RC
