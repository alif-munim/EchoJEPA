#!/bin/bash
#SBATCH -p ml-p5-48xlarge
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH -t 4-00:00:00
#SBATCH -J mae_fgap_b
#SBATCH -o /tmp/mae_fgap_b_%j.out
#SBATCH -e /tmp/mae_fgap_b_%j.err
#SBATCH -w ip-10-0-50-184
#SBATCH --signal=B:USR1@300

set -euxo pipefail

# =============================================================================
# Pilot 2/2: Frame-Gap VideoMAE ViT-B (temporal gap masking, random init)
# 100 epochs on MIMIC-IV-Echo, checkpoints at e25/e50/e75/e100.
# Purpose: intervention for temporal shortcut hypothesis. With temporal gap
# masking, spatial interpolation is impossible — model must learn temporal
# dynamics throughout training. Expect temporal features retained at e100.
#
# Frame-gap masking: 8 temporal positions split into context[0:3], gap[3:5],
# target[5:8]. Visible patches ONLY in context frames. Gap+target all masked.
# Overall mask_ratio=0.9 preserved (~27% visible in context, 0% elsewhere).
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

# AWS credentials
export AWS_SDK_LOAD_CONFIG=1
export HOME="/home/ubuntu"
export AWS_SHARED_CREDENTIALS_FILE="${HOME}/.aws/credentials"
export AWS_CONFIG_FILE="${HOME}/.aws/config"
unset AWS_EC2_METADATA_DISABLED 2>/dev/null || true
export AWS_METADATA_SERVICE_TIMEOUT=5
export AWS_METADATA_SERVICE_NUM_ATTEMPTS=3

# NCCL
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# =========================
# PREFLIGHT
# =========================
echo "[PREFLIGHT] Checking AWS credentials..."
aws sts get-caller-identity || { echo "[FATAL] AWS credentials check failed"; exit 1; }
echo "[PREFLIGHT] OK"

# =========================
# TRAINING CONFIG
# =========================
EPOCHS=100
WARMUP_EPOCHS=10
SAVE_FREQ=25
BATCH_PER_GPU=32
ACCUM_ITER=4
NUM_FRAMES=16
TARGET_FPS=8
MASK_RATIO=0.9
MASK_TYPE=frame_gap
TEMPORAL_GAP=2
LR_VAL="1.5e-4"
MIN_LR_VAL="1e-5"

# S3 paths
ART_BUCKET="sagemaker-hyperpod-lifecycle-495467399120-usw2"
S3_BASE="s3://${ART_BUCKET}/vjepa2-artifacts"
CSV_S3="${S3_BASE}/data/csv/mimic_annotations_s3.csv"
ENV_S3="${S3_BASE}/env/vjepa2-312-pack-env.tar"
RUN_S3="${S3_BASE}/runs/mae_pilot_framegap_vitb_${SLURM_JOB_ID}"

# =========================
# S3 SYNC
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
  local out="/tmp/mae_fgap_b_${SLURM_JOB_ID}.out"
  local err="/tmp/mae_fgap_b_${SLURM_JOB_ID}.err"
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
export WORKDIR="${SCRATCH}/mae_fgap_b_${SLURM_JOB_ID}"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

export VIDEOMAE_TMP="${WORKDIR}/tmp_videos"
mkdir -p "${VIDEOMAE_TMP}"

# Environment
aws s3 cp "$ENV_S3" env.tar
mkdir -p env && tar -xf env.tar -C env
set +u && source env/bin/activate && set -u
env/bin/conda-unpack || true
pip install "numpy<2.0" timm==0.4.12 boto3 tensorboardX decord --no-cache-dir

# Code (deployed by ~/deploy.sh)
REPO_DIR="/opt/vjepa2"
cd "$REPO_DIR"
pip install -e . --no-deps --no-build-isolation 2>/dev/null || true
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# Data (download to WORKDIR so absolute path in --data_path works)
mkdir -p "${WORKDIR}/data"
aws s3 cp "$CSV_S3" "${WORKDIR}/data/train.csv"

# VideoMAE scripts live in this subdirectory
VIDEOMAE_DIR="${REPO_DIR}/evals/video_classification_frozen/modelcustom/VideoMAE"
cd "$VIDEOMAE_DIR"

# Safety patch for legacy PyTorch
if [ -f "utils.py" ]; then
    sed -i 's/from torch._six import inf/from torch import inf/g' utils.py
fi

# =========================
# LAUNCH (random init, no --finetune, frame_gap masking)
# =========================
OUTPUT_DIR="${WORKDIR}/output"
mkdir -p "$OUTPUT_DIR"

# Periodic S3 sync
(
  while true; do
    sleep 900
    sync_ckpts
    upload_logs
  done
) &
SYNC_PID=$!

srun --kill-on-bad-exit=1 \
  python -u run_mae_pretraining.py \
    --data_path "${WORKDIR}/data/train.csv" \
    --mask_type "${MASK_TYPE}" --mask_ratio "${MASK_RATIO}" \
    --temporal_gap "${TEMPORAL_GAP}" \
    --model pretrain_videomae_base_patch16_224 \
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
    --save_ckpt_freq "${SAVE_FREQ}"
RC=$?

kill $SYNC_PID 2>/dev/null || true
wait $SYNC_PID 2>/dev/null || true
upload_logs
sync_ckpts
exit $RC
