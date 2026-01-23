#!/bin/bash
#SBATCH -p ml.p5e.48xlarge
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH -t 4-00:00:00
#SBATCH -J videomae_strict_retry
#SBATCH -o /tmp/videomae_%j.out
#SBATCH -e /tmp/videomae_%j.err
#SBATCH --signal=B:USR1@300

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

# =========================
# AWS CREDENTIAL CONFIG (Fix for long-running jobs)
# =========================
# Make boto3 reliably find the same config/creds as the AWS CLI
export AWS_SDK_LOAD_CONFIG=1

# Make sure boto3/worker processes look in the expected home for ~/.aws/*
export HOME="/home/ubuntu"
export AWS_SHARED_CREDENTIALS_FILE="${HOME}/.aws/credentials"
export AWS_CONFIG_FILE="${HOME}/.aws/config"

# If using instance profile/IMDS, don't accidentally disable it
unset AWS_EC2_METADATA_DISABLED 2>/dev/null || true
export AWS_METADATA_SERVICE_TIMEOUT=5
export AWS_METADATA_SERVICE_NUM_ATTEMPTS=3

# =========================
# NCCL CONFIG (Avoid 10-minute hangs on failures)
# =========================
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# =========================
# PREFLIGHT CREDENTIAL CHECK (Fail fast before hours of training)
# =========================
echo "[PREFLIGHT] Checking AWS credentials..."
aws sts get-caller-identity || { echo "[FATAL] AWS CLI credentials check failed"; exit 1; }

python - <<'PY'
import boto3
try:
    identity = boto3.client("sts").get_caller_identity()
    print(f"[PREFLIGHT] boto3 credentials OK: {identity['Arn']}")
except Exception as e:
    print(f"[FATAL] boto3 credentials check failed: {e}")
    exit(1)
PY

echo "[PREFLIGHT] AWS credentials verified successfully"

# --- COMPUTE MATCHING TARGETS (Base + Cooldown) ---
# Base: 72,000 updates
# Cooldown: 12,000 updates
# Total: 84,000 updates
TARGET_UPDATES=84000
TARGET_WARMUP_UPDATES=12000
TARGET_EFF_BS=1024

# --- VideoMAE Micro-batch Configuration ---
WORLD_SIZE=8
BATCH_PER_GPU=32
MICRO_GLOBAL_BS=$((BATCH_PER_GPU * WORLD_SIZE))
ACCUM_ITER=$((TARGET_EFF_BS / MICRO_GLOBAL_BS))

# Validate Batch Configuration
if [ $((MICRO_GLOBAL_BS * ACCUM_ITER)) -ne "${TARGET_EFF_BS}" ]; then
  echo "[FATAL] MICRO_GLOBAL_BS * ACCUM_ITER must equal TARGET_EFF_BS"
  exit 1
fi

# Protocol
TARGET_FPS=8
NUM_FRAMES=16

# S3 Config
ART_BUCKET="sagemaker-hyperpod-lifecycle-495467399120-usw2"
S3_BASE="s3://${ART_BUCKET}/vjepa2-artifacts"
CODE_S3="${S3_BASE}/code/videomae-code.tar"
CSV_S3="${S3_BASE}/data/csv/mimic_annotations_s3.csv"
ENV_S3="${S3_BASE}/env/vjepa2-312-pack-env.tar"
VITL_RAW_S3="${S3_BASE}/checkpoints/vitl_raw.pth"
RUN_S3="${S3_BASE}/runs/videomae_${SLURM_JOB_ID}"

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
  local out="/tmp/videomae_${SLURM_JOB_ID}.out"
  local err="/tmp/videomae_${SLURM_JOB_ID}.err"
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
export WORKDIR="${SCRATCH}/videomae_${SLURM_JOB_ID}"
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

# Code
echo "[INFO] Downloading code..."
aws s3 cp "$CODE_S3" code.tar
tar -xf code.tar

# Data
mkdir -p data
aws s3 cp "$CSV_S3" data/train.csv

# Checkpoint Init
echo "[INFO] Preparing Checkpoint..."
aws s3 cp "$VITL_RAW_S3" vitl_raw.pth

python - <<'PY'
import torch
from modeling_pretrain import pretrain_videomae_large_patch16_224

RAW = "vitl_raw.pth"
OUT = "pretrained_videomae_init.pth"
DECODER_DEPTH = 4

ckpt = torch.load(RAW, map_location="cpu")
src = ckpt.get("model", ckpt.get("state_dict", ckpt))

tmp = {k.replace("module.", ""): v for k, v in src.items() 
       if not k.startswith(("module.head.", "module.fc_norm.", "head.", "fc_norm."))}
src = tmp

model = pretrain_videomae_large_patch16_224(pretrained=False, decoder_depth=DECODER_DEPTH)
tgt_sd = model.state_dict()

def pick(*names):
    for n in names:
        if n in src: return src[n]
    return None

new_sd = {}
for k, v in tgt_sd.items():
    if k.startswith(("decoder.", "mask_token", "decoder_")):
        new_sd[k] = v
        continue

    sv = pick(k, k.replace("encoder.", ""), "encoder."+k)

    if sv is None and k.endswith("patch_embed.proj.weight"):
        sv = pick("patch_embed.proj.weight", "encoder.patch_embed.proj.weight")
    if sv is not None and k.endswith("patch_embed.proj.weight") and sv.ndim == 4 and v.ndim == 5:
        t = v.shape[2]
        sv = sv.unsqueeze(2).repeat(1, 1, t, 1, 1) / float(t)
        new_sd[k] = sv
        continue

    if sv is None and k.endswith("pos_embed"):
        sv = pick("pos_embed", "encoder.pos_embed")
    if sv is not None and k.endswith("pos_embed") and sv.ndim == 3 and v.ndim == 3:
        if sv.shape[1] == 197: sv_wo_cls = sv[:, 1:, :]
        else: sv_wo_cls = sv
        N = v.shape[1]
        if sv_wo_cls.shape[1] == 196 and N % 196 == 0:
            rep = N // 196
            new_sd[k] = sv_wo_cls.repeat(1, rep, 1)
            continue

    if sv is not None and sv.shape == v.shape:
        new_sd[k] = sv
    else:
        new_sd[k] = v

torch.save({"model": new_sd}, OUT)
print(f"[INFO] Wrote VideoMAE init ckpt: {OUT}")
PY

# --- Safety Patch for Legacy PyTorch (If not already fixed in tar) ---
if [ -f "utils.py" ]; then
    sed -i 's/from torch._six import inf/from torch import inf/g' utils.py
fi

# =========================================================
# COMPUTE MATCHED EPOCH CALCULATIONS
# =========================================================

NUM_SAMPLES=$(python - <<'PY'
p="data/train.csv"
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

if [ "${UPDATES_PER_EPOCH}" -le 0 ]; then
  echo "[FATAL] UPDATES_PER_EPOCH computed as 0. Check batch sizes."
  exit 1
fi

EPOCHS=$(( (TARGET_UPDATES + UPDATES_PER_EPOCH - 1) / UPDATES_PER_EPOCH ))
WARMUP_EPOCHS=$(( (TARGET_WARMUP_UPDATES + UPDATES_PER_EPOCH - 1) / UPDATES_PER_EPOCH ))

echo "[INFO] COMPUTE MATCHED CONFIG (Base + Cooldown):"
echo "       Total Updates: ${TARGET_UPDATES} -> ${EPOCHS} Epochs"
echo "       Warmup Updates: ${TARGET_WARMUP_UPDATES} -> ${WARMUP_EPOCHS} Epochs"

# Updated LR (Half of previous)
LR_VAL="8.7890625e-7"

# =========================
# LAUNCH
# =========================
OUTPUT_DIR="${WORKDIR}/output"
mkdir -p "$OUTPUT_DIR"

# DOWNLOAD CHECKPOINT (RESUME from Epoch 30)
# Note: Using 'checkpoint-29.pth' from previous run to be safe
aws s3 cp s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts/runs/videomae_97/training_folder/checkpoint-29.pth "${WORKDIR}/checkpoint-29.pth" || echo "[WARN] Checkpoint not found"

# Start periodic uploads
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
    --mask_type tube --mask_ratio 0.9 \
    --model pretrain_videomae_large_patch16_224 \
    --finetune "pretrained_videomae_init.pth" \
    --decoder_depth 4 \
    --batch_size "${BATCH_PER_GPU}" \
    --accum_iter "${ACCUM_ITER}" \
    --num_frames "${NUM_FRAMES}" \
    --sampling_rate "${TARGET_FPS}" \
    --opt adamw \
    --opt_betas 0.9 0.95 \
    --opt_eps 1e-6 \
    --lr "${LR_VAL}" \
    --min_lr 1.0e-6 \
    --weight_decay 0.04 \
    --warmup_epochs "${WARMUP_EPOCHS}" \
    --epochs "${EPOCHS}" \
    --log_dir "${OUTPUT_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_workers 6 \
    --clip_grad 0.5 \
    --resume "${WORKDIR}/checkpoint-29.pth" \
    --start_epoch 30

RC=$?

# graceful cleanup
kill $SYNC_PID 2>/dev/null || true
wait $SYNC_PID 2>/dev/null || true
upload_logs
sync_ckpts
exit $RC
