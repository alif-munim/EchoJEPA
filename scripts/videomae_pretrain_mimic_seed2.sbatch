#!/bin/bash
#SBATCH -p ml-p5-48xlarge
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH -t 7-00:00:00
#SBATCH -J mae_seed2
#SBATCH -o /tmp/mae_seed2_%j.out
#SBATCH -e /tmp/mae_seed2_%j.err
#SBATCH --signal=B:USR1@300

set -euxo pipefail

# =============================================================================
# VideoMAE-L Pretraining on MIMIC-IV-Echo — SEED 2
# =============================================================================
# Multi-seed replication of the MAE temporal shortcut trajectory.
# Identical to the original run (videomae_pretrain_mimic_matched.sbatch)
# EXCEPT: --seed 42 (original used --seed 0)
#
# Purpose: Verify that the -343% -> +1% temporal shortcut trajectory
# (MAE transiently learns then abandons temporal features) is reproducible
# across random seeds. This is the paper's signature finding and needs
# seed replication for NeurIPS.
#
# Full 100-epoch equivalent (60K updates = 200 V-JEPA epochs).
# Saves checkpoints at e24/e49/e74/e99 for frame shuffling comparison.
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

# =========================
# AWS CREDENTIAL CONFIG
# =========================
export AWS_SDK_LOAD_CONFIG=1
export HOME="/home/ubuntu"
export AWS_SHARED_CREDENTIALS_FILE="${HOME}/.aws/credentials"
export AWS_CONFIG_FILE="${HOME}/.aws/config"
unset AWS_EC2_METADATA_DISABLED 2>/dev/null || true
export AWS_METADATA_SERVICE_TIMEOUT=5
export AWS_METADATA_SERVICE_NUM_ATTEMPTS=3

# =========================
# NCCL CONFIG
# =========================
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# =========================
# PREFLIGHT CREDENTIAL CHECK
# =========================
echo "[PREFLIGHT] Checking AWS credentials..."
aws sts get-caller-identity || { echo "[FATAL] AWS CLI credentials check failed"; exit 1; }

python3 - <<'PY'
import boto3
try:
    identity = boto3.client("sts").get_caller_identity()
    print(f"[PREFLIGHT] boto3 credentials OK: {identity['Arn']}")
except Exception as e:
    print(f"[FATAL] boto3 credentials check failed: {e}")
    exit(1)
PY

echo "[PREFLIGHT] AWS credentials verified successfully"

# =========================
# COMPUTE CONFIG
# =========================
# Full 100-epoch equivalent: 200 V-JEPA epochs = 60,000 updates
# (Original seed-0 run used this same target)
TARGET_JEPA_UPDATES=60000
TARGET_WARMUP_UPDATES=12000
TARGET_EFF_BS=1024

WORLD_SIZE=8
BATCH_PER_GPU=32
MICRO_GLOBAL_BS=$((BATCH_PER_GPU * WORLD_SIZE))  # 256
ACCUM_ITER=$((TARGET_EFF_BS / MICRO_GLOBAL_BS))   # 4

if [ $((MICRO_GLOBAL_BS * ACCUM_ITER)) -ne "${TARGET_EFF_BS}" ]; then
  echo "[FATAL] MICRO_GLOBAL_BS * ACCUM_ITER must equal TARGET_EFF_BS"
  exit 1
fi

TARGET_FPS=8
NUM_FRAMES=16

# S3 Config
ART_BUCKET="sagemaker-hyperpod-lifecycle-495467399120-usw2"
S3_BASE="s3://${ART_BUCKET}/vjepa2-artifacts"
CSV_S3="${S3_BASE}/data/csv/mimic_annotations_s3.csv"
ENV_S3="${S3_BASE}/env/vjepa2-312-pack-env.tar"
VITL_RAW_S3="${S3_BASE}/checkpoints/vitl_raw.pth"
RUN_S3="${S3_BASE}/runs/videomae_seed2_${SLURM_JOB_ID}"

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
  local out="/tmp/mae_seed2_${SLURM_JOB_ID}.out"
  local err="/tmp/mae_seed2_${SLURM_JOB_ID}.err"
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
export WORKDIR="${SCRATCH}/mae_seed2_${SLURM_JOB_ID}"
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
# CODE — deploy from S3 tarball to scratch (avoids /opt/vjepa2 permission issues)
# =========================
CODE_DIR="${WORKDIR}/code"
mkdir -p "$CODE_DIR"
aws s3 cp "${S3_BASE}/setup/vjepa2-src.tar.gz" "${WORKDIR}/vjepa2-src.tar.gz" --quiet
tar xzf "${WORKDIR}/vjepa2-src.tar.gz" -C "$CODE_DIR"
cd "$CODE_DIR"

pip install -e . --no-deps --no-build-isolation 2>/dev/null || true
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# Data (write to scratch, not /opt/vjepa2 which is read-only)
mkdir -p "${WORKDIR}/data"
aws s3 cp "$CSV_S3" "${WORKDIR}/data/train.csv"

# Checkpoint Init (ImageNet ViT-L -> VideoMAE format)
# Download to scratch (repo dir is read-only)
echo "[INFO] Preparing Checkpoint..."
aws s3 cp "$VITL_RAW_S3" "${WORKDIR}/vitl_raw.pth"

python3 - <<'PY'
import torch
from modeling_pretrain import pretrain_videomae_large_patch16_224

import os
RAW = os.environ["WORKDIR"] + "/vitl_raw.pth"
OUT = os.environ["WORKDIR"] + "/pretrained_videomae_init.pth"
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

# --- Safety Patch for Legacy PyTorch ---
# utils.py is in /opt/vjepa2 which may be read-only; use sudo for the patch
if [ -f "utils.py" ] && grep -q "torch._six" utils.py; then
    sudo sed -i 's/from torch._six import inf/from torch import inf/g' utils.py 2>/dev/null || true
fi

# =========================================================
# COMPUTE UPDATE-MATCHED EPOCH COUNT
# =========================================================
DATA_CSV="${WORKDIR}/data/train.csv"

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

if [ "${UPDATES_PER_EPOCH}" -le 0 ]; then
  echo "[FATAL] UPDATES_PER_EPOCH computed as 0. Check batch sizes."
  exit 1
fi

EPOCHS=$(( (TARGET_JEPA_UPDATES + UPDATES_PER_EPOCH - 1) / UPDATES_PER_EPOCH ))
WARMUP_EPOCHS=$(( (TARGET_WARMUP_UPDATES + UPDATES_PER_EPOCH - 1) / UPDATES_PER_EPOCH ))

ACTUAL_UPDATES=$((EPOCHS * UPDATES_PER_EPOCH))
ACTUAL_WARMUP=$((WARMUP_EPOCHS * UPDATES_PER_EPOCH))

echo "[INFO] SEED 2 RUN — COMPUTE-MATCHED CONFIG:"
echo "       Seed: 42 (original run used seed 0)"
echo "       V-JEPA target: ${TARGET_JEPA_UPDATES} total updates, ${TARGET_WARMUP_UPDATES} warmup updates"
echo "       Updates per VideoMAE epoch: ${UPDATES_PER_EPOCH}"
echo "       VideoMAE epochs: ${EPOCHS} (= ${ACTUAL_UPDATES} updates)"
echo "       VideoMAE warmup: ${WARMUP_EPOCHS} epochs (= ${ACTUAL_WARMUP} updates)"
echo "       Effective BS: ${TARGET_EFF_BS}"
echo "       Key checkpoints: e24, e49, e74, e99 (for frame shuffling)"

# =========================
# LR CONFIG (identical to original)
# =========================
LR_VAL="1.5e-4"
MIN_LR_VAL="1e-5"

# =========================
# LAUNCH
# =========================
OUTPUT_DIR="${WORKDIR}/output"
mkdir -p "$OUTPUT_DIR"

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
    --data_path "${DATA_CSV}" \
    --mask_type tube --mask_ratio 0.9 \
    --model pretrain_videomae_large_patch16_224 \
    --finetune "${WORKDIR}/pretrained_videomae_init.pth" \
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
    --seed 42 \
    --log_dir "${OUTPUT_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_workers 6 \
    --clip_grad 0.5 \
    --save_ckpt_freq 5
RC=$?

# Graceful cleanup
kill $SYNC_PID 2>/dev/null || true
wait $SYNC_PID 2>/dev/null || true
upload_logs
sync_ckpts
exit $RC
