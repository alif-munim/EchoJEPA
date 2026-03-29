#!/bin/bash
#SBATCH -p ml.p5e.48xlarge
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH -t 4-00:00:00
#SBATCH -J videomae_matched
#SBATCH -o /tmp/videomae_%j.out
#SBATCH -e /tmp/videomae_%j.err
#SBATCH --signal=B:USR1@300

set -euxo pipefail

# =============================================================================
# VideoMAE-L Pretraining on MIMIC-IV-Echo (525K clips)
# Compute-matched to V-JEPA 2 / BYOL ViT-L 50-epoch runs
# =============================================================================
#
# EPOCH MATCHING EXPLANATION:
#
# V-JEPA/BYOL use ipe=300 (iterations per epoch), NOT full data passes.
# With 525K samples and BS=1024, a full data pass = ~513 optimizer updates.
# But each V-JEPA "epoch" runs exactly 300 updates.
#
#   50 V-JEPA epochs = 50 * 300 = 15,000 optimizer updates = ~29.2 real data passes
#   V-JEPA warmup    = 40 * 300 = 12,000 updates
#
# VideoMAE uses real epochs (full data passes). With BS=1024:
#   1 VideoMAE epoch = ~513 optimizer updates
#   30 VideoMAE epochs = ~15,390 updates (matches V-JEPA's 15,000)
#   24 warmup epochs = ~12,312 updates (matches V-JEPA's 12,000)
#
# V-JEPA uses CONSTANT LR after warmup (final_lr = lr = 1.75e-4).
# VideoMAE uses cosine decay (standard recipe). This is a framework difference.
#
# LR: VideoMAE standard base_lr=1.5e-4, linearly scaled by BS/256.
#   1.5e-4 * (1024/256) = 6.0e-4 peak LR (standard VideoMAE recipe)
#   min_lr: 1e-5 * 4 = 4e-5 (end of cosine)
#
# OLD BROKEN CONFIG (videomae_pretrain_mimic.sbatch):
#   LR = 8.79e-7 -> scaled to 3.52e-6 (170x too low!)
#   min_lr = 1e-6 -> scaled to 4e-6 (HIGHER than peak = inverted schedule)
#   Targeting 84,000 updates (wrong), resuming from broken checkpoint
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

# =========================
# COMPUTE-MATCHED CONFIG
# =========================
# V-JEPA 50 epochs = 15,000 updates, warmup 40 epochs = 12,000 updates
TARGET_JEPA_UPDATES=15000
TARGET_WARMUP_UPDATES=12000
TARGET_EFF_BS=1024

# VideoMAE micro-batch: 32/GPU * 8 GPUs = 256, accum 4 -> effective 1024
WORLD_SIZE=8
BATCH_PER_GPU=32
MICRO_GLOBAL_BS=$((BATCH_PER_GPU * WORLD_SIZE))  # 256
ACCUM_ITER=$((TARGET_EFF_BS / MICRO_GLOBAL_BS))   # 4

if [ $((MICRO_GLOBAL_BS * ACCUM_ITER)) -ne "${TARGET_EFF_BS}" ]; then
  echo "[FATAL] MICRO_GLOBAL_BS * ACCUM_ITER must equal TARGET_EFF_BS"
  exit 1
fi

# Protocol (matched to V-JEPA/BYOL)
TARGET_FPS=8
NUM_FRAMES=16

# S3 Config
ART_BUCKET="sagemaker-hyperpod-lifecycle-495467399120-usw2"
S3_BASE="s3://${ART_BUCKET}/vjepa2-artifacts"
CODE_S3="${S3_BASE}/code/videomae-code.tar"
CSV_S3="${S3_BASE}/data/csv/mimic_annotations_s3.csv"
ENV_S3="${S3_BASE}/env/vjepa2-312-pack-env.tar"
VITL_RAW_S3="${S3_BASE}/checkpoints/vitl_raw.pth"
RUN_S3="${S3_BASE}/runs/videomae_matched_${SLURM_JOB_ID}"

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

# Checkpoint Init (ImageNet ViT-L -> VideoMAE format)
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

# --- Safety Patch for Legacy PyTorch ---
if [ -f "utils.py" ]; then
    sed -i 's/from torch._six import inf/from torch import inf/g' utils.py
fi

# =========================================================
# COMPUTE UPDATE-MATCHED EPOCH COUNT
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

# Update-matched: same number of optimizer updates as V-JEPA 50 epochs
EPOCHS=$(( (TARGET_JEPA_UPDATES + UPDATES_PER_EPOCH - 1) / UPDATES_PER_EPOCH ))
WARMUP_EPOCHS=$(( (TARGET_WARMUP_UPDATES + UPDATES_PER_EPOCH - 1) / UPDATES_PER_EPOCH ))

ACTUAL_UPDATES=$((EPOCHS * UPDATES_PER_EPOCH))
ACTUAL_WARMUP=$((WARMUP_EPOCHS * UPDATES_PER_EPOCH))

echo "[INFO] COMPUTE-MATCHED CONFIG:"
echo "       V-JEPA target: ${TARGET_JEPA_UPDATES} total updates, ${TARGET_WARMUP_UPDATES} warmup updates"
echo "       Updates per VideoMAE epoch: ${UPDATES_PER_EPOCH}"
echo "       VideoMAE epochs: ${EPOCHS} (= ${ACTUAL_UPDATES} updates)"
echo "       VideoMAE warmup: ${WARMUP_EPOCHS} epochs (= ${ACTUAL_WARMUP} updates)"
echo "       Effective BS: ${TARGET_EFF_BS}"

# =========================
# LR CONFIG
# =========================
# Standard VideoMAE base_lr=1.5e-4, linearly scaled by effective_BS/256
# run_mae_pretraining.py line 251: args.lr = args.lr * effective_global_batch / 256
# So 1.5e-4 * 1024/256 = 6.0e-4 peak LR (standard recipe)
#
# min_lr=1e-5 * 4 = 4e-5 (end of cosine schedule)
# warmup_lr=1e-6 * 4 = 4e-6 (start of warmup)
#
# OLD BROKEN: LR=8.79e-7 -> 3.52e-6 after scaling (170x too low!)
LR_VAL="1.5e-4"
MIN_LR_VAL="1e-5"

# =========================
# LAUNCH (FRESH START)
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

# Graceful cleanup
kill $SYNC_PID 2>/dev/null || true
wait $SYNC_PID 2>/dev/null || true
upload_logs
sync_ckpts
exit $RC
