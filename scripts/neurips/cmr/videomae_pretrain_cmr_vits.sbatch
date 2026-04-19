#!/bin/bash
#SBATCH -p dev
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH -t 1-00:00:00
#SBATCH -J mae_cmr_s
#SBATCH -o /tmp/mae_cmr_s_%j.out
#SBATCH -e /tmp/mae_cmr_s_%j.err
#SBATCH --signal=B:USR1@300

set -euxo pipefail

# =============================================================================
# VideoMAE ViT-S pretraining on CMR SAX — ImageNet-21K init
# =============================================================================
# Cross-modality validation for NeurIPS temporal shortcut paper.
# Controlled comparison: JEPA vs MAE on 21.8K CMR SAX cine clips.
#
# ViT-S (384-dim, 12 blocks, 6 heads). Decoder: depth 4, dim 192.
# Data: SAX-only from MnM + MnM2 + Sunnybrook + DSB2 + CMR-Multi.
# ACDC held out for probe evaluation.
# =============================================================================

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

echo "[PREFLIGHT] Checking AWS credentials..."
aws sts get-caller-identity || { echo "[FATAL] AWS CLI credentials check failed"; exit 1; }
echo "[PREFLIGHT] OK"

# =========================
# S3 CONFIG
# =========================
ART_BUCKET="sagemaker-hyperpod-lifecycle-495467399120-usw2"
S3_BASE="s3://${ART_BUCKET}/vjepa2-artifacts"
CSV_S3="${S3_BASE}/data/csv/cmr_pretrain.csv"
ENV_S3="${S3_BASE}/env/vjepa2-312-pack-env.tar"
CKPT_S3="${S3_BASE}/checkpoints/vits_in21k.pt"
RUN_S3="${S3_BASE}/runs/mae_cmr_vits_${SLURM_JOB_ID}"

# =========================
# COMPUTE CONFIG
# =========================
# Dataset: 21,840 clips. BS=256 → 85 steps/epoch → 100 epochs = 8,500 updates.
TARGET_EFF_BS=256
WORLD_SIZE=8
BATCH_PER_GPU=32
MICRO_GLOBAL_BS=$((BATCH_PER_GPU * WORLD_SIZE))
ACCUM_ITER=$((TARGET_EFF_BS / MICRO_GLOBAL_BS))

TARGET_FPS=25
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
  local out="/tmp/mae_cmr_s_${SLURM_JOB_ID}.out"
  local err="/tmp/mae_cmr_s_${SLURM_JOB_ID}.err"
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
export WORKDIR="${SCRATCH}/mae_cmr_s_${SLURM_JOB_ID}"
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

# Safety Patches
if [ -f "utils.py" ]; then
    grep -q "torch._six" utils.py && \
      sed -i 's/from torch._six import inf/from torch import inf/g' utils.py
fi
find . -name "*.py" -exec grep -l "torch.load(" {} \; | while read f; do
    sed -i 's/torch\.load(\([^)]*\))/torch.load(\1, weights_only=False)/g' "$f"
    sed -i 's/weights_only=False, weights_only=False/weights_only=False/g' "$f"
done

# =========================
# CHECKPOINT INIT (ImageNet-21K ViT-S → VideoMAE format)
# =========================
OUTPUT_DIR="${WORKDIR}/output"
mkdir -p "$OUTPUT_DIR"

echo "[INFO] Downloading ViT-S IN21K checkpoint..."
aws s3 cp "$CKPT_S3" "${WORKDIR}/vits_in21k.pt"

# Convert timm ViT-S → VideoMAE pretrain format
python3 - "${WORKDIR}/vits_in21k.pt" "${WORKDIR}/vits_mae_init.pth" <<'PY'
import sys, torch
IN_PATH, OUT_PATH = sys.argv[1], sys.argv[2]
DECODER_DEPTH = 4

raw = torch.load(IN_PATH, map_location="cpu", weights_only=False)

# Remove head keys
for k in list(raw.keys()):
    if k.startswith(("head.", "fc_norm.", "pre_logits.")):
        del raw[k]

# Load VideoMAE model to get decoder init
from modeling_pretrain import pretrain_videomae_small_patch16_224
model = pretrain_videomae_small_patch16_224(pretrained=False, decoder_depth=DECODER_DEPTH)

# Map timm keys to VideoMAE encoder keys
new_sd = {}
for k, v in raw.items():
    if k.startswith(("cls_token", "pos_embed")):
        continue
    # patch_embed: inflate 2D → 3D (repeat across tubelet_size=2)
    if k == "patch_embed.proj.weight":
        # timm: [384, 3, 16, 16] → VideoMAE: [384, 3, 2, 16, 16]
        v3d = v.unsqueeze(2).repeat(1, 1, 2, 1, 1) / 2.0
        new_sd["encoder." + k] = v3d
        continue
    if k == "patch_embed.proj.bias":
        new_sd["encoder." + k] = v
        continue

    # Try encoder-prefixed key
    sv = None
    for candidate in [k, "encoder." + k]:
        if candidate in model.state_dict():
            sv = candidate
            break
    if sv is None:
        sv = "encoder." + k
    new_sd[sv] = v

# Add decoder weights from random init
for k, v in model.state_dict().items():
    if k.startswith(("decoder.", "mask_token", "decoder_")):
        if k not in new_sd:
            new_sd[k] = v
    # Also add any encoder keys we missed (pos_embed, cls_token for VideoMAE)
    if k.startswith("encoder.") and k not in new_sd:
        new_sd[k] = v

torch.save({"model": new_sd}, OUT_PATH)
print(f"[INFO] Wrote VideoMAE init ckpt: {OUT_PATH}")
print(f"[INFO] Keys: {len(new_sd)}, encoder: {sum(1 for k in new_sd if k.startswith('encoder.'))}, decoder: {sum(1 for k in new_sd if k.startswith('decoder.'))}")
PY

echo "[INFO] Init checkpoint ready"

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
        if first.endswith(".mp4"):
            n+=1
print(n)
PY
)
echo "[INFO] NUM_SAMPLES=${NUM_SAMPLES}"

STEPS_PER_EPOCH=$((NUM_SAMPLES / MICRO_GLOBAL_BS))
UPDATES_PER_EPOCH=$((STEPS_PER_EPOCH / ACCUM_ITER))

EPOCHS=800
WARMUP_EPOCHS=40

echo "[INFO] CMR MAE ViT-S CONFIG:"
echo "       Samples: ${NUM_SAMPLES}"
echo "       Effective BS: ${TARGET_EFF_BS}"
echo "       Steps/epoch: ${STEPS_PER_EPOCH}"
echo "       Updates/epoch: ${UPDATES_PER_EPOCH}"
echo "       Epochs: ${EPOCHS} (warmup: ${WARMUP_EPOCHS})"

# =========================
# LAUNCH
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
    --model pretrain_videomae_small_patch16_224 \
    --finetune "${WORKDIR}/vits_mae_init.pth" \
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
    --save_ckpt_freq 10
RC=$?

kill $SYNC_PID 2>/dev/null || true
wait $SYNC_PID 2>/dev/null || true
upload_logs
sync_ckpts
exit $RC
