#!/bin/bash
#SBATCH -p dev
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH -t 7-00:00:00
#SBATCH -J mae_b_21k
#SBATCH -o /opt/dlami/nvme/logs/mae_b_21k_%j.out
#SBATCH -e /opt/dlami/nvme/logs/mae_b_21k_%j.err
#SBATCH --signal=B:USR1@300

set -euxo pipefail

# =============================================================================
# VideoMAE ViT-B pretraining on MIMIC-IV-Echo — ImageNet-21K init
# =============================================================================
# NeurIPS controlled comparison: same data (525K MIMIC), same compute budget,
# different model size (ViT-B vs ViT-L). Tests whether the temporal shortcut
# scales with model capacity.
#
# Matched to ViT-L MAE schedule: 100 epochs, effective BS 1024,
# cosine LR 1.0e-4 → 1e-5 (reduced from 1.5e-4 for ViT-B stability),
# tube masking 90%, decoder depth 4.
#
# Fixed: root s3_dataset.py now retries with random samples on S3 errors
# instead of returning zero-frame dummies (which poisoned MAE training).
# =============================================================================

export AWS_DEFAULT_REGION="us-west-2"
export AWS_REGION="us-west-2"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export MASTER_PORT=$((12000 + $RANDOM % 20000))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

export HOME="/home/ubuntu"
unset AWS_SHARED_CREDENTIALS_FILE 2>/dev/null || true
unset AWS_CONFIG_FILE 2>/dev/null || true
unset AWS_EC2_METADATA_DISABLED 2>/dev/null || true
export AWS_METADATA_SERVICE_TIMEOUT=10
export AWS_METADATA_SERVICE_NUM_ATTEMPTS=5

export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1

echo "[PREFLIGHT] Checking AWS credentials..."
aws sts get-caller-identity || { echo "[FATAL] AWS CLI credentials check failed"; exit 1; }
echo "[PREFLIGHT] OK"

# =========================
# S3 CONFIG
# =========================
ART_BUCKET="sagemaker-hyperpod-lifecycle-495467399120-usw2"
S3_BASE="s3://${ART_BUCKET}/vjepa2-artifacts"
CSV_S3="${S3_BASE}/data/csv/mimic_annotations_s3.csv"
ENV_S3="${S3_BASE}/env/vjepa2-312-pack-env.tar"
CKPT_S3="${S3_BASE}/checkpoints/vitb_in21k.pt"
RUN_S3="${S3_BASE}/runs/mae_vitb_in21k_${SLURM_JOB_ID}"
# Optional: set RESUME_FROM_JOB=<prev_slurm_job_id> to pull prior checkpoints from S3.
# Example: RESUME_FROM_JOB=298 sbatch scripts/videomae_pretrain_mimic_vitb_in21k.sbatch
RESUME_FROM_JOB="${RESUME_FROM_JOB:-}"

# =========================
# COMPUTE CONFIG
# =========================
TARGET_EFF_BS=1024
WORLD_SIZE=8
BATCH_PER_GPU=32
MICRO_GLOBAL_BS=$((BATCH_PER_GPU * WORLD_SIZE))
ACCUM_ITER=$((TARGET_EFF_BS / MICRO_GLOBAL_BS))

TARGET_FPS=8
NUM_FRAMES=16
LR_VAL="1.0e-4"
MIN_LR_VAL="1e-5"

# =========================
# LOGGING / SYNC
# =========================
mkdir -p /opt/dlami/nvme/logs

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
  local out="/opt/dlami/nvme/logs/mae_b_21k_${SLURM_JOB_ID}.out"
  local err="/opt/dlami/nvme/logs/mae_b_21k_${SLURM_JOB_ID}.err"
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
export WORKDIR="${SCRATCH}/mae_b_21k_${SLURM_JOB_ID}"
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

# VideoMAE entrypoint (run_mae_pretraining.py, modeling_pretrain.py, utils.py) lives in a subdir.
# cd there so imports like "from modeling_pretrain import ..." resolve.
VIDEOMAE_DIR="${CODE_DIR}/evals/video_classification_frozen/modelcustom/VideoMAE"
cd "$VIDEOMAE_DIR"
export PYTHONPATH="$VIDEOMAE_DIR:${PYTHONPATH}"

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
# CHECKPOINT INIT (ImageNet-21K ViT-B → VideoMAE format)
# =========================
OUTPUT_DIR="${WORKDIR}/output"
mkdir -p "$OUTPUT_DIR"

# If RESUME_FROM_JOB is set, pull prior checkpoints from S3 so run_mae_pretraining.py
# --auto_resume (default) picks them up. checkpoint-latest.pth preferred over
# checkpoint-N.pth by auto_load_model (see evals/.../VideoMAE/utils.py).
if [ -n "${RESUME_FROM_JOB}" ]; then
  PRIOR_S3="${S3_BASE}/runs/mae_vitb_in21k_${RESUME_FROM_JOB}/training_folder/"
  echo "[INFO] RESUME_FROM_JOB=${RESUME_FROM_JOB}, pulling prior checkpoints from ${PRIOR_S3}"
  aws s3 sync "${PRIOR_S3}" "${OUTPUT_DIR}/" --only-show-errors || true
  ls -la "${OUTPUT_DIR}/" || true
fi

echo "[INFO] Downloading ViT-B IN21K checkpoint..."
aws s3 cp "$CKPT_S3" "${WORKDIR}/vitb_in21k.pt"

# Convert timm ViT-B → VideoMAE pretrain format
python3 - "${WORKDIR}/vitb_in21k.pt" "${WORKDIR}/vitb_mae_init.pth" <<'PY'
import sys, torch
IN_PATH, OUT_PATH = sys.argv[1], sys.argv[2]
DECODER_DEPTH = 4

raw = torch.load(IN_PATH, map_location="cpu", weights_only=False)

for k in list(raw.keys()):
    if k.startswith(("head.", "fc_norm.", "pre_logits.")):
        del raw[k]

from modeling_pretrain import pretrain_videomae_base_patch16_224
model = pretrain_videomae_base_patch16_224(pretrained=False, decoder_depth=DECODER_DEPTH)

new_sd = {}
tgt_sd = model.state_dict()
for k, v in raw.items():
    if k.startswith("cls_token"):
        continue
    # Inflate pos_embed: [1, 197, D] → [1, 1568, D] (196 spatial × 8 temporal)
    if k == "pos_embed":
        tgt_pe = tgt_sd.get("encoder.pos_embed", tgt_sd.get("pos_embed"))
        if tgt_pe is not None:
            sv = v[:, 1:, :] if v.shape[1] == 197 else v  # strip cls token
            N = tgt_pe.shape[1]
            if sv.shape[1] == 196 and N % 196 == 0:
                rep = N // 196
                new_sd["encoder.pos_embed"] = sv.repeat(1, rep, 1)
                print(f"[INFO] pos_embed: {v.shape} → {new_sd['encoder.pos_embed'].shape} (repeat {rep}x)")
        continue
    if k == "patch_embed.proj.weight":
        v3d = v.unsqueeze(2).repeat(1, 1, 2, 1, 1) / 2.0
        new_sd["encoder." + k] = v3d
        continue
    if k == "patch_embed.proj.bias":
        new_sd["encoder." + k] = v
        continue
    sv = None
    for candidate in [k, "encoder." + k]:
        if candidate in model.state_dict():
            sv = candidate
            break
    if sv is None:
        sv = "encoder." + k
    new_sd[sv] = v

for k, v in model.state_dict().items():
    if k.startswith(("decoder.", "mask_token", "decoder_")):
        if k not in new_sd:
            new_sd[k] = v
    if k.startswith("encoder.") and k not in new_sd:
        new_sd[k] = v

torch.save({"model": new_sd}, OUT_PATH)
print(f"[INFO] Wrote VideoMAE-B init ckpt: {OUT_PATH}")
print(f"[INFO] Keys: {len(new_sd)}")
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
        if first.startswith("s3://") or first.startswith("/"):
            n+=1
print(n)
PY
)
echo "[INFO] NUM_SAMPLES=${NUM_SAMPLES}"

STEPS_PER_EPOCH=$((NUM_SAMPLES / MICRO_GLOBAL_BS))
UPDATES_PER_EPOCH=$((STEPS_PER_EPOCH / ACCUM_ITER))

EPOCHS=100
WARMUP_EPOCHS=40

echo "[INFO] VideoMAE ViT-B IN21K CONFIG:"
echo "       Samples: ${NUM_SAMPLES}"
echo "       Effective BS: ${TARGET_EFF_BS}"
echo "       Batch/GPU: ${BATCH_PER_GPU}"
echo "       Accum iter: ${ACCUM_ITER}"
echo "       Steps/epoch: ${STEPS_PER_EPOCH}"
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
    --model pretrain_videomae_base_patch16_224 \
    --finetune "${WORKDIR}/vitb_mae_init.pth" \
    --no_auto_resume \
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
