#!/bin/bash
#SBATCH -p dev
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 12:00:00
#SBATCH -J pvl_lvef
#SBATCH -o /opt/dlami/nvme/logs/pvl_lvef_%j.out
#SBATCH -e /opt/dlami/nvme/logs/pvl_lvef_%j.err
#SBATCH --nodelist=ip-10-0-50-35

set -euxo pipefail

# =============================================================================
# phase_mask (Run 4, 389) + localblock (Run 2, 393) -> EchoNet-Dynamic LVEF
# Trains d=4 attentive probes at three checkpoint points:
#   phase_mask_389 : e25, e50
#   localblock_393 : e25
# Same 6-HP grid as phi_jepa_end_lvef_e25_e50_e75.sbatch for apples-to-apples.
# =============================================================================

export AWS_DEFAULT_REGION="us-west-2"
export AWS_REGION="us-west-2"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export HOME="/home/ubuntu"
unset AWS_SHARED_CREDENTIALS_FILE 2>/dev/null || true
unset AWS_CONFIG_FILE 2>/dev/null || true
unset AWS_EC2_METADATA_DISABLED 2>/dev/null || true
export AWS_METADATA_SERVICE_TIMEOUT=10
export AWS_METADATA_SERVICE_NUM_ATTEMPTS=5

aws sts get-caller-identity || exit 1

ART_BUCKET="sagemaker-hyperpod-lifecycle-495467399120-usw2"
S3_BASE="s3://${ART_BUCKET}/vjepa2-artifacts"
ENV_S3="${S3_BASE}/env/vjepa2-312-pack-env.tar"

PM_RUN_S3="${S3_BASE}/runs/phase_mask_389/checkpoints"
LB_RUN_S3="${S3_BASE}/runs/localblock_393/checkpoints"
RUN_S3="${S3_BASE}/runs/phase_vs_localblock_lvef_probes_${SLURM_JOB_ID}"

# Jobs to run: "<model>:<epoch>"
JOBS=(
  "phase_mask:25"
  "phase_mask:50"
  "localblock:25"
)

upload_logs() {
  local out="/opt/dlami/nvme/logs/pvl_lvef_${SLURM_JOB_ID}.out"
  local err="/opt/dlami/nvme/logs/pvl_lvef_${SLURM_JOB_ID}.err"
  [ -f "${out}" ] && aws s3 cp "${out}" "${RUN_S3}/logs/job.out" --only-show-errors || true
  [ -f "${err}" ] && aws s3 cp "${err}" "${RUN_S3}/logs/job.err" --only-show-errors || true
}
trap 'upload_logs' EXIT

nvidia-smi -L || true
mkdir -p /opt/dlami/nvme/logs

SCRATCH="/opt/dlami/nvme"
WORKDIR="${SCRATCH}/pvl_lvef_${SLURM_JOB_ID}"
mkdir -p "$WORKDIR"

# Env
aws s3 cp "$ENV_S3" "${WORKDIR}/env.tar"
mkdir -p "${WORKDIR}/env" && tar -xf "${WORKDIR}/env.tar" -C "${WORKDIR}/env"
set +u && source "${WORKDIR}/env/bin/activate" && set -u
"${WORKDIR}/env/bin/conda-unpack" || true
export TMPDIR="${WORKDIR}/tmp" && mkdir -p "$TMPDIR"
pip install boto3 --no-cache-dir --quiet

# Code
CODE_DIR="${WORKDIR}/code"
mkdir -p "$CODE_DIR"
aws s3 cp "${S3_BASE}/setup/vjepa2-src.tar.gz" "${WORKDIR}/vjepa2-src.tar.gz" --quiet
tar xzf "${WORKDIR}/vjepa2-src.tar.gz" -C "$CODE_DIR"
cd "$CODE_DIR"
pip install -e . --no-deps --no-build-isolation 2>/dev/null || true
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# Data
DATA_DIR="${WORKDIR}/data/csv"
mkdir -p "$DATA_DIR"
aws s3 cp "${S3_BASE}/data/csv/echonet_dynamic_train_s3_raw.csv" "$DATA_DIR/" --quiet
aws s3 cp "${S3_BASE}/data/csv/echonet_dynamic_val_s3_raw.csv" "$DATA_DIR/" --quiet

CKPT_DIR="${WORKDIR}/checkpoints"
mkdir -p "$CKPT_DIR"

# Download all checkpoints up front
for J in "${JOBS[@]}"; do
  MODEL="${J%%:*}"
  EP="${J##*:}"
  if [ "$MODEL" = "phase_mask" ]; then
    SRC="${PM_RUN_S3}/e${EP}.pt"
  else
    SRC="${LB_RUN_S3}/e${EP}.pt"
  fi
  DEST="${CKPT_DIR}/${MODEL}_e${EP}.pt"
  echo "[INFO] Downloading ${MODEL} e${EP}..."
  aws s3 cp "$SRC" "$DEST" --quiet
  ls -lh "$DEST"
done

# Generate configs
for J in "${JOBS[@]}"; do
  MODEL="${J%%:*}"
  EP="${J##*:}"
  EVAL_CFG="configs/eval/vitl/neurips/${MODEL}_e${EP}_end_lvef_d4.yaml"
  FOLDER_NAME="${MODEL}_e${EP}_end_lvef_224"
  TAG="neurips-${MODEL//_/-}-e${EP}-end-lvef-d4"
  mkdir -p "$(dirname $EVAL_CFG)"
  cat > "$EVAL_CFG" <<YAMLEOF
app: vjepa
cpus_per_task: 32
folder: evals/vitl/neurips/${FOLDER_NAME}
mem_per_gpu: 80G
nodes: 1
tasks_per_node: 8
num_workers: 8
eval_name: video_classification_frozen
resume_checkpoint: false
tag: ${TAG}
experiment:
  classifier:
    task_type: regression
    num_heads: 16
    num_probe_blocks: 4
    num_targets: 1
  data:
    dataset_type: VideoDataset
    dataset_train: ${DATA_DIR}/echonet_dynamic_train_s3_raw.csv
    dataset_val: ${DATA_DIR}/echonet_dynamic_val_s3_raw.csv
    target_mean: 55.7776
    target_std: 12.4072
    resolution: 224
    frames_per_clip: 16
    frame_step: 2
    num_segments: 2
    num_views_per_segment: 1
  optimization:
    batch_size: 1
    num_epochs: 20
    use_bfloat16: true
    use_pos_embed: false
    multihead_kwargs:
    - {final_lr: 0.0, final_weight_decay: 0.01, lr: 0.0001, start_lr: 0.0001, warmup: 0.0, weight_decay: 0.01}
    - {final_lr: 0.0, final_weight_decay: 0.1, lr: 0.0001, start_lr: 0.0001, warmup: 0.0, weight_decay: 0.1}
    - {final_lr: 0.0, final_weight_decay: 0.4, lr: 0.0001, start_lr: 0.0001, warmup: 0.0, weight_decay: 0.4}
    - {final_lr: 0.0, final_weight_decay: 0.01, lr: 0.00005, start_lr: 0.00005, warmup: 0.0, weight_decay: 0.01}
    - {final_lr: 0.0, final_weight_decay: 0.1, lr: 0.00005, start_lr: 0.00005, warmup: 0.0, weight_decay: 0.1}
    - {final_lr: 0.0, final_weight_decay: 0.4, lr: 0.00005, start_lr: 0.00005, warmup: 0.0, weight_decay: 0.4}
model_kwargs:
  checkpoint: ${CKPT_DIR}/${MODEL}_e${EP}.pt
  module_name: evals.video_classification_frozen.modelcustom.vit_encoder_multiclip
  pretrain_kwargs:
    encoder:
      checkpoint_key: target_encoder
      model_name: vit_large
      patch_size: 16
      tubelet_size: 2
      uniform_power: true
      use_rope: true
  wrapper_kwargs:
    max_frames: 128
    use_pos_embed: false
YAMLEOF
  echo "[INFO] Created config: $EVAL_CFG"
done

# Train probes sequentially
for J in "${JOBS[@]}"; do
  MODEL="${J%%:*}"
  EP="${J##*:}"
  EVAL_CFG="configs/eval/vitl/neurips/${MODEL}_e${EP}_end_lvef_d4.yaml"
  FOLDER="evals/vitl/neurips/${MODEL}_e${EP}_end_lvef_224"
  echo "============================================"
  echo "[INFO] Training ${MODEL} e${EP} LVEF probe..."
  echo "============================================"
  python3 -m evals.main --fname "$EVAL_CFG" \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7
  echo "[INFO] ${MODEL} e${EP} probe training complete"

  aws s3 sync "$FOLDER/" \
    "${RUN_S3}/${MODEL}_e${EP}_lvef/" --only-show-errors || true

  upload_logs
  rm -f "${CKPT_DIR}/${MODEL}_e${EP}.pt"
done

echo "[INFO] All phase-vs-localblock LVEF probes complete"
