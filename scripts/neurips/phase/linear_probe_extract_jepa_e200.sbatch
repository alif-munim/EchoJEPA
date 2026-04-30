#!/bin/bash
#SBATCH -p dev
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 01:00:00
#SBATCH -J lin_extr_jepa
#SBATCH -o /opt/dlami/nvme/logs/lin_extr_jepa_%j.out
#SBATCH -e /opt/dlami/nvme/logs/lin_extr_jepa_%j.err

set -euxo pipefail

# Extracts pre-pool features for JEPA IN21K e200 on EchoNet-Dynamic train+test.
# Outputs uploaded to s3://.../features/diff_probe/jepa_e200_{train,test}.pt

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
FEATURES_S3="${S3_BASE}/features/diff_probe"

mkdir -p /opt/dlami/nvme/logs
SCRATCH="/opt/dlami/nvme"
WORKDIR="${SCRATCH}/lin_extr_jepa_${SLURM_JOB_ID}"
mkdir -p "$WORKDIR"

aws s3 cp "$ENV_S3" "${WORKDIR}/env.tar"
mkdir -p "${WORKDIR}/env" && tar -xf "${WORKDIR}/env.tar" -C "${WORKDIR}/env"
set +u && source "${WORKDIR}/env/bin/activate" && set -u
"${WORKDIR}/env/bin/conda-unpack" || true
export TMPDIR="${WORKDIR}/tmp" && mkdir -p "$TMPDIR"
pip install boto3 --no-cache-dir --quiet

CODE_DIR="${WORKDIR}/code"
mkdir -p "$CODE_DIR"
aws s3 cp "${S3_BASE}/setup/vjepa2-src.tar.gz" "${WORKDIR}/vjepa2-src.tar.gz" --quiet
tar xzf "${WORKDIR}/vjepa2-src.tar.gz" -C "$CODE_DIR"
cd "$CODE_DIR"
pip install -e . --no-deps --no-build-isolation 2>/dev/null || true
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

DATA_DIR="${WORKDIR}/data/csv"
mkdir -p "$DATA_DIR"
aws s3 cp "${S3_BASE}/data/csv/echonet_dynamic_train_s3_raw.csv" "${DATA_DIR}/" --quiet
aws s3 cp "${S3_BASE}/data/csv/echonet_dynamic_test_s3_raw.csv" "${DATA_DIR}/" --quiet

CKPT_DIR="${WORKDIR}/checkpoints"
mkdir -p "$CKPT_DIR"
aws s3 cp "${S3_BASE}/runs/jepa_in21k_e200_280/training_folder/latest.pt" "${CKPT_DIR}/jepa_in21k_vitl_e200.pt" --quiet

FEAT_DIR="${WORKDIR}/features"
mkdir -p "$FEAT_DIR"

extract() {
  local SPLIT="$1"
  local CSV="${DATA_DIR}/echonet_dynamic_${SPLIT}_s3_raw.csv"
  local OUT="${FEAT_DIR}/jepa_e200_${SPLIT}.pt"
  local CFG="${WORKDIR}/cfg_${SPLIT}.yaml"
  cat > "$CFG" <<YAML
app: vjepa
folder: ${WORKDIR}/evals/jepa_e200_${SPLIT}_prepool
eval_name: feature_extraction_pre_pool
tag: lin-extr-jepa-e200-${SPLIT}
experiment:
  data:
    dataset_type: VideoDataset
    dataset_val: ${CSV}
    resolution: 224
    frames_per_clip: 16
    frame_step: 2
    num_segments: 2
    num_views_per_segment: 1
  optimization:
    batch_size: 8
    use_bfloat16: true
model_kwargs:
  checkpoint: ${CKPT_DIR}/jepa_in21k_vitl_e200.pt
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
YAML
  echo "[EXTRACT] jepa_e200 ${SPLIT}"
  FEATURES_OUTPUT="${OUT}" \
  python3 -m evals.main --fname "${CFG}" \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7
  aws s3 cp "${OUT}" "${FEATURES_S3}/jepa_e200_${SPLIT}.pt" --only-show-errors
  echo "[UPLOAD] jepa_e200_${SPLIT}.pt"
}

extract train
extract test

echo "[DONE] JEPA e200 extractions"
