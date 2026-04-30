#!/bin/bash
#SBATCH -p dev
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 3:00:00
#SBATCH -J cmr_prepool
#SBATCH -o /opt/dlami/nvme/logs/cmr_prepool_%j.out
#SBATCH -e /opt/dlami/nvme/logs/cmr_prepool_%j.err

# Root fs is 100% full on this cluster; redirect temp/cache to NVMe.
mkdir -p /opt/dlami/nvme/tmp /opt/dlami/nvme/cache/pip
export TMPDIR=/opt/dlami/nvme/tmp
export TEMP=/opt/dlami/nvme/tmp
export TMP=/opt/dlami/nvme/tmp
export PIP_CACHE_DIR=/opt/dlami/nvme/cache/pip
export XDG_CACHE_HOME=/opt/dlami/nvme/cache

set -euxo pipefail

# =============================================================================
# CMR ACDC pre-pool feature extraction for probe-family diagnostics.
# Extracts [N, S=1, T=8, D=384] fp16 .pt caches for:
#   JEPA slow-EMA (job 346)  e30, e100, e200
#   MAE (job 183)            e30, e100, e200, e300
# For both ACDC LVEF and DX CSVs (train + test).
# ViT-S encoders, num_segments=1, 16 frames, 224px, matches the attentive
# CMR inference configs (cmr_j_lvmf_444 / cmr_lvmf_283).
# Env:
#   SMOKE=1  -> JEPA e30 only (both tasks)
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
JEPA_CKPT_S3="${S3_BASE}/runs/jepa_cmr_vits_slowema_346/training_folder"
MAE_CKPT_S3="${S3_BASE}/runs/mae_cmr_vits_183/training_folder"
FEATURES_S3="${S3_BASE}/features/cmr_prepool"
RUN_S3="${S3_BASE}/runs/cmr_prepool_${SLURM_JOB_ID}"
SCRATCH="/opt/dlami/nvme"
SMOKE="${SMOKE:-0}"

mkdir -p /opt/dlami/nvme/logs

upload_logs() {
  [ -f "/opt/dlami/nvme/logs/cmr_prepool_${SLURM_JOB_ID}.out" ] && \
    aws s3 cp "/opt/dlami/nvme/logs/cmr_prepool_${SLURM_JOB_ID}.out" "${RUN_S3}/logs/job.out" --only-show-errors || true
  [ -f "/opt/dlami/nvme/logs/cmr_prepool_${SLURM_JOB_ID}.err" ] && \
    aws s3 cp "/opt/dlami/nvme/logs/cmr_prepool_${SLURM_JOB_ID}.err" "${RUN_S3}/logs/job.err" --only-show-errors || true
}
trap 'upload_logs' EXIT

nvidia-smi -L || true
pkill -9 -f "evals.main" 2>/dev/null || true; sleep 2

# -------- env + code --------
aws s3 cp "$ENV_S3" "${SCRATCH}/env.tar"
mkdir -p "${SCRATCH}/env" && tar -xf "${SCRATCH}/env.tar" -C "${SCRATCH}/env"
set +u && source "${SCRATCH}/env/bin/activate" && set -u
"${SCRATCH}/env/bin/conda-unpack" || true
pip install boto3 scikit-learn --no-cache-dir --quiet

CODE_DIR="${SCRATCH}/code_cmr_prepool_${SLURM_JOB_ID}"
mkdir -p "$CODE_DIR"
aws s3 cp "${S3_BASE}/setup/vjepa2-src.tar.gz" "${SCRATCH}/vjepa2-src-cmr_prepool.tar.gz" --quiet
tar xzf "${SCRATCH}/vjepa2-src-cmr_prepool.tar.gz" -C "$CODE_DIR"
cd "$CODE_DIR"
pip install -e . --no-deps --no-build-isolation 2>/dev/null || true
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# -------- ACDC data --------
mkdir -p "${SCRATCH}/data/csv" "${SCRATCH}/data/acdc_clips" "${SCRATCH}/checkpoints"
echo "[INFO] Downloading ACDC clips..."
aws s3 sync "${S3_BASE}/data/cmr/clips/" "${SCRATCH}/data/acdc_clips/" \
  --exclude "*" --include "acdc_*" --only-show-errors

# LVEF + DX CSVs
for NAME in acdc_train acdc_test acdc_dx_train acdc_dx_test; do
  aws s3 cp "${S3_BASE}/data/csv/${NAME}_s3.csv" "${SCRATCH}/data/csv/${NAME}_s3.csv" --quiet
  sed "s|s3://.*clips/|${SCRATCH}/data/acdc_clips/|g" "${SCRATCH}/data/csv/${NAME}_s3.csv" > "${SCRATCH}/data/csv/${NAME}_local.csv"
  echo "[INFO] ${NAME}: $(wc -l < ${SCRATCH}/data/csv/${NAME}_local.csv) clips"
done

# -------- config templates --------
CONF_DIR="${CODE_DIR}/configs/inference/vits16/cmr"
mkdir -p "${CONF_DIR}"

make_yaml() {
  # args: OUT_YAML TAG CKPT_LOCAL MODULE_NAME MODEL_NAME CSV_LOCAL FOLDER
  local OUT="$1"; local TAG="$2"; local CKPT="$3"; local MODULE="$4"; local MODEL="$5"; local CSV="$6"; local FOLDER="$7"
  cat > "${OUT}" <<YAMLEOF
app: vjepa
eval_name: feature_extraction_pre_pool
tag: ${TAG}
folder: ${FOLDER}
val_only: true

experiment:
  data:
    dataset_type: VideoDataset
    dataset_val: ${CSV}
    resolution: 224
    frames_per_clip: 16
    frame_step: 1
    num_segments: 1
    num_views_per_segment: 1
  optimization:
    batch_size: 8
    use_bfloat16: true

model_kwargs:
  checkpoint: ${CKPT}
  module_name: ${MODULE}
  pretrain_kwargs:
    encoder:
      model_name: ${MODEL}
YAMLEOF
  # JEPA-style encoders take patch_size/tubelet_size + rope config.
  # VideoMAE ViT-S ctor bakes patch_size in and only accepts tubelet_size.
  if [[ "${MODULE}" == *vit_encoder_multiclip* ]]; then
    cat >> "${OUT}" <<YAMLEOF
      patch_size: 16
      tubelet_size: 2
      checkpoint_key: target_encoder
      uniform_power: true
      use_rope: true
  wrapper_kwargs:
    max_frames: 128
    use_pos_embed: false
YAMLEOF
  else
    cat >> "${OUT}" <<YAMLEOF
      tubelet_size: 2
  wrapper_kwargs: {}
YAMLEOF
  fi
}

# -------- extraction helper --------
extract_one() {
  # args: MODEL_TAG (jepa_e30|mae_e100|...) CKPT_LOCAL MODULE MODEL_NAME
  local MODEL_TAG="$1"; local CKPT="$2"; local MODULE="$3"; local MODEL_NAME="$4"

  for TASK in lvef dx; do
    case "${TASK}" in
      lvef)
        local CSV_TRAIN="${SCRATCH}/data/csv/acdc_train_local.csv"
        local CSV_TEST="${SCRATCH}/data/csv/acdc_test_local.csv"
        ;;
      dx)
        local CSV_TRAIN="${SCRATCH}/data/csv/acdc_dx_train_local.csv"
        local CSV_TEST="${SCRATCH}/data/csv/acdc_dx_test_local.csv"
        ;;
    esac

    for SPLIT in train test; do
      local CSV_LOCAL
      [[ "${SPLIT}" == "train" ]] && CSV_LOCAL="${CSV_TRAIN}" || CSV_LOCAL="${CSV_TEST}"
      local OUT_DIR="${SCRATCH}/features/cmr_prepool/${MODEL_TAG}/${TASK}_${SPLIT}"
      mkdir -p "${OUT_DIR}"
      local YAML="${CONF_DIR}/${MODEL_TAG}_${TASK}_${SPLIT}.yaml"
      make_yaml "${YAML}" "cmr-${MODEL_TAG}-${TASK}-${SPLIT}" "${CKPT}" "${MODULE}" "${MODEL_NAME}" "${CSV_LOCAL}" "${OUT_DIR}"
      echo "[INFO] extract ${MODEL_TAG} ${TASK} ${SPLIT}"
      FEATURES_OUTPUT="${OUT_DIR}/features.pt" \
      python3 -m evals.main --fname "${YAML}" \
        --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7
      # upload
      aws s3 cp "${OUT_DIR}/features.pt" \
        "${FEATURES_S3}/${MODEL_TAG}/${TASK}_${SPLIT}.pt" --only-show-errors
      echo "[INFO] uploaded features.pt for ${MODEL_TAG} ${TASK} ${SPLIT}"
    done
  done
}

# -------- run --------
JEPA_MODULE="evals.video_classification_frozen.modelcustom.vit_encoder_multiclip"
JEPA_MODEL_NAME="vit_small"
MAE_MODULE="evals.video_classification_frozen.modelcustom.videomae_encoder"
MAE_MODEL_NAME="vit_small_patch16_224"

if [[ "${SMOKE}" == "1" ]]; then
  EPOCHS_JEPA="30"
  EPOCHS_MAE=""
else
  EPOCHS_JEPA="30 100 200"
  EPOCHS_MAE="30 100 200 300"
fi

for EP in ${EPOCHS_JEPA}; do
  MODEL_TAG="jepa_slowema_e${EP}"
  CKPT_LOCAL="${SCRATCH}/checkpoints/jepa_cmr_slowema_e${EP}.pt"
  aws s3 cp "${JEPA_CKPT_S3}/e${EP}.pt" "${CKPT_LOCAL}" --quiet
  extract_one "${MODEL_TAG}" "${CKPT_LOCAL}" "${JEPA_MODULE}" "${JEPA_MODEL_NAME}"
  rm -f "${CKPT_LOCAL}"
done

for EP in ${EPOCHS_MAE}; do
  MODEL_TAG="mae_e${EP}"
  CKPT_IDX=$((EP - 1))
  CKPT_LOCAL="${SCRATCH}/checkpoints/mae_cmr_vits_ep${CKPT_IDX}.pth"
  aws s3 cp "${MAE_CKPT_S3}/checkpoint-${CKPT_IDX}.pth" "${CKPT_LOCAL}" --quiet
  extract_one "${MODEL_TAG}" "${CKPT_LOCAL}" "${MAE_MODULE}" "${MAE_MODEL_NAME}"
  rm -f "${CKPT_LOCAL}"
done

echo "[INFO] All pre-pool extractions complete"
