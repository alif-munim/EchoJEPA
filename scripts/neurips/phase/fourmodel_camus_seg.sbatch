#!/bin/bash
#SBATCH -p dev
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 06:00:00
#SBATCH -J fm_camus_seg
#SBATCH -o /opt/dlami/nvme/logs/fm_camus_seg_%j.out
#SBATCH -e /opt/dlami/nvme/logs/fm_camus_seg_%j.err

set -euxo pipefail

# =============================================================================
# Four-model CAMUS segmentation probe (NeurIPS companion to LVEF probes).
# Models: echojepa_l_k, jepa_in21k_e200, echoprime, panecho.
# Uses evals.segmentation_frozen.eval with --num_epochs=50 and the 7-config
# grid search (matches SALT protocol). Outputs best_decoder.pt and
# grid_summary.json per model.
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
S3_CLEAN="s3://echodata25/neurips"
RUN_S3="${S3_BASE}/runs/fourmodel_camus_seg_${SLURM_JOB_ID}"

upload_logs() {
  local out="/opt/dlami/nvme/logs/fm_camus_seg_${SLURM_JOB_ID}.out"
  local err="/opt/dlami/nvme/logs/fm_camus_seg_${SLURM_JOB_ID}.err"
  [ -f "${out}" ] && aws s3 cp "${out}" "${RUN_S3}/logs/job.out" --only-show-errors || true
  [ -f "${err}" ] && aws s3 cp "${err}" "${RUN_S3}/logs/job.err" --only-show-errors || true
}
trap 'upload_logs' EXIT

nvidia-smi -L || true
mkdir -p /opt/dlami/nvme/logs

SCRATCH="/opt/dlami/nvme"
WORKDIR="${SCRATCH}/fm_camus_seg_${SLURM_JOB_ID}"
mkdir -p "$WORKDIR"

aws s3 cp "$ENV_S3" "${WORKDIR}/env.tar"
mkdir -p "${WORKDIR}/env" && tar -xf "${WORKDIR}/env.tar" -C "${WORKDIR}/env"
set +u && source "${WORKDIR}/env/bin/activate" && set -u
"${WORKDIR}/env/bin/conda-unpack" || true
export TMPDIR="${WORKDIR}/tmp" && mkdir -p "$TMPDIR"
pip install boto3 nibabel --no-cache-dir --quiet

CODE_DIR="${WORKDIR}/code"
mkdir -p "$CODE_DIR"
aws s3 cp "${S3_BASE}/setup/vjepa2-src.tar.gz" "${WORKDIR}/vjepa2-src.tar.gz" --quiet
tar xzf "${WORKDIR}/vjepa2-src.tar.gz" -C "$CODE_DIR"
cd "$CODE_DIR"
pip install -e . --no-deps --no-build-isolation 2>/dev/null || true
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# --- CAMUS data ---
CAMUS_NVME="${SCRATCH}/camus"
CAMUS_DIR="${CAMUS_NVME}/CAMUS_public"
if [ ! -d "${CAMUS_DIR}/database_nifti" ]; then
  echo "[INFO] Downloading CAMUS data from S3..."
  mkdir -p "${CAMUS_NVME}"
  aws s3 cp "${S3_CLEAN}/data/camus.zip" "${CAMUS_NVME}/camus.zip"
  cd "${CAMUS_NVME}"
  unzip -q camus.zip
  rm -f camus.zip
  cd "${CODE_DIR}"
  echo "[INFO] CAMUS extracted: $(ls ${CAMUS_DIR}/database_nifti | wc -l) patients"
fi
mkdir -p "${CODE_DIR}/data/camus"
if [ ! -e "${CODE_DIR}/data/camus/CAMUS_public" ]; then
  ln -sf "${CAMUS_DIR}" "${CODE_DIR}/data/camus/CAMUS_public"
fi

# --- Backbones ---
CKPT_DIR="${WORKDIR}/checkpoints"
mkdir -p "$CKPT_DIR"

echo "[INFO] Downloading echojepa-l-k backbone..."
aws s3 cp "${S3_BASE}/checkpoints/anneal/keep/vitl-kinetics-pt220-an55.pt" "${CKPT_DIR}/echojepa_l_k.pt" --quiet

echo "[INFO] Downloading jepa-in21k-e200 backbone..."
aws s3 cp "${S3_BASE}/runs/jepa_in21k_e200_280/training_folder/latest.pt" "${CKPT_DIR}/jepa_in21k_e200.pt" --quiet

echo "[INFO] Downloading echoprime backbone..."
aws s3 cp "${S3_BASE}/checkpoints/echoprime/echo_prime_encoder.pt" "${CKPT_DIR}/echo_prime_encoder.pt" --quiet

# PanEcho: torch.hub cache
export TORCH_HOME="${WORKDIR}/torch_cache"
PE_CACHE="${TORCH_HOME}/hub/checkpoints"
mkdir -p "$PE_CACHE"
aws s3 cp "${S3_BASE}/checkpoints/panecho/panecho.pt" "${PE_CACHE}/panecho.pt" --quiet

# --- Run segmentation probe training per model ---
# Format: "label:model_type:ckpt_path:model_name"
JOBS=(
  "echojepa_l_k:vjepa:${CKPT_DIR}/echojepa_l_k.pt:vit_large"
  "jepa_in21k_e200:vjepa:${CKPT_DIR}/jepa_in21k_e200.pt:vit_large"
  "echoprime:echoprime:${CKPT_DIR}/echo_prime_encoder.pt:mvit_v2_s"
  "panecho:panecho:unused:panecho"
)

for SPEC in "${JOBS[@]}"; do
  LABEL="${SPEC%%:*}"
  REST="${SPEC#*:}"
  MT="${REST%%:*}"
  REST2="${REST#*:}"
  CKPT="${REST2%%:*}"
  MNAME="${REST2##*:}"

  OUTPUT_DIR="${CODE_DIR}/results/segmentation/${LABEL}"
  mkdir -p "$OUTPUT_DIR"

  echo "============================================"
  echo "[INFO] CAMUS seg: ${LABEL} (type=${MT}, model_name=${MNAME})"
  echo "============================================"
  python3 -m evals.segmentation_frozen.eval \
    --model_type "${MT}" \
    --checkpoint "${CKPT}" \
    --model_name "${MNAME}" \
    --output_dir "${OUTPUT_DIR}" \
    --device cuda:0 \
    --num_epochs 50

  # Upload results
  if [ -f "${OUTPUT_DIR}/grid_summary.json" ]; then
    BEST_CONFIG=$(python3 -c "import json; print(json.load(open('${OUTPUT_DIR}/grid_summary.json')).get('best_config', ''))")
    DECODER_CKPT="${OUTPUT_DIR}/${BEST_CONFIG}/best_decoder.pt"
    aws s3 cp "${OUTPUT_DIR}/grid_summary.json" "${S3_CLEAN}/probes/camus_segmentation/${LABEL}/grid_summary.json" --only-show-errors
    [ -f "${DECODER_CKPT}" ] && aws s3 cp "${DECODER_CKPT}" "${S3_CLEAN}/probes/camus_segmentation/${LABEL}/best_decoder.pt" --only-show-errors
    # Also sync full output dir to the job's S3 run folder
    aws s3 sync "${OUTPUT_DIR}/" "${RUN_S3}/${LABEL}/" --only-show-errors || true
    echo "[INFO] ${LABEL} results uploaded."
  else
    echo "[WARN] grid_summary.json missing for ${LABEL}"
  fi

  upload_logs
done

echo "[INFO] Four-model CAMUS segmentation complete"
