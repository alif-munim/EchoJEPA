#!/bin/bash
#SBATCH -p dev
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 6:00:00
#SBATCH -J prepool_pertoken
#SBATCH -o /opt/dlami/nvme/logs/prepool_pertoken_%j.out
#SBATCH -e /opt/dlami/nvme/logs/prepool_pertoken_%j.err
#SBATCH --signal=B:USR1@300

set -euxo pipefail

# =============================================================================
# Per-token (no spatial mean) pre-pool feature extraction for the
# per-token diff-probe caveat experiment (Section H).
#
# Cache shape:  [N_clips, num_segments=2, T=8, spatial=196, D=1024] fp16
# Cache size:   ~7.6 GB test, ~44.7 GB train per checkpoint.
#
# Modes:
#   SPOT_CHECK=1 (default) -> MAE e99 only (test + train).
#   SPOT_CHECK=0           -> minimum 7-checkpoint scope
#                             (MAE e50/e99/e194, JEPA e50/e100, BYOL e100, SALT e79).
#                             Do NOT run full 7-ckpt scope until spot-check passes.
#
# Writes to  /opt/dlami/nvme/features/diff_probe_pertoken/   and syncs to
# s3://${ART_BUCKET}/vjepa2-artifacts/features/diff_probe_pertoken/
# =============================================================================

export AWS_DEFAULT_REGION="us-west-2"
export AWS_REGION="us-west-2"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export AWS_SDK_LOAD_CONFIG=1
export HOME="/home/ubuntu"
export AWS_SHARED_CREDENTIALS_FILE="${HOME}/.aws/credentials"
export AWS_CONFIG_FILE="${HOME}/.aws/config"
unset AWS_EC2_METADATA_DISABLED 2>/dev/null || true

aws sts get-caller-identity || exit 1

ART_BUCKET="sagemaker-hyperpod-lifecycle-495467399120-usw2"
S3_BASE="s3://${ART_BUCKET}/vjepa2-artifacts"
RUN_S3="${S3_BASE}/runs/diff_probe_pertoken_extract_${SLURM_JOB_ID}"
FEATURES_S3_BASE="${S3_BASE}/features/diff_probe_pertoken"
SCRATCH="/opt/dlami/nvme"
SPOT_CHECK="${SPOT_CHECK:-1}"

mkdir -p /opt/dlami/nvme/logs

# Tell the extractor to keep the spatial axis.
export FEATURE_KEEP_SPATIAL=1

declare -A JEPA_S3=(
  [50]="${S3_BASE}/checkpoints/jepa_in21k/jepa_in21k_vitl_e50.pt"
  [100]="${S3_BASE}/checkpoints/jepa_in21k/jepa_in21k_vitl_e100.pt"
)
declare -A MAE_S3=(
  [50]="${S3_BASE}/checkpoints/echomae_l_mimic_ep50.pth"
  [99]="${S3_BASE}/checkpoints/echomae_l_mimic_ep99.pth"
  [194]="${S3_BASE}/runs/videomae_e200b_179/training_folder/checkpoint-194.pth"
)
declare -A BYOL_S3=(
  [100]="${S3_BASE}/checkpoints/byol/byol_vitl_imagenet_v2_e100.pt"
)
declare -A SALT_S3=(
  [79]="${S3_BASE}/checkpoints/salt_v1/salt_s2v1_e79.pt"
)

upload_logs() {
  local o="/opt/dlami/nvme/logs/prepool_pertoken_${SLURM_JOB_ID}.out"
  local e="/opt/dlami/nvme/logs/prepool_pertoken_${SLURM_JOB_ID}.err"
  [ -f "$o" ] && aws s3 cp "$o" "${RUN_S3}/logs/job.out" --only-show-errors || true
  [ -f "$e" ] && aws s3 cp "$e" "${RUN_S3}/logs/job.err" --only-show-errors || true
}
trap 'upload_logs' EXIT

nvidia-smi -L || true
pkill -9 -f "evals.main" 2>/dev/null || true; sleep 2

# =========================
# ENV & CODE
# =========================
aws s3 cp "${S3_BASE}/env/vjepa2-312-pack-env.tar" "${SCRATCH}/env.tar"
mkdir -p "${SCRATCH}/env" && tar -xf "${SCRATCH}/env.tar" -C "${SCRATCH}/env"
set +u && source "${SCRATCH}/env/bin/activate" && set -u
"${SCRATCH}/env/bin/conda-unpack" || true
pip install boto3 --no-cache-dir --quiet

CODE_DIR="${SCRATCH}/code"
mkdir -p "$CODE_DIR"
aws s3 cp "${S3_BASE}/setup/vjepa2-src.tar.gz" "${SCRATCH}/vjepa2-src.tar.gz" --quiet
tar xzf "${SCRATCH}/vjepa2-src.tar.gz" -C "$CODE_DIR"
cd "$CODE_DIR"
pip install -e . --no-deps --no-build-isolation 2>/dev/null || true

mkdir -p "${SCRATCH}/data/csv" "${SCRATCH}/checkpoints" "${SCRATCH}/features/diff_probe_pertoken"
aws s3 cp "${S3_BASE}/data/csv/echonet_dynamic_test_s3_raw.csv" "${SCRATCH}/data/csv/"
aws s3 cp "${S3_BASE}/data/csv/echonet_dynamic_train_s3_raw.csv" "${SCRATCH}/data/csv/"

# =========================
# EXTRACTION
# =========================
run_extract_split() {
  local model_key="$1"
  local epoch="$2"
  local ckpt_local="$3"
  local config_name="$4"
  local split="$5"
  local val_csv="$6"

  local features_out="${SCRATCH}/features/diff_probe_pertoken/${model_key}_e${epoch}_${split}.pt"
  local folder_out="${SCRATCH}/evals/neurips/diff_probe_pertoken/${model_key}_e${epoch}_${split}"
  mkdir -p "${folder_out}"

  rm -f "${features_out}" "${features_out%.pt}.rank"*.pt

  OVERRIDE_CKPT="${ckpt_local}" \
  OVERRIDE_FOLDER="${folder_out}" \
  OVERRIDE_TAG="neurips-diff-probe-pertoken-${model_key}-e${epoch}-${split}" \
  OVERRIDE_VAL_DATA="${val_csv}" \
  FEATURES_OUTPUT="${features_out}" \
  FEATURE_KEEP_SPATIAL=1 \
  python3 -m evals.main \
    --fname "${CODE_DIR}/configs/feature_extraction/vitl/neurips/diff_probe/${config_name}" \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7

  echo "[INFO] ${model_key} e${epoch} (${split}) per-token extraction done"
  ls -lh "${features_out}"
  aws s3 cp "${features_out}" "${FEATURES_S3_BASE}/${model_key}_e${epoch}_${split}.pt" --only-show-errors
}

run_extract() {
  local model_key="$1"
  local epoch="$2"
  local s3_path="$3"
  local config_name="$4"

  local ckpt_local="${SCRATCH}/checkpoints/${model_key}_e${epoch}.pt"
  echo "============================================================"
  echo "[INFO] Per-token extract ${model_key} e${epoch} from ${s3_path}"
  echo "============================================================"
  aws s3 cp "${s3_path}" "${ckpt_local}" --quiet

  # test first (faster, validates pipeline)
  run_extract_split "${model_key}" "${epoch}" "${ckpt_local}" "${config_name}" \
    "test" "${SCRATCH}/data/csv/echonet_dynamic_test_s3_raw.csv"

  run_extract_split "${model_key}" "${epoch}" "${ckpt_local}" "${config_name}" \
    "train" "${SCRATCH}/data/csv/echonet_dynamic_train_s3_raw.csv"

  # free checkpoint but KEEP feature cache (Phase 3 needs it).
  rm -f "${ckpt_local}"
  upload_logs
}

if [ "${SPOT_CHECK}" = "1" ]; then
  echo "[INFO] SPOT_CHECK=1: MAE e99 per-token only"
  run_extract "mae" "99" "${MAE_S3[99]}" "echomae_e99_prepool.yaml"
else
  echo "[INFO] SPOT_CHECK=0: minimum 7-checkpoint scope"
  for EP in 50 100; do
    run_extract "jepa" "${EP}" "${JEPA_S3[$EP]}" "jepa_e100_prepool.yaml"
  done
  for EP in 50 99 194; do
    run_extract "mae" "${EP}" "${MAE_S3[$EP]}" "echomae_e99_prepool.yaml"
  done
  run_extract "byol" "100" "${BYOL_S3[100]}" "byol_prepool.yaml"
  run_extract "salt" "79"  "${SALT_S3[79]}"  "salt_prepool.yaml"
fi

echo "============================================================"
echo "[INFO] Per-token extraction complete"
echo "============================================================"
ls -lh "${SCRATCH}/features/diff_probe_pertoken/"
df -h /opt/dlami/nvme
