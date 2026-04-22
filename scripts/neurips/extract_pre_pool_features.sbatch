#!/bin/bash
#SBATCH -p dev
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 6:00:00
#SBATCH -J prepool_extract
#SBATCH -o /opt/dlami/nvme/logs/prepool_extract_%j.out
#SBATCH -e /opt/dlami/nvme/logs/prepool_extract_%j.err
#SBATCH --signal=B:USR1@300

set -euxo pipefail

# =============================================================================
# Pre-pool per-tubelet feature extraction for diff-probe analysis.
# Iterates all (model, ckpt) combos in the canonical trajectory, writes
# [N_clips, num_segments=2, T=8, D=1024] fp16 .pt files to NVMe + S3.
#
# SPOT_CHECK=1 -> extract MAE e99 only (Phase 1 gate)
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
RUN_S3="${S3_BASE}/runs/diff_probe_prepool_${SLURM_JOB_ID}"
FEATURES_S3_BASE="${S3_BASE}/features/diff_probe"
SCRATCH="/opt/dlami/nvme"
SPOT_CHECK="${SPOT_CHECK:-0}"

mkdir -p /opt/dlami/nvme/logs

# Checkpoint maps (model:epoch -> S3 path, local name, module, checkpoint_key, model_name)
# JEPA
declare -A JEPA_S3=(
  [25]="${S3_BASE}/checkpoints/jepa_in21k/jepa_in21k_vitl_e25.pt"
  [50]="${S3_BASE}/checkpoints/jepa_in21k/jepa_in21k_vitl_e50.pt"
  [75]="${S3_BASE}/checkpoints/jepa_in21k/jepa_in21k_vitl_e75.pt"
  [100]="${S3_BASE}/checkpoints/jepa_in21k/jepa_in21k_vitl_e100.pt"
)
# MAE (extended trajectory)
declare -A MAE_S3=(
  [25]="${S3_BASE}/checkpoints/echomae_l_mimic_ep24.pth"
  [50]="${S3_BASE}/checkpoints/echomae_l_mimic_ep50.pth"
  [75]="${S3_BASE}/checkpoints/echomae_l_mimic_ep74.pth"
  [99]="${S3_BASE}/checkpoints/echomae_l_mimic_ep99.pth"
  [124]="${S3_BASE}/runs/videomae_e200_159/training_folder/checkpoint-124.pth"
  [149]="${S3_BASE}/runs/videomae_e200_159/training_folder/checkpoint-149.pth"
  [174]="${S3_BASE}/runs/videomae_e200_159/training_folder/checkpoint-174.pth"
  [194]="${S3_BASE}/runs/videomae_e200b_179/training_folder/checkpoint-194.pth"
)
# BYOL (imagenet-v2 init)
declare -A BYOL_S3=(
  [24]="${S3_BASE}/checkpoints/byol/byol_vitl_imagenet_v2_e24.pt"
  [50]="${S3_BASE}/checkpoints/byol/byol_vitl_imagenet_v2_e50.pt"
  [75]="${S3_BASE}/checkpoints/byol/byol_vitl_imagenet_v2_e75.pt"
  [100]="${S3_BASE}/checkpoints/byol/byol_vitl_imagenet_v2_e100.pt"
)
# SALT S2v1 (student, absolute epochs e4/e29/e54/e79)
declare -A SALT_S3=(
  [4]="${S3_BASE}/checkpoints/salt_v1/salt_s2v1_e4.pt"
  [29]="${S3_BASE}/checkpoints/salt_v1/salt_s2v1_e29.pt"
  [54]="${S3_BASE}/checkpoints/salt_v1/salt_s2v1_e54.pt"
  [79]="${S3_BASE}/checkpoints/salt_v1/salt_s2v1_e79.pt"
)

upload_logs() {
  local o="/opt/dlami/nvme/logs/prepool_extract_${SLURM_JOB_ID}.out"
  local e="/opt/dlami/nvme/logs/prepool_extract_${SLURM_JOB_ID}.err"
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

mkdir -p "${SCRATCH}/data/csv" "${SCRATCH}/checkpoints" "${SCRATCH}/features/diff_probe"
aws s3 cp "${S3_BASE}/data/csv/echonet_dynamic_test_s3_raw.csv" "${SCRATCH}/data/csv/"
aws s3 cp "${S3_BASE}/data/csv/echonet_dynamic_train_s3_raw.csv" "${SCRATCH}/data/csv/"

# =========================
# EXTRACTION LOOP
# =========================
run_extract_split() {
  local model_key="$1"  # jepa | mae | byol | salt
  local epoch="$2"
  local ckpt_local="$3"
  local config_name="$4"
  local split="$5"  # test | train
  local val_csv="$6"

  local features_out="${SCRATCH}/features/diff_probe/${model_key}_e${epoch}_${split}.pt"
  local folder_out="${SCRATCH}/evals/neurips/diff_probe/${model_key}_e${epoch}_${split}"
  mkdir -p "${folder_out}"

  # Clean stale shards
  rm -f "${features_out}" "${features_out%.pt}.rank"*.pt

  OVERRIDE_CKPT="${ckpt_local}" \
  OVERRIDE_FOLDER="${folder_out}" \
  OVERRIDE_TAG="neurips-diff-probe-${model_key}-e${epoch}-${split}" \
  OVERRIDE_VAL_DATA="${val_csv}" \
  FEATURES_OUTPUT="${features_out}" \
  python3 -m evals.main \
    --fname "${CODE_DIR}/configs/feature_extraction/vitl/neurips/diff_probe/${config_name}" \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7

  echo "[INFO] ${model_key} e${epoch} (${split}) extraction done"
  ls -lh "${features_out}"
  aws s3 cp "${features_out}" "${FEATURES_S3_BASE}/${model_key}_e${epoch}_${split}.pt" --only-show-errors
}

run_extract() {
  local model_key="$1"  # jepa | mae | byol | salt
  local epoch="$2"
  local s3_path="$3"
  local config_name="$4"  # yaml basename in configs/feature_extraction/vitl/neurips/diff_probe/

  local ckpt_local="${SCRATCH}/checkpoints/${model_key}_e${epoch}.pt"
  echo "============================================================"
  echo "[INFO] Extracting ${model_key} e${epoch} from ${s3_path}"
  echo "============================================================"
  aws s3 cp "${s3_path}" "${ckpt_local}" --quiet

  # test split (1277 clips)
  run_extract_split "${model_key}" "${epoch}" "${ckpt_local}" "${config_name}" \
    "test" "${SCRATCH}/data/csv/echonet_dynamic_test_s3_raw.csv"

  # In SPOT_CHECK mode skip the train split; otherwise run it (7465 clips).
  if [ "${SPOT_CHECK}" != "1" ]; then
    run_extract_split "${model_key}" "${epoch}" "${ckpt_local}" "${config_name}" \
      "train" "${SCRATCH}/data/csv/echonet_dynamic_train_s3_raw.csv"
  fi

  # Keep local caches for Phase 3; delete checkpoint to save disk
  rm -f "${ckpt_local}"
  upload_logs
}

if [ "${SPOT_CHECK}" = "1" ]; then
  echo "[INFO] SPOT_CHECK=1: MAE e99 only"
  run_extract "mae" "99" "${MAE_S3[99]}" "echomae_e99_prepool.yaml"
else
  # JEPA
  for EP in 25 50 75 100; do
    run_extract "jepa" "${EP}" "${JEPA_S3[$EP]}" "jepa_e100_prepool.yaml"
  done
  # MAE
  for EP in 25 50 75 99 124 149 174 194; do
    run_extract "mae" "${EP}" "${MAE_S3[$EP]}" "echomae_e99_prepool.yaml"
  done
  # BYOL
  for EP in 24 50 75 100; do
    run_extract "byol" "${EP}" "${BYOL_S3[$EP]}" "byol_prepool.yaml"
  done
  # SALT
  for EP in 4 29 54 79; do
    run_extract "salt" "${EP}" "${SALT_S3[$EP]}" "salt_prepool.yaml"
  done
fi

echo "============================================================"
echo "[INFO] All extractions complete"
echo "============================================================"
ls -lh "${SCRATCH}/features/diff_probe/"
