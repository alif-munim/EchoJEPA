#!/bin/bash
#SBATCH -p dev
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 2:00:00
#SBATCH -J prepool_random
#SBATCH -o /opt/dlami/nvme/logs/prepool_random_%j.out
#SBATCH -e /opt/dlami/nvme/logs/prepool_random_%j.err
#SBATCH --signal=B:USR1@300

set -euxo pipefail

# =============================================================================
# Extract pre-pool features using RANDOM-INIT ViT-L (3 seeds: 42, 43, 44).
# Uses vit_encoder_multiclip_random module (skips checkpoint load, seeds
# torch.manual_seed before constructing the ViT).
# Control condition for the diff-probe analysis.
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

aws sts get-caller-identity || exit 1

ART_BUCKET="sagemaker-hyperpod-lifecycle-495467399120-usw2"
S3_BASE="s3://${ART_BUCKET}/vjepa2-artifacts"
RUN_S3="${S3_BASE}/runs/diff_probe_random_${SLURM_JOB_ID}"
FEATURES_S3_BASE="${S3_BASE}/features/diff_probe"
SCRATCH="/opt/dlami/nvme"

mkdir -p /opt/dlami/nvme/logs

upload_logs() {
  local o="/opt/dlami/nvme/logs/prepool_random_${SLURM_JOB_ID}.out"
  local e="/opt/dlami/nvme/logs/prepool_random_${SLURM_JOB_ID}.err"
  [ -f "$o" ] && aws s3 cp "$o" "${RUN_S3}/logs/job.out" --only-show-errors || true
  [ -f "$e" ] && aws s3 cp "$e" "${RUN_S3}/logs/job.err" --only-show-errors || true
}
trap 'upload_logs' EXIT

nvidia-smi -L || true

# =========================
# ENV & CODE
# =========================
aws s3 cp "${S3_BASE}/env/vjepa2-312-pack-env.tar" "${SCRATCH}/env.tar"
mkdir -p "${SCRATCH}/env" && tar -xf "${SCRATCH}/env.tar" -C "${SCRATCH}/env"
set +u && source "${SCRATCH}/env/bin/activate" && set -u
"${SCRATCH}/env/bin/conda-unpack" || true

CODE_DIR="${SCRATCH}/code"
mkdir -p "$CODE_DIR"
aws s3 cp "${S3_BASE}/setup/vjepa2-src.tar.gz" "${SCRATCH}/vjepa2-src.tar.gz" --quiet
tar xzf "${SCRATCH}/vjepa2-src.tar.gz" -C "$CODE_DIR"
cd "$CODE_DIR"
pip install -e . --no-deps --no-build-isolation 2>/dev/null || true

mkdir -p "${SCRATCH}/data/csv" "${SCRATCH}/features/diff_probe"
aws s3 cp "${S3_BASE}/data/csv/echonet_dynamic_test_s3_raw.csv"  "${SCRATCH}/data/csv/"
aws s3 cp "${S3_BASE}/data/csv/echonet_dynamic_train_s3_raw.csv" "${SCRATCH}/data/csv/"

# =========================
# EXTRACTION LOOP
# =========================
run_split() {
  local seed="$1"
  local split="$2"  # test | train
  local val_csv="$3"

  local features_out="${SCRATCH}/features/diff_probe/random_s${seed}_${split}.pt"
  local folder_out="${SCRATCH}/evals/neurips/diff_probe/random_s${seed}_${split}"
  mkdir -p "${folder_out}"

  rm -f "${features_out}" "${features_out%.pt}.rank"*.pt

  RANDOM_ENCODER_SEED="${seed}" \
  OVERRIDE_FOLDER="${folder_out}" \
  OVERRIDE_TAG="neurips-diff-probe-random-s${seed}-${split}" \
  OVERRIDE_VAL_DATA="${val_csv}" \
  FEATURES_OUTPUT="${features_out}" \
  python3 -m evals.main \
    --fname "${CODE_DIR}/configs/feature_extraction/vitl/neurips/diff_probe/random_encoder_prepool.yaml" \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7

  echo "[INFO] random_s${seed} (${split}) done"
  ls -lh "${features_out}"
  aws s3 cp "${features_out}" "${FEATURES_S3_BASE}/random_s${seed}_${split}.pt" --only-show-errors
}

for SEED in 42 43 44; do
  echo "============================================================"
  echo "[INFO] Random-encoder seed=${SEED}"
  echo "============================================================"
  run_split "${SEED}" "test"  "${SCRATCH}/data/csv/echonet_dynamic_test_s3_raw.csv"
  run_split "${SEED}" "train" "${SCRATCH}/data/csv/echonet_dynamic_train_s3_raw.csv"
  upload_logs
done

echo "============================================================"
echo "[INFO] All random-encoder extractions complete"
echo "============================================================"
ls -lh "${SCRATCH}/features/diff_probe/" | grep random
