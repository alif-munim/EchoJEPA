#!/bin/bash
#SBATCH -p dev
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 2:00:00
#SBATCH -J salt_mf
#SBATCH -o /tmp/salt_mf_%j.out
#SBATCH -e /tmp/salt_mf_%j.err

set -euxo pipefail

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
ENV_S3="${S3_BASE}/env/vjepa2-312-pack-env.tar"
RUN_S3="${S3_BASE}/runs/salt_mf_${SLURM_JOB_ID}"

upload_logs() {
  [ -f "/tmp/salt_mf_${SLURM_JOB_ID}.out" ] && aws s3 cp "/tmp/salt_mf_${SLURM_JOB_ID}.out" "${RUN_S3}/logs/job.out" --only-show-errors || true
  [ -f "/tmp/salt_mf_${SLURM_JOB_ID}.err" ] && aws s3 cp "/tmp/salt_mf_${SLURM_JOB_ID}.err" "${RUN_S3}/logs/job.err" --only-show-errors || true
}
trap 'upload_logs' EXIT

nvidia-smi -L || true
pkill -9 -f "evals.main" 2>/dev/null || true; sleep 2

SCRATCH="/opt/dlami/nvme"

aws s3 cp "$ENV_S3" "${SCRATCH}/env.tar"
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

mkdir -p "${SCRATCH}/data/csv" "${SCRATCH}/checkpoints" "${SCRATCH}/probes"
aws s3 cp "${S3_BASE}/data/csv/echonet_dynamic_train_s3_raw.csv" "${SCRATCH}/data/csv/"
aws s3 cp "${S3_BASE}/data/csv/echonet_dynamic_test_s3_raw.csv" "${SCRATCH}/data/csv/"

# SALT e29, e54, e79 (e4 already done in job 220)
for EP in 29 54 79; do
  echo "============================================================"
  echo "[INFO] SALT e${EP}"
  echo "============================================================"

  CKPT_LOCAL="${SCRATCH}/checkpoints/salt_s2v1_e${EP}.pt"
  aws s3 cp "${S3_BASE}/checkpoints/salt_v1/salt_s2v1_e${EP}.pt" "${CKPT_LOCAL}" --quiet
  echo "[INFO] Encoder downloaded: $(ls -lh ${CKPT_LOCAL} | awk '{print $5}')"

  mkdir -p "${SCRATCH}/probes/salt_e${EP}_end"
  aws s3 cp "${S3_BASE}/probes/salt_e${EP}_end_lvef/best.pt" "${SCRATCH}/probes/salt_e${EP}_end/best.pt" --quiet

  CONFIG="${CODE_DIR}/configs/inference/vitl/neurips/echonet-dynamic/salt_s2v1_e${EP}_end_test.yaml"

  echo "[INFO] CLEAN: SALT e${EP}"
  python3 -m evals.main --fname "${CONFIG}" \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7
  echo "[INFO] SALT e${EP} clean done"

  echo "[INFO] MATCHED_FRAME: SALT e${EP}"
  FRAME_SHUFFLE=100 FRAME_SHUFFLE_TYPE=matched_frame \
  python3 -m evals.main --fname "${CONFIG}" \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7
  echo "[INFO] SALT e${EP} matched_frame done"

  upload_logs
  rm -f "${CKPT_LOCAL}"
done

echo "[INFO] All SALT remaining runs complete"
