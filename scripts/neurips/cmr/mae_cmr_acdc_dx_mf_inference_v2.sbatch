#!/bin/bash
#SBATCH -p dev
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 2:00:00
#SBATCH -J cmr_dxmf
#SBATCH -o /opt/dlami/nvme/logs/cmr_dxmf_%j.out
#SBATCH -e /opt/dlami/nvme/logs/cmr_dxmf_%j.err

set -euxo pipefail

# =============================================================================
# CMR MAE ACDC diagnosis: clean + matched_frame inference (num_segments=1)
# =============================================================================
# Reruns Phase 2 with single-clip eval to avoid overlap on short CMR clips.
# Uses probes from job 281 (already trained). Local data (no S3 streaming).
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
CKPT_S3="${S3_BASE}/runs/mae_cmr_vits_183/training_folder"
PROBE_S3="${S3_BASE}/runs/cmr_dx_traj_281"
RUN_S3="${S3_BASE}/runs/cmr_dxmf_v2_${SLURM_JOB_ID}"

EPOCHS="30 50 100 200 300 400 600 800"

upload_logs() {
  local out="/opt/dlami/nvme/logs/cmr_dxmf_${SLURM_JOB_ID}.out"
  local err="/opt/dlami/nvme/logs/cmr_dxmf_${SLURM_JOB_ID}.err"
  [ -f "${out}" ] && aws s3 cp "${out}" "${RUN_S3}/logs/job.out" --only-show-errors || true
  [ -f "${err}" ] && aws s3 cp "${err}" "${RUN_S3}/logs/job.err" --only-show-errors || true
}
trap 'upload_logs' EXIT

nvidia-smi -L || true
pkill -9 -f "evals.main" 2>/dev/null || true; sleep 2
mkdir -p /opt/dlami/nvme/logs

SCRATCH="/opt/dlami/nvme"

aws s3 cp "$ENV_S3" "${SCRATCH}/env.tar"
mkdir -p "${SCRATCH}/env" && tar -xf "${SCRATCH}/env.tar" -C "${SCRATCH}/env"
set +u && source "${SCRATCH}/env/bin/activate" && set -u
"${SCRATCH}/env/bin/conda-unpack" || true
pip install boto3 scikit-learn --no-cache-dir --quiet

CODE_DIR="${SCRATCH}/code"
mkdir -p "$CODE_DIR"
aws s3 cp "${S3_BASE}/setup/vjepa2-src.tar.gz" "${SCRATCH}/vjepa2-src.tar.gz" --quiet
tar xzf "${SCRATCH}/vjepa2-src.tar.gz" -C "$CODE_DIR"
cd "$CODE_DIR"
pip install -e . --no-deps --no-build-isolation 2>/dev/null || true

# Download ACDC clips locally
mkdir -p "${SCRATCH}/data/csv" "${SCRATCH}/data/acdc_clips" "${SCRATCH}/checkpoints" "${SCRATCH}/probes"
echo "[INFO] Downloading ACDC clips..."
aws s3 sync "${S3_BASE}/data/cmr/clips/" "${SCRATCH}/data/acdc_clips/" \
  --exclude "*" --include "acdc_*" --only-show-errors
echo "[INFO] Downloaded $(ls ${SCRATCH}/data/acdc_clips/acdc_*.mp4 | wc -l) clips"

# Create local CSVs
aws s3 cp "${S3_BASE}/data/csv/acdc_dx_train_s3.csv" "${SCRATCH}/data/csv/tmp_s3.csv" --quiet
aws s3 cp "${S3_BASE}/data/csv/acdc_dx_test_s3.csv" "${SCRATCH}/data/csv/tmp_test_s3.csv" --quiet
sed "s|s3://.*clips/|${SCRATCH}/data/acdc_clips/|g" "${SCRATCH}/data/csv/tmp_s3.csv" > "${SCRATCH}/data/csv/acdc_dx_train_local.csv"
sed "s|s3://.*clips/|${SCRATCH}/data/acdc_clips/|g" "${SCRATCH}/data/csv/tmp_test_s3.csv" > "${SCRATCH}/data/csv/acdc_dx_test_local.csv"

# Download probes from job 281
echo "[INFO] Downloading probes..."
for EP in ${EPOCHS}; do
  mkdir -p "${SCRATCH}/probes/cmr_mae_e${EP}_dx"
  PROBE_PATH=$(aws s3 ls "${PROBE_S3}/mae_e${EP}_acdc_dx/" --region us-west-2 --recursive 2>/dev/null | grep "best.pt" | awk '{print $4}')
  if [ -n "${PROBE_PATH}" ]; then
    aws s3 cp "s3://${ART_BUCKET}/${PROBE_PATH}" "${SCRATCH}/probes/cmr_mae_e${EP}_dx/best.pt" --quiet
  fi
done

# Run clean + matched_frame for each checkpoint
for EP in ${EPOCHS}; do
  CKPT_IDX=$((EP - 1))
  CKPT_LOCAL="${SCRATCH}/checkpoints/mae_cmr_vits_ep${CKPT_IDX}.pth"
  aws s3 cp "${CKPT_S3}/checkpoint-${CKPT_IDX}.pth" "${CKPT_LOCAL}" --quiet

  CONFIG="${CODE_DIR}/configs/inference/vits16/cmr/mae_cmr_e${EP}_acdc_dx_test.yaml"

  echo "[INFO] CLEAN: e${EP}"
  OVERRIDE_TAG="cmr-mae-vits-e${EP}-acdc-dx-clean-v2" \
  python3 -m evals.main --fname "${CONFIG}" \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7
  echo "[INFO] e${EP} clean done"

  echo "[INFO] MATCHED_FRAME: e${EP}"
  FRAME_SHUFFLE=100 FRAME_SHUFFLE_TYPE=matched_frame \
  OVERRIDE_TAG="cmr-mae-vits-e${EP}-acdc-dx-mf-v2" \
  python3 -m evals.main --fname "${CONFIG}" \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7
  echo "[INFO] e${EP} matched_frame done"

  upload_logs
  rm -f "${CKPT_LOCAL}"
done

echo "[INFO] All CMR diagnosis clean + matched_frame (num_segments=1) complete"
