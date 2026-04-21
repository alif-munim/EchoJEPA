#!/bin/bash
#SBATCH -p dev
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 6:00:00
#SBATCH -J mae_mf
#SBATCH -o /tmp/mae_mf_%j.out
#SBATCH -e /tmp/mae_mf_%j.err
#SBATCH --signal=B:USR1@300

set -euxo pipefail

# =============================================================================
# EchoMAE-L trajectory: clean + matched_frame (time-aware shuffle)
# =============================================================================
# Uses evals.main with FRAME_SHUFFLE_TYPE=matched_frame for proper RoPE
# remapping. This is the main-text protocol from the NeurIPS paper.
# num_segments=2 prediction averaging (matches original frame-shuffling.md).
# 6 multihead_kwargs to match probe checkpoint heads.
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
ENV_S3="${S3_BASE}/env/vjepa2-312-pack-env.tar"
RUN_S3="${S3_BASE}/runs/mae_mf_traj_${SLURM_JOB_ID}"

# Encoder checkpoint S3 paths
declare -A ENCODER_S3=(
  [25]="${S3_BASE}/checkpoints/echomae_l_mimic_ep24.pth"
  [50]="${S3_BASE}/checkpoints/echomae_l_mimic_ep50.pth"
  [75]="${S3_BASE}/checkpoints/echomae_l_mimic_ep74.pth"
  [99]="${S3_BASE}/checkpoints/echomae_l_mimic_ep99.pth"
  [124]="${S3_BASE}/runs/videomae_e200_159/training_folder/checkpoint-124.pth"
  [149]="${S3_BASE}/runs/videomae_e200_159/training_folder/checkpoint-149.pth"
  [174]="${S3_BASE}/runs/videomae_e200_159/training_folder/checkpoint-174.pth"
  [194]="${S3_BASE}/runs/videomae_e200b_179/training_folder/checkpoint-194.pth"
)

# Probe checkpoint S3 paths
declare -A PROBE_S3=(
  [25]="${S3_BASE}/probes/echomae_e25_end_lvef/best.pt"
  [50]="${S3_BASE}/probes/echomae_e50_end_lvef/best.pt"
  [75]="${S3_BASE}/probes/echomae_e75_end_lvef/best.pt"
  [99]="${S3_BASE}/probes/echomae_e99_end_lvef/best.pt"
  [124]="${S3_BASE}/runs/mae_traj_lvef_202/echomae_e124_end_lvef/video_classification_frozen/neurips-echomae-l-e124-end-lvef-d4/best.pt"
  [149]="${S3_BASE}/runs/mae_traj_lvef_202/echomae_e149_end_lvef/video_classification_frozen/neurips-echomae-l-e149-end-lvef-d4/best.pt"
  [174]="${S3_BASE}/runs/mae_traj_lvef_202/echomae_e174_end_lvef/video_classification_frozen/neurips-echomae-l-e174-end-lvef-d4/best.pt"
  [194]="${S3_BASE}/runs/mae200_lvef_201/eval_output/video_classification_frozen/neurips-echomae-l-e200-end-lvef-d4/best.pt"
)

declare -A ENCODER_LOCAL=(
  [25]="echomae_l_mimic_ep24.pth"
  [50]="echomae_l_mimic_ep50.pth"
  [75]="echomae_l_mimic_ep74.pth"
  [99]="echomae_l_mimic_ep99.pth"
  [124]="echomae_l_mimic_ep124.pth"
  [149]="echomae_l_mimic_ep149.pth"
  [174]="echomae_l_mimic_ep174.pth"
  [194]="echomae_l_mimic_ep194.pth"
)

EPOCHS="25 50 75 99 124 149 174 194"

upload_logs() {
  [ -f "/tmp/mae_mf_${SLURM_JOB_ID}.out" ] && aws s3 cp "/tmp/mae_mf_${SLURM_JOB_ID}.out" "${RUN_S3}/logs/job.out" --only-show-errors || true
  [ -f "/tmp/mae_mf_${SLURM_JOB_ID}.err" ] && aws s3 cp "/tmp/mae_mf_${SLURM_JOB_ID}.err" "${RUN_S3}/logs/job.err" --only-show-errors || true
}
trap 'upload_logs' EXIT

# =========================
# SETUP
# =========================
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

# Download all probes upfront (small, ~3.5 GB each but reused)
for EP in ${EPOCHS}; do
  mkdir -p "${SCRATCH}/probes/echomae_e${EP}_end"
  aws s3 cp "${PROBE_S3[$EP]}" "${SCRATCH}/probes/echomae_e${EP}_end/best.pt" --quiet
done

# =========================
# RUN CLEAN + MATCHED_FRAME FOR EACH CHECKPOINT
# =========================
for EP in ${EPOCHS}; do
  echo "============================================================"
  echo "[INFO] Processing EchoMAE-L e${EP}"
  echo "============================================================"

  # Download encoder
  CKPT_LOCAL="${SCRATCH}/checkpoints/${ENCODER_LOCAL[$EP]}"
  aws s3 cp "${ENCODER_S3[$EP]}" "${CKPT_LOCAL}" --quiet

  CONFIG="${CODE_DIR}/configs/inference/vitl/neurips/echonet-dynamic/echomae_e${EP}_end_test.yaml"

  # --- CLEAN ---
  echo "[INFO] CLEAN: e${EP}"
  python3 -m evals.main --fname "${CONFIG}" \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7
  echo "[INFO] e${EP} clean done"

  # --- MATCHED_FRAME (time-aware shuffle, RoPE remapped) ---
  echo "[INFO] MATCHED_FRAME: e${EP}"
  FRAME_SHUFFLE=100 FRAME_SHUFFLE_TYPE=matched_frame \
  python3 -m evals.main --fname "${CONFIG}" \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7
  echo "[INFO] e${EP} matched_frame done"

  upload_logs
  rm -f "${CKPT_LOCAL}"
done

echo ""
echo "============================================================"
echo "[INFO] All 8 checkpoints × 2 conditions (clean + matched_frame) complete"
echo "============================================================"
