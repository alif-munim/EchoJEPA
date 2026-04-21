#!/bin/bash
#SBATCH -p dev
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 12:00:00
#SBATCH -J sev_traj
#SBATCH -o /tmp/sev_traj_%j.out
#SBATCH -e /tmp/sev_traj_%j.err
#SBATCH --signal=B:USR1@300

set -euxo pipefail

# =============================================================================
# Frame Shuffle Severity Gradient: MAE training trajectory (e24-e194)
# =============================================================================
# Runs frame_shuffle_severity.py for 8 MAE checkpoints.
# Uses the original standalone script (single GPU, no evals.main).
# This matches the protocol that produced the original 0.141/0.445 numbers.
# =============================================================================

export AWS_DEFAULT_REGION="us-west-2"
export AWS_REGION="us-west-2"
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
RUN_S3="${S3_BASE}/runs/sev_traj_${SLURM_JOB_ID}"

upload_logs() {
  [ -f "/tmp/sev_traj_${SLURM_JOB_ID}.out" ] && aws s3 cp "/tmp/sev_traj_${SLURM_JOB_ID}.out" "${RUN_S3}/logs/job.out" --only-show-errors || true
  [ -f "/tmp/sev_traj_${SLURM_JOB_ID}.err" ] && aws s3 cp "/tmp/sev_traj_${SLURM_JOB_ID}.err" "${RUN_S3}/logs/job.err" --only-show-errors || true
}
trap 'upload_logs' EXIT

# =========================
# SETUP
# =========================
nvidia-smi -L || true
SCRATCH="/opt/dlami/nvme"

# Env
aws s3 cp "$ENV_S3" "${SCRATCH}/env.tar"
mkdir -p "${SCRATCH}/env" && tar -xf "${SCRATCH}/env.tar" -C "${SCRATCH}/env"
set +u && source "${SCRATCH}/env/bin/activate" && set -u
"${SCRATCH}/env/bin/conda-unpack" || true
pip install boto3 decord --no-cache-dir --quiet

# Code
CODE_DIR="${SCRATCH}/code"
mkdir -p "$CODE_DIR"
aws s3 cp "${S3_BASE}/setup/vjepa2-src.tar.gz" "${SCRATCH}/vjepa2-src.tar.gz" --quiet
tar xzf "${SCRATCH}/vjepa2-src.tar.gz" -C "$CODE_DIR"
cd "$CODE_DIR"
pip install -e . --no-deps --no-build-isolation 2>/dev/null || true
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# =========================
# DOWNLOAD TEST VIDEOS (local for decord)
# =========================
echo "[INFO] Downloading EchoNet-Dynamic test videos..."
VIDEOS_DIR="${SCRATCH}/echonet_videos"
mkdir -p "${VIDEOS_DIR}"

# Download test videos listed in the CSV
aws s3 cp "${S3_BASE}/data/csv/echonet_dynamic_test_s3_raw.csv" "${SCRATCH}/test_s3.csv" --quiet
awk '{print $1}' "${SCRATCH}/test_s3.csv" | while read s3path; do
  fname=$(basename "$s3path")
  aws s3 cp "$s3path" "${VIDEOS_DIR}/${fname}" --quiet
done
echo "[INFO] Downloaded $(ls ${VIDEOS_DIR}/*.avi | wc -l) test videos"

# Create local test CSV
awk -v dir="${VIDEOS_DIR}" '{fname=$1; gsub(/.*\//, "", fname); print dir"/"fname, $2}' "${SCRATCH}/test_s3.csv" > "${SCRATCH}/test_local.csv"
echo "[INFO] Local test CSV: $(wc -l < ${SCRATCH}/test_local.csv) videos"

# =========================
# DOWNLOAD ENCODER CHECKPOINTS + PROBES
# =========================
# The script expects paths relative to the code dir. Create the expected directory structure.
mkdir -p checkpoints
mkdir -p evals/vitl/icml/echomae_e24_end_lvef_224/video_classification_frozen/icml-echomae-l-e24-end-lvef-d4
mkdir -p evals/vitl/icml/echomae_pt50_end_lvef_224/video_classification_frozen/icml-echomae-l-pt50-end-lvef-d4
mkdir -p evals/vitl/icml/echomae_e74_end_lvef_224/video_classification_frozen/icml-echomae-l-e74-end-lvef-d4
mkdir -p evals/vitl/icml/echomae_e99_end_lvef_224/video_classification_frozen/icml-echomae-l-e99-end-lvef-d4

# e24-e99: encoders from S3 checkpoints
echo "[INFO] Downloading encoder checkpoints..."
aws s3 cp "${S3_BASE}/checkpoints/echomae_l_mimic_ep24.pth" checkpoints/videomae_l_mimic_ep24.pth --quiet
aws s3 cp "${S3_BASE}/checkpoints/echomae_l_mimic_ep50.pth" checkpoints/videomae_l_mimic_ep50.pth --quiet
aws s3 cp "${S3_BASE}/checkpoints/echomae_l_mimic_ep74.pth" checkpoints/videomae_l_mimic_ep74.pth --quiet
aws s3 cp "${S3_BASE}/checkpoints/echomae_l_mimic_ep99.pth" checkpoints/videomae_l_mimic_ep99.pth --quiet

# e124-e194: encoders from run checkpoints
aws s3 cp "${S3_BASE}/runs/videomae_e200_159/training_folder/checkpoint-124.pth" checkpoints/videomae_l_mimic_ep124.pth --quiet
aws s3 cp "${S3_BASE}/runs/videomae_e200_159/training_folder/checkpoint-149.pth" checkpoints/videomae_l_mimic_ep149.pth --quiet
aws s3 cp "${S3_BASE}/runs/videomae_e200_159/training_folder/checkpoint-174.pth" checkpoints/videomae_l_mimic_ep174.pth --quiet
aws s3 cp "${S3_BASE}/runs/videomae_e200b_179/training_folder/checkpoint-194.pth" checkpoints/videomae_l_mimic_ep194.pth --quiet

# Probes: e24-e99 from ICML probes on S3
echo "[INFO] Downloading probe checkpoints..."
aws s3 cp "${S3_BASE}/probes/echomae_e25_end_lvef/best.pt" evals/vitl/icml/echomae_e24_end_lvef_224/video_classification_frozen/icml-echomae-l-e24-end-lvef-d4/best.pt --quiet
aws s3 cp "${S3_BASE}/probes/echomae_e50_end_lvef/best.pt" evals/vitl/icml/echomae_pt50_end_lvef_224/video_classification_frozen/icml-echomae-l-pt50-end-lvef-d4/best.pt --quiet
aws s3 cp "${S3_BASE}/probes/echomae_e75_end_lvef/best.pt" evals/vitl/icml/echomae_e74_end_lvef_224/video_classification_frozen/icml-echomae-l-e74-end-lvef-d4/best.pt --quiet
aws s3 cp "${S3_BASE}/probes/echomae_e99_end_lvef/best.pt" evals/vitl/icml/echomae_e99_end_lvef_224/video_classification_frozen/icml-echomae-l-e99-end-lvef-d4/best.pt --quiet

# Probes: e124-e194 from NeurIPS probe runs
mkdir -p evals/vitl/icml/echomae_e124_end_lvef_224/video_classification_frozen/icml-echomae-l-e124-end-lvef-d4
mkdir -p evals/vitl/icml/echomae_e149_end_lvef_224/video_classification_frozen/icml-echomae-l-e149-end-lvef-d4
mkdir -p evals/vitl/icml/echomae_e174_end_lvef_224/video_classification_frozen/icml-echomae-l-e174-end-lvef-d4
mkdir -p evals/vitl/icml/echomae_e194_end_lvef_224/video_classification_frozen/icml-echomae-l-e194-end-lvef-d4

aws s3 cp "${S3_BASE}/runs/mae_traj_lvef_202/echomae_e124_end_lvef/video_classification_frozen/neurips-echomae-l-e124-end-lvef-d4/best.pt" evals/vitl/icml/echomae_e124_end_lvef_224/video_classification_frozen/icml-echomae-l-e124-end-lvef-d4/best.pt --quiet
aws s3 cp "${S3_BASE}/runs/mae_traj_lvef_202/echomae_e149_end_lvef/video_classification_frozen/neurips-echomae-l-e149-end-lvef-d4/best.pt" evals/vitl/icml/echomae_e149_end_lvef_224/video_classification_frozen/icml-echomae-l-e149-end-lvef-d4/best.pt --quiet
aws s3 cp "${S3_BASE}/runs/mae_traj_lvef_202/echomae_e174_end_lvef/video_classification_frozen/neurips-echomae-l-e174-end-lvef-d4/best.pt" evals/vitl/icml/echomae_e174_end_lvef_224/video_classification_frozen/icml-echomae-l-e174-end-lvef-d4/best.pt --quiet
aws s3 cp "${S3_BASE}/runs/mae200_lvef_201/eval_output/video_classification_frozen/neurips-echomae-l-e200-end-lvef-d4/best.pt" evals/vitl/icml/echomae_e194_end_lvef_224/video_classification_frozen/icml-echomae-l-e194-end-lvef-d4/best.pt --quiet

echo "[INFO] All checkpoints downloaded"

# =========================
# RUN SEVERITY GRADIENT
# =========================
# Run e24-e99 first (reproducing original numbers), then e124-e194

MODELS="MAE-L-e24 MAE-L-e50 MAE-L-e74 MAE-L-e99 MAE-L-e124 MAE-L-e149 MAE-L-e174 MAE-L-e194"

for MODEL in ${MODELS}; do
  echo "============================================================"
  echo "[INFO] Running severity gradient: ${MODEL}"
  echo "============================================================"

  python3 scripts/neurips/frame_shuffling/frame_shuffle_severity.py \
    --test_csv "${SCRATCH}/test_local.csv" \
    --models "${MODEL}" \
    --n_seeds 3 \
    --device cuda:0

  echo "[INFO] ${MODEL} done"
  upload_logs
done

echo "[INFO] All severity gradient runs complete"
