#!/bin/bash
#SBATCH --job-name=salt_noise
#SBATCH --partition=ml-p5-48xlarge
#SBATCH --nodes=1
#SBATCH --nodelist=ip-10-0-50-184
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --gpus-per-node=8
#SBATCH --time=1:00:00
#SBATCH --output=logs/salt_noised_seg_%j.out
#SBATCH --exclusive

set -uo pipefail

REPO_DIR="/opt/vjepa2"
PYTHON="/opt/vjepa2-312/bin/python3"
SCRATCH="/opt/dlami/nvme"

cd "${REPO_DIR}"
${PYTHON} -m pip install nibabel 2>&1 | tail -1
echo "[$(date)] SALT noised segmentation on $(hostname)"

SALT_CKPT="${SCRATCH}/checkpoints/pretrain/mimic/salt_s2_vitl_224px_16f/e79.pt"
SALT_DEC="${REPO_DIR}/results/segmentation/salt_s2v1_e79/lr5e-02_wd1e-04/best_decoder.pt"

${PYTHON} scripts/rebuttal/noised_segmentation.py \
    --encoder_type vjepa \
    --encoder_checkpoint "${SALT_CKPT}" \
    --encoder_model_name vit_large \
    --decoder_checkpoint "${SALT_DEC}" \
    --device cuda:0 \
    --label salt_s2v1_e79

echo "[$(date)] Uploading..."
CSV="scripts/rebuttal/samples/salt_s2v1_e79_noised_segmentation.csv"
[ -f "${CSV}" ] && aws s3 cp "${CSV}" "s3://echodata25/neurips/results/camus_frame_shuffle/${CSV##*/}"
echo "[$(date)] Done."
