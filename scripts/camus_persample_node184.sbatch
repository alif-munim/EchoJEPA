#!/bin/bash
#SBATCH --job-name=salt_ps
#SBATCH --partition=ml-p5-48xlarge
#SBATCH --nodes=1
#SBATCH --nodelist=ip-10-0-50-184
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --gpus-per-node=8
#SBATCH --time=2:00:00
#SBATCH --output=logs/salt_persample_%j.out
#SBATCH --exclusive

# Per-sample CAMUS frame shuffling for bootstrap CIs: SALT
set -uo pipefail

export LD_LIBRARY_PATH="/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:/usr/local/cuda-12.3/compat:${LD_LIBRARY_PATH:-}"

REPO_DIR="/opt/vjepa2"
PYTHON="/opt/vjepa2-312/bin/python3"
SCRATCH="/opt/dlami/nvme"
SCRIPT="scripts/rebuttal/frame_shuffle_segmentation_persample.py"

cd "${REPO_DIR}"
echo "[$(date)] SALT per-sample CAMUS shuffling on $(hostname)"

SALT_CKPT="${SCRATCH}/checkpoints/pretrain/mimic/salt_s2_vitl_224px_16f/e79.pt"
SALT_DEC="${REPO_DIR}/results/segmentation/salt_s2v1_e79/lr5e-02_wd1e-04/best_decoder.pt"

echo "[$(date)] === Severity ==="
${PYTHON} ${SCRIPT} --encoder_type vjepa --encoder_checkpoint "${SALT_CKPT}" \
    --encoder_model_name vit_large --decoder_checkpoint "${SALT_DEC}" \
    --device cuda:0 --label salt_s2v1_e79 --mode severity

echo "[$(date)] === 6-Condition ==="
${PYTHON} ${SCRIPT} --encoder_type vjepa --encoder_checkpoint "${SALT_CKPT}" \
    --encoder_model_name vit_large --decoder_checkpoint "${SALT_DEC}" \
    --device cuda:0 --label salt_s2v1_e79 --mode 6cond

# Upload
echo "[$(date)] Uploading..."
S3="s3://echodata25/neurips/results/camus_frame_shuffle"
for mode in camus_severity_persample camus_6cond_persample; do
    CSV="scripts/rebuttal/samples/salt_s2v1_e79_${mode}.csv"
    [ -f "${CSV}" ] && aws s3 cp "${CSV}" "${S3}/salt_s2v1_e79_${mode}.csv"
done
echo "[$(date)] All done."
