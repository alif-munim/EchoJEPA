#!/bin/bash
#SBATCH --job-name=salt_ext
#SBATCH --partition=ml-p5-48xlarge
#SBATCH --nodes=1
#SBATCH --nodelist=ip-10-0-50-184
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --gpus-per-node=8
#SBATCH --time=2:00:00
#SBATCH --output=logs/salt_camus_extended_%j.out
#SBATCH --exclusive

# Extended CAMUS frame shuffling: severity + 6cond for SALT
set -uo pipefail

export LD_LIBRARY_PATH="/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:/usr/local/cuda-12.3/compat:${LD_LIBRARY_PATH:-}"

REPO_DIR="/opt/vjepa2"
PYTHON="/opt/vjepa2-312/bin/python3"
SCRATCH="/opt/dlami/nvme"
SCRIPT="scripts/rebuttal/frame_shuffle_segmentation_extended.py"

cd "${REPO_DIR}"
echo "[$(date)] SALT extended CAMUS frame shuffling on $(hostname)"

SALT_CKPT="${SCRATCH}/checkpoints/pretrain/mimic/salt_s2_vitl_224px_16f/e79.pt"
SALT_DEC="${REPO_DIR}/results/segmentation/salt_s2v1_e79/lr5e-02_wd1e-04/best_decoder.pt"

echo "[$(date)] Verifying checkpoints..."
ls -lh "${SALT_CKPT}" "${SALT_DEC}"

# Severity gradient
echo ""
echo "[$(date)] === Severity Gradient ==="
${PYTHON} ${SCRIPT} --encoder_type vjepa --encoder_checkpoint "${SALT_CKPT}" \
    --encoder_model_name vit_large --decoder_checkpoint "${SALT_DEC}" \
    --device cuda:0 --label salt_s2v1_e79 --mode severity

echo ""
echo "[$(date)] === 6-Condition ==="
${PYTHON} ${SCRIPT} --encoder_type vjepa --encoder_checkpoint "${SALT_CKPT}" \
    --encoder_model_name vit_large --decoder_checkpoint "${SALT_DEC}" \
    --device cuda:0 --label salt_s2v1_e79 --mode 6cond

# Print results
echo ""
echo "[$(date)] === Results ==="
SAMPLES="${REPO_DIR}/scripts/rebuttal/samples"
for mode in camus_severity camus_6cond; do
    CSV="${SAMPLES}/salt_s2v1_e79_${mode}.csv"
    if [ -f "${CSV}" ]; then
        echo ""; echo "--- salt_s2v1_e79 ${mode} ---"; cat "${CSV}"
    else
        echo "[WARNING] Missing: ${CSV}"
    fi
done

# Upload
echo ""
echo "[$(date)] Uploading to S3..."
S3_CLEAN="s3://echodata25/neurips"
for mode in camus_severity camus_6cond; do
    CSV="${SAMPLES}/salt_s2v1_e79_${mode}.csv"
    if [ -f "${CSV}" ]; then
        aws s3 cp "${CSV}" "${S3_CLEAN}/results/camus_frame_shuffle/salt_s2v1_e79_${mode}.csv"
    fi
done

echo "[$(date)] All done."
