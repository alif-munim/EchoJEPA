#!/bin/bash
#SBATCH --job-name=salt_camus
#SBATCH --partition=ml-p5-48xlarge
#SBATCH --nodes=1
#SBATCH --nodelist=ip-10-0-50-184
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --gpus-per-node=8
#SBATCH --time=4:00:00
#SBATCH --output=logs/salt_camus_%j.out
#SBATCH --exclusive

# SALT CAMUS segmentation probe training + frame shuffling
# Node 184 — has salt_s2v1_e79.pt from previous deploy

set -uo pipefail

export LD_LIBRARY_PATH="/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:/usr/local/cuda-12.3/compat:${LD_LIBRARY_PATH:-}"

REPO_DIR="/opt/vjepa2"
PYTHON="/opt/vjepa2-312/bin/python3"
SCRATCH="/opt/dlami/nvme"
S3_BUCKET="s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts"
S3_CLEAN="s3://echodata25/neurips"

cd "${REPO_DIR}"

echo "[$(date)] Starting SALT CAMUS probe training on $(hostname)"

# Install nibabel (required for CAMUS NIfTI loading)
echo "[$(date)] Installing nibabel..."
${PYTHON} -m pip install nibabel 2>&1 | tail -1

# ============================================================
# 1. Download CAMUS data to NVMe
# ============================================================
CAMUS_NVME="${SCRATCH}/camus"
CAMUS_DIR="${CAMUS_NVME}/CAMUS_public"

if [ ! -d "${CAMUS_DIR}/database_nifti" ]; then
    echo "[$(date)] Downloading CAMUS data from S3..."
    mkdir -p "${CAMUS_NVME}"
    aws s3 cp s3://echodata25/neurips/data/camus.zip "${CAMUS_NVME}/camus.zip"
    cd "${CAMUS_NVME}"
    unzip -q camus.zip
    rm -f camus.zip
    cd "${REPO_DIR}"
    echo "[$(date)] CAMUS data extracted: $(ls ${CAMUS_DIR}/database_nifti | wc -l) patients"
else
    echo "[$(date)] CAMUS data already present: $(ls ${CAMUS_DIR}/database_nifti | wc -l) patients"
fi

# Symlink CAMUS data to expected repo path
mkdir -p "${REPO_DIR}/data/camus"
if [ ! -e "${REPO_DIR}/data/camus/CAMUS_public" ]; then
    ln -sf "${CAMUS_DIR}" "${REPO_DIR}/data/camus/CAMUS_public"
fi

# ============================================================
# 2. Ensure SALT e79 encoder is available
# ============================================================
SALT_CKPT="${SCRATCH}/checkpoints/pretrain/mimic/salt_s2_vitl_224px_16f/e79.pt"

if [ ! -f "${SALT_CKPT}" ]; then
    # Fallback: check repo path or download from S3
    SALT_CKPT_ALT="${REPO_DIR}/checkpoints/pretrain/mimic/salt_s2v1_e79.pt"
    if [ -f "${SALT_CKPT_ALT}" ]; then
        SALT_CKPT="${SALT_CKPT_ALT}"
    else
        echo "[$(date)] Downloading SALT e79 encoder from S3..."
        mkdir -p "$(dirname ${SALT_CKPT})"
        aws s3 cp "${S3_BUCKET}/runs/salt_s2_pretrain_388/checkpoints/e79.pt" "${SALT_CKPT}"
        if [ ! -f "${SALT_CKPT}" ]; then
            echo "[ERROR] Failed to download SALT e79 encoder"
            exit 1
        fi
    fi
fi
echo "[$(date)] SALT encoder: $(ls -lh ${SALT_CKPT} | awk '{print $5}')"

# ============================================================
# 3. Train SALT CAMUS segmentation decoder (7-config grid search)
# ============================================================
OUTPUT_DIR="${REPO_DIR}/results/segmentation/salt_s2v1_e79"

echo "[$(date)] Training SALT CAMUS segmentation decoder..."
echo "[$(date)]   Encoder: ${SALT_CKPT}"
echo "[$(date)]   Output:  ${OUTPUT_DIR}"

${PYTHON} -m evals.segmentation_frozen.eval \
    --model_type vjepa \
    --checkpoint "${SALT_CKPT}" \
    --output_dir "${OUTPUT_DIR}" \
    --device cuda:0 \
    --num_epochs 50

echo "[$(date)] Training complete."

# ============================================================
# 4. Determine best config and run frame shuffling
# ============================================================
GRID_SUMMARY="${OUTPUT_DIR}/grid_summary.json"

if [ ! -f "${GRID_SUMMARY}" ]; then
    echo "[ERROR] grid_summary.json not found — training may have failed"
    exit 1
fi

# Extract best config directory name from grid_summary.json
BEST_CONFIG=$(${PYTHON} -c "
import json
with open('${GRID_SUMMARY}') as f:
    data = json.load(f)
print(data.get('best_config', ''))
")

if [ -z "${BEST_CONFIG}" ]; then
    echo "[ERROR] Could not determine best config from grid_summary.json"
    cat "${GRID_SUMMARY}"
    exit 1
fi

DECODER_CKPT="${OUTPUT_DIR}/${BEST_CONFIG}/best_decoder.pt"
echo "[$(date)] Best config: ${BEST_CONFIG}"
echo "[$(date)] Decoder: ${DECODER_CKPT}"

# Print results
echo "[$(date)] === SALT CAMUS Results ==="
cat "${GRID_SUMMARY}"

# ============================================================
# 5. Run frame shuffling on SALT CAMUS probe
# ============================================================
echo ""
echo "[$(date)] Running SALT frame shuffling on CAMUS..."

${PYTHON} scripts/rebuttal/frame_shuffle_segmentation.py \
    --encoder_type vjepa \
    --encoder_checkpoint "${SALT_CKPT}" \
    --encoder_model_name vit_large \
    --decoder_checkpoint "${DECODER_CKPT}" \
    --device cuda:0 \
    --label salt_s2v1_e79 \
    --n_seeds 3

echo "[$(date)] SALT CAMUS frame shuffling complete."

# ============================================================
# 6. Upload results to S3
# ============================================================
echo "[$(date)] Uploading results to S3..."
aws s3 cp "${GRID_SUMMARY}" "${S3_CLEAN}/probes/camus_segmentation/salt_s2v1_e79/grid_summary.json"
aws s3 cp "${DECODER_CKPT}" "${S3_CLEAN}/probes/camus_segmentation/salt_s2v1_e79/best_decoder.pt"
aws s3 cp "${REPO_DIR}/scripts/rebuttal/samples/salt_s2v1_e79_frame_shuffle_segmentation.csv" \
    "${S3_CLEAN}/results/camus_frame_shuffle/salt_s2v1_e79.csv"

echo "[$(date)] All done."
