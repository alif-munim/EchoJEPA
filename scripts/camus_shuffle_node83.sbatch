#!/bin/bash
#SBATCH --job-name=camus_shuf
#SBATCH --partition=ml-p5-48xlarge
#SBATCH --nodes=1
#SBATCH --nodelist=ip-10-0-50-83
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --gpus-per-node=8
#SBATCH --time=3:00:00
#SBATCH --output=logs/camus_shuffle_%j.out
#SBATCH --exclusive

# CAMUS frame shuffling for JEPA/BYOL/MAE (e100 init-matched)
# Node 83 — downloads encoders + decoders from S3, runs 3 models in parallel

set -uo pipefail

export LD_LIBRARY_PATH="/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:/usr/local/cuda-12.3/compat:${LD_LIBRARY_PATH:-}"

REPO_DIR="/opt/vjepa2"
PYTHON="/opt/vjepa2-312/bin/python3"
SCRATCH="/opt/dlami/nvme"
S3_BUCKET="s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts"
S3_CLEAN="s3://echodata25/neurips"

cd "${REPO_DIR}"

echo "[$(date)] Starting CAMUS frame shuffling on $(hostname)"

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
# 2. Download encoder checkpoints to NVMe (root fs is full)
# ============================================================
CKPT_DIR="${SCRATCH}/encoder_ckpts"
mkdir -p "${CKPT_DIR}"

# JEPA IN21K e100 (latest.pt from job 376 = e100, ~5.1 GiB)
JEPA_CKPT="${CKPT_DIR}/jepa_in21k_vitl_e100.pt"
if [ ! -f "${JEPA_CKPT}" ]; then
    echo "[$(date)] Downloading JEPA IN21K e100 (~5.1 GiB) to NVMe..."
    aws s3 cp "${S3_BUCKET}/runs/jepa_in21k_pretrain_376/checkpoints/latest.pt" "${JEPA_CKPT}"
    if [ ! -f "${JEPA_CKPT}" ]; then
        echo "[ERROR] JEPA download failed"; exit 1
    fi
fi

# BYOL e100 — use local NVMe copy
BYOL_CKPT="${SCRATCH}/checkpoints/pretrain/mimic/byol_vitl_224px_16f_imagenet_v2_resume/e100.pt"
if [ ! -f "${BYOL_CKPT}" ]; then
    BYOL_CKPT="${CKPT_DIR}/byol_vitl_imagenet_v2_e100.pt"
    if [ ! -f "${BYOL_CKPT}" ]; then
        echo "[$(date)] Downloading BYOL e100 (~2.5 GiB) to NVMe..."
        aws s3 cp "${S3_BUCKET}/checkpoints/byol-vitl-imagenet-v2-resume/e100.pt" "${BYOL_CKPT}"
        if [ ! -f "${BYOL_CKPT}" ]; then
            echo "[ERROR] BYOL download failed"; exit 1
        fi
    fi
fi

# MAE e99 (~3.8 GiB)
MAE_CKPT="${CKPT_DIR}/videomae_l_mimic_ep99.pth"
if [ ! -f "${MAE_CKPT}" ]; then
    echo "[$(date)] Downloading MAE e99 (~3.8 GiB) to NVMe..."
    aws s3 cp "${S3_BUCKET}/runs/videomae_resume_e54_354/training_folder/checkpoint-99.pth" "${MAE_CKPT}"
    if [ ! -f "${MAE_CKPT}" ]; then
        echo "[ERROR] MAE download failed"; exit 1
    fi
fi

echo "[$(date)] Encoder checkpoints ready:"
ls -lh "${JEPA_CKPT}" "${BYOL_CKPT}" "${MAE_CKPT}"

# ============================================================
# 3. Download trained CAMUS decoder checkpoints from S3
# ============================================================
DEC_DIR="${SCRATCH}/camus_decoders"
mkdir -p "${DEC_DIR}"

JEPA_DEC="${DEC_DIR}/jepa_in21k_e100_best_decoder.pt"
BYOL_DEC="${DEC_DIR}/byol_e100_best_decoder.pt"
MAE_DEC="${DEC_DIR}/mae_e99_best_decoder.pt"

if [ ! -f "${JEPA_DEC}" ]; then
    echo "[$(date)] Downloading JEPA decoder..."
    aws s3 cp "${S3_CLEAN}/probes/camus_segmentation/jepa_in21k_e100/best_decoder.pt" "${JEPA_DEC}"
fi
if [ ! -f "${BYOL_DEC}" ]; then
    echo "[$(date)] Downloading BYOL decoder..."
    aws s3 cp "${S3_CLEAN}/probes/camus_segmentation/byol_e100/best_decoder.pt" "${BYOL_DEC}"
fi
if [ ! -f "${MAE_DEC}" ]; then
    echo "[$(date)] Downloading MAE decoder..."
    aws s3 cp "${S3_CLEAN}/probes/camus_segmentation/mae_e99/best_decoder.pt" "${MAE_DEC}"
fi

echo "[$(date)] Decoder checkpoints ready:"
ls -lh "${JEPA_DEC}" "${BYOL_DEC}" "${MAE_DEC}"

# ============================================================
# 4. Run frame shuffling for all 3 models in parallel
# ============================================================
echo ""
echo "[$(date)] Launching frame shuffling for 3 models in parallel..."

# JEPA on cuda:0
echo "[$(date)] Starting JEPA IN21K e100..."
${PYTHON} scripts/neurips/frame_shuffle_segmentation.py \
    --encoder_type vjepa \
    --encoder_checkpoint "${JEPA_CKPT}" \
    --encoder_model_name vit_large \
    --decoder_checkpoint "${JEPA_DEC}" \
    --device cuda:0 \
    --label jepa_in21k_e100 \
    --n_seeds 3 &
PID_JEPA=$!

# BYOL on cuda:1
echo "[$(date)] Starting BYOL e100..."
${PYTHON} scripts/neurips/frame_shuffle_segmentation.py \
    --encoder_type vjepa \
    --encoder_checkpoint "${BYOL_CKPT}" \
    --encoder_model_name vit_large \
    --decoder_checkpoint "${BYOL_DEC}" \
    --device cuda:1 \
    --label byol_e100 \
    --n_seeds 3 &
PID_BYOL=$!

# MAE on cuda:2
echo "[$(date)] Starting MAE e99..."
${PYTHON} scripts/neurips/frame_shuffle_segmentation.py \
    --encoder_type videomae \
    --encoder_checkpoint "${MAE_CKPT}" \
    --decoder_checkpoint "${MAE_DEC}" \
    --device cuda:2 \
    --label mae_e99 \
    --n_seeds 3 &
PID_MAE=$!

echo "[$(date)] Waiting for all 3 models to complete..."
echo "  JEPA PID=${PID_JEPA}, BYOL PID=${PID_BYOL}, MAE PID=${PID_MAE}"

wait ${PID_JEPA}
echo "[$(date)] JEPA complete (exit=$?)"

wait ${PID_BYOL}
echo "[$(date)] BYOL complete (exit=$?)"

wait ${PID_MAE}
echo "[$(date)] MAE complete (exit=$?)"

# ============================================================
# 5. Print results
# ============================================================
echo ""
echo "[$(date)] === CAMUS Frame Shuffling Results ==="
SAMPLES="${REPO_DIR}/scripts/neurips/samples"

for f in jepa_in21k_e100 byol_e100 mae_e99; do
    CSV="${SAMPLES}/${f}_frame_shuffle_segmentation.csv"
    if [ -f "${CSV}" ]; then
        echo ""
        echo "--- ${f} ---"
        cat "${CSV}"
    else
        echo "[WARNING] Missing: ${CSV}"
    fi
done

# ============================================================
# 6. Upload results to S3
# ============================================================
echo ""
echo "[$(date)] Uploading results to S3..."
for f in jepa_in21k_e100 byol_e100 mae_e99; do
    CSV="${SAMPLES}/${f}_frame_shuffle_segmentation.csv"
    if [ -f "${CSV}" ]; then
        aws s3 cp "${CSV}" "${S3_CLEAN}/results/camus_frame_shuffle/${f}.csv"
    fi
done

echo "[$(date)] All done."
