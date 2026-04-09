#!/bin/bash
#SBATCH --job-name=mae_camus
#SBATCH --partition=ml-p5-48xlarge
#SBATCH --nodes=1
#SBATCH --nodelist=ip-10-0-50-83
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --gpus-per-node=8
#SBATCH --time=1:00:00
#SBATCH --output=logs/mae_camus_shuffle_%j.out
#SBATCH --exclusive

# MAE CAMUS frame shuffling — cuBLAS bf16 fix applied
set -uo pipefail

# Critical: PyTorch-bundled cuBLAS MUST come first for H100 bf16
export LD_LIBRARY_PATH="/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cublas/lib:/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cudnn/lib:/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/nccl/lib:/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:/usr/local/cuda-12.3/compat:${LD_LIBRARY_PATH:-}"

REPO_DIR="/opt/vjepa2"
PYTHON="/opt/vjepa2-312/bin/python3"
SCRATCH="/opt/dlami/nvme"

cd "${REPO_DIR}"

echo "[$(date)] MAE CAMUS frame shuffling on $(hostname)"

MAE_CKPT="${SCRATCH}/encoder_ckpts/videomae_l_mimic_ep99.pth"
MAE_DEC="${SCRATCH}/camus_decoders/mae_e99_best_decoder.pt"

echo "[$(date)] Encoder: ${MAE_CKPT}"
echo "[$(date)] Decoder: ${MAE_DEC}"
ls -lh "${MAE_CKPT}" "${MAE_DEC}"

${PYTHON} scripts/rebuttal/frame_shuffle_segmentation.py \
    --encoder_type videomae \
    --encoder_checkpoint "${MAE_CKPT}" \
    --decoder_checkpoint "${MAE_DEC}" \
    --device cuda:0 \
    --label mae_e99 \
    --n_seeds 3

echo "[$(date)] Done."
cat scripts/rebuttal/samples/mae_e99_frame_shuffle_segmentation.csv

# Upload
aws s3 cp scripts/rebuttal/samples/mae_e99_frame_shuffle_segmentation.csv \
    s3://echodata25/neurips/results/camus_frame_shuffle/mae_e99.csv

echo "[$(date)] All done."
