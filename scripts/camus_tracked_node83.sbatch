#!/bin/bash
#SBATCH --job-name=camus_trk
#SBATCH --partition=ml-p5-48xlarge
#SBATCH --nodes=1
#SBATCH --nodelist=ip-10-0-50-83
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --gpus-per-node=8
#SBATCH --time=1:00:00
#SBATCH --output=logs/camus_tracked_%j.out
#SBATCH --exclusive

# Version B: tracked extraction for JEPA/BYOL/MAE
set -uo pipefail

# cuBLAS bf16 fix for MAE on H100
export LD_LIBRARY_PATH="/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cublas/lib:/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cudnn/lib:/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/nccl/lib:/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:/usr/local/cuda-12.3/compat:${LD_LIBRARY_PATH:-}"

REPO_DIR="/opt/vjepa2"
PYTHON="/opt/vjepa2-312/bin/python3"
SCRATCH="/opt/dlami/nvme"
SCRIPT="scripts/rebuttal/frame_shuffle_segmentation_tracked.py"

cd "${REPO_DIR}"
echo "[$(date)] Tracked CAMUS extraction on $(hostname)"

JEPA_CKPT="${SCRATCH}/encoder_ckpts/jepa_in21k_vitl_e100.pt"
BYOL_CKPT="${SCRATCH}/checkpoints/pretrain/mimic/byol_vitl_224px_16f_imagenet_v2_resume/e100.pt"
MAE_CKPT="${SCRATCH}/encoder_ckpts/videomae_l_mimic_ep99.pth"
JEPA_DEC="${SCRATCH}/camus_decoders/jepa_in21k_e100_best_decoder.pt"
BYOL_DEC="${SCRATCH}/camus_decoders/byol_e100_best_decoder.pt"
MAE_DEC="${SCRATCH}/camus_decoders/mae_e99_best_decoder.pt"

# All 3 models in parallel on separate GPUs
${PYTHON} ${SCRIPT} --encoder_type vjepa --encoder_checkpoint "${JEPA_CKPT}" \
    --encoder_model_name vit_large --decoder_checkpoint "${JEPA_DEC}" \
    --device cuda:0 --label jepa_in21k_e100 &
PID1=$!
${PYTHON} ${SCRIPT} --encoder_type vjepa --encoder_checkpoint "${BYOL_CKPT}" \
    --encoder_model_name vit_large --decoder_checkpoint "${BYOL_DEC}" \
    --device cuda:1 --label byol_e100 &
PID2=$!
${PYTHON} ${SCRIPT} --encoder_type videomae --encoder_checkpoint "${MAE_CKPT}" \
    --decoder_checkpoint "${MAE_DEC}" \
    --device cuda:2 --label mae_e99 &
PID3=$!
wait ${PID1}; echo "[$(date)] JEPA done (exit=$?)"
wait ${PID2}; echo "[$(date)] BYOL done (exit=$?)"
wait ${PID3}; echo "[$(date)] MAE done (exit=$?)"

# Upload
echo "[$(date)] Uploading..."
S3="s3://echodata25/neurips/results/camus_frame_shuffle"
for m in jepa_in21k_e100 byol_e100 mae_e99; do
    CSV="scripts/rebuttal/samples/${m}_camus_tracked_persample.csv"
    [ -f "${CSV}" ] && aws s3 cp "${CSV}" "${S3}/${m}_camus_tracked_persample.csv"
done
echo "[$(date)] All done."
