#!/bin/bash
#SBATCH --job-name=camus_ps
#SBATCH --partition=ml-p5-48xlarge
#SBATCH --nodes=1
#SBATCH --nodelist=ip-10-0-50-83
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --gpus-per-node=8
#SBATCH --time=3:00:00
#SBATCH --output=logs/camus_persample_%j.out
#SBATCH --exclusive

# Per-sample CAMUS frame shuffling for bootstrap CIs: JEPA/BYOL/MAE
set -uo pipefail

export LD_LIBRARY_PATH="/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cublas/lib:/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cudnn/lib:/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/nccl/lib:/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:/usr/local/cuda-12.3/compat:${LD_LIBRARY_PATH:-}"

REPO_DIR="/opt/vjepa2"
PYTHON="/opt/vjepa2-312/bin/python3"
SCRATCH="/opt/dlami/nvme"
SCRIPT="scripts/rebuttal/frame_shuffle_segmentation_persample.py"

cd "${REPO_DIR}"
echo "[$(date)] Per-sample CAMUS shuffling on $(hostname)"

JEPA_CKPT="${SCRATCH}/encoder_ckpts/jepa_in21k_vitl_e100.pt"
BYOL_CKPT="${SCRATCH}/checkpoints/pretrain/mimic/byol_vitl_224px_16f_imagenet_v2_resume/e100.pt"
MAE_CKPT="${SCRATCH}/encoder_ckpts/videomae_l_mimic_ep99.pth"
JEPA_DEC="${SCRATCH}/camus_decoders/jepa_in21k_e100_best_decoder.pt"
BYOL_DEC="${SCRATCH}/camus_decoders/byol_e100_best_decoder.pt"
MAE_DEC="${SCRATCH}/camus_decoders/mae_e99_best_decoder.pt"

# Phase 1: Severity (3 models parallel)
echo "[$(date)] === Severity ==="
${PYTHON} ${SCRIPT} --encoder_type vjepa --encoder_checkpoint "${JEPA_CKPT}" \
    --encoder_model_name vit_large --decoder_checkpoint "${JEPA_DEC}" \
    --device cuda:0 --label jepa_in21k_e100 --mode severity &
PID1=$!
${PYTHON} ${SCRIPT} --encoder_type vjepa --encoder_checkpoint "${BYOL_CKPT}" \
    --encoder_model_name vit_large --decoder_checkpoint "${BYOL_DEC}" \
    --device cuda:1 --label byol_e100 --mode severity &
PID2=$!
${PYTHON} ${SCRIPT} --encoder_type videomae --encoder_checkpoint "${MAE_CKPT}" \
    --decoder_checkpoint "${MAE_DEC}" \
    --device cuda:2 --label mae_e99 --mode severity &
PID3=$!
wait ${PID1}; echo "[$(date)] JEPA severity done (exit=$?)"
wait ${PID2}; echo "[$(date)] BYOL severity done (exit=$?)"
wait ${PID3}; echo "[$(date)] MAE severity done (exit=$?)"

# Phase 2: 6-cond (3 models parallel)
echo "[$(date)] === 6-Condition ==="
${PYTHON} ${SCRIPT} --encoder_type vjepa --encoder_checkpoint "${JEPA_CKPT}" \
    --encoder_model_name vit_large --decoder_checkpoint "${JEPA_DEC}" \
    --device cuda:0 --label jepa_in21k_e100 --mode 6cond &
PID4=$!
${PYTHON} ${SCRIPT} --encoder_type vjepa --encoder_checkpoint "${BYOL_CKPT}" \
    --encoder_model_name vit_large --decoder_checkpoint "${BYOL_DEC}" \
    --device cuda:1 --label byol_e100 --mode 6cond &
PID5=$!
${PYTHON} ${SCRIPT} --encoder_type videomae --encoder_checkpoint "${MAE_CKPT}" \
    --decoder_checkpoint "${MAE_DEC}" \
    --device cuda:2 --label mae_e99 --mode 6cond &
PID6=$!
wait ${PID4}; echo "[$(date)] JEPA 6cond done (exit=$?)"
wait ${PID5}; echo "[$(date)] BYOL 6cond done (exit=$?)"
wait ${PID6}; echo "[$(date)] MAE 6cond done (exit=$?)"

# Upload
echo "[$(date)] Uploading..."
S3="s3://echodata25/neurips/results/camus_frame_shuffle"
for m in jepa_in21k_e100 byol_e100 mae_e99; do
    for mode in camus_severity_persample camus_6cond_persample; do
        CSV="scripts/rebuttal/samples/${m}_${mode}.csv"
        [ -f "${CSV}" ] && aws s3 cp "${CSV}" "${S3}/${m}_${mode}.csv"
    done
done
echo "[$(date)] All done."
