#!/bin/bash
#SBATCH --job-name=lvef_nps
#SBATCH --partition=ml-p5-48xlarge
#SBATCH --nodes=1
#SBATCH --nodelist=ip-10-0-50-83
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --gpus-per-node=8
#SBATCH --time=4:00:00
#SBATCH --output=logs/lvef_noised_persample_%j.out
#SBATCH --exclusive

set -uo pipefail

# cuBLAS bf16 fix for MAE on H100
export LD_LIBRARY_PATH="/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cublas/lib:/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cudnn/lib:/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/nccl/lib:/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:/usr/local/cuda-12.3/compat:${LD_LIBRARY_PATH:-}"

REPO_DIR="/opt/vjepa2"
PYTHON="/opt/vjepa2-312/bin/python3"
SCRATCH="/opt/dlami/nvme"
SCRIPT="scripts/rebuttal/noised_inference_persample.py"
TEST_CSV="${SCRATCH}/echonet_dynamic/test_local.csv"
S3_PROBES="s3://echodata25/neurips/probes/end_lvef_e100"

cd "${REPO_DIR}"
echo "[$(date)] LVEF noised per-sample on $(hostname)"

# Encoder checkpoints (already on NVMe)
JEPA_CKPT="${SCRATCH}/encoder_ckpts/jepa_in21k_vitl_e100.pt"
BYOL_CKPT="${SCRATCH}/checkpoints/pretrain/mimic/byol_vitl_224px_16f_imagenet_v2_resume/e100.pt"
MAE_CKPT="${SCRATCH}/encoder_ckpts/videomae_l_mimic_ep99.pth"

# Download LVEF probes from S3
PROBE_DIR="${SCRATCH}/lvef_probes"
mkdir -p "${PROBE_DIR}"
for m in jepa_in21k_e100 byol_e100 mae_e99; do
    PROBE="${PROBE_DIR}/${m}_best.pt"
    if [ ! -f "${PROBE}" ]; then
        S3_KEY="${m}"
        [ "${m}" = "mae_e99" ] && S3_KEY="mae_e99"
        echo "Downloading probe: ${m}"
        aws s3 cp "${S3_PROBES}/${S3_KEY}/best.pt" "${PROBE}" || { echo "FAILED: ${m} probe download"; exit 1; }
    fi
done

JEPA_PROBE="${PROBE_DIR}/jepa_in21k_e100_best.pt"
BYOL_PROBE="${PROBE_DIR}/byol_e100_best.pt"
MAE_PROBE="${PROBE_DIR}/mae_e99_best.pt"

# Verify all files exist
for f in "${JEPA_CKPT}" "${BYOL_CKPT}" "${MAE_CKPT}" "${JEPA_PROBE}" "${BYOL_PROBE}" "${MAE_PROBE}" "${TEST_CSV}"; do
    [ ! -f "${f}" ] && echo "MISSING: ${f}" && exit 1
done

# Run all 3 models in parallel
${PYTHON} ${SCRIPT} --encoder_type vjepa --encoder_checkpoint "${JEPA_CKPT}" \
    --encoder_model_name vit_large --encoder_key target_encoder \
    --probe_checkpoint "${JEPA_PROBE}" \
    --test_csv "${TEST_CSV}" --device cuda:0 --label jepa_in21k_e100 &
PID1=$!
${PYTHON} ${SCRIPT} --encoder_type vjepa --encoder_checkpoint "${BYOL_CKPT}" \
    --encoder_model_name vit_large --encoder_key target_encoder \
    --probe_checkpoint "${BYOL_PROBE}" \
    --test_csv "${TEST_CSV}" --device cuda:1 --label byol_e100 &
PID2=$!
${PYTHON} ${SCRIPT} --encoder_type videomae --encoder_checkpoint "${MAE_CKPT}" \
    --probe_checkpoint "${MAE_PROBE}" \
    --test_csv "${TEST_CSV}" --device cuda:2 --label mae_e99 &
PID3=$!
wait ${PID1}; echo "[$(date)] JEPA done (exit=$?)"
wait ${PID2}; echo "[$(date)] BYOL done (exit=$?)"
wait ${PID3}; echo "[$(date)] MAE done (exit=$?)"

# Upload
echo "[$(date)] Uploading..."
S3="s3://echodata25/neurips/results/lvef_noised"
for m in jepa_in21k_e100 byol_e100 mae_e99; do
    CSV="scripts/rebuttal/samples/${m}_noised_lvef_persample.csv"
    [ -f "${CSV}" ] && aws s3 cp "${CSV}" "${S3}/${m}_noised_lvef_persample.csv"
done
echo "[$(date)] All done."
