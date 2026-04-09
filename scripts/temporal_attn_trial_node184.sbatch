#!/bin/bash
#SBATCH --job-name=temporal_attn_trial
#SBATCH --partition=ml-p5-48xlarge
#SBATCH --nodes=1
#SBATCH --nodelist=ip-10-0-50-184
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=0:30:00
#SBATCH --output=logs/temporal_attn_trial_%j.out

set -uo pipefail

# Fix cuBLAS bf16 on H100s
export LD_LIBRARY_PATH="/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cublas/lib:/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"

REPO_DIR="/opt/vjepa2"
PYTHON="/opt/vjepa2-312/bin/python3"
SCRATCH="/opt/dlami/nvme"
SCRIPT="scripts/rebuttal/temporal_attention_trial.py"
TEST_CSV="${SCRATCH}/echonet_dynamic/test_local.csv"
SAVE_DIR="scripts/rebuttal/temporal_attention"
HYP="s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts/runs"

cd "${REPO_DIR}"
echo "[$(date)] Temporal attention trial: SALT S2 e29 on $(hostname)"

SALT_E29="${SCRATCH}/checkpoints/salt_s2v1_e29.pt"

if [ ! -f "${SALT_E29}" ]; then
    echo "[$(date)] Downloading SALT S2 e29..."
    aws s3 cp "${HYP}/salt_s2_pretrain_388/checkpoints/e29.pt" "${SALT_E29}" || { echo "FAILED: download"; exit 1; }
fi

[ ! -f "${TEST_CSV}" ] && echo "MISSING: ${TEST_CSV}" && exit 1

echo ""
echo "============================================"
echo "[$(date)] SALT S2 v1 e29"
echo "============================================"
${PYTHON} ${SCRIPT} \
    --checkpoint "${SALT_E29}" \
    --checkpoint_key encoder \
    --encoder_type vjepa \
    --model_name vit_large \
    --label salt_s2v1_e29 \
    --test_csv "${TEST_CSV}" \
    --n_videos 10 \
    --device cuda:0

echo ""
echo "[$(date)] Done."
