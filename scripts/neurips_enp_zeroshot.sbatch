#!/bin/bash
#SBATCH --job-name=enp-zshot
#SBATCH --partition=ml-p5-48xlarge
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=/opt/vjepa2/logs/enp_zeroshot_%j.out
#SBATCH --nodelist=ip-10-0-50-184
#SBATCH --exclusive

set -eo pipefail

cd /opt/vjepa2
source /opt/vjepa2-312/bin/activate
set -u

# cuBLAS bf16 fix for H100 — use PyTorch's bundled CUDA libs
export LD_LIBRARY_PATH="/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cublas/lib:/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"

# S3 paths
HYP="s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts"
CLEAN="s3://echodata25/neurips"
NVME="/opt/dlami/nvme"
RUN_S3="${HYP}/runs/neurips_enp_zeroshot_${SLURM_JOB_ID}"

mkdir -p ${NVME}/checkpoints ${NVME}/data/csv ${NVME}/probes/end_lvef_e100 logs

# ==============================================================================
# REFRESH CREDENTIALS (IMDS)
# ==============================================================================
echo "[INFO] Refreshing credentials..."
TOKEN=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
ROLE_NAME=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/iam/security-credentials/)
CREDS_JSON=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/iam/security-credentials/$ROLE_NAME)
export AWS_ACCESS_KEY_ID=$(echo "$CREDS_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['AccessKeyId'])")
export AWS_SECRET_ACCESS_KEY=$(echo "$CREDS_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['SecretAccessKey'])")
export AWS_SESSION_TOKEN=$(echo "$CREDS_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['Token'])")

# ==============================================================================
# DOWNLOAD DATA, ENCODERS, AND PROBES
# ==============================================================================
echo "[INFO] Downloading pediatric test CSV..."
aws s3 cp ${HYP}/csv_pediatric/echonet_pediatric_test_s3.csv ${NVME}/data/csv/ --quiet

echo "[INFO] Downloading encoder checkpoints..."
aws s3 cp ${HYP}/runs/vjepa_mimic_pretrain_125/training_folder/e100.pt ${NVME}/checkpoints/echojepa_l_in21k_e100.pt --quiet &
aws s3 cp ${HYP}/runs/byol_pretrain_resume_342/checkpoints/e100.pt ${NVME}/checkpoints/echobyol_l_e100.pt --quiet &
aws s3 cp ${HYP}/runs/videomae_resume_e54_354/training_folder/checkpoint-99.pth ${NVME}/checkpoints/echomae_l_mimic_ep99.pth --quiet &
aws s3 cp ${HYP}/runs/salt_s2_pretrain_388/checkpoints/e79.pt ${NVME}/checkpoints/salt_s2_vitl_e79.pt --quiet &

echo "[INFO] Downloading END LVEF probe checkpoints..."
for m in jepa_in21k_e100 byol_e100 mae_e99 salt_v1_e79; do
    mkdir -p ${NVME}/probes/end_lvef_e100/$m
    aws s3 cp ${CLEAN}/probes/end_lvef_e100/$m/best.pt ${NVME}/probes/end_lvef_e100/$m/best.pt --quiet &
done

wait
echo "[INFO] All downloads complete."

pip install -e . --no-deps --no-build-isolation 2>/dev/null || true
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# Detect GPUs
GPU_COUNT=$(nvidia-smi -L | wc -l)
DEVICES=()
for ((i=0; i<GPU_COUNT; i++)); do DEVICES+=("cuda:${i}"); done
echo "[INFO] ${GPU_COUNT} GPUs: ${DEVICES[*]}"

# ==============================================================================
# RUN 4 ZERO-SHOT INFERENCE RUNS
# ==============================================================================
CONFIGS=(
    "configs/inference/vitl/neurips/echonet-pediatric/echojepa_e100.yaml"
    "configs/inference/vitl/neurips/echonet-pediatric/echobyol_e100.yaml"
    "configs/inference/vitl/neurips/echonet-pediatric/echomae_e99.yaml"
    "configs/inference/vitl/neurips/echonet-pediatric/salt_s2v1_e79.yaml"
)
NAMES=("JEPA" "BYOL" "MAE" "SALT")

for i in "${!CONFIGS[@]}"; do
    echo ""
    echo "============================================================"
    echo "[INFO] ${NAMES[$i]} zero-shot inference — $(date)"
    echo "============================================================"

    python3 -m evals.main \
        --fname "${CONFIGS[$i]}" \
        --devices "${DEVICES[@]}" \
        --val_only \
        2>&1 | tee "logs/enp_zshot_${NAMES[$i],,}_${SLURM_JOB_ID}.log"

    echo "[INFO] ${NAMES[$i]} done — $(date)"
done

# ==============================================================================
# BOOTSTRAP CIs
# ==============================================================================
echo ""
echo "============================================================"
echo "[INFO] Computing bootstrap CIs..."
echo "============================================================"

python3 scripts/neurips/enp_zeroshot_bootstrap.py \
    --results_dir ${NVME}/evals/neurips/enp_zeroshot \
    --scaler_path data/scalers/pediatric_ef_scaler.pkl \
    --n_bootstrap 10000 --seed 42

# ==============================================================================
# UPLOAD RESULTS
# ==============================================================================
echo "[INFO] Uploading results to S3..."
aws s3 sync ${NVME}/evals/neurips/enp_zeroshot/ ${RUN_S3}/enp_zeroshot/ --quiet
aws s3 cp /opt/vjepa2/logs/enp_zeroshot_${SLURM_JOB_ID}.out ${RUN_S3}/logs/master.out --quiet 2>/dev/null || true

echo "[INFO] DONE. Results at: ${RUN_S3}/enp_zeroshot/"
