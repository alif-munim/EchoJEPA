#!/bin/bash
#SBATCH --job-name=enp-lvef-4m
#SBATCH --partition=ml-p5-48xlarge
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=/opt/vjepa2/logs/enp_lvef_4models_%j.out
#SBATCH --nodelist=ip-10-0-50-184
#SBATCH --exclusive

set -eo pipefail

cd /opt/vjepa2
source /opt/vjepa2-312/bin/activate
set -u

# cuBLAS bf16 fix for H100
export LD_LIBRARY_PATH=/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/lib:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}

# S3 paths
HYP="s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts"
CLEAN="s3://echodata25/neurips"
NVME="/opt/dlami/nvme"
RUN_S3="${HYP}/runs/neurips_enp_lvef_4models_${SLURM_JOB_ID}"

mkdir -p ${NVME}/checkpoints ${NVME}/data/csv logs

# ==============================================================================
# REFRESH CREDENTIALS (IMDS)
# ==============================================================================
echo "[INFO] Refreshing credentials from Instance Metadata..."
TOKEN=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
ROLE_NAME=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/iam/security-credentials/)
CREDS_JSON=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/iam/security-credentials/$ROLE_NAME)
export AWS_ACCESS_KEY_ID=$(echo "$CREDS_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['AccessKeyId'])")
export AWS_SECRET_ACCESS_KEY=$(echo "$CREDS_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['SecretAccessKey'])")
export AWS_SESSION_TOKEN=$(echo "$CREDS_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['Token'])")

# ==============================================================================
# DOWNLOAD DATA & CHECKPOINTS
# ==============================================================================
echo "[INFO] Downloading pediatric CSVs..."
aws s3 cp ${HYP}/csv_pediatric/echonet_pediatric_train_s3.csv ${NVME}/data/csv/ --quiet
aws s3 cp ${HYP}/csv_pediatric/echonet_pediatric_val_s3.csv ${NVME}/data/csv/ --quiet

echo "[INFO] Downloading encoder checkpoints..."
# JEPA IN21K e100 (from old run 125, same init)
aws s3 cp ${HYP}/runs/vjepa_mimic_pretrain_125/training_folder/e100.pt ${NVME}/checkpoints/echojepa_l_in21k_e100.pt --quiet
# BYOL e100
aws s3 cp ${HYP}/runs/byol_pretrain_resume_342/checkpoints/e100.pt ${NVME}/checkpoints/echobyol_l_e100.pt --quiet
# MAE e99
aws s3 cp ${HYP}/runs/videomae_resume_e54_354/training_folder/checkpoint-99.pth ${NVME}/checkpoints/echomae_l_mimic_ep99.pth --quiet
# SALT v1 e79
aws s3 cp ${HYP}/runs/salt_s2_pretrain_388/checkpoints/e79.pt ${NVME}/checkpoints/salt_s2_vitl_e79.pt --quiet

echo "[INFO] All downloads complete."

pip install -e . --no-deps --no-build-isolation 2>/dev/null || true
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# Detect GPUs
GPU_COUNT=$(nvidia-smi -L | wc -l)
DEVICES=()
for ((i=0; i<GPU_COUNT; i++)); do DEVICES+=("cuda:${i}"); done
echo "[INFO] ${GPU_COUNT} GPUs: ${DEVICES[*]}"

# Background S3 sync every 10 min
(
    while true; do
        sleep 600
        for d in ${NVME}/evals/neurips/echo*_enp_lvef ${NVME}/evals/neurips/salt*_enp_lvef; do
            [ -d "$d" ] && aws s3 sync "$d" "${RUN_S3}/$(basename $d)/" --quiet 2>/dev/null || true
        done
    done
) &
SYNC_PID=$!

on_exit() {
    kill $SYNC_PID 2>/dev/null || true
    wait $SYNC_PID 2>/dev/null || true
    echo "[INFO] Final S3 upload..."
    for d in ${NVME}/evals/neurips/echo*_enp_lvef ${NVME}/evals/neurips/salt*_enp_lvef; do
        [ -d "$d" ] && aws s3 sync "$d" "${RUN_S3}/$(basename $d)/" --quiet 2>/dev/null || true
    done
}
trap on_exit EXIT

# ==============================================================================
# RUN 4 PROBE TRAININGS SEQUENTIALLY
# ==============================================================================
CONFIGS=(
    "configs/eval/vitl/neurips/echojepa_l_e100_enp_lvef_d1.yaml"
    "configs/eval/vitl/neurips/echobyol_l_e100_enp_lvef_d1.yaml"
    "configs/eval/vitl/neurips/echomae_l_e99_enp_lvef_d1.yaml"
    "configs/eval/vitl/neurips/salt_s2v1_e79_enp_lvef_d1.yaml"
)
NAMES=("JEPA" "BYOL" "MAE" "SALT")

OVERALL_RC=0
for i in "${!CONFIGS[@]}"; do
    echo ""
    echo "============================================================"
    echo "[INFO] Starting ${NAMES[$i]} pediatric probe — $(date)"
    echo "       Config: ${CONFIGS[$i]}"
    echo "============================================================"

    set +e
    python3 -m evals.main \
        --fname "${CONFIGS[$i]}" \
        --devices "${DEVICES[@]}" \
        2>&1 | tee "logs/enp_${NAMES[$i],,}_${SLURM_JOB_ID}.log"
    RC=${PIPESTATUS[0]}
    set -e

    if [ "$RC" -ne 0 ]; then
        echo "[ERROR] ${NAMES[$i]} failed with exit code ${RC}"
        OVERALL_RC=1
    else
        echo "[INFO] ${NAMES[$i]} completed successfully — $(date)"
    fi
done

echo ""
echo "============================================================"
echo "[INFO] All 4 pediatric probes done — $(date)"
echo "============================================================"

exit $OVERALL_RC
