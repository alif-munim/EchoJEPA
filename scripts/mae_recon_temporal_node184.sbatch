#!/bin/bash
#SBATCH --job-name=mae_recon
#SBATCH --partition=ml-p5-48xlarge
#SBATCH --nodes=1
#SBATCH --nodelist=ip-10-0-50-184
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=1:00:00
#SBATCH --output=logs/mae_recon_temporal_%j.out

set -uo pipefail

REPO_DIR="/opt/vjepa2"
PYTHON="/opt/vjepa2-312/bin/python3"
SCRATCH="/opt/dlami/nvme"
SCRIPT="scripts/rebuttal/mae_reconstruction_temporal.py"
TEST_CSV="${SCRATCH}/echonet_dynamic/test_local.csv"
SAVE_DIR="scripts/rebuttal/mae_recon_vis"
S3_BUCKET="s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts/runs"

cd "${REPO_DIR}"
echo "[$(date)] MAE reconstruction temporal sensitivity on $(hostname)"

# --- Download MAE checkpoints (pretraining checkpoints WITH decoder) ---
CKPT_DIR="${SCRATCH}/mae_pretrain_ckpts"
mkdir -p "${CKPT_DIR}"

declare -A MAE_CKPTS
# e24, e49: original run (job 245). e74, e99: resumed run (job 354).
MAE_CKPTS[mae_e24]="videomae_matched_2n_245/training_folder/checkpoint-24.pth"
MAE_CKPTS[mae_e49]="videomae_matched_2n_245/training_folder/checkpoint-49.pth"
MAE_CKPTS[mae_e74]="videomae_resume_e54_354/training_folder/checkpoint-74.pth"
MAE_CKPTS[mae_e99]="videomae_resume_e54_354/training_folder/checkpoint-99.pth"

for label in mae_e24 mae_e49 mae_e74 mae_e99; do
    s3_key="${MAE_CKPTS[$label]}"
    local_path="${CKPT_DIR}/${label}.pth"
    if [ ! -f "${local_path}" ]; then
        echo "[$(date)] Downloading ${label}..."
        aws s3 cp "${S3_BUCKET}/${s3_key}" "${local_path}" \
            || { echo "FAILED: ${label} download"; exit 1; }
    else
        echo "[$(date)] ${label} already cached"
    fi
done

# --- Verify test CSV exists ---
[ ! -f "${TEST_CSV}" ] && echo "MISSING: ${TEST_CSV}" && exit 1

# --- Run for each checkpoint ---
for label in mae_e24 mae_e49 mae_e74 mae_e99; do
    echo ""
    echo "============================================"
    echo "[$(date)] Running ${label}"
    echo "============================================"
    ${PYTHON} ${SCRIPT} \
        --checkpoint "${CKPT_DIR}/${label}.pth" \
        --test_csv "${TEST_CSV}" \
        --label "${label}" \
        --save_dir "${SAVE_DIR}" \
        --n_videos 100 --n_vis 5 \
        --device cuda:0
done

# --- Upload results ---
echo ""
echo "[$(date)] Uploading results..."
aws s3 sync "${SAVE_DIR}/" "s3://echodata25/neurips/mae_recon_temporal/" \
    --exclude "*.pyc" --quiet

echo "[$(date)] Done. Summary:"
cat "${SAVE_DIR}/temporal_sensitivity_summary.csv"
