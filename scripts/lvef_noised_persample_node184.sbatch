#!/bin/bash
#SBATCH --job-name=salt_lvnp
#SBATCH --partition=ml-p5-48xlarge
#SBATCH --nodes=1
#SBATCH --nodelist=ip-10-0-50-184
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --gpus-per-node=8
#SBATCH --time=4:00:00
#SBATCH --output=logs/salt_lvef_noised_persample_%j.out
#SBATCH --exclusive

set -uo pipefail

REPO_DIR="/opt/vjepa2"
PYTHON="/opt/vjepa2-312/bin/python3"
SCRATCH="/opt/dlami/nvme"
SCRIPT="scripts/rebuttal/noised_inference_persample.py"
TEST_CSV="${SCRATCH}/echonet_dynamic/test_local.csv"

cd "${REPO_DIR}"
echo "[$(date)] SALT LVEF noised per-sample on $(hostname)"

# NOTE: salt_s2v1_e79.pt is the correct checkpoint matching the probe.
# salt_s2_vitl_224px_16f/e79.pt is a DIFFERENT training run (different loss/lr/weights).
SALT_CKPT="${REPO_DIR}/checkpoints/pretrain/mimic/salt_s2v1_e79.pt"

# Download SALT LVEF probe
PROBE_DIR="${SCRATCH}/lvef_probes"
mkdir -p "${PROBE_DIR}"
SALT_PROBE="${PROBE_DIR}/salt_v1_e79_best.pt"
if [ ! -f "${SALT_PROBE}" ]; then
    echo "Downloading SALT LVEF probe..."
    aws s3 cp "s3://echodata25/neurips/probes/end_lvef_e100/salt_v1_e79/best.pt" "${SALT_PROBE}" \
        || { echo "FAILED: SALT probe download"; exit 1; }
fi

# Verify
[ ! -f "${SALT_CKPT}" ] && echo "MISSING: ${SALT_CKPT}" && exit 1
[ ! -f "${SALT_PROBE}" ] && echo "MISSING: ${SALT_PROBE}" && exit 1
[ ! -f "${TEST_CSV}" ] && echo "MISSING: ${TEST_CSV}" && exit 1

${PYTHON} ${SCRIPT} --encoder_type vjepa --encoder_checkpoint "${SALT_CKPT}" \
    --encoder_model_name vit_large --encoder_key target_encoder \
    --probe_checkpoint "${SALT_PROBE}" \
    --test_csv "${TEST_CSV}" --device cuda:0 --label salt_v1_e79

# Upload
echo "[$(date)] Uploading..."
CSV="scripts/rebuttal/samples/salt_v1_e79_noised_lvef_persample.csv"
[ -f "${CSV}" ] && aws s3 cp "${CSV}" "s3://echodata25/neurips/results/lvef_noised/salt_v1_e79_noised_lvef_persample.csv"
echo "[$(date)] Done."
