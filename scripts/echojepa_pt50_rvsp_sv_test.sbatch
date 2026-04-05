#!/bin/bash
#SBATCH -p ml-p5-48xlarge
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 1-00:00:00
#SBATCH -J jep_svt
#SBATCH -o /tmp/jep_svt-%j.out
#SBATCH -e /tmp/jep_svt-%j.err
#SBATCH --chdir=/tmp
#SBATCH --nodelist=ip-10-0-50-83

set -euxo pipefail

export PATH="/opt/vjepa2-312/bin:$PATH"

# =========================
# AWS CONFIG
# =========================
export AWS_DEFAULT_REGION="us-west-2"
export AWS_REGION="us-west-2"
export AWS_SDK_LOAD_CONFIG=1
export HOME="/home/ubuntu"
export AWS_SHARED_CREDENTIALS_FILE="${HOME}/.aws/credentials"
export AWS_CONFIG_FILE="${HOME}/.aws/config"
unset AWS_EC2_METADATA_DISABLED 2>/dev/null || true
export AWS_METADATA_SERVICE_TIMEOUT=5
export AWS_METADATA_SERVICE_NUM_ATTEMPTS=3

# =========================
# S3 CONFIGURATION
# =========================
ART_BUCKET="sagemaker-hyperpod-lifecycle-495467399120-usw2"
S3_BASE="s3://${ART_BUCKET}/vjepa2-artifacts"

ENV_S3="${S3_BASE}/env/vjepa2-312-pack-env.tar"

# Probe checkpoints (from training runs)
A4C_PROBE_S3="${S3_BASE}/runs/echojepa_pt50_rvsp_a4c_301/training_folder/video_classification_frozen/icml-echojepa-l-pt50-rvsp-a4c-d4/best.pt"
PSAX_PROBE_S3="${S3_BASE}/runs/echojepa_pt50_rvsp_psax_305/training_folder/video_classification_frozen/icml-echojepa-l-pt50-rvsp-psax-d4/best.pt"

# Encoder checkpoint
ENC_CKPT_S3="${S3_BASE}/checkpoints/echojepa_l_mimic_ep50.pt"
ENC_CKPT_LOCAL="/opt/dlami/nvme/checkpoints/echojepa_l_mimic_ep50.pt"

# Test CSVs
A4C_TEST_S3="${S3_BASE}/data/csv/rvsp_test_a4c.csv"
PSAX_TEST_S3="${S3_BASE}/data/csv/rvsp_test_psax.csv"

# Run output
RUN_S3="${S3_BASE}/runs/echojepa_pt50_rvsp_sv_test_${SLURM_JOB_ID}"

# =========================
# LOGGING
# =========================
upload_logs() {
  aws s3 cp "/tmp/jep_svt-${SLURM_JOB_ID}.out" "${RUN_S3}/logs/job.out" --only-show-errors 2>/dev/null || true
  aws s3 cp "/tmp/jep_svt-${SLURM_JOB_ID}.err" "${RUN_S3}/logs/job.err" --only-show-errors 2>/dev/null || true
}
trap 'rc=$?; upload_logs; exit $rc' EXIT

# =========================
# HOST SETUP
# =========================
echo "[INFO] host=$(hostname) job=${SLURM_JOB_ID}"
nvidia-smi -L || true

SCRATCH="/opt/dlami/nvme"
[ ! -d "$SCRATCH" ] && SCRATCH="/tmp"
WORKDIR="${SCRATCH}/jep_svt_${SLURM_JOB_ID}"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

# =========================
# ENVIRONMENT
# =========================
echo "[INFO] Setting up environment..."
if [ ! -d "env/bin" ]; then
    aws s3 cp "$ENV_S3" env.tar
    mkdir -p env && tar -xf env.tar -C env
    set +u && source env/bin/activate && set -u
    env/bin/conda-unpack || true
    pip install "numpy<2.0" boto3 --no-cache-dir
    rm -f env.tar
else
    set +u && source env/bin/activate && set -u
fi

# =========================
# CODE
# =========================
REPO_DIR="/opt/vjepa2"
cd "$REPO_DIR"
pip install -e . --no-deps --no-build-isolation 2>/dev/null || true
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# =========================
# DOWNLOAD DATA, PROBES, ENCODER
# =========================
echo "[INFO] Downloading test CSVs..."
mkdir -p /opt/dlami/nvme/data/csv
aws s3 cp "$A4C_TEST_S3" /opt/dlami/nvme/data/csv/rvsp_test_a4c.csv
aws s3 cp "$PSAX_TEST_S3" /opt/dlami/nvme/data/csv/rvsp_test_psax.csv

echo "[INFO] Downloading probe checkpoints..."
mkdir -p /opt/dlami/nvme/probes
aws s3 cp "$A4C_PROBE_S3" /opt/dlami/nvme/probes/echojepa_l_pt50_rvsp_a4c_best.pt
aws s3 cp "$PSAX_PROBE_S3" /opt/dlami/nvme/probes/echojepa_l_pt50_rvsp_psax_best.pt

echo "[INFO] Downloading encoder checkpoint..."
mkdir -p "$(dirname "$ENC_CKPT_LOCAL")"
aws s3 cp "$ENC_CKPT_S3" "$ENC_CKPT_LOCAL"

mkdir -p /opt/dlami/nvme/predictions

# =========================
# RUN A4C TEST INFERENCE
# =========================
GPU_COUNT=$(nvidia-smi -L | wc -l)
DEVICES=()
for ((i=0; i<GPU_COUNT; i++)); do DEVICES+=("cuda:${i}"); done

echo "[INFO] Running A4C test inference..."
set +e
python -m evals.main \
  --fname configs/inference/vitl/icml/echojepa_l_pt50_rvsp_a4c_test.yaml \
  --devices "${DEVICES[@]}" \
  2>&1 | tee "a4c_test_${SLURM_JOB_ID}.log"
A4C_RC=${PIPESTATUS[0]}
set -e

# =========================
# RUN PSAX TEST INFERENCE
# =========================
echo "[INFO] Running PSAX test inference..."
set +e
python -m evals.main \
  --fname configs/inference/vitl/icml/echojepa_l_pt50_rvsp_psax_test.yaml \
  --devices "${DEVICES[@]}" \
  2>&1 | tee "psax_test_${SLURM_JOB_ID}.log"
PSAX_RC=${PIPESTATUS[0]}
set -e

# =========================
# UPLOAD RESULTS
# =========================
echo "[INFO] Uploading results..."
aws s3 cp /opt/dlami/nvme/predictions/icml-echojepa-l-pt50-rvsp-a4c-test.csv "${RUN_S3}/predictions/a4c_test.csv" --only-show-errors 2>/dev/null || true
aws s3 cp /opt/dlami/nvme/predictions/icml-echojepa-l-pt50-rvsp-psax-test.csv "${RUN_S3}/predictions/psax_test.csv" --only-show-errors 2>/dev/null || true
aws s3 cp "a4c_test_${SLURM_JOB_ID}.log" "${RUN_S3}/logs/a4c_test.log" --only-show-errors || true
aws s3 cp "psax_test_${SLURM_JOB_ID}.log" "${RUN_S3}/logs/psax_test.log" --only-show-errors || true

# Upload training folder outputs
for dir in /opt/dlami/nvme/evals/icml/echojepa_pt50_rvsp_a4c_test /opt/dlami/nvme/evals/icml/echojepa_pt50_rvsp_psax_test; do
  if [ -d "$dir" ]; then
    aws s3 sync "$dir" "${RUN_S3}/training_folder/$(basename $dir)/" \
      --exclude "*.pth" --include "*.pt" --include "*.csv" --include "*.log" --include "*.json" \
      --only-show-errors || true
  fi
done

echo "[INFO] A4C RC=$A4C_RC, PSAX RC=$PSAX_RC"
if [ "$A4C_RC" -ne 0 ] || [ "$PSAX_RC" -ne 0 ]; then
  echo "[ERROR] One or both inference runs failed"
  exit 1
fi

echo "[INFO] DONE. Results at: ${RUN_S3}"
