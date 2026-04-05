#!/bin/bash
#SBATCH -p ml-p5-48xlarge
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 1-00:00:00
#SBATCH -J rvsp_ng
#SBATCH -o /tmp/rvsp_ng-%j.out
#SBATCH -e /tmp/rvsp_ng-%j.err
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

# Multi-view RVSP probe + encoder + test CSV
PROBE_S3="${S3_BASE}/runs/echojepa_pt50_rvsp_full/video_classification_frozen/icml-echojepa-l-pt50-rvsp-d4-full/best.pt"
ENC_S3="${S3_BASE}/checkpoints/echojepa_l_mimic_ep50.pt"
TEST_CSV_S3="${S3_BASE}/data/csv/rvsp_test.csv"

CFG_REL="configs/inference/vitl/icml/echojepa_l_pt50_rvsp_multi_test_hp.yaml"

RUN_S3="${S3_BASE}/runs/echojepa_pt50_rvsp_noise_grid_${SLURM_JOB_ID}"

# =========================
# LOGGING
# =========================
upload_logs() {
  aws s3 cp "/tmp/rvsp_ng-${SLURM_JOB_ID}.out" "${RUN_S3}/logs/job.out" --only-show-errors 2>/dev/null || true
  aws s3 cp "/tmp/rvsp_ng-${SLURM_JOB_ID}.err" "${RUN_S3}/logs/job.err" --only-show-errors 2>/dev/null || true
}
trap 'rc=$?; upload_logs; exit $rc' EXIT

echo "[INFO] host=$(hostname) job=${SLURM_JOB_ID}"
nvidia-smi -L || true

SCRATCH="/opt/dlami/nvme"
WORKDIR="${SCRATCH}/rvsp_ng_${SLURM_JOB_ID}"
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
export LD_LIBRARY_PATH="/opt/conda/lib:${LD_LIBRARY_PATH:-}"

# =========================
# DOWNLOAD
# =========================
echo "[INFO] Downloading probe, encoder, test CSV..."
mkdir -p /opt/dlami/nvme/probes /opt/dlami/nvme/data/csv /opt/dlami/nvme/checkpoints
aws s3 cp "$PROBE_S3" /opt/dlami/nvme/probes/echojepa_l_pt50_rvsp_multi_best.pt
aws s3 cp "$ENC_S3" /opt/dlami/nvme/checkpoints/echojepa_l_mimic_ep50.pt
aws s3 cp "$TEST_CSV_S3" /opt/dlami/nvme/data/csv/rvsp_test.csv

OUTDIR="/opt/dlami/nvme/noise_results/rvsp_multi"
mkdir -p "$OUTDIR"

# =========================
# RUN NOISE GRID — clean + 9 conditions across 8 GPUs
# =========================
echo "[INFO] Running clean baseline on cuda:0..."
TMPDIR=/tmp python -m evals.main \
  --fname "$CFG_REL" \
  --devices cuda:0 \
  --val_only 2>&1 | tee "$OUTDIR/clean.log"
echo "[INFO] Clean baseline done."

echo "[INFO] Running noise grid across GPUs..."
GPU=0
PIDS=""
for PTYPE in depth_attenuation gaussian_shadow haze_artifact; do
  for SEV in mild moderate severe; do
    echo "[INFO] Launching $PTYPE/$SEV on cuda:$GPU"
    TMPDIR=/tmp PERTURBATION_TYPE=$PTYPE PERTURBATION_SEVERITY=$SEV \
      python -m evals.main \
        --fname "$CFG_REL" \
        --devices "cuda:$GPU" \
        --val_only 2>&1 | tee "$OUTDIR/${PTYPE}_${SEV}.log" &
    PIDS="$PIDS $!"
    GPU=$(( (GPU + 1) % 8 ))
    sleep 3
  done
done

echo "[INFO] Waiting for all noise conditions..."
for PID in $PIDS; do
  wait $PID || echo "[WARN] PID $PID failed"
done

echo "[INFO] All conditions complete. Uploading results..."
aws s3 sync "$OUTDIR" "${RUN_S3}/noise_results/" --only-show-errors || true

# Also upload the eval output folder
EVAL_OUT="/opt/dlami/nvme/evals/icml/rvsp_noise_grid"
if [ -d "$EVAL_OUT" ]; then
  aws s3 sync "$EVAL_OUT" "${RUN_S3}/eval_output/" \
    --exclude "*.pth" --include "*.pt" --include "*.csv" --include "*.log" --include "*.json" \
    --only-show-errors || true
fi

echo "[INFO] DONE. Results at: ${RUN_S3}"
