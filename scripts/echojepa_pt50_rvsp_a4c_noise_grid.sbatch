#!/bin/bash
#SBATCH -p ml-p5-48xlarge
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 1-00:00:00
#SBATCH -J jep_noi
#SBATCH -o /tmp/jep_noi-%j.out
#SBATCH -e /tmp/jep_noi-%j.err
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

# Probe + encoder
A4C_PROBE_S3="${S3_BASE}/runs/echojepa_pt50_rvsp_a4c_301/training_folder/video_classification_frozen/icml-echojepa-l-pt50-rvsp-a4c-d4/best.pt"
ENC_CKPT_S3="${S3_BASE}/checkpoints/echojepa_l_mimic_ep50.pt"

# Test CSV (single-view A4C, raw labels from train set)
TEST_CSV_S3="${S3_BASE}/data/csv/rvsp_val_a4c.csv"

# Run output
RUN_S3="${S3_BASE}/runs/echojepa_pt50_rvsp_a4c_noise_${SLURM_JOB_ID}"

# =========================
# LOGGING
# =========================
upload_logs() {
  aws s3 cp "/tmp/jep_noi-${SLURM_JOB_ID}.out" "${RUN_S3}/logs/job.out" --only-show-errors 2>/dev/null || true
  aws s3 cp "/tmp/jep_noi-${SLURM_JOB_ID}.err" "${RUN_S3}/logs/job.err" --only-show-errors 2>/dev/null || true
}
trap 'rc=$?; upload_logs; exit $rc' EXIT

# =========================
# HOST SETUP
# =========================
echo "[INFO] host=$(hostname) job=${SLURM_JOB_ID}"
nvidia-smi -L || true

SCRATCH="/opt/dlami/nvme"
WORKDIR="${SCRATCH}/jep_noi_${SLURM_JOB_ID}"
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
echo "[INFO] Downloading probe + encoder + test CSV..."
mkdir -p /opt/dlami/nvme/probes /opt/dlami/nvme/data/csv /opt/dlami/nvme/checkpoints
aws s3 cp "$A4C_PROBE_S3" /opt/dlami/nvme/probes/echojepa_l_pt50_rvsp_a4c_best.pt
aws s3 cp "$ENC_CKPT_S3" /opt/dlami/nvme/checkpoints/echojepa_l_mimic_ep50.pt
aws s3 cp "$TEST_CSV_S3" /opt/dlami/nvme/data/csv/rvsp_val_a4c.csv

OUTDIR="/opt/dlami/nvme/noise_results/rvsp_a4c"
mkdir -p "$OUTDIR"

# =========================
# RUN NOISE GRID — 10 conditions in parallel across 8 GPUs
# =========================
# Conditions: clean + 3 perturbation types × 3 severity levels = 10
# GPU allocation: clean on GPU 0, then 9 conditions on GPUs 0-7 (some GPUs run 2)

COMMON_ARGS="
  --encoder_type vjepa
  --encoder_checkpoint /opt/dlami/nvme/checkpoints/echojepa_l_mimic_ep50.pt
  --encoder_model_name vit_large
  --encoder_key target_encoder
  --probe_checkpoint /opt/dlami/nvme/probes/echojepa_l_pt50_rvsp_a4c_best.pt
  --task_type regression
  --test_csv /opt/dlami/nvme/data/csv/rvsp_val_a4c.csv
  --resolution 224
  --frames 16
  --frame_step 1
  --label JEPA-A4C
"

echo "[INFO] Running clean baseline on GPU 0..."
python scripts/neurips/noised_inference.py $COMMON_ARGS \
  --perturbation_types \
  --severity_levels \
  --device cuda:0 \
  --output "$OUTDIR/clean.csv" \
  2>&1 | tee "$OUTDIR/clean.log" &
PID_CLEAN=$!

echo "[INFO] Running noise grid across GPUs 1-7..."
GPU=1
for PTYPE in depth_attenuation gaussian_shadow haze_artifact; do
  for SEV in mild moderate severe; do
    echo "[INFO] Launching $PTYPE/$SEV on cuda:$GPU"
    python scripts/neurips/noised_inference.py $COMMON_ARGS \
      --perturbation_types "$PTYPE" \
      --severity_levels "$SEV" \
      --skip_clean \
      --device "cuda:$GPU" \
      --output "$OUTDIR/${PTYPE}_${SEV}.csv" \
      2>&1 | tee "$OUTDIR/${PTYPE}_${SEV}.log" &
    GPU=$(( (GPU + 1) % 8 ))
    # Stagger launches slightly to avoid OOM
    sleep 5
  done
done

echo "[INFO] Waiting for all conditions to complete..."
wait

echo "[INFO] All conditions complete. Uploading results..."
aws s3 sync "$OUTDIR" "${RUN_S3}/noise_results/" --only-show-errors || true

echo "[INFO] DONE. Results at: ${RUN_S3}/noise_results/"
