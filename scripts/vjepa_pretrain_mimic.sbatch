#!/bin/bash
#SBATCH -p ml.p5e.48xlarge
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 4-12:00:00
#SBATCH -J vjepa2_pretrain_mimic_vitl
#SBATCH -o /tmp/vjepa2_pretrain-%j.out
#SBATCH -e /tmp/vjepa2_pretrain-%j.err
#SBATCH --signal=B:USR1@300

set -euxo pipefail

# === ENVIRONMENT VARIABLES ===
export AWS_DEFAULT_REGION="us-west-2"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MASTER_PORT=$((12000 + $RANDOM % 20000))

# === S3 CONFIGURATION ===
ART_BUCKET="sagemaker-hyperpod-lifecycle-495467399120-usw2"
ART_PREFIX="vjepa2-artifacts"
S3_BASE="s3://${ART_BUCKET}/${ART_PREFIX}"

# Artifact Locations
ENV_S3="${S3_BASE}/env/vjepa2-312-pack-env.tar"
CODE_S3="${S3_BASE}/code/vjepa2-code.tar"
BOTO3_WHL_S3="${S3_BASE}/wheels/boto3-wheelhouse.tar"
RUN_S3="${S3_BASE}/runs/${SLURM_JOB_ID}"

# Data & Checkpoint Locations (S3)
CSV_S3="${S3_BASE}/data/csv/mimic_annotations_s3.csv"
CKPT_S3="${S3_BASE}/checkpoints/vitl.pt"

# Config Location (inside repo or S3 override)
CFG_REL="configs/train/vitl16/pretrain-mimic-224px-16f.yaml"
CFG_S3="${S3_BASE}/${CFG_REL}"

# === ROBUST LOGGING & SYNC SETUP ===

sync_ckpts() {
  echo "[INFO] Syncing checkpoints to S3..."
  # Sync only relevant artifacts to reduce S3 load
  # We assume "output" is in the current directory (Repo Root)
  aws s3 sync "output" "${RUN_S3}/training_folder/" \
    --exclude "*" \
    --include "*.pt" --include "*.pth" --include "*.ckpt" \
    --include "*.yaml" --include "*.json" --include "*.csv" \
    --only-show-errors || true
}

upload_logs() {
  echo "[INFO] Uploading Logs..."
  aws s3 cp "/tmp/vjepa2_pretrain-${SLURM_JOB_ID}.out" "${RUN_S3}/logs/job.out" --only-show-errors || true
  aws s3 cp "/tmp/vjepa2_pretrain-${SLURM_JOB_ID}.err" "${RUN_S3}/logs/job.err" --only-show-errors || true
  # Also upload the python runlog if it exists
  if [ -f "$RUNLOG" ]; then
      aws s3 cp "$RUNLOG" "${RUN_S3}/logs/${RUNLOG}" --only-show-errors || true
  fi
}

on_exit() {
  rc=$?
  echo "[INFO] Job Exiting with code $rc. Performing final sync..."
  upload_logs
  sync_ckpts
  exit $rc
}

# Trap setup
# 1. Standard exit/crash
trap on_exit EXIT
# 2. Preemption or Kill signals
trap 'echo "[WARN] Received SIGTERM/SIGINT"; on_exit' TERM INT
# 3. Timeout Warning (from --signal)
trap 'echo "[WARN] SIGUSR1: Time limit approaching"; sync_ckpts; upload_logs' USR1

# === HOST SETUP ===
echo "[INFO] host=$(hostname) job=${SLURM_JOB_ID}"
nvidia-smi -L || true

# Setup Scratch Workspace (NVMe)
SCRATCH="/opt/dlami/nvme"
if [ ! -d "$SCRATCH" ]; then SCRATCH="/tmp"; fi
WORKDIR="${SCRATCH}/vjepa2_pretrain_${SLURM_JOB_ID}"
mkdir -p "$WORKDIR"
cd "$WORKDIR"
echo "[INFO] WORKDIR=$WORKDIR"

# === 1. SETUP ENVIRONMENT ===
echo "[INFO] Downloading Environment..."
aws s3 cp "$ENV_S3" env.tar
mkdir -p env
tar -xf env.tar -C env
set +u
source env/bin/activate
set -u
env/bin/conda-unpack || true

# Install boto3 (offline)
aws s3 cp "$BOTO3_WHL_S3" boto3-wheelhouse.tar
mkdir -p wheelhouse
tar -xf boto3-wheelhouse.tar -C wheelhouse
python -m pip install --no-index --find-links wheelhouse boto3

# === 2. SETUP CODE ===
echo "[INFO] Downloading Code..."
aws s3 cp "$CODE_S3" code.tar
tar -xf code.tar
# Find repo root dynamically
REPO_DIR=$(find . -maxdepth 10 -type f -name run_inf.sh -print -quit | xargs -r dirname)
cd "${REPO_DIR}"
pip install -e . --no-deps --no-build-isolation || true
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# Ensure config exists (fetch from S3 if missing from tar)
if [ ! -f "$CFG_REL" ]; then
  echo "[WARN] Config not found in code.tar, fetching from S3..."
  mkdir -p "$(dirname "$CFG_REL")"
  aws s3 cp "$CFG_S3" "$CFG_REL"
fi

# === 3. PREPARE INPUTS ===
echo "[INFO] Downloading Inputs..."

# Define Local Paths (Relative to Repo)
export LOCAL_CSV="data/csv/mimic_annotations_s3.csv"
export LOCAL_CKPT="checkpoints/vitl.pt"
export CFG_REL

# Download CSV
mkdir -p "$(dirname "$LOCAL_CSV")"
aws s3 cp "$CSV_S3" "$LOCAL_CSV"

# Download Checkpoint
mkdir -p "$(dirname "$LOCAL_CKPT")"
aws s3 cp "$CKPT_S3" "$LOCAL_CKPT"

# === 4. PATCH CONFIG (FIXED) ===
echo "[INFO] Patching Config Paths in $CFG_REL ..."

python - <<'PY'
import re
from pathlib import Path
import os

config_rel = os.environ["CFG_REL"]
local_csv = os.environ["LOCAL_CSV"]
local_ckpt = os.environ["LOCAL_CKPT"]

config_path = Path(config_rel)
text = config_path.read_text()

# Get absolute path to the repo root on this node
repo_root = os.getcwd()
csv_path = os.path.join(repo_root, local_csv)
ckpt_path = os.path.join(repo_root, local_ckpt)
output_dir = os.path.join(repo_root, "output")

# Regex Replacements
# 1. Replace the CSV path.
text = re.sub(r'(\s*-\s*).*/mimic_annotations_s3.csv', f'\\g<1>{csv_path}', text)

# 2. Replace the Checkpoint path (anneal_ckpt)
text = re.sub(r'anneal_ckpt:.*vitl.pt', f'anneal_ckpt: {ckpt_path}', text)

# 3. Replace the Output Folder
text = re.sub(r'^\s*folder:.*$', f'folder: {output_dir}', text, flags=re.MULTILINE)

config_path.write_text(text)
print(f"Patched config saved to {config_path}")
print(f"  > Dataset: {csv_path}")
print(f"  > Checkpoint: {ckpt_path}")
print(f"  > Output: {output_dir}")
PY

# === 5. RUN TRAINING ===
GPU_COUNT=$(nvidia-smi -L | wc -l)
DEVICES=()
for ((i=0; i<GPU_COUNT; i++)); do DEVICES+=("cuda:${i}"); done
echo "[INFO] Launching on devices: ${DEVICES[*]}"

RUNLOG="train_vitl_${SLURM_JOB_ID}.log"

# --- START BACKGROUND SYNC ---
echo "[INFO] Starting background checkpoint sync (every 10 mins)..."
(
  while true; do
    sleep 600
    sync_ckpts
  done
) &
SYNC_PID=$!

set +e
python -m app.main \
  --fname "$CFG_REL" \
  --devices "${DEVICES[@]}" \
  2>&1 | tee "$RUNLOG"
RC=${PIPESTATUS[0]}
set -e

# --- STOP BACKGROUND SYNC ---
echo "[INFO] Training finished. Stopping background sync..."
kill $SYNC_PID 2>/dev/null || true
wait $SYNC_PID 2>/dev/null || true

# === 6. UPLOAD RESULTS ===
echo "[INFO] Uploading Results to ${RUN_S3}..."

# Upload Logs & Config
aws s3 cp "$RUNLOG" "${RUN_S3}/logs/${RUNLOG}" || true
aws s3 cp "$CFG_REL" "${RUN_S3}/config/train_config.yaml" || true

# Final Sync is handled by on_exit trap, but we can do an explicit one here
# ensuring we get everything including non-checkpoint files if needed.
aws s3 sync "output" "${RUN_S3}/training_folder/" || true

echo "[INFO] Done. Exit Code: $RC"
exit "$RC"