#!/bin/bash
#SBATCH -p ml.p5e.48xlarge
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 4-12:00:00
#SBATCH -J vjepa2_pretrain
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
S3_BASE="s3://${ART_BUCKET}/vjepa2-artifacts"

# Artifact Locations
ENV_S3="${S3_BASE}/env/vjepa2-312-pack-env.tar"
CODE_S3="${S3_BASE}/code/vjepa2-code.tar"
BOTO3_WHL_S3="${S3_BASE}/wheels/boto3-wheelhouse.tar"
RUN_S3="${S3_BASE}/runs/vjepa_mimic_pretrain_${SLURM_JOB_ID}"

# Data & Config (S3) - UPDATED: new config for continuation from epoch 130
CSV_S3="${S3_BASE}/data/csv/mimic_annotations_s3.csv"
CFG_S3="${S3_BASE}/configs/train/vitl16/pretrain-mimic-224px-16f.yaml"
CFG_REL="configs/train/vitl16/pretrain-mimic-224px-16f.yaml"

# S3 Location of the Raw Checkpoint (for init if needed)
VITL_RAW_S3="${S3_BASE}/checkpoints/vitl_raw.pth"

# RESUME CHECKPOINT: epoch 130 from previous run
# s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts/runs/vjepa_mimic_pretrain_129/training_folder/log_r7.csv
RESUME_CKPT_S3="${S3_BASE}/runs/vjepa_mimic_pretrain_137/training_folder/e150.pt"

# Define Log Name Globally
RUNLOG="train_vitl_${SLURM_JOB_ID}.log"

# === ROBUST LOGGING & SYNC SETUP ===
sync_ckpts() {
  echo "[INFO] Syncing checkpoints to S3..."
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

trap on_exit EXIT
trap 'echo "[WARN] Received SIGTERM/SIGINT"; on_exit' TERM INT
trap 'echo "[WARN] SIGUSR1: Time limit approaching"; sync_ckpts; upload_logs' USR1

# === HOST SETUP ===
echo "[INFO] host=$(hostname) job=${SLURM_JOB_ID}"
nvidia-smi -L || true

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
REPO_DIR=$(find . -maxdepth 10 -type f -name run_inf.sh -print -quit | xargs -r dirname)
cd "${REPO_DIR}"
pip install -e . --no-deps --no-build-isolation || true
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# Ensure config exists
if [ ! -f "$CFG_REL" ]; then
  echo "[WARN] Config not found in code.tar, fetching from S3..."
  mkdir -p "$(dirname "$CFG_REL")"
  aws s3 cp "$CFG_S3" "$CFG_REL"
fi

# === 3. PREPARE INPUTS ===
echo "[INFO] Downloading Inputs..."

export LOCAL_CSV="data/csv/mimic_annotations_s3.csv"

export LOCAL_CKPT="checkpoints/vitl_in21k.pt"      
export LOCAL_CKPT_RAW="checkpoints/vitl_raw.pth"   
export CFG_REL

mkdir -p "$(dirname "$LOCAL_CSV")"
aws s3 cp "$CSV_S3" "$LOCAL_CSV"

# --- Download raw checkpoint from S3 ---
mkdir -p "$(dirname "$LOCAL_CKPT_RAW")"
echo "[INFO] Downloading ViT-L Raw Checkpoint from S3..."
aws s3 cp "$VITL_RAW_S3" "$LOCAL_CKPT_RAW"

# --- Process Checkpoint (Strip heads, format for V-JEPA) ---
python - <<'PY'
import os, torch

raw_path = os.environ["LOCAL_CKPT_RAW"]
out_path = os.environ["LOCAL_CKPT"]

if not os.path.exists(raw_path):
    raise FileNotFoundError(f"Raw checkpoint not found at {raw_path}")

print(f"[INFO] Loading raw ckpt: {raw_path}")
ckpt = torch.load(raw_path, map_location="cpu")
sd = ckpt.get("model", ckpt.get("state_dict", ckpt))

# strip 'module.' and drop classifier head keys
clean = {}
for k, v in sd.items():
    if k.startswith("module."):
        k = k[len("module."):]
    if k.startswith("head.") or k.startswith("fc_norm."):
        continue
    clean[k] = v

os.makedirs(os.path.dirname(out_path), exist_ok=True)
torch.save(clean, out_path)
print(f"[INFO] Wrote stripped state_dict: {out_path} (keys={len(clean)})")
PY

# === 3.5 DOWNLOAD RESUME CHECKPOINT ===
echo "[INFO] Setting up resume checkpoint from epoch 90..."
mkdir -p output
aws s3 cp "$RESUME_CKPT_S3" "output/latest.pt"
echo "[INFO] Resume checkpoint downloaded to output/latest.pt"

# === 4. PATCH CONFIG PATHS ===
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

repo_root = os.getcwd()
csv_path = os.path.join(repo_root, local_csv)
ckpt_path = os.path.join(repo_root, local_ckpt)
output_dir = os.path.join(repo_root, "output")

# dataset path
text = re.sub(r'(\s*-\s*).*/mimic_annotations_s3.csv', f'\\g<1>{csv_path}', text)

# anneal_ckpt path (points to our renamed vitl_in21k.pt)
text = re.sub(r'anneal_ckpt:.*', f'anneal_ckpt: {ckpt_path}', text)

# output folder
text = re.sub(r'^\s*folder:.*$', f'folder: {output_dir}', text, flags=re.MULTILINE)

config_path.write_text(text)
print(f"Patched config saved to {config_path}")
print(f"  > Dataset: {csv_path}")
print(f"  > Init ckpt: {ckpt_path}")
print(f"  > Output: {output_dir}")
PY

# === 5. RUN TRAINING ===
GPU_COUNT=$(nvidia-smi -L | wc -l)
DEVICES=()
for ((i=0; i<GPU_COUNT; i++)); do DEVICES+=("cuda:${i}"); done
echo "[INFO] Launching on devices: ${DEVICES[*]}"

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

echo "[INFO] Training finished. Stopping background sync..."
kill $SYNC_PID 2>/dev/null || true
wait $SYNC_PID 2>/dev/null || true

echo "[INFO] Uploading Results to ${RUN_S3}..."
aws s3 cp "$RUNLOG" "${RUN_S3}/logs/${RUNLOG}" || true
aws s3 cp "$CFG_REL" "${RUN_S3}/config/train_config.yaml" || true
aws s3 sync "output" "${RUN_S3}/training_folder/" || true

echo "[INFO] Done. Exit Code: $RC"
exit "$RC"
