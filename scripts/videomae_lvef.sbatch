#!/bin/bash
#SBATCH -p partition-1
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH -t 4-12:00:00
#SBATCH -J videomae_train
#SBATCH -o /tmp/videomae_train-%j.out
#SBATCH -e /tmp/videomae_train-%j.err

set -euxo pipefail

export AWS_DEFAULT_REGION="us-west-2"

# --- Bucket Config ---
ART_BUCKET="sagemaker-hyperpod-lifecycle-495467399120-usw2"
ART_PREFIX="vjepa2-artifacts"
S3_BASE="s3://${ART_BUCKET}/${ART_PREFIX}"

# --- Artifacts ---
ENV_S3="${S3_BASE}/env/vjepa2-312-pack-env.tar"
CODE_S3="${S3_BASE}/code/vjepa2-code.tar"
BOTO3_WHL_S3="${S3_BASE}/wheels/boto3-wheelhouse.tar"

# --- Training Config ---
CFG_REL="configs/eval/vitg-384/videomae_lvef.yaml"
CFG_S3="${S3_BASE}/${CFG_REL}"

# --- Data ---
TRAIN_CSV_REL="${TRAIN_CSV_REL:-data/csv/a4c_b_lvef_train_224px.csv}"
VAL_CSV_REL="${VAL_CSV_REL:-data/csv/a4c_b_lvef_val_224px.csv}"

TRAIN_CSV_S3="${TRAIN_CSV_S3:-${S3_BASE}/${TRAIN_CSV_REL}}"
VAL_CSV_S3="${VAL_CSV_S3:-${S3_BASE}/${VAL_CSV_REL}}"

# --- Checkpoint Config ---
# Google Drive file ID for VideoMAE ViT-L pretrain checkpoint
GDRIVE_ID="1qLOXWb_MGEvaI7tvuAe94CV7S2HXRwT3"
# Local path expected by your YAML config
CKPT_LOCAL="/opt/dlami/nvme/checkpoints/videomae_vitl_pretrain.pth"
# S3 cache location (to avoid re-downloading from Google Drive)
CKPT_S3="${S3_BASE}/checkpoints/videomae_vitl_pretrain_gdrive.pth"

RUN_S3="${S3_BASE}/runs/videomae_lvef_${SLURM_JOB_ID}"

upload_logs() {
  aws s3 cp "/tmp/videomae_train-${SLURM_JOB_ID}.out" "${RUN_S3}/logs/videomae_train-${SLURM_JOB_ID}.out" 2>/dev/null || true
  aws s3 cp "/tmp/videomae_train-${SLURM_JOB_ID}.err" "${RUN_S3}/logs/videomae_train-${SLURM_JOB_ID}.err" 2>/dev/null || true
}
trap 'rc=$?; upload_logs; exit $rc' EXIT

echo "[INFO] host=$(hostname) job=${SLURM_JOB_ID}"
command -v aws
aws --version
nvidia-smi -L || true

SCRATCH="/opt/dlami/nvme"
if [ ! -d "$SCRATCH" ]; then SCRATCH="/tmp"; fi
WORKDIR="${SCRATCH}/videomae_train_${SLURM_JOB_ID}"
mkdir -p "$WORKDIR"

# --- Torch Hub Cache Setup ---
export TORCH_HOME="${WORKDIR}/torch_cache"
mkdir -p "$TORCH_HOME"

cd "$WORKDIR"
echo "[INFO] WORKDIR=$WORKDIR"

# --- Setup Environment ---
aws s3 cp "$ENV_S3" env.tar
mkdir -p env
tar -xf env.tar -C env
set +u
source env/bin/activate
set -u
env/bin/conda-unpack || true

# Install boto3
aws s3 cp "$BOTO3_WHL_S3" boto3-wheelhouse.tar
mkdir -p wheelhouse
tar -xf boto3-wheelhouse.tar -C wheelhouse
python -m pip install --no-index --find-links wheelhouse boto3

# Install gdown for Google Drive downloads
pip install gdown --quiet

# --- Setup Code ---
aws s3 cp "$CODE_S3" code.tar
tar -xf code.tar
REPO_DIR=$(find . -maxdepth 10 -type f -name run_inf.sh -print -quit | xargs -r dirname)
cd "${REPO_DIR}"

# Ensure config exists
if [ ! -f "$CFG_REL" ]; then
  echo "[WARN] ${CFG_REL} not found in code.tar; trying S3: ${CFG_S3}"
  mkdir -p "$(dirname "$CFG_REL")"
  aws s3 cp "$CFG_S3" "$CFG_REL"
fi

# --- Patch Paths ---
python - <<'PY'
from pathlib import Path
import os, re

repo = os.getcwd()
studio = "/home/sagemaker-user/user-default-efs/vjepa2"
efs_pat = re.compile(r"/mnt/custom-file-systems/efs/[^ \n\"']+/vjepa2")

roots = ["configs", "app", "evals", "src"]
exts  = {".yaml", ".py"}

for r in roots:
    p = Path(r)
    if not p.exists(): continue
    for f in p.rglob("*"):
        if f.suffix in exts and f.is_file():
            try:
                t = f.read_text()
                t2 = t.replace(studio, repo)
                t2 = efs_pat.sub(repo, t2)
                if t2 != t:
                    f.write_text(t2)
                    print(f"Patched {f}")
            except: pass
PY

# Make code importable
pip install -e . --no-deps --no-build-isolation || true
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# --- Download Inputs ---
# 1. CSVs
mkdir -p "$(dirname "$TRAIN_CSV_REL")"
aws s3 cp "$TRAIN_CSV_S3" "$TRAIN_CSV_REL"
aws s3 cp "$VAL_CSV_S3" "$VAL_CSV_REL"

# 2. VideoMAE Checkpoint - with S3 caching to avoid repeated Google Drive downloads
echo "[INFO] Downloading VideoMAE ViT-L pretrain checkpoint..."
mkdir -p "$(dirname "$CKPT_LOCAL")"

# First, try to get from S3 cache
if aws s3 cp "$CKPT_S3" "$CKPT_LOCAL" 2>/dev/null; then
    echo "[INFO] Downloaded checkpoint from S3 cache"
else
    echo "[INFO] Checkpoint not in S3 cache, downloading from Google Drive..."
    
    # Download from Google Drive with retry logic
    DOWNLOAD_SUCCESS=false
    for attempt in 1 2 3; do
        echo "[INFO] Google Drive download attempt $attempt/3..."
        if gdown "https://drive.google.com/uc?id=${GDRIVE_ID}" -O "$CKPT_LOCAL" --quiet; then
            DOWNLOAD_SUCCESS=true
            break
        fi
        echo "[WARN] Attempt $attempt failed, waiting before retry..."
        sleep 10
    done
    
    if [ "$DOWNLOAD_SUCCESS" = false ]; then
        echo "[ERROR] Failed to download VideoMAE checkpoint from Google Drive after 3 attempts"
        exit 1
    fi
    
    # Cache to S3 for future runs
    echo "[INFO] Caching checkpoint to S3 for future runs..."
    aws s3 cp "$CKPT_LOCAL" "$CKPT_S3" || echo "[WARN] Failed to cache checkpoint to S3"
fi

# Verify checkpoint
if [ ! -f "$CKPT_LOCAL" ]; then
    echo "[ERROR] Checkpoint file not found at $CKPT_LOCAL"
    exit 1
fi

CKPT_SIZE=$(stat -c%s "$CKPT_LOCAL" 2>/dev/null || stat -f%z "$CKPT_LOCAL" 2>/dev/null || echo "0")
echo "[INFO] Checkpoint size: ${CKPT_SIZE} bytes"

# VideoMAE ViT-L pretrain checkpoint should be ~1.1GB
if [ "$CKPT_SIZE" -lt 1000000000 ]; then
    echo "[WARN] Checkpoint seems small (${CKPT_SIZE} bytes). Expected ~1.1GB for ViT-L pretrain"
    echo "[WARN] The file may be corrupted or incomplete"
fi

# Verify it's a valid PyTorch checkpoint
python - "$CKPT_LOCAL" <<'PY'
import sys
import torch

ckpt_path = sys.argv[1]
try:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model" in ckpt:
        keys = list(ckpt["model"].keys())
    elif "state_dict" in ckpt:
        keys = list(ckpt["state_dict"].keys())
    else:
        keys = list(ckpt.keys())
    
    print(f"[INFO] Checkpoint verified: {len(keys)} keys")
    
    # Check if it's a pretrain checkpoint (has encoder prefix)
    has_encoder = any(k.startswith("encoder.") for k in keys)
    has_decoder = any(k.startswith("decoder") for k in keys)
    
    if has_encoder:
        print("[INFO] Detected PRETRAIN checkpoint format (has encoder.* keys)")
    if has_decoder:
        print("[INFO] Checkpoint contains decoder weights (will be skipped)")
    
    # Show sample keys
    print(f"[INFO] Sample keys: {keys[:5]}")
    
except Exception as e:
    print(f"[ERROR] Failed to load checkpoint: {e}")
    sys.exit(1)
PY

# --- Parse Output Folder from Config ---
CFG_FOLDER="$(python - <<PY
import re
from pathlib import Path
try:
    t = Path("${CFG_REL}").read_text()
    m = re.search(r'^\s*folder:\s*(.+)\s*$', t, re.M)
    print(m.group(1).strip().strip('"').strip("'") if m else "")
except: pass
PY
)"

if [ -z "$CFG_FOLDER" ]; then
    echo "[WARN] Could not parse 'folder' from config. Defaulting to 'output'."
    CFG_FOLDER="output"
fi

# Ensure the folder exists so training doesn't crash on start
mkdir -p "$CFG_FOLDER"

# --- Background Sync Process ---
(
  while true; do
    sleep 900  # 15 minutes
    echo "[INFO] Background syncing checkpoints to S3..."
    aws s3 sync "$CFG_FOLDER" "${RUN_S3}/training_folder/" \
      --exclude "*.pth" --include "*.pt" --include "*.csv" --include "*.log" || true
  done
) &
SYNC_PID=$!

trap 'kill $SYNC_PID; rc=$?; upload_logs; exit $rc' EXIT

# --- Run Training ---

# 1. Detect GPUs dynamically
GPU_COUNT=$(nvidia-smi -L | wc -l)
DEVICES=()
for ((i=0; i<GPU_COUNT; i++)); do DEVICES+=("cuda:${i}"); done
echo "[INFO] Detected ${GPU_COUNT} GPUs: ${DEVICES[*]}"

RUNLOG="videomae_train_${SLURM_JOB_ID}.log"

set +e
python -m evals.main \
  --fname "$CFG_REL" \
  --devices "${DEVICES[@]}" \
  2>&1 | tee "$RUNLOG"
RC=${PIPESTATUS[0]}
set -e

# --- Upload Results ---
aws s3 cp "$RUNLOG" "${RUN_S3}/logs/${RUNLOG}" || true
aws s3 cp "$CFG_REL" "${RUN_S3}/config/videomae_lvef.yaml" || true

echo "[INFO] Uploading training artifacts..."
aws s3 sync "$CFG_FOLDER" "${RUN_S3}/training_folder/" \
  --exclude "*.pth" --include "*.pt" --include "*.csv" --include "*.log" || true

if [ "$RC" -ne 0 ]; then
  echo "[ERROR] Training failed with exit code ${RC}"
  exit "$RC"
fi

echo "[INFO] DONE. Results under: ${RUN_S3}"