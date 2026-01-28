#!/bin/bash
#SBATCH -p ml.p5e.48xlarge 
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH -t 4-12:00:00           # Increased time for training (adjust as needed)
#SBATCH -J vjepa_train
#SBATCH -o /tmp/vitl_rvsp-%j.out
#SBATCH -e /tmp/vitl_rvsp-%j.err

set -euxo pipefail

export AWS_DEFAULT_REGION="us-west-2"

ART_BUCKET="sagemaker-hyperpod-lifecycle-495467399120-usw2"
ART_PREFIX="vjepa2-artifacts"
S3_BASE="s3://${ART_BUCKET}/${ART_PREFIX}"

# --- Artifacts ---
ENV_S3="${S3_BASE}/env/vjepa2-312-pack-env.tar"
CODE_S3="${S3_BASE}/code/vjepa2-code.tar"
BOTO3_WHL_S3="${S3_BASE}/wheels/boto3-wheelhouse.tar"

# --- Training Config ---
# Ensure this matches the path inside your code.tar or is uploaded separately
CFG_REL="configs/eval/vitl/rvsp.yaml"
CFG_S3="${S3_BASE}/${CFG_REL}"

# --- Data (override-friendly) ---
TRAIN_CSV_REL="${TRAIN_CSV_REL:-data/csv/rvsp_train.csv}"
VAL_CSV_REL="${VAL_CSV_REL:-data/csv/rvsp_val.csv}"

# Allow overriding S3 paths too; otherwise derive from S3_BASE + relpath
TRAIN_CSV_S3="${TRAIN_CSV_S3:-${S3_BASE}/${TRAIN_CSV_REL}}"
VAL_CSV_S3="${VAL_CSV_S3:-${S3_BASE}/${VAL_CSV_REL}}"


# --- Pretrained Backbone ---
ENC_CKPT_REL="checkpoints/anneal/keep/vitl-pt-210-an25.pt"
ENC_CKPT_S3="${S3_BASE}/${ENC_CKPT_REL}"

RUN_S3="${S3_BASE}/runs/vitl_rvsp_${SLURM_JOB_ID}"

upload_logs() {
  aws s3 cp "/tmp/vjepa_train-${SLURM_JOB_ID}.out" "${RUN_S3}/logs/vjepa_train-${SLURM_JOB_ID}.out" 2>/dev/null || true
  aws s3 cp "/tmp/vjepa_train-${SLURM_JOB_ID}.err" "${RUN_S3}/logs/vjepa_train-${SLURM_JOB_ID}.err" 2>/dev/null || true
}
trap 'rc=$?; upload_logs; exit $rc' EXIT

echo "[INFO] host=$(hostname) job=${SLURM_JOB_ID}"
command -v aws
aws --version
nvidia-smi -L || true

SCRATCH="/opt/dlami/nvme"
if [ ! -d "$SCRATCH" ]; then SCRATCH="/tmp"; fi
WORKDIR="${SCRATCH}/vjepa2_train_${SLURM_JOB_ID}"
mkdir -p "$WORKDIR"
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
# Replaces hardcoded Studio paths with the current SLURM scratch path
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

# 2. Backbone Checkpoint
mkdir -p "$(dirname "$ENC_CKPT_REL")"
aws s3 cp "$ENC_CKPT_S3" "$ENC_CKPT_REL"

# --- Parse Output Folder from Config ---
# We need to know where the script writes logs so we can upload them later
CFG_FOLDER="$(python - <<'PY'
import re
from pathlib import Path
# Note: Using the specific config file name
t = Path("configs/eval/vitl/rvsp.yaml").read_text()
m = re.search(r'^\s*folder:\s*(.+)\s*$', t, re.M)
print(m.group(1).strip().strip('"').strip("'") if m else "")
PY
)"

test -n "$CFG_FOLDER" || { echo "[ERROR] Could not parse 'folder' from config"; exit 5; }
mkdir -p "$CFG_FOLDER"

# --- Run Training ---

# Uses evals.main (direct execution) instead of main_distributed (submitit)

# 1. Detect GPUs dynamically (same logic as your inference script)
GPU_COUNT=$(nvidia-smi -L | wc -l)
DEVICES=()
for ((i=0; i<GPU_COUNT; i++)); do DEVICES+=("cuda:${i}"); done
echo "[INFO] Detected ${GPU_COUNT} GPUs: ${DEVICES[*]}"

RUNLOG="rvsp_train_${SLURM_JOB_ID}.log"

set +e
# 2. Run directly with --devices
python -m evals.main \
  --fname "$CFG_REL" \
  --devices "${DEVICES[@]}" \
  2>&1 | tee "$RUNLOG"
RC=${PIPESTATUS[0]}
set -e


# --- Upload Results ---
aws s3 cp "$RUNLOG" "${RUN_S3}/logs/${RUNLOG}" || true
aws s3 cp "$CFG_REL" "${RUN_S3}/config/vitl_rvsp_regression.yaml" || true

# Upload the entire output folder (logs, checkpoints, best.pt, etc.)
echo "[INFO] Uploading training artifacts..."
aws s3 sync "$CFG_FOLDER" "${RUN_S3}/training_folder/" \
  --exclude "*.pth" --include "*.pt" --include "*.csv" --include "*.log" || true

if [ "$RC" -ne 0 ]; then
  echo "[ERROR] Training failed with exit code ${RC}"
  exit "$RC"
fi

echo "[INFO] DONE. Results under: ${RUN_S3}"