#!/bin/bash
#SBATCH -p partition-1
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 4-12:00:00
#SBATCH -J vjepa_lvef_224
#SBATCH -o /tmp/vjepa_224px-lvef-%j.out
#SBATCH -e /tmp/vjepa_224px-lvef-%j.err

set -euxo pipefail

# === CREDENTIAL FIX (CRITICAL) ===
# 1. Unset stale credentials passed from your local shell
unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN

# 2. WIPE STALE CONFIG FILES
rm -rf ~/.aws
mkdir -p ~/.aws

# 3. Force Region configuration
export AWS_DEFAULT_REGION="us-west-2"
export AWS_REGION="us-west-2"
export AWS_SDK_LOAD_CONFIG=1

# 4. Explicitly configure CLI to be safe
aws configure set default.region us-west-2
aws configure set default.s3.signature_version s3v4
# =================================

# === S3 CONFIGURATION ===
ART_BUCKET="sagemaker-hyperpod-lifecycle-495467399120-usw2"
ART_PREFIX="vjepa2-artifacts"
S3_BASE="s3://${ART_BUCKET}/${ART_PREFIX}"

# Artifacts
ENV_S3="${S3_BASE}/env/vjepa2-312-pack-env.tar"
CODE_S3="${S3_BASE}/code/vjepa2-code.tar"
BOTO3_WHL_S3="${S3_BASE}/wheels/boto3-wheelhouse.tar"

# === TRAINING CONFIGURATION ===
# 1. Point to your new Regression YAML
CFG_REL="configs/eval/vitg-384/lvef/vjepa_lvef_224px.yaml"
CFG_S3="${S3_BASE}/${CFG_REL}"

# 2. Update Data Paths to match your new CSV filenames
TRAIN_CSV_REL="data/csv/a4c_b_lvef_train_224px.csv"
VAL_CSV_REL="data/csv/a4c_b_lvef_val_224px_256.csv"

TRAIN_CSV_S3="${S3_BASE}/${TRAIN_CSV_REL}"
VAL_CSV_S3="${S3_BASE}/${VAL_CSV_REL}"

# 3. Backbone Checkpoint
ENC_CKPT_REL="checkpoints/anneal/keep/pt-280-an81.pt"
ENC_CKPT_S3="${S3_BASE}/${ENC_CKPT_REL}"

# Run Output Location in S3
RUN_S3="${S3_BASE}/runs/vjepa_lvef_224px_${SLURM_JOB_ID}"

# === LOGGING & TRAPS ===
upload_logs() {
  # Upload master logs (stdout/stderr)
  aws s3 cp "/tmp/vjepa_224px-lvef-${SLURM_JOB_ID}.out" "${RUN_S3}/logs/job.out" --region us-west-2 2>/dev/null || true
  aws s3 cp "/tmp/vjepa_224px-lvef-${SLURM_JOB_ID}.err" "${RUN_S3}/logs/job.err" --region us-west-2 2>/dev/null || true
}
trap 'rc=$?; upload_logs; exit $rc' EXIT

# === HOST SETUP ===
echo "[INFO] host=$(hostname) job=${SLURM_JOB_ID}"
nvidia-smi -L || true

SCRATCH="/opt/dlami/nvme"
if [ ! -d "$SCRATCH" ]; then SCRATCH="/tmp"; fi
WORKDIR="${SCRATCH}/vjepa_lvef_${SLURM_JOB_ID}"
mkdir -p "$WORKDIR"
cd "$WORKDIR"
echo "[INFO] WORKDIR=$WORKDIR"

# === SETUP ENVIRONMENT ===
aws s3 cp "$ENV_S3" env.tar --region us-west-2
mkdir -p env
tar -xf env.tar -C env
set +u
source env/bin/activate
set -u
env/bin/conda-unpack || true

# Install boto3
aws s3 cp "$BOTO3_WHL_S3" boto3-wheelhouse.tar --region us-west-2
mkdir -p wheelhouse
tar -xf boto3-wheelhouse.tar -C wheelhouse
python -m pip install --no-index --find-links wheelhouse boto3

# === SETUP CODE ===
aws s3 cp "$CODE_S3" code.tar --region us-west-2
tar -xf code.tar
REPO_DIR=$(find . -maxdepth 10 -type f -name run_inf.sh -print -quit | xargs -r dirname)
cd "${REPO_DIR}"

# Ensure config exists (fetch from S3 if not in code.tar)
if [ ! -f "$CFG_REL" ]; then
  echo "[WARN] ${CFG_REL} not found in code.tar; trying S3: ${CFG_S3}"
  mkdir -p "$(dirname "$CFG_REL")"
  aws s3 cp "$CFG_S3" "$CFG_REL" --region us-west-2
fi

# === PATCH PATHS ===
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
            except Exception as e:
                print(f"Skipping {f}: {e}")
PY

# Make code importable
pip install -e . --no-deps --no-build-isolation || true
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# === DOWNLOAD INPUTS ===
echo "[INFO] Downloading Data & Checkpoints..."

# 1. CSVs
mkdir -p "$(dirname "$TRAIN_CSV_REL")"
aws s3 cp "$TRAIN_CSV_S3" "$TRAIN_CSV_REL" --region us-west-2
aws s3 cp "$VAL_CSV_S3" "$VAL_CSV_REL" --region us-west-2

# 2. Backbone Checkpoint
mkdir -p "$(dirname "$ENC_CKPT_REL")"
aws s3 cp "$ENC_CKPT_S3" "$ENC_CKPT_REL" --region us-west-2

# === PARSE OUTPUT FOLDER ===
# FIX: Export variable so Python can see it
export CFG_REL 

CFG_FOLDER="$(python - <<'PY'
import re
import os
from pathlib import Path

# Use the environment variable for flexibility
cfg_rel = os.environ["CFG_REL"]
if os.path.exists(cfg_rel):
    t = Path(cfg_rel).read_text()
    m = re.search(r'^\s*folder:\s*(.+)\s*$', t, re.M)
    print(m.group(1).strip().strip('"').strip("'") if m else "")
else:
    print("")
PY
)"

if [ -z "$CFG_FOLDER" ]; then
    echo "[ERROR] Could not parse 'folder' from config $CFG_REL"
    exit 5
fi

mkdir -p "$CFG_FOLDER"
echo "[INFO] Training will output to: $CFG_FOLDER"

# === RUN TRAINING ===
GPU_COUNT=$(nvidia-smi -L | wc -l)
DEVICES=()
for ((i=0; i<GPU_COUNT; i++)); do DEVICES+=("cuda:${i}"); done
echo "[INFO] Detected ${GPU_COUNT} GPUs: ${DEVICES[*]}"

RUNLOG="lvef_train_${SLURM_JOB_ID}.log"

# === BACKGROUND SYNC (INTERMITTENT) ===
echo "[INFO] Starting background sync process (Every 10 mins)..."
(
    while true; do
        sleep 600
        # Sync Output Folder (Checkpoints, etc.)
        aws s3 sync "$CFG_FOLDER" "${RUN_S3}/training_folder/" \
            --exclude "*.pth" \
            --include "*.pt" \
            --include "*.csv" \
            --include "*.log" \
            --include "*.json" \
            --region us-west-2 > /dev/null 2>&1 || true
            
        # Sync Training Log
        if [ -f "$RUNLOG" ]; then
            aws s3 cp "$RUNLOG" "${RUN_S3}/logs/${RUNLOG}" --region us-west-2 > /dev/null 2>&1 || true
        fi
    done
) &
SYNC_PID=$!
# ======================================

set +e
python -m evals.main \
  --fname "$CFG_REL" \
  --devices "${DEVICES[@]}" \
  2>&1 | tee "$RUNLOG"
RC=${PIPESTATUS[0]}
set -e

# Stop background sync
kill $SYNC_PID 2>/dev/null || true
wait $SYNC_PID 2>/dev/null || true

# === FINAL UPLOAD ===
echo "[INFO] Uploading final artifacts..."

aws s3 cp "$RUNLOG" "${RUN_S3}/logs/${RUNLOG}" --region us-west-2 || true
aws s3 cp "$CFG_REL" "${RUN_S3}/config/run_config.yaml" --region us-west-2 || true

aws s3 sync "$CFG_FOLDER" "${RUN_S3}/training_folder/" \
  --exclude "*.pth" \
  --include "*.pt" \
  --include "*.csv" \
  --include "*.log" \
  --include "*.json" \
  --region us-west-2 || true

if [ "$RC" -ne 0 ]; then
  echo "[ERROR] Training failed with exit code ${RC}"
  exit "$RC"
fi

echo "[INFO] DONE. Results uploaded to: ${RUN_S3}"