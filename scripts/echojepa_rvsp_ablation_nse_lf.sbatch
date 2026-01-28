#!/bin/bash
#SBATCH -p partition-1
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH -t 7-12:00:00
#SBATCH -J vjepa_train
#SBATCH -o /tmp/vjepa_rvsp_ablation-%j.out
#SBATCH -e /tmp/vjepa_rvsp_ablation-%j.err

set -euxo pipefail

export AWS_DEFAULT_REGION="us-west-2"
export AWS_REGION="us-west-2"

ART_BUCKET="sagemaker-hyperpod-lifecycle-495467399120-usw2"
ART_PREFIX="vjepa2-artifacts"
S3_BASE="s3://${ART_BUCKET}/${ART_PREFIX}"

# --- Artifacts ---
ENV_S3="${S3_BASE}/env/vjepa2-312-pack-env.tar"
CODE_S3="${S3_BASE}/code/vjepa2-code.tar"
BOTO3_WHL_S3="${S3_BASE}/wheels/boto3-wheelhouse.tar"

# --- Training Config ---
CFG_REL="configs/eval/vitg-384/rvsp/ablations/echojepa_nse_lf.yaml"
CFG_S3="${S3_BASE}/${CFG_REL}"

# --- Data (override-friendly) ---
TRAIN_CSV_REL="${TRAIN_CSV_REL:-data/csv/rvsp_train.csv}"
VAL_CSV_REL="${VAL_CSV_REL:-data/csv/rvsp_val.csv}"

TRAIN_CSV_S3="${TRAIN_CSV_S3:-${S3_BASE}/${TRAIN_CSV_REL}}"
VAL_CSV_S3="${VAL_CSV_S3:-${S3_BASE}/${VAL_CSV_REL}}"

# --- Pretrained Backbone ---
ENC_CKPT_REL="checkpoints/anneal/keep/pt-280-an81.pt"
ENC_CKPT_S3="${S3_BASE}/${ENC_CKPT_REL}"

RUN_S3="${S3_BASE}/runs/vjepa_rvsp_ablation_nse_lf_${SLURM_JOB_ID}"

# --- Trap to upload SLURM logs on exit ---
upload_logs() {
  echo "[INFO] Uploading logs..."
  aws s3 cp "/tmp/vjepa_rvsp_train-${SLURM_JOB_ID}.out" "${RUN_S3}/logs/master.out" --region us-west-2 2>/dev/null || true
  aws s3 cp "/tmp/vjepa_rvsp_train-${SLURM_JOB_ID}.err" "${RUN_S3}/logs/master.err" --region us-west-2 2>/dev/null || true
}
trap 'rc=$?; upload_logs; exit $rc' EXIT

echo "[INFO] host=$(hostname) job=${SLURM_JOB_ID}"
command -v aws
aws --version
nvidia-smi -L || true

# --- CREDENTIAL SAFETY CHECK (Fixes 400 Bad Request) ---
rm -rf ~/.aws
mkdir -p ~/.aws
aws configure set default.s3.signature_version s3v4
aws configure set default.region us-west-2
echo "[INFO] Verifying Identity..."
aws sts get-caller-identity
# -------------------------------------------------------

SCRATCH="/opt/dlami/nvme"
if [ ! -d "$SCRATCH" ]; then SCRATCH="/tmp"; fi
WORKDIR="${SCRATCH}/vjepa2_train_${SLURM_JOB_ID}"
mkdir -p "$WORKDIR"
cd "$WORKDIR"
echo "[INFO] WORKDIR=$WORKDIR"

# --- Setup Environment ---
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

# --- Setup Code ---
aws s3 cp "$CODE_S3" code.tar --region us-west-2
tar -xf code.tar
REPO_DIR=$(find . -maxdepth 10 -type f -name run_inf.sh -print -quit | xargs -r dirname)
cd "${REPO_DIR}"

# Ensure config exists
if [ ! -f "$CFG_REL" ]; then
  echo "[WARN] ${CFG_REL} not found in code.tar; trying S3: ${CFG_S3}"
  mkdir -p "$(dirname "$CFG_REL")"
  aws s3 cp "$CFG_S3" "$CFG_REL" --region us-west-2
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
mkdir -p "$(dirname "$TRAIN_CSV_REL")"
aws s3 cp "$TRAIN_CSV_S3" "$TRAIN_CSV_REL" --region us-west-2
aws s3 cp "$VAL_CSV_S3" "$VAL_CSV_REL" --region us-west-2

mkdir -p "$(dirname "$ENC_CKPT_REL")"
aws s3 cp "$ENC_CKPT_S3" "$ENC_CKPT_REL" --region us-west-2

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
    echo "[WARN] Could not parse 'folder', defaulting to 'output'"
    CFG_FOLDER="output"
fi
mkdir -p "$CFG_FOLDER"

# --- Run Training with Periodic Sync ---

# 1. Detect GPUs
GPU_COUNT=$(nvidia-smi -L | wc -l)
DEVICES=()
for ((i=0; i<GPU_COUNT; i++)); do DEVICES+=("cuda:${i}"); done
echo "[INFO] Detected ${GPU_COUNT} GPUs: ${DEVICES[*]}"

RUNLOG="rvsp_train_${SLURM_JOB_ID}.log"

# 2. Define Sync Function
sync_loop() {
    echo "[bg-sync] Starting periodic sync to ${RUN_S3}/training_folder/"
    while true; do
        sleep 600 # Sync every 10 minutes (600s)
        # We suppress output to avoid cluttering the main log
        aws s3 sync "$CFG_FOLDER" "${RUN_S3}/training_folder/" \
            --exclude "*.pth" --include "*.pt" --include "*.csv" --include "*.log" --quiet --region us-west-2 || true
    done
}

# 3. Start Sync in Background
sync_loop &
SYNC_PID=$!
echo "[INFO] Background sync process started with PID: $SYNC_PID"

set +e # Allow script to continue if python fails, so we can upload logs/final state
# 4. Run Python Training
python -m evals.main \
  --fname "$CFG_REL" \
  --devices "${DEVICES[@]}" \
  2>&1 | tee "$RUNLOG"
RC=${PIPESTATUS[0]}
set -e # Re-enable error checking

# 5. Stop Sync Loop
kill "$SYNC_PID" || true
wait "$SYNC_PID" 2>/dev/null || true

# --- Final Result Upload ---
echo "[INFO] Uploading final logs and artifacts..."

aws s3 cp "$RUNLOG" "${RUN_S3}/logs/${RUNLOG}" --region us-west-2 || true
aws s3 cp "$CFG_REL" "${RUN_S3}/config/rvsp_regression.yaml" --region us-west-2 || true

# Final Sync of Output Folder
aws s3 sync "$CFG_FOLDER" "${RUN_S3}/training_folder/" \
  --exclude "*.pth" --include "*.pt" --include "*.csv" --include "*.log" --region us-west-2 || true

if [ "$RC" -ne 0 ]; then
  echo "[ERROR] Training failed with exit code ${RC}"
  exit "$RC"
fi

echo "[INFO] DONE. Results under: ${RUN_S3}"