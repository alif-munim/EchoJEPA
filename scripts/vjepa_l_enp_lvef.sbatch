#!/bin/bash
#SBATCH -p partition-1
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH -t 4-12:00:00
#SBATCH -J vjepa_enp_vitl_lvef
#SBATCH -o /tmp/vjepa_enp_vitl_lvef-%j.out
#SBATCH -e /tmp/vjepa_enp_vitl_lvef-%j.err

set -euxo pipefail

# ==============================================================================
# 0. CREDENTIAL CLEANUP (Start Fresh)
# ==============================================================================
unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN
rm -rf ~/.aws
mkdir -p ~/.aws
export AWS_DEFAULT_REGION="us-west-2"
export AWS_REGION="us-west-2"
export AWS_SDK_LOAD_CONFIG=1
aws configure set default.region us-west-2
aws configure set default.s3.signature_version s3v4

# === S3 CONFIGURATION ===
ART_BUCKET="sagemaker-hyperpod-lifecycle-495467399120-usw2"
ART_PREFIX="vjepa2-artifacts"
S3_BASE="s3://${ART_BUCKET}/${ART_PREFIX}"

# Artifacts
ENV_S3="${S3_BASE}/env/vjepa2-312-pack-env.tar"
CODE_S3="${S3_BASE}/code/vjepa2-code.tar"
BOTO3_WHL_S3="${S3_BASE}/wheels/boto3-wheelhouse.tar"

# === TRAINING CONFIGURATION ===
# 1. New ViT-L Pediatric YAML
CFG_REL="configs/eval/vitl/enp_lvef.yaml"
CFG_S3="${S3_BASE}/${CFG_REL}"

# 2. Data Paths (PEDIATRIC)
# Source: S3 location where you synced the CSVs
TRAIN_CSV_S3="${S3_BASE}/csv_pediatric/echonet_pediatric_train_s3.csv"
VAL_CSV_S3="${S3_BASE}/csv_pediatric/echonet_pediatric_val_s3.csv"

# Dest: Local paths expected by the YAML
TRAIN_CSV_REL="data/csv/echonet_pediatric_train.csv"
VAL_CSV_REL="data/csv/echonet_pediatric_val.csv"

# 3. ViT-L Backbone Checkpoint (Matches YAML)
ENC_CKPT_REL="checkpoints/anneal/keep/vitl-pt-210-an25.pt"
ENC_CKPT_S3="${S3_BASE}/${ENC_CKPT_REL}"

# Run Output Location in S3
RUN_S3="${S3_BASE}/runs/vjepa_enp_vitl_lvef_${SLURM_JOB_ID}"

# === LOGGING TRAP ===
# Updated to match the #SBATCH -o/-e paths
upload_logs() {
  echo "[INFO] Trapped EXIT: Uploading logs..."
  aws s3 cp "/tmp/vjepa_enp_vitl_lvef-${SLURM_JOB_ID}.out" "${RUN_S3}/logs/master.out" 2>/dev/null || true
  aws s3 cp "/tmp/vjepa_enp_vitl_lvef-${SLURM_JOB_ID}.err" "${RUN_S3}/logs/master.err" 2>/dev/null || true
}
trap 'rc=$?; upload_logs; exit $rc' EXIT

# === HOST SETUP ===
echo "[INFO] host=$(hostname) job=${SLURM_JOB_ID}"
nvidia-smi -L || true

# Setup Scratch on NVMe
SCRATCH="/opt/dlami/nvme"
if [ ! -d "$SCRATCH" ]; then SCRATCH="/tmp"; fi
WORKDIR="${SCRATCH}/vjepa_enp_vitl_lvef_${SLURM_JOB_ID}"
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

# Ensure config exists
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
                if t2 != t: f.write_text(t2)
            except: pass
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

# === PARSE CONFIG (FOLDER & TAG) ===
export CFG_REL 
eval $(python - <<'PY'
import re, os
from pathlib import Path
cfg_rel = os.environ["CFG_REL"]
folder, tag = "", ""
if os.path.exists(cfg_rel):
    t = Path(cfg_rel).read_text()
    m_folder = re.search(r'^\s*folder:\s*(.+)\s*$', t, re.M)
    if m_folder: folder = m_folder.group(1).strip().strip('"').strip("'")
    m_tag = re.search(r'^\s*tag:\s*(.+)\s*$', t, re.M)
    if m_tag: tag = m_tag.group(1).strip().strip('"').strip("'")
print(f"export CFG_FOLDER='{folder}'")
print(f"export CFG_TAG='{tag}'")
PY
)
if [ -z "$CFG_FOLDER" ]; then CFG_FOLDER="output"; fi
mkdir -p "$CFG_FOLDER"

# ==============================================================================
# CRITICAL: REFRESH CREDENTIALS FOR DECORD
# Decord/FFmpeg requires env vars to stream from S3. 
# We fetch fresh ones from the Instance Metadata Service (IMDS).
# ==============================================================================
echo "[INFO] Refreshing credentials from Instance Metadata..."
TOKEN=$(curl -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600" 2>/dev/null)
ROLE_NAME=$(curl -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/iam/security-credentials/ 2>/dev/null)
CREDS_JSON=$(curl -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/iam/security-credentials/$ROLE_NAME 2>/dev/null)

export AWS_ACCESS_KEY_ID=$(echo "$CREDS_JSON" | python3 -c "import sys, json; print(json.load(sys.stdin)['AccessKeyId'])")
export AWS_SECRET_ACCESS_KEY=$(echo "$CREDS_JSON" | python3 -c "import sys, json; print(json.load(sys.stdin)['SecretAccessKey'])")
export AWS_SESSION_TOKEN=$(echo "$CREDS_JSON" | python3 -c "import sys, json; print(json.load(sys.stdin)['Token'])")
# ==============================================================================

# === RUN TRAINING ===
GPU_COUNT=$(nvidia-smi -L | wc -l)
DEVICES=()
for ((i=0; i<GPU_COUNT; i++)); do DEVICES+=("cuda:${i}"); done
echo "[INFO] Detected ${GPU_COUNT} GPUs: ${DEVICES[*]}"

RUNLOG="lvef_train_${SLURM_JOB_ID}.log"

# Background Sync
(
    while true; do
        sleep 600
        aws s3 sync "$CFG_FOLDER" "${RUN_S3}/training_folder/" --exclude "*.pth" --include "*.pt" --include "*.csv" --include "*.log" --quiet --region us-west-2 || true
    done
) &
SYNC_PID=$!

set +e
python -m evals.main \
  --fname "$CFG_REL" \
  --devices "${DEVICES[@]}" \
  2>&1 | tee "$RUNLOG"
RC=${PIPESTATUS[0]}
set -e

kill $SYNC_PID || true
wait $SYNC_PID 2>/dev/null || true

# === FINAL UPLOAD ===
echo "[INFO] Uploading final artifacts..."
aws s3 cp "$RUNLOG" "${RUN_S3}/logs/job_final.log" --region us-west-2 || true
aws s3 cp "$CFG_REL" "${RUN_S3}/config/run_config.yaml" --region us-west-2 || true
aws s3 sync "$CFG_FOLDER" "${RUN_S3}/training_folder/" --exclude "*.pth" --include "*.pt" --include "*.csv" --include "*.log" --region us-west-2 || true

if [ "$RC" -ne 0 ]; then
  echo "[ERROR] Training failed with exit code ${RC}"
  exit "$RC"
fi

echo "[INFO] DONE. Results under: ${RUN_S3}"