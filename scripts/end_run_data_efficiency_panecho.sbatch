#!/bin/bash
#SBATCH -p partition-1
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 4-12:00:00
#SBATCH -J panecho_eff
#SBATCH -o /tmp/panecho_lveff-%j.out
#SBATCH -e /tmp/panecho_lveff-%j.err

set -euxo pipefail

# ==============================================================================
# 0. CREDENTIAL & REGION SETUP
# ==============================================================================
export AWS_DEFAULT_REGION="us-west-2"
export AWS_REGION="us-west-2"
export AWS_SDK_LOAD_CONFIG=1

# Clean slate for credentials
rm -rf ~/.aws
mkdir -p ~/.aws
aws configure set default.s3.signature_version s3v4
aws configure set default.region us-west-2

# Verify identity (debug step)
aws sts get-caller-identity || echo "[WARN] Instance profile identity check failed."

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================

# Bucket & Prefix
ART_BUCKET="sagemaker-hyperpod-lifecycle-495467399120-usw2"
ART_PREFIX="vjepa2-artifacts"
S3_BASE="s3://${ART_BUCKET}/${ART_PREFIX}"

# Artifacts
ENV_S3="${S3_BASE}/env/vjepa2-312-pack-env.tar"
CODE_S3="${S3_BASE}/code/vjepa2-code.tar"
BOTO3_WHL_S3="${S3_BASE}/wheels/boto3-wheelhouse.tar"

# Data CSVs
TRAIN_FULL_NAME="echonet_dynamic_train_s3.csv"
VAL_NAME="echonet_dynamic_val_s3.csv"
TRAIN_FULL_S3="${S3_BASE}/data/csv/${TRAIN_FULL_NAME}"
VAL_S3="${S3_BASE}/data/csv/${VAL_NAME}"

# PanEcho Weights 
# NOTE: Ensure this filename matches what is in your bucket!
PANECHO_CKPT_S3="${S3_BASE}/checkpoints/panecho_checkpoint.pth"
PANECHO_CKPT_LOCAL="/opt/dlami/nvme/checkpoints/panecho_checkpoint.pth"

# Base Config
BASE_CFG_REL="configs/eval/vitg-384/lvef/end_panecho_lvef.yaml"
BASE_CFG_S3="${S3_BASE}/${BASE_CFG_REL}"

# Video Dataset Location
VIDEO_S3_PREFIX="${S3_BASE}/data/EchoNet-Dynamic/Videos"

# Output S3 Location
JOB_S3="${S3_BASE}/runs/efficiency_exp_panecho_${SLURM_JOB_ID}"

# ==============================================================================
# 2. SETUP & HYDRATION
# ==============================================================================

# Error Trap & Log Upload
upload_logs() {
  echo "[INFO] Uploading master logs..."
  aws s3 cp "/tmp/panecho_lveff-${SLURM_JOB_ID}.out" "${JOB_S3}/logs/master.out" --region us-west-2 2>/dev/null || true
  aws s3 cp "/tmp/panecho_lveff-${SLURM_JOB_ID}.err" "${JOB_S3}/logs/master.err" --region us-west-2 2>/dev/null || true

  if [ -n "${REFRESH_PID:-}" ]; then
    echo "Killing Background Service (PID $REFRESH_PID)"
    kill "$REFRESH_PID" || true
  fi
}
trap 'rc=$?; upload_logs; exit $rc' EXIT

echo "[INFO] host=$(hostname) job=${SLURM_JOB_ID}"
nvidia-smi -L || true

# Workspace Setup (Prefer NVMe)
SCRATCH="/opt/dlami/nvme"
if [ ! -d "$SCRATCH" ]; then SCRATCH="/tmp"; fi
WORKDIR="${SCRATCH}/panecho_eff_${SLURM_JOB_ID}"
mkdir -p "$WORKDIR"
cd "$WORKDIR"
echo "[INFO] WORKDIR=$WORKDIR"

# Environment
aws s3 cp "$ENV_S3" env.tar --region us-west-2
mkdir -p env
tar -xf env.tar -C env
set +u
source env/bin/activate
set -u
env/bin/conda-unpack || true

# Boto3
aws s3 cp "$BOTO3_WHL_S3" boto3-wheelhouse.tar --region us-west-2
mkdir -p wheelhouse
tar -xf boto3-wheelhouse.tar -C wheelhouse
python -m pip install --no-index --find-links wheelhouse boto3

# Code Download
aws s3 cp "$CODE_S3" code.tar --region us-west-2
tar -xf code.tar

# --- ROBUSTNESS FIX: Auto-detect Repo Root ---
# This handles cases where the tarball creates a subdirectory
REPO_DIR=$(find . -maxdepth 4 -type f -name setup.py -print -quit | xargs -r dirname)
if [ -n "$REPO_DIR" ] && [ "$REPO_DIR" != "." ]; then
    echo "[INFO] Changing directory to detected repo root: $REPO_DIR"
    cd "$REPO_DIR"
fi

# --- ROBUSTNESS FIX: Config Fallback ---
if [ ! -f "$BASE_CFG_REL" ]; then
    echo "[WARN] Config not found in tarball. Downloading from S3..."
    mkdir -p "$(dirname "$BASE_CFG_REL")"
    aws s3 cp "$BASE_CFG_S3" "$BASE_CFG_REL" --region us-west-2
fi

# Path Patching + PyTorch 2.6 Fix
python - <<'PY'
from pathlib import Path
import os, re
repo = os.getcwd()
studio = "/home/sagemaker-user/user-default-efs/vjepa2"
efs_pat = re.compile(r"/mnt/custom-file-systems/efs/[^ \n\"']+/vjepa2")

roots = ["src", "configs", "evals"]

for r in roots:
    p = Path(r)
    if not p.exists(): continue
    for f in p.rglob("*"):
        if f.suffix in {".yaml", ".py"} and f.is_file():
            try:
                t = f.read_text()
                
                # 1. Path Fixes
                t2 = t.replace(studio, repo)
                t2 = efs_pat.sub(repo, t2)
                
                # 2. PyTorch 2.6 Fix (Weights Only)
                if 'torch.load' in t2 and 'weights_only' not in t2:
                    t2 = t2.replace('torch.load(checkpoint, map_location="cpu")', 
                                    'torch.load(checkpoint, map_location="cpu", weights_only=False)')
                    t2 = t2.replace('torch.load(f, map_location=map_location)', 
                                    'torch.load(f, map_location=map_location, weights_only=False)')
                    # Specific to typical wrappers
                    t2 = t2.replace('torch.load(ckpt_path, map_location="cpu")', 
                                    'torch.load(ckpt_path, map_location="cpu", weights_only=False)')

                if t2 != t:
                    f.write_text(t2)
            except: pass
PY

# Install Code
pip install -e . --no-deps --no-build-isolation || true
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# ==============================================================================
# 3. DOWNLOAD DATA & WEIGHTS
# ==============================================================================

# Download CSVs
mkdir -p data/csv
aws s3 cp "$TRAIN_FULL_S3" "data/csv/${TRAIN_FULL_NAME}" --region us-west-2
aws s3 cp "$VAL_S3" "data/csv/${VAL_NAME}" --region us-west-2

# --- PANECHO WEIGHTS ---
mkdir -p "$(dirname "$PANECHO_CKPT_LOCAL")"
echo "[INFO] Downloading PanEcho weights from $PANECHO_CKPT_S3"
aws s3 cp "$PANECHO_CKPT_S3" "$PANECHO_CKPT_LOCAL" --region us-west-2

# ==============================================================================
# 4. VIDEO SYNC & CSV PATCHING
# ==============================================================================
echo "[INFO] Syncing video dataset from S3 to local NVMe..."

LOCAL_VIDEO_DIR="${WORKDIR}/videos"
mkdir -p "$LOCAL_VIDEO_DIR"

aws s3 sync "$VIDEO_S3_PREFIX" "$LOCAL_VIDEO_DIR" --quiet --region us-west-2

echo "[INFO] Patching CSVs to point to local video files..."

ESC_S3_PREFIX=$(printf '%s\n' "$VIDEO_S3_PREFIX" | sed 's:[][\/.^$*]:\\&:g')
ESC_LOCAL_DIR=$(printf '%s\n' "${LOCAL_VIDEO_DIR}" | sed 's:[][\/.^$*]:\\&:g')

sed -i "s|${ESC_S3_PREFIX}|${ESC_LOCAL_DIR}|g" "data/csv/${TRAIN_FULL_NAME}"
sed -i "s|${ESC_S3_PREFIX}|${ESC_LOCAL_DIR}|g" "data/csv/${VAL_NAME}"

# ==============================================================================
# 5. HELPER SCRIPTS
# ==============================================================================

# Stratified Subset Script
cat << 'EOF' > make_stratified_subset.py
import argparse, random, os, math
from collections import defaultdict

def make_stratified_subset(input_path, output_path, percentage, seed=42, min_per_class=1):
    with open(input_path, 'r') as f: lines = f.readlines()
    
    label_to_lines = defaultdict(list)
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 2:
            try:
                val = float(parts[1])
                bin_key = int(val / 5) * 5
                label_to_lines[bin_key].append(line)
            except ValueError:
                label_to_lines["err"].append(line)

    rng = random.Random(seed)
    final_subset = []
    print(f"Stratifying {len(lines)} samples (Regression Mode), Target: {percentage}%, Seed: {seed}")

    for label, class_lines in label_to_lines.items():
        total = len(class_lines)
        target = math.ceil(total * (percentage / 100.0))
        count = max(target, min(min_per_class, total))
        rng.shuffle(class_lines)
        final_subset.extend(class_lines[:count])

    rng.shuffle(final_subset)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f: f.writelines(final_subset)
    print(f"Saved {len(final_subset)} samples to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--percent", type=float, required=True)
    parser.add_argument("--min", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    make_stratified_subset(args.input, args.out, args.percent, args.seed, args.min)
EOF

# Config Generator (Injects PanEcho Weights)
cat << 'EOF' > generate_run_config.py
import yaml, argparse, os
parser = argparse.ArgumentParser()
parser.add_argument("--base", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--tag", required=True)
parser.add_argument("--train_csv", required=True)
parser.add_argument("--val_csv", required=True)
parser.add_argument("--panecho_ckpt", required=True)
args = parser.parse_args()

with open(args.base) as f:
    cfg = yaml.safe_load(f)

cfg['tag'] = args.tag
cwd = os.getcwd()

# --- PATCH PATHS ---
cfg['experiment']['data']['dataset_train'] = os.path.join(cwd, args.train_csv)
cfg['experiment']['data']['dataset_val'] = os.path.join(cwd, args.val_csv)

# INJECT PANECHO WEIGHTS INTO WRAPPER_KWARGS
if 'wrapper_kwargs' not in cfg['model_kwargs']:
    cfg['model_kwargs']['wrapper_kwargs'] = {}

cfg['model_kwargs']['wrapper_kwargs']['checkpoint_path'] = args.panecho_ckpt
# -------------------

if 'folder' in cfg:
    cfg['folder'] = os.path.join(cwd, "results", "regression")

with open(args.out, 'w') as f:
    yaml.dump(cfg, f)
print(f"Generated config: {args.out}")
EOF

# ==============================================================================
# 6. BACKGROUND SERVICE
# ==============================================================================
start_background_services() {
    echo "[INFO] Starting Background Service (Auth + Sync)..."
    
    # Ensure local aws dir exists for the loop
    mkdir -p ~/.aws

    while true; do
        # Refresh IMDSv2 Token
        TOKEN=$(curl -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600" -s)
        
        if [ -n "$TOKEN" ]; then
            ROLE_NAME=$(curl -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/iam/security-credentials/ -s)
            CREDS_JSON=$(curl -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/iam/security-credentials/$ROLE_NAME -s)

            KEY=$(echo "$CREDS_JSON" | python3 -c "import sys, json; print(json.load(sys.stdin)['AccessKeyId'])")
            SECRET=$(echo "$CREDS_JSON" | python3 -c "import sys, json; print(json.load(sys.stdin)['SecretAccessKey'])")
            SESSION=$(echo "$CREDS_JSON" | python3 -c "import sys, json; print(json.load(sys.stdin)['Token'])")

            cat <<FILE > ~/.aws/credentials.tmp
[default]
aws_access_key_id = $KEY
aws_secret_access_key = $SECRET
aws_session_token = $SESSION
FILE
            mv ~/.aws/credentials.tmp ~/.aws/credentials
        fi

        # Sync results if directory exists
        if [ -d "results/regression/video_classification_frozen" ]; then
             aws s3 sync "results/regression/video_classification_frozen" "${JOB_S3}" \
                --exclude "*.pth" --include "*.pt" --include "*.csv" --include "*.log" \
                --no-progress --region us-west-2 > /dev/null 2>&1 || true
        fi

        sleep 900
    done
}

start_background_services &
REFRESH_PID=$!

# ==============================================================================
# 7. MAIN TRAINING LOOP
# ==============================================================================
GPU_COUNT=$(nvidia-smi -L | wc -l)
DEVICES=()
for ((i=0; i<GPU_COUNT; i++)); do DEVICES+=("cuda:${i}"); done

run_training() {
    local PCT_NAME=$1
    local SEED=$2
    local CSV_FILE=$3

    if [ "$SEED" == "none" ]; then
        local TAG="panecho_lvef_${PCT_NAME}"
    else
        local TAG="panecho_lvef_${PCT_NAME}_s${SEED}"
    fi

    echo "------------------------------------------------------------------"
    echo "STARTING RUN: $TAG"
    echo "DATASET: $CSV_FILE"
    echo "------------------------------------------------------------------"

    local RUN_CFG="configs/run_${TAG}.yaml"
    
    # Generate Run Config
    python generate_run_config.py \
        --base "$BASE_CFG_REL" \
        --out "$RUN_CFG" \
        --tag "$TAG" \
        --train_csv "$CSV_FILE" \
        --val_csv "data/csv/${VAL_NAME}" \
        --panecho_ckpt "$PANECHO_CKPT_LOCAL"

    local LOG_FILE="logs_${TAG}.txt"

    set +e
    python -m evals.main \
      --fname "$RUN_CFG" \
      --devices "${DEVICES[@]}" \
      2>&1 | tee "$LOG_FILE"
    local RC=$?
    set -e

    echo "[INFO] Finalizing sync for $TAG..."
    local OUT_DIR="results/regression/video_classification_frozen/${TAG}"

    aws s3 sync "$OUT_DIR" "${JOB_S3}/${TAG}" \
        --exclude "*.pth" --include "*.pt" --include "*.csv" --include "*.log" \
        --region us-west-2 || true

    aws s3 cp "$LOG_FILE" "${JOB_S3}/logs/${LOG_FILE}" --region us-west-2 || true

    if [ $RC -ne 0 ]; then
        echo "[ERROR] Run $TAG failed with exit code $RC"
    fi
}

# --- 1% Experiments (3 Seeds) ---
for s in 1 2 3; do
    CSV="data/csv/train_1pct_s${s}.csv"
    python make_stratified_subset.py \
        --input "data/csv/${TRAIN_FULL_NAME}" \
        --out "$CSV" \
        --percent 1.0 \
        --seed "$s" \
        --min 3
    run_training "1pct" "$s" "$CSV"
done

# --- 10% Experiments (3 Seeds) ---
for s in 1 2 3; do
    CSV="data/csv/train_10pct_s${s}.csv"
    python make_stratified_subset.py \
        --input "data/csv/${TRAIN_FULL_NAME}" \
        --out "$CSV" \
        --percent 10.0 \
        --seed "$s" \
        --min 3
    run_training "10pct" "$s" "$CSV"
done

# --- 100% Experiment (1 Run) ---
run_training "100pct" "none" "data/csv/${TRAIN_FULL_NAME}"

echo "[INFO] ALL EXPERIMENTS COMPLETED."