#!/bin/bash
#SBATCH -p ml.p5e.48xlarge
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 4-12:00:00
#SBATCH -J echoprime_eff
#SBATCH -o /tmp/echoprime_eff-%j.out
#SBATCH -e /tmp/echoprime_eff-%j.err

set -euxo pipefail

# ==============================================================================
# 0. CREDENTIAL FIX (CRITICAL)
# ==============================================================================
unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN
rm -rf ~/.aws
mkdir -p ~/.aws
export AWS_DEFAULT_REGION="us-west-2"
export AWS_REGION="us-west-2"
export AWS_SDK_LOAD_CONFIG=1
aws configure set default.region us-west-2
aws configure set default.s3.signature_version s3v4

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
TRAIN_FULL_NAME="uhn_views_22k_train.csv"
VAL_NAME="uhn_views_22k_val.csv"
TRAIN_FULL_S3="${S3_BASE}/data/csv/${TRAIN_FULL_NAME}"
VAL_S3="${S3_BASE}/data/csv/${VAL_NAME}"

# Pretrained Backbone (VideoMAE)
# S3 Source: The specific run checkpoint you identified
# ENC_CKPT_S3="${S3_BASE}/runs/videomae_115/training_folder/checkpoint-163.pth"
# Local Dest: Where we save it on disk
# ENC_CKPT_LOCAL="/opt/dlami/nvme/checkpoints/videomae_vitl_pretrain.pth"

# Base Config (Relative path inside code.tar)
BASE_CFG_REL="configs/eval/vitg-384/view/echoprime_view_classification.yaml"

# Video Dataset Location (S3 -> Local NVMe)
VIDEO_S3_PREFIX="s3://echodata25/results/uhn_studies_22k_607_224px/"

# Output S3 Location for this specific job
JOB_S3="${S3_BASE}/runs/efficiency_exp_echoprime_${SLURM_JOB_ID}"

# ==============================================================================
# 2. SETUP & HYDRATION
# ==============================================================================

# Error Trap & Log Upload
upload_logs() {
  echo "[INFO] Uploading master logs..."
  aws s3 cp "/tmp/echoprime_eff-${SLURM_JOB_ID}.out" "${JOB_S3}/logs/master.out" 2>/dev/null || true
  aws s3 cp "/tmp/echoprime_eff-${SLURM_JOB_ID}.err" "${JOB_S3}/logs/master.err" 2>/dev/null || true

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
WORKDIR="${SCRATCH}/echoprime_eff_${SLURM_JOB_ID}"
mkdir -p "$WORKDIR"
cd "$WORKDIR"
echo "[INFO] WORKDIR=$WORKDIR"

# Environment & Boto3
aws s3 cp "$ENV_S3" env.tar
mkdir -p env
tar -xf env.tar -C env
set +u
source env/bin/activate
set -u
env/bin/conda-unpack || true

aws s3 cp "$BOTO3_WHL_S3" boto3-wheelhouse.tar
mkdir -p wheelhouse
tar -xf boto3-wheelhouse.tar -C wheelhouse
python -m pip install --no-index --find-links wheelhouse boto3

# Code Download & Extraction
aws s3 cp "$CODE_S3" code.tar
tar -xf code.tar

# Verify Base Config Exists
if [ ! -f "$BASE_CFG_REL" ]; then
    echo "[ERROR] Base config not found at $BASE_CFG_REL inside code.tar"
    exit 1
fi

# Path Patching (Studio -> Cluster)
# We run this to fix any imports or absolute paths in the src code
python - <<'PY'
from pathlib import Path
import os, re
repo = os.getcwd()
studio = "/home/sagemaker-user/user-default-efs/vjepa2"
efs_pat = re.compile(r"/mnt/custom-file-systems/efs/[^ \n\"']+/vjepa2")
roots = ["src", "configs"]
for r in roots:
    p = Path(r)
    if not p.exists(): continue
    for f in p.rglob("*"):
        if f.suffix in {".yaml", ".py"} and f.is_file():
            try:
                t = f.read_text()
                t2 = t.replace(studio, repo)
                t2 = efs_pat.sub(repo, t2)
                if t2 != t:
                    f.write_text(t2)
            except: pass
PY

# Install Code
pip install -e . --no-deps --no-build-isolation || true
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# ==============================================================================
# 3. DOWNLOAD DATA & CHECKPOINT
# ==============================================================================

# Download CSVs
mkdir -p data/csv
aws s3 cp "$TRAIN_FULL_S3" "data/csv/${TRAIN_FULL_NAME}"
aws s3 cp "$VAL_S3" "data/csv/${VAL_NAME}"

# Download Checkpoint
# mkdir -p "$(dirname "$ENC_CKPT_LOCAL")"
# echo "[INFO] Downloading Checkpoint from $ENC_CKPT_S3"
# aws s3 cp "$ENC_CKPT_S3" "$ENC_CKPT_LOCAL"

# ==============================================================================
# 4. VIDEO SYNC & CSV PATCHING (CRITICAL PERFORMANCE STEP)
# ==============================================================================
echo "[INFO] Syncing video dataset from S3 to local NVMe..."

LOCAL_VIDEO_DIR="${WORKDIR}/videos"
mkdir -p "$LOCAL_VIDEO_DIR"

# Remove --quiet if you want progress output.
aws s3 sync "$VIDEO_S3_PREFIX" "$LOCAL_VIDEO_DIR" --quiet

echo "[INFO] Patching CSVs to point to local video files..."

# Escape paths for sed
ESC_S3_PREFIX=$(printf '%s\n' "$VIDEO_S3_PREFIX" | sed 's:[][\/.^$*]:\\&:g')
ESC_LOCAL_DIR=$(printf '%s\n' "${LOCAL_VIDEO_DIR}/" | sed 's:[][\/.^$*]:\\&:g')

# Update CSVs in place
sed -i "s|${ESC_S3_PREFIX}|${ESC_LOCAL_DIR}|g" "data/csv/${TRAIN_FULL_NAME}"
sed -i "s|${ESC_S3_PREFIX}|${ESC_LOCAL_DIR}|g" "data/csv/${VAL_NAME}"

echo "[INFO] Local video cache ready: $LOCAL_VIDEO_DIR"

# ==============================================================================
# 5. HELPER SCRIPTS
# ==============================================================================

# Script to subset data for 1% and 10% runs
cat << 'EOF' > make_stratified_subset.py
import argparse, random, os, math
from collections import defaultdict

def make_stratified_subset(input_path, output_path, percentage, seed=42, min_per_class=1):
    with open(input_path, 'r') as f: lines = f.readlines()
    label_to_lines = defaultdict(list)
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 2: label_to_lines[parts[1]].append(line)

    rng = random.Random(seed)
    final_subset = []
    print(f"Stratifying {len(lines)} samples, Target: {percentage}%, Seed: {seed}")

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

# Script to inject specific run settings into YAML
cat << 'EOF' > generate_run_config.py
import yaml, argparse, os
parser = argparse.ArgumentParser()
parser.add_argument("--base", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--tag", required=True)
parser.add_argument("--train_csv", required=True)
parser.add_argument("--val_csv", required=True)
parser.add_argument("--ckpt", required=True)
args = parser.parse_args()

with open(args.base) as f:
    cfg = yaml.safe_load(f)

cfg['tag'] = args.tag
cwd = os.getcwd()

# --- PATCH PATHS ---
cfg['experiment']['data']['dataset_train'] = os.path.join(cwd, args.train_csv)
cfg['experiment']['data']['dataset_val'] = os.path.join(cwd, args.val_csv)
cfg['model_kwargs']['checkpoint'] = os.path.join(cwd, args.ckpt)
# -------------------

if 'folder' in cfg:
    cfg['folder'] = os.path.join(cwd, "results", "classifier")

with open(args.out, 'w') as f:
    yaml.dump(cfg, f)
print(f"Generated config: {args.out}")
EOF

# ==============================================================================
# 6. BACKGROUND SERVICE
# ==============================================================================
start_background_services() {
    echo "[INFO] Starting Background Service (Auth + Sync)..."

    while true; do
        # Refresh IMDSv2 Token
        TOKEN=$(curl -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600" -s)
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

        # Sync results if directory exists
        if [ -d "results/classifier/video_classification_frozen" ]; then
             aws s3 sync "results/classifier/video_classification_frozen" "${JOB_S3}" \
                --exclude "*.pth" --include "*.pt" --include "*.csv" --include "*.log" \
                --no-progress > /dev/null 2>&1 || true
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
        local TAG="uhn22k_videomae_${PCT_NAME}"
    else
        local TAG="uhn22k_videomae_${PCT_NAME}_s${SEED}"
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
        --val_csv "data/csv/${VAL_NAME}"

    local LOG_FILE="logs_${TAG}.txt"

    set +e
    python -m evals.main \
      --fname "$RUN_CFG" \
      --devices "${DEVICES[@]}" \
      2>&1 | tee "$LOG_FILE"
    local RC=$?
    set -e

    echo "[INFO] Finalizing sync for $TAG..."
    local OUT_DIR="results/classifier/video_classification_frozen/${TAG}"

    aws s3 sync "$OUT_DIR" "${JOB_S3}/${TAG}" \
        --exclude "*.pth" --include "*.pt" --include "*.csv" --include "*.log" || true

    aws s3 cp "$LOG_FILE" "${JOB_S3}/logs/${LOG_FILE}" || true

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