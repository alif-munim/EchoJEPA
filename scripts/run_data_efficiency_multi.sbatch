#!/bin/bash
#SBATCH -p partition-1
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH -t 12-12:00:00
#SBATCH -J vjepa_efficiency
#SBATCH -o /tmp/vjepa_eff-%j.out
#SBATCH -e /tmp/vjepa_eff-%j.err

set -euxo pipefail

# ==============================================================================
# CONFIGURATION
# ==============================================================================
export AWS_DEFAULT_REGION="us-west-2"

# Bucket & Prefix
ART_BUCKET="sagemaker-hyperpod-lifecycle-495467399120-usw2"
ART_PREFIX="vjepa2-artifacts"
S3_BASE="s3://${ART_BUCKET}/${ART_PREFIX}"

# Artifacts
ENV_S3="${S3_BASE}/env/vjepa2-312-pack-env.tar"
CODE_S3="${S3_BASE}/code/vjepa2-code.tar"
BOTO3_WHL_S3="${S3_BASE}/wheels/boto3-wheelhouse.tar"

# Base Config (We will modify this on the fly)
BASE_CFG_REL="configs/eval/vitg-384/view/echojepa_view_classification_224px_multi.yaml"
BASE_CFG_S3="${S3_BASE}/${BASE_CFG_REL}"

# Data
TRAIN_FULL_NAME="uhn_views_22k_train.csv"
VAL_NAME="uhn_views_22k_val.csv"
TRAIN_FULL_S3="${S3_BASE}/data/csv/${TRAIN_FULL_NAME}"
VAL_S3="${S3_BASE}/data/csv/${VAL_NAME}"

# Pretrained Backbone
ENC_CKPT_REL="checkpoints/anneal/keep/pt-280-an81.pt"
ENC_CKPT_S3="${S3_BASE}/${ENC_CKPT_REL}"

# Output S3 Location for this specific job
JOB_S3="${S3_BASE}/runs/efficiency_exp_${SLURM_JOB_ID}"

# ==============================================================================
# SETUP & HYDRATION
# ==============================================================================

# 1. Error Trap
upload_logs() {
  echo "[INFO] Uploading master logs..."
  aws s3 cp "/tmp/vjepa_eff-${SLURM_JOB_ID}.out" "${JOB_S3}/logs/master.out" 2>/dev/null || true
  aws s3 cp "/tmp/vjepa_eff-${SLURM_JOB_ID}.err" "${JOB_S3}/logs/master.err" 2>/dev/null || true
  
  # Kill the background refresher if it exists
  if [ -n "${REFRESH_PID:-}" ]; then
    echo "Killing Background Service (PID $REFRESH_PID)"
    kill "$REFRESH_PID" || true
  fi
}
trap 'rc=$?; upload_logs; exit $rc' EXIT

echo "[INFO] host=$(hostname) job=${SLURM_JOB_ID}"
nvidia-smi -L || true

# 2. Workspace Setup
SCRATCH="/opt/dlami/nvme"
if [ ! -d "$SCRATCH" ]; then SCRATCH="/tmp"; fi
WORKDIR="${SCRATCH}/vjepa_eff_${SLURM_JOB_ID}"
mkdir -p "$WORKDIR"
cd "$WORKDIR"
echo "[INFO] WORKDIR=$WORKDIR"

# 3. Environment & Boto3
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

# 4. Code Download
aws s3 cp "$CODE_S3" code.tar
tar -xf code.tar

# 5. Config Download
if [ ! -f "$BASE_CFG_REL" ]; then
  mkdir -p "$(dirname "$BASE_CFG_REL")"
  aws s3 cp "$BASE_CFG_S3" "$BASE_CFG_REL"
fi

# 6. Path Patching (Studio -> Cluster)
python - <<'PY'
from pathlib import Path
import os, re
repo = os.getcwd()
studio = "/home/sagemaker-user/user-default-efs/vjepa2"
efs_pat = re.compile(r"/mnt/custom-file-systems/efs/[^ \n\"']+/vjepa2")
roots = ["configs", "app", "evals", "src"]
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
                    print(f"Patched {f}")
            except: pass
PY

# 7. Install Code
pip install -e . --no-deps --no-build-isolation || true
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# 8. Download Data & Checkpoint
# 8. Download Data & Checkpoint
mkdir -p data/csv
aws s3 cp "$TRAIN_FULL_S3" "data/csv/${TRAIN_FULL_NAME}"
aws s3 cp "$VAL_S3" "data/csv/${VAL_NAME}"
mkdir -p "$(dirname "$ENC_CKPT_REL")"
aws s3 cp "$ENC_CKPT_S3" "$ENC_CKPT_REL"

# --- NEW: CACHE VIDEOS LOCALLY ---
echo "[INFO] Syncing video dataset to local NVMe..."
LOCAL_VIDEO_DIR="${WORKDIR}/videos"
mkdir -p "$LOCAL_VIDEO_DIR"

# Assuming your videos are all under this common prefix based on your logs:
# s3://echodata25/results/uhn_studies_22k_607_224px/
# We sync that folder to local. 
# NOTE: Adjust the S3 path below if your CSV uses multiple different buckets/prefixes.
VIDEO_S3_PREFIX="s3://echodata25/results/uhn_studies_22k_607_224px/"

# Fast sync using AWS CLI
aws s3 sync "$VIDEO_S3_PREFIX" "$LOCAL_VIDEO_DIR" --quiet

echo "[INFO] Patching CSVs to point to local video files..."
# We replace the S3 prefix in the CSVs with the local directory path
# Escape slashes for sed
ESC_S3_PREFIX=$(printf '%s\n' "$VIDEO_S3_PREFIX" | sed 's:[][\/.^$*]:\\&:g')
ESC_LOCAL_DIR=$(printf '%s\n' "$LOCAL_VIDEO_DIR/" | sed 's:[][\/.^$*]:\\&:g')

sed -i "s|${ESC_S3_PREFIX}|${ESC_LOCAL_DIR}|g" "data/csv/${TRAIN_FULL_NAME}"
sed -i "s|${ESC_S3_PREFIX}|${ESC_LOCAL_DIR}|g" "data/csv/${VAL_NAME}"

echo "[INFO] Data preparation complete."
# ---------------------------------


mkdir -p "$(dirname "$ENC_CKPT_REL")"
aws s3 cp "$ENC_CKPT_S3" "$ENC_CKPT_REL"

# ==============================================================================
# HELPER SCRIPTS
# ==============================================================================

# Ensure make_stratified_subset.py exists
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

# Helper to generate run-specific YAML
cat << 'EOF' > generate_run_config.py
import yaml, argparse, os
parser = argparse.ArgumentParser()
parser.add_argument("--base", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--tag", required=True)
parser.add_argument("--train_csv", required=True)
args = parser.parse_args()

with open(args.base) as f:
    cfg = yaml.safe_load(f)

# Update Tag
cfg['tag'] = args.tag

# Update Dataset
# Note: Path patching already ran, so we assume relative paths or CWD
cwd = os.getcwd()
cfg['experiment']['data']['dataset_train'] = os.path.join(cwd, args.train_csv)

# Ensure folder structure is correct (remove hardcodes if any remain)
if 'folder' in cfg:
    # Point to local scratch 'results' folder
    cfg['folder'] = os.path.join(cwd, "results", "classifier")

with open(args.out, 'w') as f:
    yaml.dump(cfg, f)
print(f"Generated config: {args.out}")
EOF

# ==============================================================================
# BACKGROUND SERVICE (CREDENTIALS + SYNC)
# ==============================================================================
# This function runs in the background to:
# 1. Update ~/.aws/credentials (Fixes 400 Bad Request / Expiry)
# 2. Sync results to S3 continuously (Prevents data loss on crash)
start_background_services() {
    mkdir -p ~/.aws
    echo "[INFO] Starting Background Service (Auth + Sync)..."
    
    while true; do
        # --- TASK 1: REFRESH CREDENTIALS ---
        # Fetch fresh keys from the cluster metadata service
        TOKEN=$(curl -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600" -s)
        ROLE_NAME=$(curl -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/iam/security-credentials/ -s)
        CREDS_JSON=$(curl -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/iam/security-credentials/$ROLE_NAME -s)
        
        KEY=$(echo "$CREDS_JSON" | python3 -c "import sys, json; print(json.load(sys.stdin)['AccessKeyId'])")
        SECRET=$(echo "$CREDS_JSON" | python3 -c "import sys, json; print(json.load(sys.stdin)['SecretAccessKey'])")
        SESSION=$(echo "$CREDS_JSON" | python3 -c "import sys, json; print(json.load(sys.stdin)['Token'])")
        
        # Write to atomic temp file then move
        cat <<EOF > ~/.aws/credentials.tmp
[default]
aws_access_key_id = $KEY
aws_secret_access_key = $SECRET
aws_session_token = $SESSION
EOF
        mv ~/.aws/credentials.tmp ~/.aws/credentials
        
        # --- TASK 2: SYNC RESULTS ---
        # Sync the entire 'video_classification_frozen' folder.
        # This structure matches local: results/classifier/video_classification_frozen/{TAG}
        # S3 will be: JOB_S3/{TAG}/...
        if [ -d "results/classifier/video_classification_frozen" ]; then
             aws s3 sync "results/classifier/video_classification_frozen" "${JOB_S3}" \
                --exclude "*.pth" --include "*.pt" --include "*.csv" --include "*.log" \
                --no-progress > /dev/null 2>&1 || true
        fi
        
        # Sleep for 15 minutes (900s) - well within the 1-hour token expiry
        sleep 900
    done
}

# Start background service
start_background_services &
REFRESH_PID=$!

# === ADD THIS WAIT BLOCK ===
echo "[INFO] Waiting for AWS credentials to be initialized..."
while [ ! -f ~/.aws/credentials ]; do
    sleep 1
done
echo "[INFO] Credentials initialized. Starting training."
# ===========================


# ==============================================================================
# MAIN TRAINING LOOP
# ==============================================================================

# Detect GPUs
GPU_COUNT=$(nvidia-smi -L | wc -l)
DEVICES=()
for ((i=0; i<GPU_COUNT; i++)); do DEVICES+=("cuda:${i}"); done

run_training() {
    local PCT_NAME=$1   # e.g., "1pct"
    local SEED=$2       # e.g., 42 (or "none" for 100%)
    local CSV_FILE=$3   # e.g., "data/csv/train_1pct_s42.csv"
    
    # Construct Tag
    if [ "$SEED" == "none" ]; then
        local TAG="uhn22k_multi_${PCT_NAME}"
    else
        local TAG="uhn22k_multi_${PCT_NAME}_s${SEED}"
    fi

    echo "------------------------------------------------------------------"
    echo "STARTING RUN: $TAG"
    echo "DATASET: $CSV_FILE"
    echo "------------------------------------------------------------------"

    # 1. Generate Config
    local RUN_CFG="configs/run_${TAG}.yaml"
    python generate_run_config.py \
        --base "$BASE_CFG_REL" \
        --out "$RUN_CFG" \
        --tag "$TAG" \
        --train_csv "$CSV_FILE"

    # 2. Run Training
    local LOG_FILE="logs_${TAG}.txt"
    
    # We use a subshell or allow failure so one bad run doesn't kill the whole job
    set +e
    python -m evals.main \
      --fname "$RUN_CFG" \
      --devices "${DEVICES[@]}" \
      2>&1 | tee "$LOG_FILE"
    local RC=$?
    set -e

    # 3. Final Sync for this specific run (Just to be sure)
    echo "[INFO] Finalizing sync for $TAG..."
    local OUT_DIR="results/classifier/video_classification_frozen/${TAG}"
    
    aws s3 sync "$OUT_DIR" "${JOB_S3}/${TAG}" \
        --exclude "*.pth" --include "*.pt" --include "*.csv" --include "*.log" || true
    
    aws s3 cp "$LOG_FILE" "${JOB_S3}/logs/${LOG_FILE}" || true

    if [ $RC -ne 0 ]; then
        echo "[ERROR] Run $TAG failed with exit code $RC"
    fi
}

# --- 1. Run 1% Experiments (3 Seeds) ---
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

# --- 2. Run 10% Experiments (3 Seeds) ---
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

# --- 3. Run 100% Experiment (Once) ---
run_training "100pct" "none" "data/csv/${TRAIN_FULL_NAME}"

echo "[INFO] ALL EXPERIMENTS COMPLETED."