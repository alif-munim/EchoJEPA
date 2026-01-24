#!/bin/bash
#SBATCH -p partition-1
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH -t 7-12:00:00
#SBATCH -J echoprime_lvef
#SBATCH -o /tmp/echoprime_lvef-%j.out
#SBATCH -e /tmp/echoprime_lvef-%j.err

set -euxo pipefail

# ==============================================================================
# CONFIGURATION
# ==============================================================================
export AWS_DEFAULT_REGION="us-west-2"
export AWS_REGION="us-west-2"

# Bucket & Prefix
ART_BUCKET="sagemaker-hyperpod-lifecycle-495467399120-usw2"
ART_PREFIX="vjepa2-artifacts"
S3_BASE="s3://${ART_BUCKET}/${ART_PREFIX}"

# Artifacts
ENV_S3="${S3_BASE}/env/vjepa2-312-pack-env.tar"
CODE_S3="${S3_BASE}/code/vjepa2-code.tar"
BOTO3_WHL_S3="${S3_BASE}/wheels/boto3-wheelhouse.tar"

# Base Config
BASE_CFG_REL="configs/eval/vitg-384/view/echojepa_view_classification_224px_multi.yaml"
BASE_CFG_S3="${S3_BASE}/${BASE_CFG_REL}"

# Data CSVs (LVEF Regression)
TRAIN_FULL_NAME="a4c_b_lvef_train_224px.csv"
VAL_NAME="a4c_b_lvef_val_224px.csv"
TRAIN_FULL_S3="${S3_BASE}/data/csv/${TRAIN_FULL_NAME}"
VAL_S3="${S3_BASE}/data/csv/${VAL_NAME}"

# Output S3 Location
JOB_S3="${S3_BASE}/runs/echoprime_lvef_${SLURM_JOB_ID}"

# --- FIX: Correct Video Location for LVEF Dataset ---
VIDEO_S3_PREFIX="s3://echodata25/results/uhn-lvef-224/"
# ----------------------------------------------------

# ==============================================================================
# SETUP & HYDRATION
# ==============================================================================

# 1. Error Trap
upload_logs() {
  echo "[INFO] Uploading master logs..."
  aws s3 cp "/tmp/echoprime_lvef-${SLURM_JOB_ID}.out" "${JOB_S3}/logs/master.out" --region us-west-2 2>/dev/null || true
  aws s3 cp "/tmp/echoprime_lvef-${SLURM_JOB_ID}.err" "${JOB_S3}/logs/master.err" --region us-west-2 2>/dev/null || true

  if [ -n "${REFRESH_PID:-}" ]; then
    echo "Killing Background Service (PID $REFRESH_PID)"
    kill "$REFRESH_PID" || true
  fi
}
trap 'rc=$?; upload_logs; exit $rc' EXIT

echo "[INFO] host=$(hostname) job=${SLURM_JOB_ID}"
nvidia-smi -L || true

# --- NEW: CREDENTIAL SAFETY CHECK ---
rm -rf ~/.aws
mkdir -p ~/.aws

# Force S3 Signature Version 4 & Region Config
aws configure set default.s3.signature_version s3v4
aws configure set default.region us-west-2

# Also write a config file for tools that look there instead of credentials
cat <<EOF > ~/.aws/config
[default]
region = us-west-2
s3 =
    signature_version = s3v4
EOF

echo "[INFO] Verifying Identity..."
aws sts get-caller-identity
# ------------------------------------

# 2. Workspace Setup
SCRATCH="/opt/dlami/nvme"
if [ ! -d "$SCRATCH" ]; then SCRATCH="/tmp"; fi
WORKDIR="${SCRATCH}/echoprime_lvef_${SLURM_JOB_ID}"
mkdir -p "$WORKDIR"
cd "$WORKDIR"
echo "[INFO] WORKDIR=$WORKDIR"

# 3. Environment & Boto3
aws s3 cp "$ENV_S3" env.tar --region us-west-2
mkdir -p env
tar -xf env.tar -C env
set +u
source env/bin/activate
set -u
env/bin/conda-unpack || true

aws s3 cp "$BOTO3_WHL_S3" boto3-wheelhouse.tar --region us-west-2
mkdir -p wheelhouse
tar -xf boto3-wheelhouse.tar -C wheelhouse
python -m pip install --no-index --find-links wheelhouse boto3

# --- CRITICAL FIX: EchoPrime Deps ---
echo "[INFO] Installing EchoPrime dependencies..."
pip install opencv-python-headless

# 4. Code Download
aws s3 cp "$CODE_S3" code.tar --region us-west-2
tar -xf code.tar

# 5. Config Download
if [ ! -f "$BASE_CFG_REL" ]; then
  mkdir -p "$(dirname "$BASE_CFG_REL")"
  aws s3 cp "$BASE_CFG_S3" "$BASE_CFG_REL" --region us-west-2
fi

# 6. Path Patching
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

# 8. Download CSVs
mkdir -p data/csv
aws s3 cp "$TRAIN_FULL_S3" "data/csv/${TRAIN_FULL_NAME}" --region us-west-2
aws s3 cp "$VAL_S3" "data/csv/${VAL_NAME}" --region us-west-2

# ==============================================================================
# ECHOPRIME WEIGHTS SETUP
# ==============================================================================
echo "[INFO] Setting up EchoPrime weights..."
EP_DIR="${WORKDIR}/EchoPrime"
mkdir -p "$EP_DIR"

wget -q https://github.com/echonet/EchoPrime/releases/download/v1.0.0/model_data.zip --no-check-certificate -O "${EP_DIR}/model_data.zip"
unzip -q "${EP_DIR}/model_data.zip" "model_data/weights/*" -d "$EP_DIR"

echo "[INFO] EchoPrime weights ready at ${EP_DIR}/model_data/weights"

# ==============================================================================
# DATASET SYNC & CSV PATCH
# ==============================================================================
echo "[INFO] Syncing video dataset from S3 to local NVMe..."

LOCAL_VIDEO_DIR="${WORKDIR}/videos"
mkdir -p "$LOCAL_VIDEO_DIR"

# Using --region here is critical if the public data is in a specific region
aws s3 sync "$VIDEO_S3_PREFIX" "$LOCAL_VIDEO_DIR" --quiet --region us-west-2

echo "[INFO] Patching CSVs to point to local video files..."
# Be extremely careful with special chars in S3 URIs
ESC_S3_PREFIX=$(printf '%s\n' "$VIDEO_S3_PREFIX" | sed 's:[][\/.^$*]:\\&:g')
ESC_LOCAL_DIR=$(printf '%s\n' "${LOCAL_VIDEO_DIR}/" | sed 's:[][\/.^$*]:\\&:g')

sed -i "s|${ESC_S3_PREFIX}|${ESC_LOCAL_DIR}|g" "data/csv/${TRAIN_FULL_NAME}"
sed -i "s|${ESC_S3_PREFIX}|${ESC_LOCAL_DIR}|g" "data/csv/${VAL_NAME}"

echo "[INFO] Patching complete. Verifying one entry:"
head -n 2 "data/csv/${TRAIN_FULL_NAME}"

# ==============================================================================
# CONFIGURATION GENERATOR
# ==============================================================================

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

cfg['tag'] = args.tag

# --- FORCE ECHOPRIME CONFIGURATION ---
cfg['model_kwargs']['module_name'] = 'evals.video_classification_frozen.modelcustom.echo_prime_encoder'
cfg['model_kwargs']['checkpoint'] = None
cfg['model_kwargs']['pretrain_kwargs'] = {}

cwd = os.getcwd()
cfg['model_kwargs']['wrapper_kwargs'] = {
    'echo_prime_root': os.path.join(cwd, 'EchoPrime'),
    'force_fp32': True,
    'bin_size': 50
}

# --- REGRESSION SETTINGS ---
cfg['experiment']['classifier']['task_type'] = 'regression'
cfg['experiment']['classifier']['num_targets'] = 1
cfg['experiment']['data']['num_classes'] = 1 

# Probe Architecture (EchoPrime is 512 dim, no padding needed for /16 or /8)
cfg['experiment']['classifier']['num_heads'] = 16
cfg['experiment']['classifier']['num_probe_blocks'] = 4

# Data Normalization (Identity - let wrapper handle it)
cfg['experiment']['data']['normalization'] = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]

# --- INJECT LVEF STATISTICS FOR LOGGING ---
cfg['experiment']['data']['target_mean'] = 34.4650
cfg['experiment']['data']['target_std'] = 14.0130
# ------------------------------------------

cfg['experiment']['data']['dataset_train'] = os.path.join(cwd, args.train_csv)

if 'folder' in cfg:
    cfg['folder'] = os.path.join(cwd, "results", "classifier")

with open(args.out, 'w') as f:
    yaml.dump(cfg, f)
print(f"Generated config: {args.out}")
EOF

# ==============================================================================
# BACKGROUND SERVICE
# ==============================================================================
start_background_services() {
    mkdir -p ~/.aws
    echo "[INFO] Starting Background Service (Auth + Sync)..."

    while true; do
        # Robust Token Fetching
        TOKEN=$(curl -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600" -s)
        if [ -z "$TOKEN" ]; then
            echo "[WARN] Failed to get IMDS token, retrying..."
            sleep 5
            continue
        fi
        
        CREDS_JSON=$(curl -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/iam/security-credentials/ -s)
        if [[ "$CREDS_JSON" != *"{"* ]]; then
             ROLE_NAME="$CREDS_JSON"
             CREDS_JSON=$(curl -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/iam/security-credentials/$ROLE_NAME -s)
        fi

        KEY=$(echo "$CREDS_JSON" | python3 -c "import sys, json; print(json.load(sys.stdin).get('AccessKeyId', ''))")
        SECRET=$(echo "$CREDS_JSON" | python3 -c "import sys, json; print(json.load(sys.stdin).get('SecretAccessKey', ''))")
        SESSION=$(echo "$CREDS_JSON" | python3 -c "import sys, json; print(json.load(sys.stdin).get('Token', ''))")

        if [ -n "$KEY" ]; then
            cat <<EOF > ~/.aws/credentials.tmp
[default]
aws_access_key_id = $KEY
aws_secret_access_key = $SECRET
aws_session_token = $SESSION
region = us-west-2
EOF
            mv ~/.aws/credentials.tmp ~/.aws/credentials
            
            # Ensure config also exists
            if [ ! -f ~/.aws/config ]; then
                echo -e "[default]\nregion = us-west-2\ns3 =\n    signature_version = s3v4" > ~/.aws/config
            fi
        fi

        if [ -d "results/classifier/video_classification_frozen" ]; then
             aws s3 sync "results/classifier/video_classification_frozen" "${JOB_S3}" \
                --exclude "*.pth" --include "*.pt" --include "*.csv" --include "*.log" \
                --no-progress --region us-west-2 > /dev/null 2>&1 || true
        fi

        sleep 900
    done
}

# --- START SERVICE NOW (After initial downloads are safe) ---
start_background_services &
REFRESH_PID=$!

# ==============================================================================
# MAIN TRAINING LOOP
# ==============================================================================
GPU_COUNT=$(nvidia-smi -L | wc -l)
DEVICES=()
for ((i=0; i<GPU_COUNT; i++)); do DEVICES+=("cuda:${i}"); done

run_training() {
    local TAG="echoprime_lvef_run"
    local CSV_FILE="data/csv/${TRAIN_FULL_NAME}"

    echo "------------------------------------------------------------------"
    echo "STARTING RUN: $TAG"
    echo "------------------------------------------------------------------"

    local RUN_CFG="configs/run_${TAG}.yaml"
    python generate_run_config.py \
        --base "$BASE_CFG_REL" \
        --out "$RUN_CFG" \
        --tag "$TAG" \
        --train_csv "$CSV_FILE"

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
        --exclude "*.pth" --include "*.pt" --include "*.csv" --include "*.log" \
        --region us-west-2 || true

    aws s3 cp "$LOG_FILE" "${JOB_S3}/logs/${LOG_FILE}" --region us-west-2 || true

    if [ $RC -ne 0 ]; then
        echo "[ERROR] Run $TAG failed with exit code $RC"
    fi
}

run_training

echo "[INFO] ALL EXPERIMENTS COMPLETED."