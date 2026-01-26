#!/bin/bash
#SBATCH -p partition-1
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 4-12:00:00
#SBATCH -J panecho_eff
#SBATCH -o /tmp/panecho_eff-%j.out
#SBATCH -e /tmp/panecho_eff-%j.err

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

# Base Config (Relative path inside code.tar)
BASE_CFG_REL="configs/eval/vitg-384/view/panecho_view_classification.yaml"

# Video Dataset Location (S3 -> Local NVMe)
VIDEO_S3_PREFIX="s3://echodata25/results/uhn_studies_22k_607_224px/"

# Output S3 Location
JOB_S3="${S3_BASE}/runs/efficiency_exp_panecho_${SLURM_JOB_ID}"

# ==============================================================================
# 2. SETUP & HYDRATION
# ==============================================================================

# Error Trap & Log Upload
upload_logs() {
  echo "[INFO] Uploading master logs..."
  aws s3 cp "/tmp/panecho_eff-${SLURM_JOB_ID}.out" "${JOB_S3}/logs/master.out" 2>/dev/null || true
  aws s3 cp "/tmp/panecho_eff-${SLURM_JOB_ID}.err" "${JOB_S3}/logs/master.err" 2>/dev/null || true

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

# Verify Base Config Exists (Create placeholder if missing for safety)
if [ ! -f "$BASE_CFG_REL" ]; then
    echo "[WARN] Base config $BASE_CFG_REL not found. Assuming user will inject via cat."
    mkdir -p "$(dirname "$BASE_CFG_REL")"
fi

# Path Patching (Studio -> Cluster)
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
pip install opencv-python-headless
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# ==============================================================================
# 3. DATA DOWNLOAD & SYNC
# ==============================================================================

# Download CSVs
mkdir -p data/csv
aws s3 cp "$TRAIN_FULL_S3" "data/csv/${TRAIN_FULL_NAME}"
aws s3 cp "$VAL_S3" "data/csv/${VAL_NAME}"

echo "[INFO] Syncing video dataset from S3 to local NVMe..."
LOCAL_VIDEO_DIR="${WORKDIR}/videos"
mkdir -p "$LOCAL_VIDEO_DIR"
aws s3 sync "$VIDEO_S3_PREFIX" "$LOCAL_VIDEO_DIR" --quiet

echo "[INFO] Patching CSVs to point to local video files..."
ESC_S3_PREFIX=$(printf '%s\n' "$VIDEO_S3_PREFIX" | sed 's:[][\/.^$*]:\\&:g')
ESC_LOCAL_DIR=$(printf '%s\n' "${LOCAL_VIDEO_DIR}/" | sed 's:[][\/.^$*]:\\&:g')

sed -i "s|${ESC_S3_PREFIX}|${ESC_LOCAL_DIR}|g" "data/csv/${TRAIN_FULL_NAME}"
sed -i "s|${ESC_S3_PREFIX}|${ESC_LOCAL_DIR}|g" "data/csv/${VAL_NAME}"

# ==============================================================================
# 4. PANECHO SETUP (UPDATED)
# ==============================================================================
echo "[INFO] Setting up PanEcho..."

# 1. Identify location
MC_DIR="evals/video_classification_frozen/modelcustom"
mkdir -p "$MC_DIR"

# 2. Inject the Wrapper Code
cat << 'EOF' > "${MC_DIR}/panecho_encoder.py"
"""
Custom model wrapper for PanEcho to work with V-JEPA 2 eval system.
"""
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

import logging
import torch
import torch.nn as nn
import sys, os, importlib, importlib.util

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class PanEchoWrapper(nn.Module):
    def __init__(self, panecho_model, embed_dim=768):
        super().__init__()
        self.panecho_model = panecho_model
        self.embed_dim = embed_dim

    @torch.no_grad()
    def forward(self, x, clip_indices=None):
        if isinstance(x, list):
            tokens = [dij for di in x for dij in di] 
            if len(tokens) == 0: raise ValueError("Empty clip list received.")
            x_tok = torch.stack(tokens, dim=1)
        elif torch.is_tensor(x):
            if x.ndim == 5: x_tok = x.unsqueeze(1)
            elif x.ndim == 6: x_tok = x
            else: raise ValueError(f"Unsupported tensor input shape: {tuple(x.shape)}")
        else:
            raise ValueError(f"Unsupported input type: {type(x)}")

        B, N, C, T, H, W = x_tok.shape
        x_flat = x_tok.reshape(B * N, C, T, H, W)
        out = self.panecho_model(x_flat)

        if isinstance(out, dict):
            if "embedding" in out: emb = out["embedding"]
            elif "last_hidden_state" in out:
                hs = out["last_hidden_state"]
                emb = hs[:, 0] if hs.ndim == 3 else hs
            else: emb = next(iter(out.values()))
        else:
            emb = out

        if emb.ndim == 3 and emb.shape[1] == 1: emb = emb[:, 0]
        if emb.ndim != 2 or emb.shape[1] != self.embed_dim:
            raise ValueError(f"Unexpected PanEcho embedding shape: {tuple(emb.shape)}")

        emb = emb.reshape(B, N, self.embed_dim) 
        return [emb[:, i:i+1, :] for i in range(N)]

def init_module(
    resolution: int,
    frames_per_clip: int,
    checkpoint: str,
    model_kwargs: dict,
    wrapper_kwargs: dict,
    device=None,
):
    logger.info("Loading PanEcho model from local source...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Try finding PanEcho in the same dir (typical structure)
    panecho_root = os.path.join(current_dir, "PanEcho")
    
    # 2. If not there, try finding it at the repo root
    if not os.path.exists(panecho_root):
        repo_root = os.path.abspath(os.path.join(current_dir, "../../../../"))
        panecho_root_alt = os.path.join(repo_root, "PanEcho")
        if os.path.exists(panecho_root_alt):
            panecho_root = panecho_root_alt

    if not os.path.exists(panecho_root):
        # List dir to help debug if it fails
        logger.error(f"Cannot find PanEcho at {panecho_root}")
        logger.error(f"Contents of {current_dir}: {os.listdir(current_dir)}")
        raise FileNotFoundError(f"PanEcho source not found at {panecho_root}")

    logger.info(f"Using PanEcho source at: {panecho_root}")

    vjepa_modules = {k: v for k, v in sys.modules.items() if k == "src" or k.startswith("src.")}
    original_path = list(sys.path)
    for k in list(vjepa_modules.keys()): sys.modules.pop(k, None)
    sys.path.insert(0, panecho_root)

    try:
        importlib.invalidate_caches()
        hubconf_path = os.path.join(panecho_root, "hubconf.py")
        spec = importlib.util.spec_from_file_location("hubconf", hubconf_path)
        hubconf = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(hubconf)

        panecho_model = hubconf.PanEcho(
            pretrained=True,
            clip_len=frames_per_clip,
            backbone_only=True,
        )
        panecho_model.eval()
        for p in panecho_model.parameters(): p.requires_grad_(False)
        if device is not None: panecho_model.to(device)

    finally:
        panecho_modules = {k: v for k, v in sys.modules.items() if k == "src" or k.startswith("src.")}
        for k in list(panecho_modules.keys()): sys.modules.pop(k, None)
        sys.path = original_path
        sys.modules.update(vjepa_modules)

    logger.info("PanEcho loaded successfully.")
    return PanEchoWrapper(panecho_model, embed_dim=768)
EOF

echo "[INFO] Verifying PanEcho folder existence (from code.tar)..."
if [ -d "${MC_DIR}/PanEcho" ]; then
    echo "[INFO] Found PanEcho at ${MC_DIR}/PanEcho"
else
    echo "[WARN] PanEcho NOT found at ${MC_DIR}/PanEcho. Searching..."
    find . -maxdepth 4 -type d -name "PanEcho"
fi

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
parser.add_argument("--ckpt", required=False) # Ignored for PanEcho
args = parser.parse_args()

with open(args.base) as f:
    cfg = yaml.safe_load(f)

cfg['tag'] = args.tag
cwd = os.getcwd()

# --- PATCH PATHS ---
cfg['experiment']['data']['dataset_train'] = os.path.join(cwd, args.train_csv)
cfg['experiment']['data']['dataset_val'] = os.path.join(cwd, args.val_csv)

# PanEcho Setup
cfg['model_kwargs']['module_name'] = 'evals.video_classification_frozen.modelcustom.panecho_encoder'
# Checkpoint is handled internally by PanEcho wrapper (Torch Hub), so we force null here
cfg['model_kwargs']['checkpoint'] = None 
cfg['model_kwargs']['pretrain_kwargs'] = {}
cfg['model_kwargs']['wrapper_kwargs'] = {}

# Ensure output folder
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
        local TAG="uhn22k_panecho_${PCT_NAME}"
    else
        local TAG="uhn22k_panecho_${PCT_NAME}_s${SEED}"
    fi

    echo "------------------------------------------------------------------"
    echo "STARTING RUN: $TAG"
    echo "DATASET: $CSV_FILE"
    echo "------------------------------------------------------------------"

    local RUN_CFG="configs/run_${TAG}.yaml"
    
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