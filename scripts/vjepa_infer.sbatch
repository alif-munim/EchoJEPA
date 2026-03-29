#!/bin/bash
#SBATCH -p partition-1
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH -t 4-12:00:00
#SBATCH -J vjepa_inf
#SBATCH -o /tmp/vjepa_inf-%j.out
#SBATCH -e /tmp/vjepa_inf-%j.err

# ------------------------------------------------------------------
# CONFIGURATION
# Set default behavior (false = use original checkpoint)
FIX_CHECKPOINT="${FIX_CHECKPOINT:-false}"

if [[ "${FIX_CHECKPOINT}" == "true" ]]; then
    echo "[INFO] FIX_CHECKPOINT=true. Will run checkpoint fixer."
else
    echo "[INFO] FIX_CHECKPOINT=false. Will use original checkpoint."
fi
# ------------------------------------------------------------------

set -euxo pipefail

export AWS_DEFAULT_REGION="us-west-2"

ART_BUCKET="sagemaker-hyperpod-lifecycle-495467399120-usw2"
ART_PREFIX="vjepa2-artifacts"
S3_BASE="s3://${ART_BUCKET}/${ART_PREFIX}"

# Artifacts
ENV_S3="${S3_BASE}/env/vjepa2-312-pack-env.tar"
BOTO3_WHL_S3="${S3_BASE}/wheels/boto3-wheelhouse.tar"

# Inference config
CFG_REL="configs/inference/vitg-384/lvef.yaml"
CFG_S3="${S3_BASE}/${CFG_REL}"   # fallback if needed

# Test CSV
TEST_CSV_REL="data/csv/a4c_b_lvef_test.csv"
TEST_CSV_S3="${S3_BASE}/${TEST_CSV_REL}"

# Probe checkpoint (trained head)
PROBE_TAG="rvsf-a4c-full-vitg16-336-16f-pt-e279-an-e78-fs2-ns2-regression-v2"
PROBE_DIR="evals/vitg-384/rvsf_a4c/video_classification_frozen/${PROBE_TAG}"
PROBE_IN_REL="${PROBE_DIR}/latest.pt"
PROBE_OUT_REL="${PROBE_DIR}/latest_fixed.pt"
PROBE_S3="${S3_BASE}/${PROBE_IN_REL}"

# Encoder checkpoint required by model_kwargs.checkpoint in lvef.yaml
ENC_CKPT_REL="checkpoints/anneal/keep/pt-280-an81.pt"
ENC_CKPT_S3="${S3_BASE}/${ENC_CKPT_REL}"

RUN_S3="${S3_BASE}/runs/${SLURM_JOB_ID}"

upload_logs() {
  aws s3 cp "/tmp/vjepa_inf-${SLURM_JOB_ID}.out" "${RUN_S3}/logs/vjepa_inf-${SLURM_JOB_ID}.out" 2>/dev/null || true
  aws s3 cp "/tmp/vjepa_inf-${SLURM_JOB_ID}.err" "${RUN_S3}/logs/vjepa_inf-${SLURM_JOB_ID}.err" 2>/dev/null || true
}
trap 'rc=$?; upload_logs; exit $rc' EXIT

echo "[INFO] host=$(hostname) job=${SLURM_JOB_ID}"
command -v aws
aws --version
nvidia-smi -L || true

SCRATCH="/opt/dlami/nvme"
if [ ! -d "$SCRATCH" ]; then SCRATCH="/tmp"; fi
WORKDIR="${SCRATCH}/vjepa2_${SLURM_JOB_ID}"
mkdir -p "$WORKDIR"
cd "$WORKDIR"
echo "[INFO] WORKDIR=$WORKDIR"

# Sanity: required artifacts exist in S3
aws s3 ls "$ENV_S3" >/dev/null
aws s3 ls "$BOTO3_WHL_S3" >/dev/null
aws s3 ls "$TEST_CSV_S3" >/dev/null
aws s3 ls "$PROBE_S3" >/dev/null
aws s3 ls "$ENC_CKPT_S3" >/dev/null

# ---- Unpack env ----
aws s3 cp "$ENV_S3" env.tar
mkdir -p env
tar -xf env.tar -C env
set +u
source env/bin/activate
set -u
env/bin/conda-unpack || true

python -V
which python
pip -V

# ---- Install boto3 from S3 wheelhouse (no PyPI) ----
aws s3 cp "$BOTO3_WHL_S3" boto3-wheelhouse.tar
mkdir -p wheelhouse
tar -xf boto3-wheelhouse.tar -C wheelhouse
python -m pip install --no-index --find-links wheelhouse boto3
python -c "import boto3; print('[INFO] boto3', boto3.__version__)"

# =========================
# CODE (use deploy.sh-deployed code at /opt/vjepa2)
# =========================
REPO_DIR="/opt/vjepa2"
cd "$REPO_DIR"

if [ ! -f "$CFG_REL" ]; then
    echo "[FATAL] Config $CFG_REL not found in $REPO_DIR"
    exit 1
fi

pip install -e . --no-deps --no-build-isolation 2>/dev/null || true
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# ---- Materialize inputs required by lvef.yaml (after patch these paths point inside $PWD) ----
mkdir -p "$(dirname "$TEST_CSV_REL")"
aws s3 cp "$TEST_CSV_S3" "$TEST_CSV_REL"
test -f "$TEST_CSV_REL"

mkdir -p "$(dirname "$ENC_CKPT_REL")"
aws s3 cp "$ENC_CKPT_S3" "$ENC_CKPT_REL"
test -f "$ENC_CKPT_REL"

mkdir -p "$(dirname "$PROBE_IN_REL")"
aws s3 cp "$PROBE_S3" "$PROBE_IN_REL"
test -f "$PROBE_IN_REL"

# ------------------------------------------------------------------
# CONDITIONAL CHECKPOINT FIXING
# ------------------------------------------------------------------
if [[ "${FIX_CHECKPOINT}" == "true" ]]; then
    echo "[INFO] FIX_CHECKPOINT is true. Running fix_inference_checkpoint.py..."
    
    # Fix probe checkpoint (remove DDP "module." prefixes)
    python classifier/fix_inference_checkpoint.py \
      --input "$PROBE_IN_REL" \
      --output "$PROBE_OUT_REL"
    test -f "$PROBE_OUT_REL"

    # Force lvef.yaml to use latest_fixed.pt for probe_checkpoint
    python - <<'PY'
from pathlib import Path
import re

cfg = Path("configs/inference/vitg-384/lvef.yaml")
t = cfg.read_text()

# Regex to find the probe_checkpoint line and change the filename to latest_fixed.pt
# Handles both ".../latest.pt" and generic ".../somename.pt"
t2 = re.sub(r'^(\s*probe_checkpoint:\s*.+?/latest)\.pt\s*$', r'\1_fixed.pt', t, flags=re.M)
if t2 == t:
    t2 = re.sub(r'^(\s*probe_checkpoint:\s*.+)\.pt\s*$', r'\1_fixed.pt', t, flags=re.M)

cfg.write_text(t2)
print("[INFO] patched lvef.yaml: probe_checkpoint -> latest_fixed.pt")
PY

else
    echo "[INFO] FIX_CHECKPOINT is false (or unset). Using original checkpoint."
    # Ensure original checkpoint exists
    if [ ! -f "$PROBE_IN_REL" ]; then
        echo "[ERROR] Original checkpoint $PROBE_IN_REL missing."
        exit 6
    fi
fi
# ------------------------------------------------------------------

# Parse output locations from config (post-patch)
CFG_FOLDER="$(python - <<'PY'
import re
from pathlib import Path
t = Path("configs/inference/vitg-384/lvef.yaml").read_text()
m = re.search(r'^\s*folder:\s*(.+)\s*$', t, re.M)
print(m.group(1).strip().strip('"').strip("'") if m else "")
PY
)"
PRED_PATH="$(python - <<'PY'
import re
from pathlib import Path
t = Path("configs/inference/vitg-384/lvef.yaml").read_text()
m = re.search(r'^\s*predictions_save_path:\s*(.+)\s*$', t, re.M)
print(m.group(1).strip().strip('"').strip("'") if m else "")
PY
)"

test -n "$PRED_PATH" || { echo "[ERROR] Could not parse predictions_save_path from $CFG_REL"; exit 5; }
test -n "$CFG_FOLDER" || { echo "[ERROR] Could not parse folder from $CFG_REL"; exit 5; }

mkdir -p "$(dirname "$PRED_PATH")"
mkdir -p "$CFG_FOLDER"

# ---- Run inference (correct CLI: --fname + --devices) ----
GPU_COUNT=$(nvidia-smi -L | wc -l)
DEVICES=()
for ((i=0; i<GPU_COUNT; i++)); do DEVICES+=("cuda:${i}"); done

RUNLOG="lvef_infer_${SLURM_JOB_ID}.log"

set +e
python -m evals.main \
  --fname "$CFG_REL" \
  --devices "${DEVICES[@]}" 2>&1 | tee "$RUNLOG"
RC=${PIPESTATUS[0]}
set -e

if [ "$RC" -ne 0 ]; then
  echo "[ERROR] evals.main failed with exit code ${RC}"
  aws s3 cp "$RUNLOG" "${RUN_S3}/logs/${RUNLOG}" 2>/dev/null || true
  exit "$RC"
fi

# ---- Verify predictions exist and are non-empty ----
if [ ! -s "$PRED_PATH" ]; then
  echo "[ERROR] Predictions file missing/empty: $PRED_PATH"
  echo "[DEBUG] predictions/:"; ls -lh preds 2>/dev/null || true
  echo "[DEBUG] predictions/:"; ls -lh predictions 2>/dev/null || true
  aws s3 cp "$RUNLOG" "${RUN_S3}/logs/${RUNLOG}" 2>/dev/null || true
  exit 10
fi

# ---- Upload outputs ----
aws s3 cp "$CFG_REL" "${RUN_S3}/config/lvef.yaml" || true
aws s3 cp "$RUNLOG" "${RUN_S3}/logs/${RUNLOG}" || true
aws s3 cp "$PRED_PATH" "${RUN_S3}/predictions/$(basename "$PRED_PATH")" || true

# Upload inference folder contents (logs/metrics/etc)
aws s3 sync "$CFG_FOLDER" "${RUN_S3}/inference_folder/" \
  --exclude "*" --include "*.csv" --include "*.json" --include "*.txt" --include "*.png" --include "*.log" || true

echo "[INFO] DONE. Results under: ${RUN_S3}"