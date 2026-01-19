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
CODE_S3="${S3_BASE}/code/vjepa2-code.tar"
BOTO3_WHL_S3="${S3_BASE}/wheels/boto3-wheelhouse.tar"

# Inference config (now included in code tar)
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
aws s3 ls "$CODE_S3" >/dev/null
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

# ---- Unpack code ----
aws s3 cp "$CODE_S3" code.tar
tar -xf code.tar
echo "[INFO] Locating repo root after extracting code.tar..."
REPO_DIR=$(find . -maxdepth 10 -type f -name run_inf.sh -print -quit | xargs -r dirname)
if [ -z "${REPO_DIR}" ]; then
  echo "[ERROR] Could not find run_inf.sh after extracting code.tar"
  echo "[DEBUG] Top-level dirs:"; find . -maxdepth 2 -type d -print
  echo "[DEBUG] First tar entries:"; tar -tf code.tar | head -n 50
  exit 2
fi
cd "${REPO_DIR}"
echo "[INFO] repo_dir=${REPO_DIR}"

# Ensure config exists (prefer from code.tar; fallback to S3 if missing)
if [ ! -f "$CFG_REL" ]; then
  echo "[WARN] ${CFG_REL} not found in code.tar; trying S3: ${CFG_S3}"
  mkdir -p "$(dirname "$CFG_REL")"
  aws s3 cp "$CFG_S3" "$CFG_REL"
fi
test -f "$CFG_REL" || {
  echo "[ERROR] Missing inference config: $CFG_REL"
  echo "[DEBUG] Available inference configs:"; find configs/inference/vitg-384 -maxdepth 3 -type f -name '*.yaml' -print || true
  exit 3
}

# ---- Patch hardcoded Studio/EFS paths -> local repo path (compute node) ----
python - <<'PY'
from pathlib import Path
import os, re

repo = os.getcwd()
studio = "/home/sagemaker-user/user-default-efs/vjepa2"
efs_pat = re.compile(r"/mnt/custom-file-systems/efs/[^ \n\"']+/vjepa2")

roots = ["run_inf.sh", "configs", "app", "evals", "src", "classifier", "data"]
# Do NOT rewrite CSV contents
exts  = {".py",".sh",".yaml",".yml",".json",".toml",".txt",".md"}

patched = 0
for r in roots:
    p = Path(r)
    if not p.exists():
        continue
    it = [p] if p.is_file() else p.rglob("*")
    for f in it:
        if not f.is_file():
            continue
        if f.suffix.lower() not in exts and f.name != "run_inf.sh":
            continue
        try:
            t = f.read_text()
        except Exception:
            continue

        t2 = t.replace(studio, repo)
        t2 = efs_pat.sub(repo, t2)

        if t2 != t:
            f.write_text(t2)
            patched += 1

print(f"[INFO] patched hardcoded paths in {patched} files -> repo={repo}")
PY

# Fail fast if any hardcoded base paths remain in runtime files
if grep -R -n -E "/home/sagemaker-user/user-default-efs/vjepa2|/mnt/custom-file-systems/efs/.*/vjepa2" \
    --include='*.py' --include='*.sh' --include='*.yaml' --include='*.yml' --include='*.json' --include='*.toml' --include='*.txt' --include='*.md' \
    run_inf.sh configs app evals src classifier 2>/dev/null | head -n 1; then
  echo "[ERROR] Still found hardcoded Studio/EFS paths after patch (runtime files)"
  exit 4
fi

# Make code importable
pip install -e . --no-deps --no-build-isolation || true
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
  echo "[DEBUG] preds/:"; ls -lh preds 2>/dev/null || true
  echo "[DEBUG] predictions/:"; ls -lh predictions 2>/dev/null || true
  aws s3 cp "$RUNLOG" "${RUN_S3}/logs/${RUNLOG}" 2>/dev/null || true
  exit 10
fi

# ---- Upload outputs ----
aws s3 cp "$CFG_REL" "${RUN_S3}/config/lvef.yaml" || true
aws s3 cp "$RUNLOG" "${RUN_S3}/logs/${RUNLOG}" || true
aws s3 cp "$PRED_PATH" "${RUN_S3}/preds/$(basename "$PRED_PATH")" || true

# Upload inference folder contents (logs/metrics/etc)
aws s3 sync "$CFG_FOLDER" "${RUN_S3}/inference_folder/" \
  --exclude "*" --include "*.csv" --include "*.json" --include "*.txt" --include "*.png" --include "*.log" || true

echo "[INFO] DONE. Results under: ${RUN_S3}"