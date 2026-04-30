#!/bin/bash
#SBATCH -p dev
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 4:00:00
#SBATCH -J cmr_probe
#SBATCH -o /opt/dlami/nvme/logs/cmr_probe_%j.out
#SBATCH -e /opt/dlami/nvme/logs/cmr_probe_%j.err

# Root fs is 100% full on this cluster; redirect temp/cache to NVMe.
mkdir -p /opt/dlami/nvme/tmp /opt/dlami/nvme/cache/pip
export TMPDIR=/opt/dlami/nvme/tmp
export TEMP=/opt/dlami/nvme/tmp
export TMP=/opt/dlami/nvme/tmp
export PIP_CACHE_DIR=/opt/dlami/nvme/cache/pip
export XDG_CACHE_HOME=/opt/dlami/nvme/cache

set -euxo pipefail

# =============================================================================
# CMR probe-family sweep: runs all 7 variants × 2 tasks × per-encoder.
# Reads cached features from S3 (features/cmr_prepool/), runs
# scripts/neurips/cmr/probe/cmr_probe_train.py with appropriate args,
# writes JSONs to S3.
# Env:
#   SMOKE=1  -> JEPA e30 only, raw+tattn_raw only (fast validation)
# =============================================================================

export AWS_DEFAULT_REGION="us-west-2"
export AWS_REGION="us-west-2"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export HOME="/home/ubuntu"
unset AWS_SHARED_CREDENTIALS_FILE 2>/dev/null || true
unset AWS_CONFIG_FILE 2>/dev/null || true
unset AWS_EC2_METADATA_DISABLED 2>/dev/null || true

aws sts get-caller-identity || exit 1

ART_BUCKET="sagemaker-hyperpod-lifecycle-495467399120-usw2"
S3_BASE="s3://${ART_BUCKET}/vjepa2-artifacts"
ENV_S3="${S3_BASE}/env/vjepa2-312-pack-env.tar"
FEATURES_S3="${S3_BASE}/features/cmr_prepool"
RUN_S3="${S3_BASE}/runs/cmr_probe_sweep_${SLURM_JOB_ID}"
SCRATCH="/opt/dlami/nvme"
SMOKE="${SMOKE:-0}"

mkdir -p /opt/dlami/nvme/logs

upload_logs() {
  [ -f "/opt/dlami/nvme/logs/cmr_probe_${SLURM_JOB_ID}.out" ] && \
    aws s3 cp "/opt/dlami/nvme/logs/cmr_probe_${SLURM_JOB_ID}.out" "${RUN_S3}/logs/job.out" --only-show-errors || true
  [ -f "/opt/dlami/nvme/logs/cmr_probe_${SLURM_JOB_ID}.err" ] && \
    aws s3 cp "/opt/dlami/nvme/logs/cmr_probe_${SLURM_JOB_ID}.err" "${RUN_S3}/logs/job.err" --only-show-errors || true
}
trap 'upload_logs' EXIT

nvidia-smi -L || true

# -------- env + code --------
aws s3 cp "$ENV_S3" "${SCRATCH}/env.tar"
mkdir -p "${SCRATCH}/env" && tar -xf "${SCRATCH}/env.tar" -C "${SCRATCH}/env"
set +u && source "${SCRATCH}/env/bin/activate" && set -u
"${SCRATCH}/env/bin/conda-unpack" || true
pip install boto3 scikit-learn --no-cache-dir --quiet

CODE_DIR="${SCRATCH}/code_cmr_probe_${SLURM_JOB_ID}"
mkdir -p "$CODE_DIR"
aws s3 cp "${S3_BASE}/setup/vjepa2-src.tar.gz" "${SCRATCH}/vjepa2-src-cmr_probe.tar.gz" --quiet
tar xzf "${SCRATCH}/vjepa2-src-cmr_probe.tar.gz" -C "$CODE_DIR"
cd "$CODE_DIR"
pip install -e . --no-deps --no-build-isolation 2>/dev/null || true
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# -------- configs --------
OUT_DIR="${SCRATCH}/cmr_probe_out/${SLURM_JOB_ID}"
FEAT_DIR="${SCRATCH}/features/cmr_prepool"
mkdir -p "${OUT_DIR}" "${FEAT_DIR}"

# Download all feature caches
echo "[INFO] Pulling feature caches..."
aws s3 sync "${FEATURES_S3}/" "${FEAT_DIR}/" --only-show-errors

if [[ "${SMOKE}" == "1" ]]; then
  MODELS="jepa_slowema_e30"
  VARIANTS="raw"
  TASKS="lvef dx"
else
  MODELS="jepa_slowema_e30 jepa_slowema_e100 jepa_slowema_e200 mae_e30 mae_e100 mae_e200 mae_e300"
  # Full variant set. single_t0..7 handled separately below so we can pick
  # best_token_idx for tattn_best_rep.
  VARIANTS_LINEAR="raw diff mean"
  VARIANTS_TATTN="tattn_raw tattn_mean_rep"
  TASKS="lvef dx"
fi

run_probe() {
  # args: MODEL VARIANT TASK [BEST_TOK_ARG]
  local MODEL="$1"; local VARIANT="$2"; local TASK="$3"; local BTK_ARG="${4:-}"
  local TRAIN_CACHE="${FEAT_DIR}/${MODEL}/${TASK}_train.pt"
  local TEST_CACHE="${FEAT_DIR}/${MODEL}/${TASK}_test.pt"
  if [ ! -f "${TRAIN_CACHE}" ] || [ ! -f "${TEST_CACHE}" ]; then
    echo "[WARN] missing cache for ${MODEL} ${TASK} — skipping"
    return
  fi
  local OUT_JSON="${OUT_DIR}/${MODEL}__${TASK}__${VARIANT}.json"
  local LR_SWEEP="--lr-sweep 1e-3"
  local MAX_EP=30
  if [[ "${VARIANT}" == tattn_* ]]; then
    LR_SWEEP="--lr-sweep 1e-4 3e-4 1e-3"
    MAX_EP=50
  fi
  if [[ "${TASK}" == "dx" ]]; then
    MAX_EP=50
  fi
  echo "[INFO] run ${MODEL} ${VARIANT} ${TASK}"
  python3 -m scripts.neurips.cmr.probe.cmr_probe_train \
    --train-cache "${TRAIN_CACHE}" \
    --test-cache "${TEST_CACHE}" \
    --out-json "${OUT_JSON}" \
    --model "${MODEL}" \
    --task "${TASK}" \
    --variant "${VARIANT}" \
    --num-classes 5 \
    --seeds 0 1 2 3 4 \
    ${LR_SWEEP} \
    --wd-sweep 1e-4 1e-2 \
    --batch-size 64 \
    --max-epochs ${MAX_EP} \
    ${BTK_ARG}
  # Upload
  aws s3 cp "${OUT_JSON}" "${RUN_S3}/jsons/$(basename ${OUT_JSON})" --only-show-errors || true
}

# Extract T for each model: peek one cache
pick_T_for_model() {
  local MODEL="$1"
  python3 - "${FEAT_DIR}/${MODEL}" <<'PY'
import sys, torch, os
d = sys.argv[1]
for fn in os.listdir(d):
    if fn.endswith(".pt"):
        obj = torch.load(os.path.join(d, fn), map_location="cpu", weights_only=False)
        feats = obj["features"]
        if feats.dim() == 3:
            print(feats.shape[1])
        else:
            print(feats.shape[2])
        break
PY
}

if [[ "${SMOKE}" == "1" ]]; then
  for MODEL in ${MODELS}; do
    for TASK in ${TASKS}; do
      for VARIANT in ${VARIANTS}; do
        run_probe "${MODEL}" "${VARIANT}" "${TASK}"
      done
    done
  done
else
  for MODEL in ${MODELS}; do
    # Infer T
    T=$(pick_T_for_model "${MODEL}")
    echo "[INFO] ${MODEL} has T=${T}"
    for TASK in ${TASKS}; do
      # Linear variants
      for VARIANT in ${VARIANTS_LINEAR}; do
        run_probe "${MODEL}" "${VARIANT}" "${TASK}"
      done
      # Single-token per t
      BEST_T=-1
      BEST_VAL=-1e9
      for ((k=0; k<T; k++)); do
        VARIANT="single_t${k}"
        run_probe "${MODEL}" "${VARIANT}" "${TASK}"
        # Read best val metric (mean across seeds) to pick best_token_idx
        VM=$(python3 - "${OUT_DIR}/${MODEL}__${TASK}__${VARIANT}.json" <<'PY'
import json, sys, numpy as np
d = json.load(open(sys.argv[1]))
vals = [r["val_metric"] for r in d["rows"]]
print(f"{np.mean(vals):.6f}")
PY
)
        if awk "BEGIN {exit !(${VM} > ${BEST_VAL})}"; then
          BEST_VAL="${VM}"
          BEST_T="${k}"
        fi
      done
      echo "[INFO] ${MODEL} ${TASK}: best_token_idx=${BEST_T} (val=${BEST_VAL})"
      # Pooled T-attn variants
      for VARIANT in ${VARIANTS_TATTN}; do
        run_probe "${MODEL}" "${VARIANT}" "${TASK}"
      done
      # Best-token-repeat
      run_probe "${MODEL}" "tattn_best_rep" "${TASK}" "--best-token-idx ${BEST_T}"
    done
  done
fi

# Final sync of all JSONs
aws s3 sync "${OUT_DIR}/" "${RUN_S3}/jsons/" --only-show-errors || true
echo "[INFO] CMR probe sweep complete"
