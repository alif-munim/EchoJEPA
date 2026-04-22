#!/bin/bash
#SBATCH -p dev
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 1:00:00
#SBATCH -J diff_probe_expctrl
#SBATCH -o /opt/dlami/nvme/logs/diff_probe_expctrl_%j.out
#SBATCH -e /opt/dlami/nvme/logs/diff_probe_expctrl_%j.err
#SBATCH --signal=B:USR1@300

set -euxo pipefail

# =============================================================================
# Expanded shuffle controls for diff-probe: MAE e25, e124, e194 + JEPA e50.
# Combined with prior 3 controls (MAE e50, e99; JEPA e100) gives coverage
# across each model's trajectory. All use shuffle-control + linear-A + diff +
# 5 seeds + wd=1e-4.
#
# Purpose: test whether diff-probe R² on temporally-shuffled representations
# is near zero at every checkpoint. Non-zero R² at any checkpoint would mean
# the diff probe is capturing non-temporal EF-correlated structure, not
# genuine temporal information.
# =============================================================================

export AWS_DEFAULT_REGION="us-west-2"
export AWS_REGION="us-west-2"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export AWS_SDK_LOAD_CONFIG=1
export HOME="/home/ubuntu"
export AWS_SHARED_CREDENTIALS_FILE="${HOME}/.aws/credentials"
export AWS_CONFIG_FILE="${HOME}/.aws/config"

aws sts get-caller-identity || exit 1

ART_BUCKET="sagemaker-hyperpod-lifecycle-495467399120-usw2"
S3_BASE="s3://${ART_BUCKET}/vjepa2-artifacts"
RUN_S3="${S3_BASE}/runs/diff_probe_expctrl_${SLURM_JOB_ID}"
FEATURES_S3_BASE="${S3_BASE}/features/diff_probe"
RESULTS_S3_BASE="${S3_BASE}/results/diff_probe"
SCRATCH="/opt/dlami/nvme"

mkdir -p /opt/dlami/nvme/logs

upload_logs() {
  local o="/opt/dlami/nvme/logs/diff_probe_expctrl_${SLURM_JOB_ID}.out"
  local e="/opt/dlami/nvme/logs/diff_probe_expctrl_${SLURM_JOB_ID}.err"
  [ -f "$o" ] && aws s3 cp "$o" "${RUN_S3}/logs/job.out" --only-show-errors || true
  [ -f "$e" ] && aws s3 cp "$e" "${RUN_S3}/logs/job.err" --only-show-errors || true
}
trap 'upload_logs' EXIT

nvidia-smi -L || true

# =========================
# ENV & CODE
# =========================
aws s3 cp "${S3_BASE}/env/vjepa2-312-pack-env.tar" "${SCRATCH}/env.tar"
mkdir -p "${SCRATCH}/env" && tar -xf "${SCRATCH}/env.tar" -C "${SCRATCH}/env"
set +u && source "${SCRATCH}/env/bin/activate" && set -u
"${SCRATCH}/env/bin/conda-unpack" || true

CODE_DIR="${SCRATCH}/code"
mkdir -p "$CODE_DIR"
aws s3 cp "${S3_BASE}/setup/vjepa2-src.tar.gz" "${SCRATCH}/vjepa2-src.tar.gz" --quiet
tar xzf "${SCRATCH}/vjepa2-src.tar.gz" -C "$CODE_DIR"
cd "$CODE_DIR"
pip install -e . --no-deps --no-build-isolation 2>/dev/null || true

mkdir -p "${SCRATCH}/features/diff_probe" "${SCRATCH}/results/diff_probe/expctrl"

# =========================
# CACHE DOWNLOAD
# =========================
MODELS=(mae_e25 mae_e124 mae_e194 jepa_e50)
for name in "${MODELS[@]}"; do
  for split in test train; do
    f="${name}_${split}.pt"
    if [ ! -f "${SCRATCH}/features/diff_probe/${f}" ]; then
      aws s3 cp "${FEATURES_S3_BASE}/${f}" "${SCRATCH}/features/diff_probe/${f}" --quiet
    fi
  done
done

# =========================
# LAUNCH 4 SHUFFLE CONTROLS IN PARALLEL
# =========================
gpu=0
for m in "${MODELS[@]}"; do
  CUDA_VISIBLE_DEVICES="${gpu}" python3 -u scripts/neurips/diff_probe_train.py \
    --train-cache "${SCRATCH}/features/diff_probe/${m}_train.pt" \
    --test-cache  "${SCRATCH}/features/diff_probe/${m}_test.pt" \
    --model "${m}" \
    --out-json "${SCRATCH}/results/diff_probe/expctrl/${m}_shuffle.json" \
    --shuffle-control --archs linear-A --inputs diff --seeds 0 1 2 3 4 \
    > "${SCRATCH}/logs/expctrl_${m}.out" 2>&1 &
  gpu=$((gpu + 1))
done
wait
echo "[INFO] expanded shuffle controls done"

# =========================
# AGGREGATE + GATE
# =========================
python3 -u - <<'PYGATE'
import glob, json, sys
files = sorted(glob.glob("/opt/dlami/nvme/results/diff_probe/expctrl/*.json"))
ok = True
for f in files:
    d = json.load(open(f))
    r2s = [r["test_r2"] for r in d["results"]]
    mean_r2 = sum(r2s) / len(r2s)
    max_r2 = max(r2s)
    status = "OK" if max_r2 < 0.05 else "FAIL"
    print(f"[expctrl] {d['model']:<15} n={len(r2s):2}  mean_r2={mean_r2:+.4f}  max_r2={max_r2:+.4f}  {status}",
          flush=True)
    if max_r2 >= 0.05:
        ok = False
sys.exit(0 if ok else 42)
PYGATE
GATE_RC=$?

aws s3 sync "${SCRATCH}/results/diff_probe/expctrl" \
  "${RESULTS_S3_BASE}/${SLURM_JOB_ID}/expctrl" --only-show-errors
upload_logs

if [ "${GATE_RC}" -ne 0 ]; then
  echo "[WARN] one or more expanded shuffle controls exceed R² >= 0.05 — diff-probe may be capturing non-temporal structure"
  exit 42
fi
echo "[INFO] all expanded shuffle controls PASS"
