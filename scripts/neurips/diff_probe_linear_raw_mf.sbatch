#!/bin/bash
#SBATCH -p dev
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 2:00:00
#SBATCH -J diff_probe_lin_raw_mf
#SBATCH -o /opt/dlami/nvme/logs/diff_probe_lin_raw_mf_%j.out
#SBATCH -e /opt/dlami/nvme/logs/diff_probe_lin_raw_mf_%j.err
#SBATCH --signal=B:USR1@300

set -euxo pipefail

# =============================================================================
# Linear-A probes on RAW features, evaluated under both clean and
# representation-space matched_frame (T-axis permutation of cached test features).
# Fills the missing 2x2 cell: {linear, attentive} x {clean, matched_frame}.
#
# Caveat: jobs 216/220 matched_frame additionally RoPE-remaps positions at the
# encoder level. We cannot undo RoPE from cached features, so our shuffle is
# strictly post-encoder T-axis permutation — the closest analog using the
# existing Phase 1 feature caches.
#
# All 20 primary (model, checkpoint) combinations, 5 seeds, wd=1e-4, linear-A,
# raw input. Produces per-probe test_r2 (clean) and test_r2_mf (shuffle) with
# bootstrap CIs.
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
RUN_S3="${S3_BASE}/runs/diff_probe_lin_raw_mf_${SLURM_JOB_ID}"
FEATURES_S3_BASE="${S3_BASE}/features/diff_probe"
RESULTS_S3_BASE="${S3_BASE}/results/diff_probe"
SCRATCH="/opt/dlami/nvme"

mkdir -p /opt/dlami/nvme/logs

upload_logs() {
  local o="/opt/dlami/nvme/logs/diff_probe_lin_raw_mf_${SLURM_JOB_ID}.out"
  local e="/opt/dlami/nvme/logs/diff_probe_lin_raw_mf_${SLURM_JOB_ID}.err"
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

mkdir -p "${SCRATCH}/features/diff_probe" "${SCRATCH}/results/diff_probe/lin_raw_mf" "${SCRATCH}/logs"

# =========================
# CACHE DOWNLOAD (20 primary checkpoints)
# =========================
MODELS=(
  jepa_e25 jepa_e50 jepa_e75 jepa_e100
  mae_e25 mae_e50 mae_e75 mae_e99 mae_e124 mae_e149 mae_e174 mae_e194
  byol_e24 byol_e50 byol_e75 byol_e100
  salt_e4 salt_e29 salt_e54 salt_e79
)

for name in "${MODELS[@]}"; do
  for split in test train; do
    f="${name}_${split}.pt"
    if [ ! -f "${SCRATCH}/features/diff_probe/${f}" ]; then
      aws s3 cp "${FEATURES_S3_BASE}/${f}" "${SCRATCH}/features/diff_probe/${f}" --quiet
    fi
  done
done

# =========================
# LAUNCH 20 LINEAR-A RAW + MF PROBES IN 3 WAVES (8 GPUs parallel)
# =========================
idx=0
for m in "${MODELS[@]}"; do
  gpu=$((idx % 8))
  CUDA_VISIBLE_DEVICES="${gpu}" python3 -u scripts/neurips/diff_probe_train.py \
    --train-cache "${SCRATCH}/features/diff_probe/${m}_train.pt" \
    --test-cache  "${SCRATCH}/features/diff_probe/${m}_test.pt" \
    --model "${m}" \
    --out-json "${SCRATCH}/results/diff_probe/lin_raw_mf/${m}.json" \
    --archs linear-A --inputs raw --seeds 0 1 2 3 4 \
    --inference-mf \
    > "${SCRATCH}/logs/linraw_${m}.out" 2>&1 &
  idx=$((idx + 1))
  if [ $((idx % 8)) = 0 ]; then wait; fi
done
wait
echo "[INFO] linear-A raw + MF probes done"

# =========================
# AGGREGATE
# =========================
python3 -u - <<'PYAGG'
import glob, json, csv
rows = []
for f in sorted(glob.glob("/opt/dlami/nvme/results/diff_probe/lin_raw_mf/*.json")):
    d = json.load(open(f))
    for r in d["results"]:
        rows.append(r)
if not rows:
    print("[WARN] no rows")
    raise SystemExit(1)
keys = set()
for r in rows:
    keys.update(r.keys())
fieldnames = sorted(keys)
out = "/opt/dlami/nvme/results/diff_probe/lin_raw_mf_all.csv"
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(rows)
print(f"wrote {out}  ({len(rows)} rows, {len(fieldnames)} cols)")
PYAGG

aws s3 sync "${SCRATCH}/results/diff_probe/lin_raw_mf" \
  "${RESULTS_S3_BASE}/${SLURM_JOB_ID}/lin_raw_mf" --only-show-errors
aws s3 cp "${SCRATCH}/results/diff_probe/lin_raw_mf_all.csv" \
  "${RESULTS_S3_BASE}/${SLURM_JOB_ID}/lin_raw_mf_all.csv" --only-show-errors || true
upload_logs

echo "[DONE] diff_probe_linear_raw_mf"
ls -lh "${SCRATCH}/results/diff_probe/lin_raw_mf/"
