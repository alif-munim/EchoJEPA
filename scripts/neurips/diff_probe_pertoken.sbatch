#!/bin/bash
#SBATCH -p dev
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 4:00:00
#SBATCH -J diff_probe_pertoken
#SBATCH -o /opt/dlami/nvme/logs/diff_probe_pertoken_%j.out
#SBATCH -e /opt/dlami/nvme/logs/diff_probe_pertoken_%j.err
#SBATCH --signal=B:USR1@300

set -euxo pipefail

# =============================================================================
# Phase 3 per-token diff probe training.
#
# SPOT_CHECK=1 (default): MAE e99 only; runs variants (b) linear-spatial,
#                         (c-A) attn_a, (c-B) attn_b, x 5 seeds x 3 shuffles
#                         (none, temporal, spatial). Two shuffles are gates
#                         (R^2 < 0.05); both must pass before expanding scope.
#
# SPOT_CHECK=0           : 7-ckpt minimum scope. DO NOT RUN until spot-check
#                          gates pass.
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
RUN_S3="${S3_BASE}/runs/diff_probe_pertoken_${SLURM_JOB_ID}"
FEATURES_S3_BASE="${S3_BASE}/features/diff_probe_pertoken"
RESULTS_S3_BASE="${S3_BASE}/results/diff_probe_pertoken"
SCRATCH="/opt/dlami/nvme"
SPOT_CHECK="${SPOT_CHECK:-1}"

mkdir -p /opt/dlami/nvme/logs

upload_logs() {
  local o="/opt/dlami/nvme/logs/diff_probe_pertoken_${SLURM_JOB_ID}.out"
  local e="/opt/dlami/nvme/logs/diff_probe_pertoken_${SLURM_JOB_ID}.err"
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

mkdir -p "${SCRATCH}/features/diff_probe_pertoken" "${SCRATCH}/results/diff_probe_pertoken" "${SCRATCH}/logs"

# =========================
# FETCH CACHES
# =========================
if [ "${SPOT_CHECK}" = "1" ]; then
  MODELS=(mae_e99)
else
  MODELS=(mae_e50 mae_e99 mae_e194 jepa_e50 jepa_e100 byol_e100 salt_e79)
fi

for name in "${MODELS[@]}"; do
  for split in test train; do
    f="${name}_${split}.pt"
    if [ ! -f "${SCRATCH}/features/diff_probe_pertoken/${f}" ]; then
      aws s3 cp "${FEATURES_S3_BASE}/${f}" "${SCRATCH}/features/diff_probe_pertoken/${f}" --quiet
    fi
  done
done

# =========================
# VERIFY SPOT-CHECK CACHE SHAPE
# =========================
if [ "${SPOT_CHECK}" = "1" ]; then
  python3 - <<'PYCHECK'
import torch
d = torch.load("/opt/dlami/nvme/features/diff_probe_pertoken/mae_e99_test.pt",
               map_location="cpu", weights_only=False, mmap=True)
f = d["features"]
print(f"[CHECK] test features shape: {tuple(f.shape)} dtype={f.dtype}")
expected = (1277, 2, 8, 196, 1024)
assert tuple(f.shape) == expected, f"Got {tuple(f.shape)}, expected {expected}"
# NaN + variance sanity (sample a slice to keep it cheap)
samp = f[0:10].float()
assert not torch.isnan(samp).any(), "NaN present"
std = float(samp.std())
print(f"[CHECK] sample std across first 10 clips: {std:.4f}")
assert std > 0.01, f"variance suspiciously low: {std}"
print("[CHECK] PASS")
PYCHECK
fi

# =========================
# TRAIN PROBES (GPU-parallel over seeds within each model)
# =========================
for m in "${MODELS[@]}"; do
  mkdir -p "${SCRATCH}/results/diff_probe_pertoken/${m}"
  # 5 seeds x 3 shuffles = 15 configs. Run one seed per GPU (8 GPUs, 5 seeds fits).
  for seed in 0 1 2 3 4; do
    gpu=$((seed % 8))
    CUDA_VISIBLE_DEVICES="${gpu}" python3 -u scripts/neurips/diff_probe_train_pertoken.py \
      --train-cache "${SCRATCH}/features/diff_probe_pertoken/${m}_train.pt" \
      --test-cache  "${SCRATCH}/features/diff_probe_pertoken/${m}_test.pt" \
      --model "${m}" \
      --seeds ${seed} \
      --archs linear_spatial attn_a attn_b \
      --shuffles none temporal spatial \
      --out-json "${SCRATCH}/results/diff_probe_pertoken/${m}/seed${seed}.json" \
      --attn-out "${SCRATCH}/results/diff_probe_pertoken/${m}/seed${seed}_attn.npz" \
      > "${SCRATCH}/logs/pertoken_${m}_s${seed}.out" 2>&1 &
  done
  wait
  echo "[INFO] ${m} probes done"
done

# =========================
# AGGREGATE
# =========================
python3 -u - <<'PYAGG'
import glob, json, csv
rows = []
for f in sorted(glob.glob("/opt/dlami/nvme/results/diff_probe_pertoken/*/seed*.json")):
    d = json.load(open(f))
    rows.extend(d["results"])
keys = set()
for r in rows:
    keys.update(r.keys())
fieldnames = sorted(keys)
out = "/opt/dlami/nvme/results/diff_probe_pertoken/all.csv"
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(rows)
print(f"wrote {out}  ({len(rows)} rows)")

# Control gate check
from collections import defaultdict
by_key = defaultdict(list)
for r in rows:
    by_key[(r["model"], r["arch"], r["shuffle"])].append(r["test_r2"])
print()
print("Gate check (R^2 < 0.05 required on shuffle controls):")
print(f"{'model':<12} {'arch':<16} {'shuffle':<10} {'mean_r2':>10} {'max_r2':>10} {'pass':>6}")
for (m, a, s), vs in sorted(by_key.items()):
    mean = sum(vs)/len(vs)
    mx = max(vs)
    if s == "none":
        status = "(info)"
    else:
        status = "PASS" if mx < 0.05 else "FAIL"
    print(f"{m:<12} {a:<16} {s:<10} {mean:>10.4f} {mx:>10.4f} {status:>6}")
PYAGG

aws s3 sync "${SCRATCH}/results/diff_probe_pertoken" \
  "${RESULTS_S3_BASE}/${SLURM_JOB_ID}" --only-show-errors

aws s3 cp "${SCRATCH}/results/diff_probe_pertoken/all.csv" \
  "${RESULTS_S3_BASE}/${SLURM_JOB_ID}/all.csv" --only-show-errors || true
upload_logs

echo "[DONE] diff_probe_pertoken"
ls -lh "${SCRATCH}/results/diff_probe_pertoken/"
