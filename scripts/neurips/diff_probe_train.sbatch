#!/bin/bash
#SBATCH -p dev
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 4:00:00
#SBATCH -J diff_probe_train
#SBATCH -o /opt/dlami/nvme/logs/diff_probe_train_%j.out
#SBATCH -e /opt/dlami/nvme/logs/diff_probe_train_%j.err
#SBATCH --signal=B:USR1@300

set -euxo pipefail

# =============================================================================
# Two-phase diff-probe training:
#   PHASE 1 (controls): 3 temporal-shuffle + 3 random-encoder probes, all
#     linear-A-diff only. ALL must hit test_r2 < 0.05 — otherwise abort.
#   PHASE 2 (primary): 20 ckpts × {linear-A, linear-B, mlp} × {diff, raw}
#     × 5 seeds, plus linear-A wd sweep. Only runs if controls pass.
#
# Modes (env var):
#   PHASE=controls | primary | both (default: both)
#
# Requires: Phase 1 extraction (job 312) complete, all 40 caches in S3 at
#   ${FEATURES_S3_BASE}/, plus 3 random-encoder caches.
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
RUN_S3="${S3_BASE}/runs/diff_probe_train_${SLURM_JOB_ID}"
FEATURES_S3_BASE="${S3_BASE}/features/diff_probe"
RESULTS_S3_BASE="${S3_BASE}/results/diff_probe"
SCRATCH="/opt/dlami/nvme"
PHASE="${PHASE:-both}"

mkdir -p /opt/dlami/nvme/logs

upload_logs() {
  local o="/opt/dlami/nvme/logs/diff_probe_train_${SLURM_JOB_ID}.out"
  local e="/opt/dlami/nvme/logs/diff_probe_train_${SLURM_JOB_ID}.err"
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

mkdir -p "${SCRATCH}/features/diff_probe" "${SCRATCH}/results/diff_probe"

# =========================
# CACHE DOWNLOAD
# =========================
download_cache() {
  local name="$1"  # e.g. jepa_e100
  for split in test train; do
    local f="${name}_${split}.pt"
    if [ ! -f "${SCRATCH}/features/diff_probe/${f}" ]; then
      aws s3 cp "${FEATURES_S3_BASE}/${f}" "${SCRATCH}/features/diff_probe/${f}" --quiet
    fi
  done
}

PRIMARY_MODELS=(
  jepa_e25 jepa_e50 jepa_e75 jepa_e100
  mae_e25 mae_e50 mae_e75 mae_e99 mae_e124 mae_e149 mae_e174 mae_e194
  byol_e24 byol_e50 byol_e75 byol_e100
  salt_e4 salt_e29 salt_e54 salt_e79
)

RANDOM_SEEDS=(42 43 44)

# Launcher — runs one probe-training job on a given GPU.
#   $1=gpu_id  $2=model_name  $3=out_json  $4-N=extra_args
run_probe() {
  local gpu_id="$1"; shift
  local model_name="$1"; shift
  local out_json="$1"; shift
  CUDA_VISIBLE_DEVICES="${gpu_id}" python3 -u scripts/neurips/diff_probe_train.py \
    --train-cache "${SCRATCH}/features/diff_probe/${model_name}_train.pt" \
    --test-cache  "${SCRATCH}/features/diff_probe/${model_name}_test.pt" \
    --model "${model_name}" \
    --out-json "${out_json}" \
    "$@" \
    > "${SCRATCH}/logs/probe_${model_name}_$(basename "${out_json}" .json).out" 2>&1
}

# =========================
# PHASE 1: CONTROLS
# =========================
if [ "${PHASE}" = "controls" ] || [ "${PHASE}" = "both" ]; then
  echo "============================================================"
  echo "[PHASE 1] CONTROLS — 3 shuffle + 3 random-encoder"
  echo "============================================================"

  # Download caches needed for controls: MAE e50, MAE e99, JEPA e100
  for m in mae_e50 mae_e99 jepa_e100; do download_cache "$m"; done
  # Random-encoder caches
  for s in "${RANDOM_SEEDS[@]}"; do
    for split in test train; do
      aws s3 cp "${FEATURES_S3_BASE}/random_s${s}_${split}.pt" \
        "${SCRATCH}/features/diff_probe/random_s${s}_${split}.pt" --quiet
    done
  done

  mkdir -p "${SCRATCH}/results/diff_probe/controls"

  # Launch 6 controls (3 shuffle + 3 random) in parallel on 6 GPUs.
  gpu=0
  for m in mae_e50 mae_e99 jepa_e100; do
    run_probe "$gpu" "$m" \
      "${SCRATCH}/results/diff_probe/controls/${m}_shuffle.json" \
      --shuffle-control --archs linear-A --inputs diff --seeds 0 1 2 3 4 &
    gpu=$((gpu + 1))
  done
  for s in "${RANDOM_SEEDS[@]}"; do
    run_probe "$gpu" "random_s${s}" \
      "${SCRATCH}/results/diff_probe/controls/random_s${s}.json" \
      --archs linear-A --inputs diff --seeds 0 1 2 3 4 &
    gpu=$((gpu + 1))
  done
  wait
  echo "[PHASE 1] controls done"

  # Aggregate + gate on R² < 0.05
  python3 -u - <<'PYGATE'
import glob, json, sys
files = sorted(glob.glob("/opt/dlami/nvme/results/diff_probe/controls/*.json"))
ok = True
for f in files:
    d = json.load(open(f))
    r2s = [r["test_r2"] for r in d["results"]]
    mean_r2 = sum(r2s) / len(r2s)
    max_r2 = max(r2s)
    status = "OK" if max_r2 < 0.05 else "FAIL"
    print(f"[control] {d['model']:<15} shuffle={d.get('shuffle_control'):5}  "
          f"n={len(r2s):2}  mean_r2={mean_r2:+.4f}  max_r2={max_r2:+.4f}  {status}",
          flush=True)
    if max_r2 >= 0.05:
        ok = False
sys.exit(0 if ok else 42)
PYGATE
  GATE_RC=$?
  # Upload control results
  aws s3 sync "${SCRATCH}/results/diff_probe/controls" \
    "${RESULTS_S3_BASE}/${SLURM_JOB_ID}/controls" --only-show-errors
  upload_logs

  if [ "${GATE_RC}" -ne 0 ]; then
    echo "[PHASE 1] CONTROL FAILURE — aborting before primary"
    exit 42
  fi
  echo "[PHASE 1] ALL CONTROLS PASS"
fi

# =========================
# PHASE 2: PRIMARY
# =========================
if [ "${PHASE}" = "primary" ] || [ "${PHASE}" = "both" ]; then
  echo "============================================================"
  echo "[PHASE 2] PRIMARY — 20 ckpts × {lin-A, lin-B, mlp} × {diff, raw} × 5 seeds"
  echo "============================================================"

  for m in "${PRIMARY_MODELS[@]}"; do download_cache "$m"; done

  mkdir -p "${SCRATCH}/results/diff_probe/primary"

  # Dispatch: 1 probe-train call = 1 model (spans all archs × inputs × seeds
  # × wd sweep in a single Python process — ~60 probes per call, ~2 min/GPU).
  # We parallelize 8 models at a time across 8 GPUs. 20 models / 8 ≈ 3 waves.
  idx=0
  for m in "${PRIMARY_MODELS[@]}"; do
    gpu=$((idx % 8))
    run_probe "$gpu" "$m" \
      "${SCRATCH}/results/diff_probe/primary/${m}.json" \
      --archs linear-A linear-B mlp --inputs diff raw \
      --seeds 0 1 2 3 4 --wd-sweep 1e-4 1e-2 &
    idx=$((idx + 1))
    if [ $((idx % 8)) = 0 ]; then wait; fi
  done
  wait
  echo "[PHASE 2] primary done"

  # Aggregate summary
  python3 -u - <<'PYAGG'
import glob, json, csv
rows = []
for f in sorted(glob.glob("/opt/dlami/nvme/results/diff_probe/primary/*.json")):
    d = json.load(open(f))
    for r in d["results"]:
        rows.append(r)
out = "/opt/dlami/nvme/results/diff_probe/primary_all.csv"
if rows:
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}  ({len(rows)} rows)")
PYAGG

  aws s3 sync "${SCRATCH}/results/diff_probe/primary" \
    "${RESULTS_S3_BASE}/${SLURM_JOB_ID}/primary" --only-show-errors
  aws s3 cp "${SCRATCH}/results/diff_probe/primary_all.csv" \
    "${RESULTS_S3_BASE}/${SLURM_JOB_ID}/primary_all.csv" --only-show-errors || true
  upload_logs
  echo "[PHASE 2] complete"
fi

echo "============================================================"
echo "[DONE] diff-probe training complete"
echo "============================================================"
ls -lh "${SCRATCH}/results/diff_probe/"
