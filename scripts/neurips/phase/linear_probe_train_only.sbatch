#!/bin/bash
#SBATCH -p dev
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 00:30:00
#SBATCH -J lin_train
#SBATCH -o /opt/dlami/nvme/logs/lin_train_%j.out
#SBATCH -e /opt/dlami/nvme/logs/lin_train_%j.err

set -euxo pipefail

# Trains linear-A-raw probes on the 4 cached (ckpt, split) pairs:
#   jepa_e100, jepa_e200, mae_e99, mae_e194
# Requires the extract_* jobs to have finished (feature caches on S3).

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
FEATURES_S3="${S3_BASE}/features/diff_probe"
RUN_S3="${S3_BASE}/runs/linear_probe_verify_${SLURM_JOB_ID}"

upload_logs() {
  local o="/opt/dlami/nvme/logs/lin_train_${SLURM_JOB_ID}.out"
  local e="/opt/dlami/nvme/logs/lin_train_${SLURM_JOB_ID}.err"
  [ -f "$o" ] && aws s3 cp "$o" "${RUN_S3}/logs/job.out" --only-show-errors || true
  [ -f "$e" ] && aws s3 cp "$e" "${RUN_S3}/logs/job.err" --only-show-errors || true
}
trap 'upload_logs' EXIT

mkdir -p /opt/dlami/nvme/logs
SCRATCH="/opt/dlami/nvme"
WORKDIR="${SCRATCH}/lin_train_${SLURM_JOB_ID}"
mkdir -p "$WORKDIR"

aws s3 cp "$ENV_S3" "${WORKDIR}/env.tar"
mkdir -p "${WORKDIR}/env" && tar -xf "${WORKDIR}/env.tar" -C "${WORKDIR}/env"
set +u && source "${WORKDIR}/env/bin/activate" && set -u
"${WORKDIR}/env/bin/conda-unpack" || true
export TMPDIR="${WORKDIR}/tmp" && mkdir -p "$TMPDIR"
pip install boto3 --no-cache-dir --quiet

CODE_DIR="${WORKDIR}/code"
mkdir -p "$CODE_DIR"
aws s3 cp "${S3_BASE}/setup/vjepa2-src.tar.gz" "${WORKDIR}/vjepa2-src.tar.gz" --quiet
tar xzf "${WORKDIR}/vjepa2-src.tar.gz" -C "$CODE_DIR"
cd "$CODE_DIR"
pip install -e . --no-deps --no-build-isolation 2>/dev/null || true
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

FEAT_DIR="${WORKDIR}/features"
RES_DIR="${WORKDIR}/results"
mkdir -p "$FEAT_DIR" "$RES_DIR"

# Pull all 4 pairs of caches
for MODEL in jepa_e100 jepa_e200 mae_e99 mae_e194; do
  for SPLIT in train test; do
    aws s3 cp "${FEATURES_S3}/${MODEL}_${SPLIT}.pt" "${FEAT_DIR}/${MODEL}_${SPLIT}.pt" --only-show-errors
    ls -lh "${FEAT_DIR}/${MODEL}_${SPLIT}.pt"
  done
done

run_probe() {
  local M="$1"
  local GPU="$2"
  CUDA_VISIBLE_DEVICES="${GPU}" python3 -u scripts/neurips/diff_probe_train.py \
    --train-cache "${FEAT_DIR}/${M}_train.pt" \
    --test-cache  "${FEAT_DIR}/${M}_test.pt" \
    --model "$M" \
    --out-json "${RES_DIR}/${M}.json" \
    --archs linear-A --inputs raw \
    --seeds 0 1 2 3 4 --wd-sweep 1e-4 1e-2 \
    > "/opt/dlami/nvme/logs/lin_train_probe_${M}_${SLURM_JOB_ID}.out" 2>&1
}

# Launch 4 probes in parallel on GPUs 0-3
run_probe jepa_e100 0 &
run_probe jepa_e200 1 &
run_probe mae_e99   2 &
run_probe mae_e194  3 &
wait

# Summary
python3 -u - <<PYEOF
import json, glob, os
import numpy as np
res_dir = "${RES_DIR}"
print("\n========== LINEAR-A RAW CLEAN TEST R² (mean ± std over 5 seeds × 2 WDs = 10 runs) ==========")
print(f"{'model':<15} {'mean':>8} {'std':>8} {'n':>6}")
for f in sorted(glob.glob(os.path.join(res_dir, "*.json"))):
    model = os.path.basename(f).replace('.json', '')
    try:
        d = json.load(open(f))
    except Exception as e:
        print(f"{model}: ERROR {e}"); continue
    r2s = [r['test_r2'] for r in d.get('results', [])
           if r.get('arch') == 'linear-A' and r.get('input') == 'raw']
    if r2s:
        a = np.array(r2s)
        print(f"{model:<15} {a.mean():>8.3f} {a.std():>8.3f} {len(a):>6}")
    else:
        print(f"{model}: no linear-A raw results")
PYEOF

aws s3 sync "${RES_DIR}/" "${RUN_S3}/results/" --only-show-errors
echo "[DONE] linear-probe training"
