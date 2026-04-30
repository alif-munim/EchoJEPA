#!/bin/bash
#SBATCH -p dev
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 01:00:00
#SBATCH -J pta_probe
#SBATCH -o /opt/dlami/nvme/logs/pta_probe_%j.out
#SBATCH -e /opt/dlami/nvme/logs/pta_probe_%j.err

set -euxo pipefail

# Pooled temporal-attention probe on cached [N, S=2, T=8, D] features.
# Bridges between linear-A raw and full-token attentive probes.
# Runs 4 models in parallel on 4 GPUs, 5 seeds x 3 LRs x 2 WDs each.

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
RUN_S3="${S3_BASE}/runs/pooled_temporal_attn_probe_verify_${SLURM_JOB_ID}"

upload_logs() {
  local o="/opt/dlami/nvme/logs/pta_probe_${SLURM_JOB_ID}.out"
  local e="/opt/dlami/nvme/logs/pta_probe_${SLURM_JOB_ID}.err"
  [ -f "$o" ] && aws s3 cp "$o" "${RUN_S3}/logs/job.out" --only-show-errors || true
  [ -f "$e" ] && aws s3 cp "$e" "${RUN_S3}/logs/job.err" --only-show-errors || true
}
trap 'upload_logs' EXIT

mkdir -p /opt/dlami/nvme/logs
SCRATCH="/opt/dlami/nvme"
WORKDIR="${SCRATCH}/pta_probe_${SLURM_JOB_ID}"
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

for M in jepa_e100 jepa_e200 mae_e99 mae_e194; do
  for SPLIT in train test; do
    aws s3 cp "${FEATURES_S3}/${M}_${SPLIT}.pt" "${FEAT_DIR}/${M}_${SPLIT}.pt" --only-show-errors
  done
done

run_probe() {
  local M="$1"
  local GPU="$2"
  CUDA_VISIBLE_DEVICES="${GPU}" python3 -u scripts/neurips/pooled_temporal_attn_probe_train.py \
    --train-cache "${FEAT_DIR}/${M}_train.pt" \
    --test-cache  "${FEAT_DIR}/${M}_test.pt" \
    --model "$M" \
    --out-json "${RES_DIR}/${M}.json" \
    --seeds 0 1 2 3 4 \
    --lrs 1e-4 3e-4 1e-3 \
    --wds 1e-4 1e-2 \
    --n-heads 8 --mlp-ratio 4 --dropout 0.1 \
    --batch-size 64 --max-epochs 50 --min-epochs 15 --patience 7 \
    > "/opt/dlami/nvme/logs/pta_probe_${M}_${SLURM_JOB_ID}.out" 2>&1
}

run_probe jepa_e100 0 &
run_probe jepa_e200 1 &
run_probe mae_e99   2 &
run_probe mae_e194  3 &
wait

python3 -u - <<PYEOF
import json, glob, os
import numpy as np
res_dir = "${RES_DIR}"
print("\n========== POOLED TEMPORAL-ATTN  CLEAN TEST R²  (best HP per seed, 5 seeds) ==========")
print(f"{'model':<15} {'r2_mean':>8} {'r2_std':>8}  {'mae_mean':>9}  {'pearson':>9}")
for f in sorted(glob.glob(os.path.join(res_dir, "*.json"))):
    model = os.path.basename(f).replace('.json', '')
    d = json.load(open(f))
    s = d.get('summary', {})
    print(f"{model:<15} {s.get('test_r2_mean', 0.0):>8.3f} {s.get('test_r2_std', 0.0):>8.3f}  "
          f"{s.get('test_mae_mean', 0.0):>9.3f}  {s.get('test_pearson_mean', 0.0):>9.3f}")
PYEOF

aws s3 sync "${RES_DIR}/" "${RUN_S3}/results/" --only-show-errors
echo "[DONE] pooled-temporal-attn probe"
