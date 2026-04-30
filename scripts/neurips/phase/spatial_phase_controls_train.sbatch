#!/bin/bash
#SBATCH -p dev
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 02:30:00
#SBATCH -J phase_ctrl
#SBATCH -o /opt/dlami/nvme/logs/phase_ctrl_%j.out
#SBATCH -e /opt/dlami/nvme/logs/phase_ctrl_%j.err

set -euxo pipefail

# Spatial / phase / temporal-information controls on cached pooled features.
# Runs 4 models (jepa_e100, jepa_e200, mae_e99, mae_e194) in parallel on 4 GPUs.
# Per model:
#   - 10 linear-A variants: raw, diff, mean, single_t0..single_t7
#     (5 seeds × 2 WDs each)
#   - pta_mean_repeat (5 seeds × 3 LRs × 2 WDs)
#   - pta_best_repeat (5 seeds × 3 LRs × 2 WDs, best token by val per seed)
# ~ 100 train-from-scratch probes per model, ~ 400 total.

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
RUN_S3="${S3_BASE}/runs/spatial_phase_controls_${SLURM_JOB_ID}"

upload_logs() {
  local o="/opt/dlami/nvme/logs/phase_ctrl_${SLURM_JOB_ID}.out"
  local e="/opt/dlami/nvme/logs/phase_ctrl_${SLURM_JOB_ID}.err"
  [ -f "$o" ] && aws s3 cp "$o" "${RUN_S3}/logs/job.out" --only-show-errors || true
  [ -f "$e" ] && aws s3 cp "$e" "${RUN_S3}/logs/job.err" --only-show-errors || true
}
trap 'upload_logs' EXIT

mkdir -p /opt/dlami/nvme/logs
SCRATCH="/opt/dlami/nvme"
WORKDIR="${SCRATCH}/phase_ctrl_${SLURM_JOB_ID}"
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

run_ctrl() {
  local M="$1"
  local GPU="$2"
  CUDA_VISIBLE_DEVICES="${GPU}" python3 -u scripts/neurips/spatial_phase_controls_train.py \
    --train-cache "${FEAT_DIR}/${M}_train.pt" \
    --test-cache  "${FEAT_DIR}/${M}_test.pt" \
    --model "$M" \
    --out-json "${RES_DIR}/${M}.json" \
    --seeds 0 1 2 3 4 \
    --batch-size 64 \
    > "/opt/dlami/nvme/logs/phase_ctrl_${M}_${SLURM_JOB_ID}.out" 2>&1
}

run_ctrl jepa_e100 0 &
run_ctrl jepa_e200 1 &
run_ctrl mae_e99   2 &
run_ctrl mae_e194  3 &
wait

python3 -u - <<PYEOF
import json, glob, os
import numpy as np
res_dir = "${RES_DIR}"
print("\n" + "="*120)
print("SPATIAL / PHASE / TEMPORAL CONTROLS — test R² (5-seed mean ± std)")
print("="*120)
hdr = f"{'model':<15} {'lin_raw':>11} {'lin_diff':>11} {'lin_mean':>11} {'lin_tavg':>11} {'lin_tbest':>11} {'pta_meanR':>11} {'pta_bestR':>11}"
print(hdr)
print("-" * len(hdr))
for f in sorted(glob.glob(os.path.join(res_dir, "*.json"))):
    model = os.path.basename(f).replace('.json','')
    d = json.load(open(f))
    s = d.get('summaries', {})
    def ms(key, mean_key='test_r2_mean', std_key='test_r2_std'):
        v = s.get(key)
        if not v:
            return "     —     "
        if isinstance(v, list):
            return "     —     "
        if 'test_r2_mean_of_tokens' in v:
            return f"{v['test_r2_mean_of_tokens']:>5.3f}±{v['test_r2_std_of_tokens']:>4.3f}"
        return f"{v[mean_key]:>5.3f}±{v[std_key]:>4.3f}"
    print(f"{model:<15} {ms('linear_raw'):>11} {ms('linear_diff'):>11} {ms('linear_mean'):>11} "
          f"{ms('linear_single_t_avg'):>11} {ms('linear_single_t_best_val'):>11} "
          f"{ms('pta_mean_repeat'):>11} {ms('pta_best_repeat'):>11}")

print()
print("="*60)
print("PER-TOKEN LINEAR R² (best WD per seed, 5-seed mean ± std)")
print("="*60)
for f in sorted(glob.glob(os.path.join(res_dir, "*.json"))):
    model = os.path.basename(f).replace('.json','')
    d = json.load(open(f))
    per_tok = d.get('summaries', {}).get('linear_single_t_per_token', [])
    row_str = " ".join([f"t{r['token_index']}={r['test_r2_mean']:.3f}" for r in per_tok])
    print(f"  {model:<15} {row_str}")
PYEOF

aws s3 sync "${RES_DIR}/" "${RUN_S3}/results/" --only-show-errors
echo "[DONE] spatial/phase controls"
