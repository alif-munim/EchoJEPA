#!/bin/bash
# One-shot smoke test for run_cross_view_retrieval_diag.py.
# Runs a single checkpoint at num-studies=10 to verify the pipeline
# end-to-end (sampler build, encode, retrieval, slot geometry, view
# classifier) before the full diag sweep fires.
#
#SBATCH -p dev
#SBATCH --nodelist=ip-10-0-50-146
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -t 0-00:20:00
#SBATCH -J mv2sv_diag_smoke
#SBATCH -o /opt/dlami/nvme/logs/mv2sv_diag_smoke-%j.out
#SBATCH -e /opt/dlami/nvme/logs/mv2sv_diag_smoke-%j.err
#SBATCH --chdir=/opt/dlami/nvme

set -euxo pipefail

export PATH="/opt/vjepa2-312/bin:$PATH"
export TMPDIR="/opt/dlami/nvme/tmp"
mkdir -p "$TMPDIR" /opt/dlami/nvme/logs
export AWS_DEFAULT_REGION="us-west-2"
export AWS_REGION="us-west-2"
export HOME="/home/ubuntu"
export AWS_SHARED_CREDENTIALS_FILE="${HOME}/.aws/credentials"
export AWS_CONFIG_FILE="${HOME}/.aws/config"
export LD_LIBRARY_PATH="/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cublas/lib:/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"

ART_BUCKET="sagemaker-hyperpod-lifecycle-495467399120-usw2"
S3_BASE="s3://${ART_BUCKET}/vjepa2-artifacts"
RUN_S3="${S3_BASE}/runs/mv2sv_diag_smoke_${SLURM_JOB_ID}"

REPO_DIR="/opt/dlami/nvme/src/vjepa2_smoke"
rm -rf "$REPO_DIR"; mkdir -p "$REPO_DIR"
aws s3 cp "${S3_BASE}/setup/vjepa2-src-mv2sv-ctrl.tar.gz" "${TMPDIR}/src_smoke.tar.gz" --only-show-errors
tar xzf "${TMPDIR}/src_smoke.tar.gz" -C "$REPO_DIR"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:/opt/dlami/nvme/pyarrow_site:${PYTHONPATH:-}"

if ! python -c "import pyarrow" 2>/dev/null; then
  pip install --target=/opt/dlami/nvme/pyarrow_site --no-cache-dir pyarrow 2>&1 | tail -3
fi

mkdir -p /opt/dlami/nvme/probe /opt/dlami/nvme/data/view_labels
[ ! -f /opt/dlami/nvme/probe/phase_annotations.parquet ] && aws s3 cp "${S3_BASE}/data/phase_annotations/phase_annotations.parquet" /opt/dlami/nvme/probe/phase_annotations.parquet --only-show-errors
[ ! -f /opt/dlami/nvme/data/view_labels/mimic_view_predictions.csv ] && aws s3 cp "${S3_BASE}/data/view_labels/mimic_view_predictions.csv" /opt/dlami/nvme/data/view_labels/mimic_view_predictions.csv --only-show-errors
[ ! -f /opt/dlami/nvme/checkpoints/jepa_in21k_vitl_e100.pt ] && aws s3 cp "${S3_BASE}/checkpoints/jepa_in21k/jepa_in21k_vitl_e100.pt" /opt/dlami/nvme/checkpoints/jepa_in21k_vitl_e100.pt --only-show-errors

OUT_DIR="/opt/dlami/nvme/diag_smoke_${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR"

# Run the diag on Base e100 with just 10 studies. Exercise every new
# code path: slot-geometry (no factorized head, so geometry still runs
# on encoder_pool), view-classifier, both retrieval modes.
python scripts/neurips/phase/run_cross_view_retrieval_diag.py \
  --checkpoint /opt/dlami/nvme/checkpoints/jepa_in21k_vitl_e100.pt \
  --feature-mode encoder_pool \
  --retrieval-mode same_study_same_view \
  --slot-geometry \
  --view-classifier \
  --source-views A4C \
  --target-views A2C,A5C,A3C,PLAX,PSAX-MV \
  --num-studies 10 \
  --out "$OUT_DIR/base_e100_ssv.json"

python scripts/neurips/phase/run_cross_view_retrieval_diag.py \
  --checkpoint /opt/dlami/nvme/checkpoints/jepa_in21k_vitl_e100.pt \
  --feature-mode encoder_pool \
  --retrieval-mode same_study_any_view \
  --slot-geometry \
  --source-views A4C \
  --target-views A2C,A5C,A3C,PLAX,PSAX-MV \
  --num-studies 10 \
  --out "$OUT_DIR/base_e100_sav.json"

echo "=== smoke outputs ==="
ls -la "$OUT_DIR"
for f in "$OUT_DIR"/*.json; do
  echo "--- $f ---"
  python3 -c "import json; d=json.load(open('$f')); import pprint; pprint.pprint({k: (v if not isinstance(v,dict) else {kk:vv for kk,vv in list(v.items())[:5]}) for k,v in d.items()})"
done

aws s3 sync "$OUT_DIR" "${RUN_S3}/" --only-show-errors
echo "[INFO] smoke complete"
