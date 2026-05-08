#!/bin/bash
# F/G/H sweep:
#   F — z_phase (or encoder_pool fallback) phase-decodability linear probe
#   G — frame-shuffle sensitivity (cos(clean, shuffled))
#   H — intra-clip temporal dissimilarity (cos(first-half, second-half))
#
# Runs on the idle compute node (146) while ctrl 658 is still running on 56.
# ~1.5h budget; 200 studies per ckpt, 4 encodes per study.
#
#SBATCH -p dev
#SBATCH --nodelist=ip-10-0-50-146
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -t 0-02:00:00
#SBATCH -J mv2sv_fgh
#SBATCH -o /opt/dlami/nvme/logs/mv2sv_fgh-%j.out
#SBATCH -e /opt/dlami/nvme/logs/mv2sv_fgh-%j.err
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
RUN_S3="${S3_BASE}/runs/mv2sv_fgh_${SLURM_JOB_ID}"

REPO_DIR="/opt/dlami/nvme/src/vjepa2_fgh"
rm -rf "$REPO_DIR"; mkdir -p "$REPO_DIR"
aws s3 cp "${S3_BASE}/setup/vjepa2-src-mv2sv-ctrl.tar.gz" "${TMPDIR}/src_fgh.tar.gz" --only-show-errors
tar xzf "${TMPDIR}/src_fgh.tar.gz" -C "$REPO_DIR"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:/opt/dlami/nvme/pyarrow_site:${PYTHONPATH:-}"

if ! python -c "import pyarrow" 2>/dev/null; then
  pip install --target=/opt/dlami/nvme/pyarrow_site --no-cache-dir pyarrow 2>&1 | tail -3
fi

mkdir -p /opt/dlami/nvme/probe /opt/dlami/nvme/data/view_labels
[ ! -f /opt/dlami/nvme/probe/phase_annotations.parquet ] && aws s3 cp "${S3_BASE}/data/phase_annotations/phase_annotations.parquet" /opt/dlami/nvme/probe/phase_annotations.parquet --only-show-errors
[ ! -f /opt/dlami/nvme/data/view_labels/mimic_view_predictions.csv ] && aws s3 cp "${S3_BASE}/data/view_labels/mimic_view_predictions.csv" /opt/dlami/nvme/data/view_labels/mimic_view_predictions.csv --only-show-errors

CKPT_DIR="/opt/dlami/nvme/fgh_ckpts_${SLURM_JOB_ID}"
mkdir -p "$CKPT_DIR"
OUT_DIR="/opt/dlami/nvme/fgh_${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR"

upload_artifacts() {
  [ -d "$OUT_DIR" ] && aws s3 sync "$OUT_DIR" "${RUN_S3}/" --only-show-errors 2>/dev/null || true
}
trap 'rc=$?; upload_artifacts; exit $rc' EXIT

fetch() {
  local src="$1"; local dst="$2"
  [ -f "$dst" ] || aws s3 cp "$src" "$dst" --only-show-errors
}

run_one() {
  local tag="$1"; local ckpt="$2"; local mode="$3"; local phase_flag="$4"
  local out="${OUT_DIR}/${tag}__${mode}.json"
  echo "=== [${tag}] mode=${mode} phase_probe=${phase_flag} ==="
  python scripts/neurips/phase/run_temporal_phase_diagnostics.py \
    --checkpoint "$ckpt" --feature-mode "$mode" \
    --num-studies 200 --out "$out" $phase_flag 2>&1 | tail -20 || \
    { echo "[WARN] ${tag}/${mode} failed"; rm -f "$out"; }
  [ -f "$out" ] && aws s3 cp "$out" "${RUN_S3}/$(basename "$out")" --only-show-errors 2>/dev/null || true
}

# --- Standard V-JEPA ckpts: encoder_pool + --also-phase-probe (falls back to pool) ---
fetch "${S3_BASE}/checkpoints/jepa_in21k/jepa_in21k_vitl_e100.pt"                       "${CKPT_DIR}/base_e100.pt"
fetch "${S3_BASE}/runs/final_phase_rel25_paper_593/checkpoints/latest.pt"               "${CKPT_DIR}/v4_e25.pt"
fetch "${S3_BASE}/runs/final_paired_iv25_paper_608/checkpoints/latest.pt"               "${CKPT_DIR}/ctrl608_e25.pt"

run_one "base_e100"         "${CKPT_DIR}/base_e100.pt"    encoder_pool --also-phase-probe
run_one "v4_phase_rel_593"  "${CKPT_DIR}/v4_e25.pt"       encoder_pool --also-phase-probe
run_one "ctrl_608"          "${CKPT_DIR}/ctrl608_e25.pt"  encoder_pool --also-phase-probe

# --- MV2SV pilot 655 e5: BOTH encoder_pool and z_phase (F) ---
PILOT_S3="${S3_BASE}/runs/mv2sv_pilot_5ep_655/ckpt"
if aws s3 ls "${PILOT_S3}/e5.pt" >/dev/null 2>&1; then
  fetch "${PILOT_S3}/e5.pt" "${CKPT_DIR}/pilot_e5.pt"
  run_one "mv2sv_pilot_655_e5" "${CKPT_DIR}/pilot_e5.pt" encoder_pool --also-phase-probe
  run_one "mv2sv_pilot_655_e5" "${CKPT_DIR}/pilot_e5.pt" z_phase      --also-phase-probe
  run_one "mv2sv_pilot_655_e5" "${CKPT_DIR}/pilot_e5.pt" z_shared     ""
  run_one "mv2sv_pilot_655_e5" "${CKPT_DIR}/pilot_e5.pt" z_view       ""
else
  echo "[INFO] pilot e5 not yet on S3"
fi

echo "[INFO] === F/G/H sweep complete ==="
ls -la "$OUT_DIR"
