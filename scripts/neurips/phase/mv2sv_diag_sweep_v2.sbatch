#!/bin/bash
# v2 diag sweep — extends the retrieval run with:
#   (C) --retrieval-mode same_study_any_view   (study-identity vs view-specific)
#   (D) --slot-geometry                          (B/C/D cosine stats)
#   (B-probe) --view-classifier                  (catches "learned a view classifier")
# Also picks up MV2SV ctrl 658 checkpoints (not present when 660 ran).
#
# One run per (checkpoint, feature-mode, retrieval-mode).
#
#SBATCH -p dev
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 0-05:00:00
#SBATCH -J mv2sv_diag_v2
#SBATCH -o /opt/dlami/nvme/logs/mv2sv_diag_v2-%j.out
#SBATCH -e /opt/dlami/nvme/logs/mv2sv_diag_v2-%j.err
#SBATCH --chdir=/opt/dlami/nvme

set -euxo pipefail

export PATH="/opt/vjepa2-312/bin:$PATH"
export TMPDIR="/opt/dlami/nvme/tmp"
mkdir -p "$TMPDIR" /opt/dlami/nvme/logs

export AWS_DEFAULT_REGION="us-west-2"
export AWS_REGION="us-west-2"
export AWS_SDK_LOAD_CONFIG=1
export HOME="/home/ubuntu"
export AWS_SHARED_CREDENTIALS_FILE="${HOME}/.aws/credentials"
export AWS_CONFIG_FILE="${HOME}/.aws/config"
unset AWS_EC2_METADATA_DISABLED 2>/dev/null || true
export LD_LIBRARY_PATH="/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cublas/lib:/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"

ART_BUCKET="sagemaker-hyperpod-lifecycle-495467399120-usw2"
S3_BASE="s3://${ART_BUCKET}/vjepa2-artifacts"
SRC_TARBALL_S3="${S3_BASE}/setup/vjepa2-src-mv2sv-ctrl.tar.gz"

REPO_DIR="/opt/dlami/nvme/src/vjepa2_diag_v2"
RUN_S3="${S3_BASE}/runs/mv2sv_retrieval_diag_v2_${SLURM_JOB_ID}"
OUT_DIR="/opt/dlami/nvme/retrieval_diag_v2_${SLURM_JOB_ID}"
JOB_LOG="/opt/dlami/nvme/logs/mv2sv_diag_v2_${SLURM_JOB_ID}.log"

upload_artifacts() {
  [ -d "$OUT_DIR" ] && aws s3 sync "$OUT_DIR" "${RUN_S3}/" --only-show-errors 2>/dev/null || true
  [ -f "$JOB_LOG" ] && aws s3 cp "$JOB_LOG" "${RUN_S3}/logs/job.log" --only-show-errors 2>/dev/null || true
}
trap 'rc=$?; upload_artifacts; exit $rc' EXIT

echo "[INFO] host=$(hostname) job=${SLURM_JOB_ID}"
nvidia-smi -L || true

mkdir -p "$REPO_DIR" "$OUT_DIR"
rm -rf "$REPO_DIR"/*
aws s3 cp "$SRC_TARBALL_S3" "${TMPDIR}/src.tar.gz" --only-show-errors
tar xzf "${TMPDIR}/src.tar.gz" -C "$REPO_DIR"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:/opt/dlami/nvme/pyarrow_site:${PYTHONPATH:-}"

# Ensure pyarrow available (node-56-safe)
if ! python -c "import pyarrow" 2>/dev/null; then
  pip install --target=/opt/dlami/nvme/pyarrow_site --no-cache-dir pyarrow 2>&1 | tail -3
fi

# Stage data
mkdir -p /opt/dlami/nvme/probe /opt/dlami/nvme/data/view_labels
[ ! -f /opt/dlami/nvme/probe/phase_annotations.parquet ] && aws s3 cp "${S3_BASE}/data/phase_annotations/phase_annotations.parquet" /opt/dlami/nvme/probe/phase_annotations.parquet --only-show-errors
[ ! -f /opt/dlami/nvme/data/view_labels/mimic_view_predictions.csv ] && aws s3 cp "${S3_BASE}/data/view_labels/mimic_view_predictions.csv" /opt/dlami/nvme/data/view_labels/mimic_view_predictions.csv --only-show-errors

CKPT_DIR="/opt/dlami/nvme/diag_v2_ckpts_${SLURM_JOB_ID}"
mkdir -p "$CKPT_DIR"

fetch() {
  local src="$1"; local dst="$2"
  [ -f "$dst" ] || aws s3 cp "$src" "$dst" --only-show-errors
}

# run_one <tag> <ckpt> <feature-mode> <retrieval-mode> [--view-classifier]
# Always emits --slot-geometry.
run_one() {
  local tag="$1"; local ckpt="$2"; local mode="$3"; local ret="$4"; shift 4
  local extra="$*"
  local out="${OUT_DIR}/${tag}__${mode}__${ret}.json"
  echo "=== [${tag}] mode=${mode} retrieval=${ret} ${extra} ==="
  python scripts/neurips/phase/run_cross_view_retrieval_diag.py \
    --checkpoint "$ckpt" \
    --feature-mode "$mode" \
    --retrieval-mode "$ret" \
    --slot-geometry \
    $extra \
    --source-views A4C \
    --target-views A2C,A5C,A3C,PLAX,PSAX-MV \
    --num-studies 1000 \
    --out "$out" 2>&1 | tail -30 || { echo "[WARN] ${tag}/${mode}/${ret} failed"; rm -f "$out"; }
  [ -f "$out" ] && aws s3 cp "$out" "${RUN_S3}/$(basename "$out")" --only-show-errors 2>/dev/null || true
}

# Fetch ckpts in the same shape as v1 (ensures availability)
fetch "${S3_BASE}/checkpoints/jepa_in21k/jepa_in21k_vitl_e100.pt"                       "${CKPT_DIR}/base_e100.pt"
fetch "${S3_BASE}/runs/jepa_in21k_e200_280/training_folder/e125.pt"                     "${CKPT_DIR}/base_e125.pt"
fetch "${S3_BASE}/runs/final_sv_25_pairs24_548/checkpoints/latest.pt"                   "${CKPT_DIR}/fb_sv_548.pt"
fetch "${S3_BASE}/runs/final_phase_25_pairs24_542/checkpoints/latest.pt"                "${CKPT_DIR}/v3_e25.pt"
fetch "${S3_BASE}/runs/final_phase_rel25_paper_593/checkpoints/latest.pt"               "${CKPT_DIR}/v4_e25.pt"
fetch "${S3_BASE}/runs/final_paired_iv25_paper_608/checkpoints/latest.pt"               "${CKPT_DIR}/ctrl608_e25.pt"

# Standard V-JEPA ckpts: encoder_pool only, both retrieval modes + view classifier (once per ckpt)
for SPEC in \
  "base_e100:${CKPT_DIR}/base_e100.pt" \
  "base_e125:${CKPT_DIR}/base_e125.pt" \
  "fb_sv_548:${CKPT_DIR}/fb_sv_548.pt" \
  "v3_fb_phase_542_e25:${CKPT_DIR}/v3_e25.pt" \
  "v4_phase_rel_593_e25:${CKPT_DIR}/v4_e25.pt" \
  "ctrl_608_e25:${CKPT_DIR}/ctrl608_e25.pt"; do
  TAG="${SPEC%%:*}"; CKPT="${SPEC#*:}"
  run_one "$TAG" "$CKPT" encoder_pool same_study_same_view --view-classifier
  run_one "$TAG" "$CKPT" encoder_pool same_study_any_view
done

# MV2SV pilot 655 — all feature modes; view classifier only for encoder_pool (1 per ckpt)
PILOT_S3="${S3_BASE}/runs/mv2sv_pilot_5ep_655/ckpt"
for EP in 1 5; do
  if aws s3 ls "${PILOT_S3}/e${EP}.pt" >/dev/null 2>&1; then
    fetch "${PILOT_S3}/e${EP}.pt" "${CKPT_DIR}/pilot_e${EP}.pt"
    TAG="mv2sv_pilot_655_e${EP}"
    CKPT="${CKPT_DIR}/pilot_e${EP}.pt"
    run_one "$TAG" "$CKPT" encoder_pool same_study_same_view --view-classifier
    run_one "$TAG" "$CKPT" encoder_pool same_study_any_view
    for MODE in z_shared z_view concat_shared_phase concat_all; do
      run_one "$TAG" "$CKPT" "$MODE" same_study_same_view
      run_one "$TAG" "$CKPT" "$MODE" same_study_any_view
    done
  else
    echo "[INFO] pilot e${EP} not on S3; skipping"
  fi
done

# MV2SV intraview-only control 658 — encoder_pool only
CTRL_S3="${S3_BASE}/runs/mv2sv_ctrl_5ep_658/ckpt"
for EP in 1 5; do
  if aws s3 ls "${CTRL_S3}/e${EP}.pt" >/dev/null 2>&1; then
    fetch "${CTRL_S3}/e${EP}.pt" "${CKPT_DIR}/ctrl658_e${EP}.pt"
    TAG="mv2sv_ctrl_658_e${EP}"
    CKPT="${CKPT_DIR}/ctrl658_e${EP}.pt"
    run_one "$TAG" "$CKPT" encoder_pool same_study_same_view --view-classifier
    run_one "$TAG" "$CKPT" encoder_pool same_study_any_view
  else
    echo "[INFO] ctrl 658 e${EP} not on S3; skipping"
  fi
done

echo "[INFO] === retrieval diag v2 sweep complete ==="
ls -la "$OUT_DIR"
