#!/bin/bash
# Priority B + C: cross-view retrieval diagnostic sweep.
# Runs after MV2SV pilot 655 finishes (chain via --dependency=afterany:655).
#
# Sweep:
#   - Base e100 (canonical, jepa_in21k_vitl_e100.pt)
#   - Base +25 e125 (jepa_in21k_e200_280/e125.pt, matched compute SV)
#   - V3 fb_phase_542 e25 (positive-only cross-view)
#   - V4 phase-rel 593 e5 and e25 (phase-relational InfoNCE)
#   - Control 608 e5 and e25 (paired intraview-only)
#   - MV2SV pilot 655 e1, e3, e5 (all MV2SV feature modes)
#   - MV2SV intraview-only ctrl 656 e5 (will be present when this fires)
#
# feature modes:
#   standard V-JEPA ckpts -> encoder_pool only
#   MV2SV ckpts (655/656) -> encoder_pool + z_shared + z_view + concat_shared_phase + concat_all
#
#SBATCH -p dev
#SBATCH --nodelist=ip-10-0-50-146
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 0-04:00:00
#SBATCH -J mv2sv_diag_sweep
#SBATCH -o /opt/dlami/nvme/logs/mv2sv_diag_sweep-%j.out
#SBATCH -e /opt/dlami/nvme/logs/mv2sv_diag_sweep-%j.err
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
SRC_TARBALL_S3="${S3_BASE}/setup/vjepa2-src-mv2sv-ctrl.tar.gz"  # contains the diag script

REPO_DIR="/opt/dlami/nvme/src/vjepa2_diag"
RUN_S3="${S3_BASE}/runs/mv2sv_retrieval_diag_${SLURM_JOB_ID}"
OUT_DIR="/opt/dlami/nvme/retrieval_diag_${SLURM_JOB_ID}"
JOB_LOG="/opt/dlami/nvme/logs/mv2sv_diag_sweep_${SLURM_JOB_ID}.log"

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

# Stage data (cache hits on NVMe if a previous job already pulled them)
mkdir -p /opt/dlami/nvme/probe /opt/dlami/nvme/data/view_labels
[ ! -f /opt/dlami/nvme/probe/phase_annotations.parquet ] && aws s3 cp "${S3_BASE}/data/phase_annotations/phase_annotations.parquet" /opt/dlami/nvme/probe/phase_annotations.parquet --only-show-errors
[ ! -f /opt/dlami/nvme/data/view_labels/mimic_view_predictions.csv ] && aws s3 cp "${S3_BASE}/data/view_labels/mimic_view_predictions.csv" /opt/dlami/nvme/data/view_labels/mimic_view_predictions.csv --only-show-errors

CKPT_DIR="/opt/dlami/nvme/diag_ckpts_${SLURM_JOB_ID}"
mkdir -p "$CKPT_DIR"

# -----------------------------------------------------------------------------
# run_one <tag> <ckpt_local_path> <feature-mode>
# -----------------------------------------------------------------------------
run_one() {
  local tag="$1"
  local ckpt="$2"
  local mode="$3"
  local out="${OUT_DIR}/${tag}__${mode}.json"
  echo "=== [${tag}] feature_mode=${mode} ckpt=${ckpt} ==="
  python scripts/neurips/phase/run_cross_view_retrieval_diag.py \
    --checkpoint "$ckpt" \
    --feature-mode "$mode" \
    --source-views A4C \
    --target-views A2C,A5C,A3C,PLAX,PSAX-MV \
    --num-studies 200 \
    --out "$out" 2>&1 | tail -40 || { echo "[WARN] ${tag}/${mode} failed; continuing"; rm -f "$out"; }
  # Upload incrementally so partial failures don't lose prior results.
  aws s3 cp "$out" "${RUN_S3}/$(basename "$out")" --only-show-errors 2>/dev/null || true
}

# -----------------------------------------------------------------------------
# fetch <s3_uri> <local_path>  — idempotent cache
# -----------------------------------------------------------------------------
fetch() {
  local src="$1"; local dst="$2"
  [ -f "$dst" ] || aws s3 cp "$src" "$dst" --only-show-errors
}

# --- Standard V-JEPA ckpts (encoder_pool only) -------------------------------
fetch "${S3_BASE}/checkpoints/jepa_in21k/jepa_in21k_vitl_e100.pt" "${CKPT_DIR}/base_e100.pt"
run_one "base_e100" "${CKPT_DIR}/base_e100.pt" "encoder_pool"

fetch "${S3_BASE}/runs/jepa_in21k_e200_280/training_folder/e125.pt" "${CKPT_DIR}/base_e125.pt"
run_one "base_e125" "${CKPT_DIR}/base_e125.pt" "encoder_pool"

fetch "${S3_BASE}/runs/final_sv_25_pairs24_548/checkpoints/latest.pt" "${CKPT_DIR}/fb_sv_548_e25.pt"
run_one "fb_sv_548_e25" "${CKPT_DIR}/fb_sv_548_e25.pt" "encoder_pool"

fetch "${S3_BASE}/runs/final_phase_25_pairs24_542/checkpoints/e5.pt" "${CKPT_DIR}/v3_e5.pt"
run_one "v3_fb_phase_542_e5" "${CKPT_DIR}/v3_e5.pt" "encoder_pool"
fetch "${S3_BASE}/runs/final_phase_25_pairs24_542/checkpoints/latest.pt" "${CKPT_DIR}/v3_e25.pt"
run_one "v3_fb_phase_542_e25" "${CKPT_DIR}/v3_e25.pt" "encoder_pool"

fetch "${S3_BASE}/runs/final_phase_rel25_paper_593/checkpoints/e5.pt" "${CKPT_DIR}/v4_e5.pt"
run_one "v4_phase_rel_593_e5" "${CKPT_DIR}/v4_e5.pt" "encoder_pool"
fetch "${S3_BASE}/runs/final_phase_rel25_paper_593/checkpoints/latest.pt" "${CKPT_DIR}/v4_e25.pt"
run_one "v4_phase_rel_593_e25" "${CKPT_DIR}/v4_e25.pt" "encoder_pool"

fetch "${S3_BASE}/runs/final_paired_iv25_paper_608/checkpoints/e5.pt" "${CKPT_DIR}/ctrl608_e5.pt"
run_one "ctrl_608_e5" "${CKPT_DIR}/ctrl608_e5.pt" "encoder_pool"
fetch "${S3_BASE}/runs/final_paired_iv25_paper_608/checkpoints/latest.pt" "${CKPT_DIR}/ctrl608_e25.pt"
run_one "ctrl_608_e25" "${CKPT_DIR}/ctrl608_e25.pt" "encoder_pool"

# 613 (no-hardneg ablation) — only if e5+ ckpt exists; otherwise skip.
if aws s3 ls "${S3_BASE}/runs/final_phase_rel_nohardneg25_paper_613/checkpoints/e5.pt" >/dev/null 2>&1; then
  fetch "${S3_BASE}/runs/final_phase_rel_nohardneg25_paper_613/checkpoints/e5.pt" "${CKPT_DIR}/noHN_e5.pt"
  run_one "noHN_613_e5" "${CKPT_DIR}/noHN_e5.pt" "encoder_pool"
else
  echo "[INFO] 613 no-hardneg e5 not yet on S3; skipping"
fi

# --- MV2SV pilot 655 (all feature modes) -------------------------------------
# Sync from the node-local run dir if available, else from S3 (pilot uploads
# e{N}.pt to RUN_S3/ckpt/ via upload_artifacts at each trap).
PILOT_S3="${S3_BASE}/runs/mv2sv_pilot_5ep_655/ckpt"
for EP in 1 3 5; do
  if aws s3 ls "${PILOT_S3}/e${EP}.pt" >/dev/null 2>&1; then
    fetch "${PILOT_S3}/e${EP}.pt" "${CKPT_DIR}/pilot_e${EP}.pt"
    for MODE in encoder_pool z_shared z_view concat_shared_phase concat_all; do
      run_one "mv2sv_pilot_655_e${EP}" "${CKPT_DIR}/pilot_e${EP}.pt" "$MODE"
    done
  else
    echo "[INFO] pilot e${EP} not yet on S3; skipping"
  fi
done

# --- MV2SV intraview-only control 656 (encoder_pool only; all aux lambdas=0 so
#     factorized_head never trained, slot modes don't make sense) -------------
CTRL_S3="${S3_BASE}/runs/mv2sv_ctrl_5ep_656/ckpt"
for EP in 1 3 5; do
  if aws s3 ls "${CTRL_S3}/e${EP}.pt" >/dev/null 2>&1; then
    fetch "${CTRL_S3}/e${EP}.pt" "${CKPT_DIR}/ctrl_mv2sv_e${EP}.pt"
    run_one "mv2sv_ctrl_656_e${EP}" "${CKPT_DIR}/ctrl_mv2sv_e${EP}.pt" "encoder_pool"
  else
    echo "[INFO] mv2sv ctrl 656 e${EP} not yet on S3; skipping"
  fi
done

echo "[INFO] === retrieval diag sweep complete ==="
ls -la "$OUT_DIR"
