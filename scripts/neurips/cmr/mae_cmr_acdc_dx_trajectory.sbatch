#!/bin/bash
#SBATCH -p dev
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 1-00:00:00
#SBATCH -J cmr_dx
#SBATCH -o /opt/dlami/nvme/logs/cmr_dx_%j.out
#SBATCH -e /opt/dlami/nvme/logs/cmr_dx_%j.err
#SBATCH --signal=B:USR1@300

set -euxo pipefail

# =============================================================================
# VideoMAE ViT-S CMR → ACDC 5-class diagnosis: probe training + clean/matched_frame
# =============================================================================
# Phase 1: Train d=4 attentive probes at 8 MAE checkpoints (e30-e800)
# Phase 2: Run clean + matched_frame inference for each trained probe
#
# ACDC official challenge split: 100 train / 50 test patients
# Classes: DCM=0, HCM=1, MINF=2, NOR=3, RV=4 (balanced, 20 train + 10 test each)
# Predictions CSVs saved for per-class post-hoc analysis.
# =============================================================================

export AWS_DEFAULT_REGION="us-west-2"
export AWS_REGION="us-west-2"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export HOME="/home/ubuntu"
# Use instance metadata service directly — do NOT set AWS_SHARED_CREDENTIALS_FILE
unset AWS_SHARED_CREDENTIALS_FILE 2>/dev/null || true
unset AWS_CONFIG_FILE 2>/dev/null || true
unset AWS_EC2_METADATA_DISABLED 2>/dev/null || true
export AWS_METADATA_SERVICE_TIMEOUT=10
export AWS_METADATA_SERVICE_NUM_ATTEMPTS=5

aws sts get-caller-identity || exit 1

ART_BUCKET="sagemaker-hyperpod-lifecycle-495467399120-usw2"
S3_BASE="s3://${ART_BUCKET}/vjepa2-artifacts"
ENV_S3="${S3_BASE}/env/vjepa2-312-pack-env.tar"
CKPT_S3="${S3_BASE}/runs/mae_cmr_vits_183/training_folder"
RUN_S3="${S3_BASE}/runs/cmr_dx_traj_${SLURM_JOB_ID}"

EPOCHS="30 50 100 200 300 400 600 800"

upload_logs() {
  local out="/opt/dlami/nvme/logs/cmr_dx_${SLURM_JOB_ID}.out"
  local err="/opt/dlami/nvme/logs/cmr_dx_${SLURM_JOB_ID}.err"
  [ -f "${out}" ] && aws s3 cp "${out}" "${RUN_S3}/logs/job.out" --only-show-errors || true
  [ -f "${err}" ] && aws s3 cp "${err}" "${RUN_S3}/logs/job.err" --only-show-errors || true
}

sync_results() {
  for ep in ${EPOCHS}; do
    for suffix in acdc_dx acdc_dx_clean acdc_dx_mf; do
      local outdir="/opt/dlami/nvme/evals/cmr/mae_e${ep}_${suffix}"
      if [ -d "${outdir}" ]; then
        aws s3 sync "${outdir}" "${RUN_S3}/mae_e${ep}_${suffix}/" \
          --exclude "*" --include "*.csv" --include "*.json" --include "*.pt" --include "*.npz" \
          --only-show-errors || true
      fi
    done
  done
}

on_exit() { rc=$?; upload_logs; sync_results; exit $rc; }
trap on_exit EXIT
trap 'sync_results; upload_logs' USR1

# =========================
# SETUP
# =========================
nvidia-smi -L || true
pkill -9 -f "evals.main" 2>/dev/null || true; sleep 2
mkdir -p /opt/dlami/nvme/logs

SCRATCH="/opt/dlami/nvme"

aws s3 cp "$ENV_S3" "${SCRATCH}/env.tar"
mkdir -p "${SCRATCH}/env" && tar -xf "${SCRATCH}/env.tar" -C "${SCRATCH}/env"
set +u && source "${SCRATCH}/env/bin/activate" && set -u
"${SCRATCH}/env/bin/conda-unpack" || true
pip install boto3 scikit-learn --no-cache-dir --quiet

CODE_DIR="${SCRATCH}/code"
mkdir -p "$CODE_DIR"
aws s3 cp "${S3_BASE}/setup/vjepa2-src.tar.gz" "${SCRATCH}/vjepa2-src.tar.gz" --quiet
tar xzf "${SCRATCH}/vjepa2-src.tar.gz" -C "$CODE_DIR"
cd "$CODE_DIR"
pip install -e . --no-deps --no-build-isolation 2>/dev/null || true

mkdir -p "${SCRATCH}/data/csv" "${SCRATCH}/data/acdc_clips" "${SCRATCH}/checkpoints" "${SCRATCH}/probes"

# =========================
# DOWNLOAD ACDC CLIPS LOCALLY (avoids S3 credential expiration during training)
# =========================
echo "[INFO] Downloading ACDC clips to local NVMe..."
aws s3 sync "${S3_BASE}/data/cmr/clips/" "${SCRATCH}/data/acdc_clips/" \
  --exclude "*" --include "acdc_*" --only-show-errors
echo "[INFO] Downloaded $(ls ${SCRATCH}/data/acdc_clips/acdc_*.mp4 | wc -l) ACDC clips"

# Create local CSVs pointing to downloaded clips
aws s3 cp "${S3_BASE}/data/csv/acdc_dx_train_s3.csv" "${SCRATCH}/data/csv/acdc_dx_s3.csv" --quiet
aws s3 cp "${S3_BASE}/data/csv/acdc_dx_test_s3.csv" "${SCRATCH}/data/csv/acdc_dx_test_s3.csv" --quiet

# Rewrite S3 paths to local paths
sed "s|s3://.*clips/|${SCRATCH}/data/acdc_clips/|g" "${SCRATCH}/data/csv/acdc_dx_s3.csv" > "${SCRATCH}/data/csv/acdc_dx_train_local.csv"
sed "s|s3://.*clips/|${SCRATCH}/data/acdc_clips/|g" "${SCRATCH}/data/csv/acdc_dx_test_s3.csv" > "${SCRATCH}/data/csv/acdc_dx_test_local.csv"
echo "[INFO] Local train CSV: $(wc -l < ${SCRATCH}/data/csv/acdc_dx_train_local.csv) clips"
echo "[INFO] Local test CSV: $(wc -l < ${SCRATCH}/data/csv/acdc_dx_test_local.csv) clips"

# =============================================================================
# PHASE 1: PROBE TRAINING
# =============================================================================
echo "============================================================"
echo "[PHASE 1] Training diagnosis probes for 8 MAE checkpoints"
echo "============================================================"

for EP in ${EPOCHS}; do
  CKPT_IDX=$((EP - 1))
  echo "[INFO] Training probe: CMR MAE e${EP} (checkpoint-${CKPT_IDX})"

  CKPT_LOCAL="${SCRATCH}/checkpoints/mae_cmr_vits_ep${CKPT_IDX}.pth"
  aws s3 cp "${CKPT_S3}/checkpoint-${CKPT_IDX}.pth" "${CKPT_LOCAL}" --quiet

  CONFIG="${CODE_DIR}/configs/eval/vits16/cmr/mae_cmr_e${EP}_acdc_dx_d4.yaml"

  python3 -m evals.main \
    --fname "${CONFIG}" \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7
  PROBE_RC=$?
  echo "[INFO] e${EP} probe training done (rc=${PROBE_RC})"

  # Copy best.pt to probes dir for inference phase
  PROBE_SRC=$(find /opt/dlami/nvme/evals/cmr/mae_e${EP}_acdc_dx -name "best.pt" 2>/dev/null | head -1)
  if [ -n "${PROBE_SRC}" ]; then
    mkdir -p "${SCRATCH}/probes/cmr_mae_e${EP}_dx"
    cp "${PROBE_SRC}" "${SCRATCH}/probes/cmr_mae_e${EP}_dx/best.pt"
    echo "[INFO] Probe saved: ${SCRATCH}/probes/cmr_mae_e${EP}_dx/best.pt"
  fi

  sync_results; upload_logs
  rm -f "${CKPT_LOCAL}"
done

echo ""
echo "============================================================"
echo "[PHASE 2] Clean + matched_frame inference"
echo "============================================================"

# =============================================================================
# PHASE 2: CLEAN + MATCHED_FRAME INFERENCE
# =============================================================================
for EP in ${EPOCHS}; do
  CKPT_IDX=$((EP - 1))
  CKPT_LOCAL="${SCRATCH}/checkpoints/mae_cmr_vits_ep${CKPT_IDX}.pth"
  aws s3 cp "${CKPT_S3}/checkpoint-${CKPT_IDX}.pth" "${CKPT_LOCAL}" --quiet

  CONFIG="${CODE_DIR}/configs/inference/vits16/cmr/mae_cmr_e${EP}_acdc_dx_test.yaml"

  # --- CLEAN ---
  echo "[INFO] CLEAN inference: e${EP}"
  OVERRIDE_TAG="cmr-mae-vits-e${EP}-acdc-dx-clean" \
  OVERRIDE_PRED_PATH="/opt/dlami/nvme/evals/cmr/mae_e${EP}_acdc_dx_clean/predictions.csv" \
  python3 -m evals.main --fname "${CONFIG}" \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7
  echo "[INFO] e${EP} clean done"

  # --- MATCHED_FRAME ---
  echo "[INFO] MATCHED_FRAME inference: e${EP}"
  FRAME_SHUFFLE=100 FRAME_SHUFFLE_TYPE=matched_frame \
  OVERRIDE_TAG="cmr-mae-vits-e${EP}-acdc-dx-mf" \
  OVERRIDE_PRED_PATH="/opt/dlami/nvme/evals/cmr/mae_e${EP}_acdc_dx_mf/predictions.csv" \
  python3 -m evals.main --fname "${CONFIG}" \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7
  echo "[INFO] e${EP} matched_frame done"

  sync_results; upload_logs
  rm -f "${CKPT_LOCAL}"
done

echo ""
echo "============================================================"
echo "[INFO] All probe training + inference complete"
echo "============================================================"
