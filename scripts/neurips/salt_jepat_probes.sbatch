#!/bin/bash
#SBATCH -p dev
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 6:00:00
#SBATCH -J salt_jprb
#SBATCH -o /opt/dlami/nvme/logs/salt_jepat_probes_%j.out
#SBATCH -e /opt/dlami/nvme/logs/salt_jepat_probes_%j.err

set -euxo pipefail

# =============================================================================
# SALT (JEPA teacher, job 335): probe training + clean/matched-frame inference
# =============================================================================
# Phase 1: Train d=4 attentive probe on EchoNet-Dynamic (6-head HP grid)
# Phase 2: Clean test inference
# Phase 3: Matched-frame test inference (RoPE remap)
# =============================================================================

export AWS_DEFAULT_REGION="us-west-2"
export AWS_REGION="us-west-2"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export HOME="/home/ubuntu"
unset AWS_SHARED_CREDENTIALS_FILE 2>/dev/null || true
unset AWS_CONFIG_FILE 2>/dev/null || true
unset AWS_EC2_METADATA_DISABLED 2>/dev/null || true
export AWS_METADATA_SERVICE_TIMEOUT=10
export AWS_METADATA_SERVICE_NUM_ATTEMPTS=5

aws sts get-caller-identity || exit 1

ART_BUCKET="sagemaker-hyperpod-lifecycle-495467399120-usw2"
S3_BASE="s3://${ART_BUCKET}/vjepa2-artifacts"
ENV_S3="${S3_BASE}/env/vjepa2-312-pack-env.tar"
S2_CKPT_S3="${S3_BASE}/runs/salt_s2_jepa_teacher_335/checkpoints/latest.pt"
RUN_S3="${S3_BASE}/runs/salt_jepat_probes_${SLURM_JOB_ID}"

PROBE_CFG="configs/eval/vitl/neurips/salt_s2_jepat_echonet_lvef_d4.yaml"
INFER_CFG="configs/inference/vitl/neurips/echonet-dynamic/salt_s2_jepat_end_test.yaml"

upload_logs() {
  local out="/opt/dlami/nvme/logs/salt_jepat_probes_${SLURM_JOB_ID}.out"
  local err="/opt/dlami/nvme/logs/salt_jepat_probes_${SLURM_JOB_ID}.err"
  [ -f "${out}" ] && aws s3 cp "${out}" "${RUN_S3}/logs/job.out" --only-show-errors || true
  [ -f "${err}" ] && aws s3 cp "${err}" "${RUN_S3}/logs/job.err" --only-show-errors || true
}
trap 'upload_logs' EXIT

# =========================
# SETUP
# =========================
nvidia-smi -L || true
mkdir -p /opt/dlami/nvme/logs

SCRATCH="/opt/dlami/nvme"
WORKDIR="${SCRATCH}/salt_jepat_probes_${SLURM_JOB_ID}"
mkdir -p "$WORKDIR"

# Env
aws s3 cp "$ENV_S3" "${WORKDIR}/env.tar"
mkdir -p "${WORKDIR}/env" && tar -xf "${WORKDIR}/env.tar" -C "${WORKDIR}/env"
set +u && source "${WORKDIR}/env/bin/activate" && set -u
"${WORKDIR}/env/bin/conda-unpack" || true
export TMPDIR="${WORKDIR}/tmp" && mkdir -p "$TMPDIR"
pip install boto3 --no-cache-dir --quiet

# Code
CODE_DIR="${WORKDIR}/code"
mkdir -p "$CODE_DIR"
aws s3 cp "${S3_BASE}/setup/vjepa2-src.tar.gz" "${WORKDIR}/vjepa2-src.tar.gz" --quiet
tar xzf "${WORKDIR}/vjepa2-src.tar.gz" -C "$CODE_DIR"
cd "$CODE_DIR"
pip install -e . --no-deps --no-build-isolation 2>/dev/null || true
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# Inject configs (from s3://.../setup/)
mkdir -p configs/eval/vitl/neurips configs/inference/vitl/neurips/echonet-dynamic
aws s3 cp "${S3_BASE}/setup/salt_s2_jepat_echonet_lvef_d4.yaml" "$PROBE_CFG" --quiet
aws s3 cp "${S3_BASE}/setup/salt_s2_jepat_end_test.yaml" "$INFER_CFG" --quiet

# Data — probe training needs train + val; inference needs train + test.
mkdir -p /opt/dlami/nvme/data/csv
aws s3 cp "${S3_BASE}/data/csv/echonet_dynamic_train_s3_raw.csv" /opt/dlami/nvme/data/csv/ --quiet
aws s3 cp "${S3_BASE}/data/csv/echonet_dynamic_val_s3_raw.csv"   /opt/dlami/nvme/data/csv/ --quiet
aws s3 cp "${S3_BASE}/data/csv/echonet_dynamic_test_s3_raw.csv"  /opt/dlami/nvme/data/csv/ --quiet

# S2 checkpoint — try local first (same node as S2), then S3
S2_LOCAL="/opt/dlami/nvme/checkpoints/pretrain/mimic/salt_s2_jepa_teacher_v1/latest.pt"
if [ ! -f "$S2_LOCAL" ]; then
  echo "[INFO] Downloading S2 checkpoint from S3..."
  mkdir -p "$(dirname $S2_LOCAL)"
  aws s3 cp "$S2_CKPT_S3" "$S2_LOCAL"
fi
ls -lh "$S2_LOCAL"

# =========================
# PHASE 1: PROBE TRAINING
# =========================
echo "[INFO] Phase 1: Training d=4 attentive probe..."
python3 -m evals.main --fname "$PROBE_CFG" \
  --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7
echo "[INFO] Probe training complete"

# Find best probe checkpoint
PROBE_DIR="${CODE_DIR}/evals/vitl/neurips/salt_s2_jepat_echonet_lvef_d4/video_classification_frozen/neurips-salt-s2-jepat-echonet-lvef-d4"
BEST_PROBE="${PROBE_DIR}/best.pt"
if [ ! -f "$BEST_PROBE" ]; then
  echo "[WARN] best.pt not found at expected path, searching..."
  BEST_PROBE=$(find "${CODE_DIR}/evals" -name "best.pt" -path "*jepat*" | head -1)
fi
echo "[INFO] Best probe: $BEST_PROBE"

PROBE_DEST="/opt/dlami/nvme/probes/salt_jepat_end"
mkdir -p "$PROBE_DEST"
cp "$BEST_PROBE" "$PROBE_DEST/best.pt"

aws s3 cp "$BEST_PROBE" "${RUN_S3}/probe/best.pt" --only-show-errors || true

# =========================
# PHASE 2: CLEAN INFERENCE
# =========================
echo "[INFO] Phase 2: Clean test inference..."
python3 -m evals.main --fname "$INFER_CFG" \
  --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7
echo "[INFO] Clean inference complete"

aws s3 sync /opt/dlami/nvme/evals/neurips/salt_jepat_end_test/ \
  "${RUN_S3}/clean/" --only-show-errors || true

# =========================
# PHASE 3: MATCHED-FRAME INFERENCE
# =========================
echo "[INFO] Phase 3: Matched-frame inference..."
FRAME_SHUFFLE=100 FRAME_SHUFFLE_TYPE=matched_frame \
OVERRIDE_TAG="neurips-salt-s2-jepat-end-test-mf" \
OVERRIDE_PRED_PATH="/opt/dlami/nvme/evals/neurips/salt_jepat_end_test_mf/predictions.csv" \
python3 -m evals.main --fname "$INFER_CFG" \
  --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7
echo "[INFO] Matched-frame inference complete"

aws s3 sync /opt/dlami/nvme/evals/neurips/salt_jepat_end_test_mf/ \
  "${RUN_S3}/matched_frame/" --only-show-errors || true

upload_logs
echo "[INFO] All phases complete for SALT JEPA teacher experiment"
