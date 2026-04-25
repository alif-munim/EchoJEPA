#!/bin/bash
#SBATCH -p dev
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 2-00:00:00
#SBATCH -J salt_s2e99t
#SBATCH -o /opt/dlami/nvme/logs/salt_s2e99t_%j.out
#SBATCH -e /opt/dlami/nvme/logs/salt_s2e99t_%j.err
#SBATCH --signal=B:USR1@300

set -euxo pipefail

# =============================================================================
# SALT Stage 2: Student against V-Pixel e99 teacher
# =============================================================================
# Tests whether SALT's "inherited disuse" pattern persists with a post-shortcut
# teacher. All student hyperparameters match job 388 (primary SALT v1 arm).
# Only change: teacher checkpoint (V-Pixel e99 instead of e20).
#
# PREREQUISITE: Stage 1 extended training must complete first.
# The teacher checkpoint should be at:
#   /opt/dlami/nvme/checkpoints/pretrain/mimic/salt_s1_vitl_224px_16f_e100/latest.pt
# or on S3 at:
#   s3://.../runs/salt_s1_e100_resume_<JOB_ID>/checkpoints/latest.pt
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

aws sts get-caller-identity || { echo "[FATAL] AWS credentials check failed"; exit 1; }

# =========================
# S3 CONFIG
# =========================
ART_BUCKET="sagemaker-hyperpod-lifecycle-495467399120-usw2"
S3_BASE="s3://${ART_BUCKET}/vjepa2-artifacts"
DATA_CSV_S3="${S3_BASE}/data/csv/mimic_annotations_s3.csv"
ENV_S3="${S3_BASE}/env/vjepa2-312-pack-env.tar"

# =====================================================================
# TEACHER CHECKPOINT — update S1_RUN_S3 with the actual S1 e100 job ID
# after Stage 1 completes. If both stages run on the same node
# sequentially, the local path may already exist.
# =====================================================================
S1_RUN_S3="${S3_BASE}/runs/salt_s1_e100_resume_329/checkpoints/latest.pt"
S1_LOCAL="/opt/dlami/nvme/checkpoints/pretrain/mimic/salt_s1_vitl_224px_16f_e100/latest.pt"

RUN_S3="${S3_BASE}/runs/salt_s2_vpixel_e99_teacher_${SLURM_JOB_ID}"

CFG_REL="configs/train/vitl16/pretrain-salt-s2-vpixel-e99-teacher.yaml"
CKPT_DIR="/opt/dlami/nvme/checkpoints/pretrain/mimic/salt_s2_vpixel_e99_teacher_v1"

# =========================
# LOGGING / SYNC
# =========================
mkdir -p /opt/dlami/nvme/logs

sync_ckpts() {
  if [ -d "$CKPT_DIR" ]; then
    aws s3 sync "$CKPT_DIR" "${RUN_S3}/checkpoints/" \
      --exclude "*" \
      --include "e*.pt" --include "latest.pt" --include "*.csv" --include "*.yaml" \
      --only-show-errors || true
  fi
}

upload_logs() {
  local out="/opt/dlami/nvme/logs/salt_s2e99t_${SLURM_JOB_ID}.out"
  local err="/opt/dlami/nvme/logs/salt_s2e99t_${SLURM_JOB_ID}.err"
  [ -f "${out}" ] && aws s3 cp "${out}" "${RUN_S3}/logs/job.out" --only-show-errors || true
  [ -f "${err}" ] && aws s3 cp "${err}" "${RUN_S3}/logs/job.err" --only-show-errors || true
}

on_exit() {
  rc=$?
  sync_ckpts
  upload_logs
  exit $rc
}
trap on_exit EXIT
trap 'sync_ckpts; upload_logs' USR1

# =========================
# SETUP
# =========================
nvidia-smi -L || true

SCRATCH="/opt/dlami/nvme"
WORKDIR="${SCRATCH}/salt_s2_e99t_${SLURM_JOB_ID}"
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

# Inject config (not in tarball)
mkdir -p configs/train/vitl16
aws s3 cp "${S3_BASE}/setup/pretrain-salt-s2-vpixel-e99-teacher.yaml" \
  "configs/train/vitl16/pretrain-salt-s2-vpixel-e99-teacher.yaml" --quiet

# Data
mkdir -p /opt/dlami/nvme/data/csv
aws s3 cp "$DATA_CSV_S3" /opt/dlami/nvme/data/csv/mimic_annotations_s3.csv

# =========================
# STAGE TEACHER CHECKPOINT
# =========================
echo "[INFO] Checking for S1 e100 teacher checkpoint..."
if [ -f "$S1_LOCAL" ]; then
  echo "[INFO] Teacher checkpoint found locally: $S1_LOCAL"
else
  echo "[INFO] Downloading teacher checkpoint from S3..."
  S1_DIR=$(dirname "$S1_LOCAL")
  mkdir -p "$S1_DIR"
  aws s3 cp "$S1_RUN_S3" "$S1_LOCAL"
fi
ls -lh "$S1_LOCAL"

mkdir -p "$CKPT_DIR"

# =========================
# PERIODIC S3 SYNC
# =========================
(
  while true; do
    sleep 900
    sync_ckpts
    upload_logs
  done
) &
SYNC_PID=$!

# =========================
# RUN SALT STAGE 2 (80 epochs, V-Pixel e99 teacher)
# =========================
echo "[INFO] Launching SALT Stage 2 with V-Pixel e99 teacher..."
python3 -m app.main \
  --fname "$CFG_REL" \
  --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7
RC=$?

kill $SYNC_PID 2>/dev/null || true
wait $SYNC_PID 2>/dev/null || true
sync_ckpts
upload_logs
exit $RC
