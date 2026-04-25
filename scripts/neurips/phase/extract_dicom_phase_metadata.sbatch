#!/bin/bash
#SBATCH -p dev
#SBATCH -N 1
#SBATCH --gres=gpu:0
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH -t 4:00:00
#SBATCH -J phi_jepa_gate1
#SBATCH -o /opt/dlami/nvme/logs/phi_jepa_gate1_%j.out
#SBATCH -e /opt/dlami/nvme/logs/phi_jepa_gate1_%j.err

set -euxo pipefail

# =============================================================================
# phi-JEPA Gate 1: DICOM metadata extraction pilot.
#
# Scans MIMIC-IV-Echo raw DICOMs on S3 and reports HR + FrameTime + study-level
# HR stdev stats. Produces one CSV row per clip plus a human-readable Gate 1
# acceptance report.
#
# Gate 1 passes if:
#   - HR valid (40-180 bpm) on >= 95% of clips
#   - within-study HR stdev <= 15 bpm on >= 85% of studies
# =============================================================================

export AWS_DEFAULT_REGION="us-west-2"
export AWS_REGION="us-west-2"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export HOME="/home/ubuntu"
unset AWS_SHARED_CREDENTIALS_FILE 2>/dev/null || true
unset AWS_CONFIG_FILE 2>/dev/null || true
export AWS_METADATA_SERVICE_TIMEOUT=10
export AWS_METADATA_SERVICE_NUM_ATTEMPTS=5

aws sts get-caller-identity || { echo "[FATAL] AWS credentials check failed"; exit 1; }

# =========================
# Knobs (override via env)
# =========================
N_STUDIES="${N_STUDIES:-1000}"
MAX_CLIPS_PER_STUDY="${MAX_CLIPS_PER_STUDY:-0}"
WORKERS="${WORKERS:-32}"
SEED="${SEED:-0}"

# =========================
# S3 layout for outputs
# =========================
ART_BUCKET="sagemaker-hyperpod-lifecycle-495467399120-usw2"
S3_BASE="s3://${ART_BUCKET}/vjepa2-artifacts"
ENV_S3="${S3_BASE}/env/vjepa2-312-pack-env.tar"
RUN_S3="${S3_BASE}/runs/phi_jepa_gate1_${SLURM_JOB_ID}"

SCRATCH="/opt/dlami/nvme"
OUT_DIR="${SCRATCH}/phi_jepa_gate1_${SLURM_JOB_ID}"
OUT_CSV="${OUT_DIR}/clip_phase_metadata.csv"
OUT_REPORT="${OUT_DIR}/gate1_report.txt"
mkdir -p "${OUT_DIR}" /opt/dlami/nvme/logs

upload_all() {
  [ -f "${OUT_CSV}"    ] && aws s3 cp "${OUT_CSV}"    "${RUN_S3}/clip_phase_metadata.csv" --only-show-errors || true
  [ -f "${OUT_REPORT}" ] && aws s3 cp "${OUT_REPORT}" "${RUN_S3}/gate1_report.txt"       --only-show-errors || true
  local out="/opt/dlami/nvme/logs/phi_jepa_gate1_${SLURM_JOB_ID}.out"
  local err="/opt/dlami/nvme/logs/phi_jepa_gate1_${SLURM_JOB_ID}.err"
  [ -f "${out}" ] && aws s3 cp "${out}" "${RUN_S3}/logs/job.out" --only-show-errors || true
  [ -f "${err}" ] && aws s3 cp "${err}" "${RUN_S3}/logs/job.err" --only-show-errors || true
}
trap upload_all EXIT

# =========================
# Env + code
# =========================
aws s3 cp "${ENV_S3}" "${SCRATCH}/env.tar"
mkdir -p "${SCRATCH}/env" && tar -xf "${SCRATCH}/env.tar" -C "${SCRATCH}/env"
set +u && source "${SCRATCH}/env/bin/activate" && set -u
"${SCRATCH}/env/bin/conda-unpack" || true
export TMPDIR="${SCRATCH}/tmp_gate1" && mkdir -p "${TMPDIR}"
pip install --no-cache-dir --quiet boto3 pydicom numpy

CODE_DIR="${SCRATCH}/code_gate1"
mkdir -p "${CODE_DIR}"
aws s3 cp "${S3_BASE}/setup/vjepa2-src.tar.gz" "${SCRATCH}/vjepa2-src.tar.gz" --quiet
tar xzf "${SCRATCH}/vjepa2-src.tar.gz" -C "${CODE_DIR}"
cd "${CODE_DIR}"

# =========================
# Run
# =========================
echo "[INFO] phi-JEPA Gate 1 pilot"
echo "       n_studies       = ${N_STUDIES}"
echo "       max_clips/study = ${MAX_CLIPS_PER_STUDY}"
echo "       workers         = ${WORKERS}"
echo "       seed            = ${SEED}"
echo "       out_csv         = ${OUT_CSV}"
echo "       out_report      = ${OUT_REPORT}"
echo "       s3_sink         = ${RUN_S3}"

python3 -u scripts/neurips/phase/extract_dicom_phase_metadata.py \
  --out "${OUT_CSV}" \
  --report "${OUT_REPORT}" \
  --n-studies "${N_STUDIES}" \
  --max-clips-per-study "${MAX_CLIPS_PER_STUDY}" \
  --workers "${WORKERS}" \
  --seed "${SEED}"
RC=$?

upload_all
echo "[INFO] Gate 1 rc=${RC} (0=PASS by thresholds, 1=review)"
exit ${RC}
