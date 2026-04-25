#!/bin/bash
#SBATCH -p dev
#SBATCH -N 1
#SBATCH --gres=gpu:0
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH -t 8:00:00
#SBATCH -J phi_jepa_full_meta
#SBATCH -o /opt/dlami/nvme/logs/phi_jepa_full_meta_%j.out
#SBATCH -e /opt/dlami/nvme/logs/phi_jepa_full_meta_%j.err

set -euxo pipefail

# =============================================================================
# phi-JEPA: Full-corpus DICOM metadata extraction.
#
# Scans ALL MIMIC-IV-Echo raw DICOMs on S3 (~525K clips across all studies)
# and produces the authoritative mimic_clip_phase_metadata.csv consumed by
# the phi-JEPA dataloader.
#
# Expected runtime: ~30-60 min at ~270 clips/s with 48 workers (Gate 1 ran
# at ~215 clips/s with 32 workers).
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
# N_STUDIES=-1 means FULL CORPUS (no subsampling).
N_STUDIES="${N_STUDIES:--1}"
MAX_CLIPS_PER_STUDY="${MAX_CLIPS_PER_STUDY:-0}"
WORKERS="${WORKERS:-48}"
SEED="${SEED:-0}"

# =========================
# S3 layout for outputs
# =========================
ART_BUCKET="sagemaker-hyperpod-lifecycle-495467399120-usw2"
S3_BASE="s3://${ART_BUCKET}/vjepa2-artifacts"
ENV_S3="${S3_BASE}/env/vjepa2-312-pack-env.tar"
RUN_S3="${S3_BASE}/runs/phi_jepa_full_meta_${SLURM_JOB_ID}"
# Canonical artifact path (stable) — reused across re-runs.
CANON_CSV_S3="${S3_BASE}/phase/mimic_clip_phase_metadata.csv"
CANON_REPORT_S3="${S3_BASE}/phase/mimic_clip_phase_metadata_report.txt"

SCRATCH="/opt/dlami/nvme"
OUT_DIR="${SCRATCH}/phi_jepa_full_meta_${SLURM_JOB_ID}"
OUT_CSV="${OUT_DIR}/mimic_clip_phase_metadata.csv"
OUT_REPORT="${OUT_DIR}/mimic_clip_phase_metadata_report.txt"
mkdir -p "${OUT_DIR}" /opt/dlami/nvme/logs

upload_all() {
  [ -f "${OUT_CSV}"    ] && aws s3 cp "${OUT_CSV}"    "${RUN_S3}/mimic_clip_phase_metadata.csv" --only-show-errors || true
  [ -f "${OUT_REPORT}" ] && aws s3 cp "${OUT_REPORT}" "${RUN_S3}/mimic_clip_phase_metadata_report.txt" --only-show-errors || true
  # Also write to canonical path for dataloader consumption.
  [ -f "${OUT_CSV}"    ] && aws s3 cp "${OUT_CSV}"    "${CANON_CSV_S3}"    --only-show-errors || true
  [ -f "${OUT_REPORT}" ] && aws s3 cp "${OUT_REPORT}" "${CANON_REPORT_S3}" --only-show-errors || true
  local out="/opt/dlami/nvme/logs/phi_jepa_full_meta_${SLURM_JOB_ID}.out"
  local err="/opt/dlami/nvme/logs/phi_jepa_full_meta_${SLURM_JOB_ID}.err"
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
export TMPDIR="${SCRATCH}/tmp_full_meta" && mkdir -p "${TMPDIR}"
pip install --no-cache-dir --quiet boto3 pydicom numpy

CODE_DIR="${SCRATCH}/code_full_meta"
mkdir -p "${CODE_DIR}"
aws s3 cp "${S3_BASE}/setup/vjepa2-src.tar.gz" "${SCRATCH}/vjepa2-src.tar.gz" --quiet
tar xzf "${SCRATCH}/vjepa2-src.tar.gz" -C "${CODE_DIR}"
cd "${CODE_DIR}"

# =========================
# Run
# =========================
echo "[INFO] phi-JEPA FULL-CORPUS metadata extraction"
echo "       n_studies       = ${N_STUDIES}  (negative = full corpus)"
echo "       max_clips/study = ${MAX_CLIPS_PER_STUDY}"
echo "       workers         = ${WORKERS}"
echo "       seed            = ${SEED}"
echo "       out_csv         = ${OUT_CSV}"
echo "       out_report      = ${OUT_REPORT}"
echo "       s3_run          = ${RUN_S3}"
echo "       s3_canonical    = ${CANON_CSV_S3}"

python3 -u scripts/neurips/phase/extract_dicom_phase_metadata.py \
  --out "${OUT_CSV}" \
  --report "${OUT_REPORT}" \
  --n-studies "${N_STUDIES}" \
  --max-clips-per-study "${MAX_CLIPS_PER_STUDY}" \
  --workers "${WORKERS}" \
  --seed "${SEED}"
RC=$?

upload_all
echo "[INFO] Full-corpus metadata rc=${RC}"
exit ${RC}
