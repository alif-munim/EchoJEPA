#!/bin/bash
# Target-difficulty audit — P2 gate. Runs after cache_cclip.sbatch.
#
# Output:  reports/echoset_jepa/target_difficulty.{md,json}
# Gate:    exit 0 on pass, exit 1 on fail. Operator must investigate a fail
#          before PR-N4 (training launcher) can run.
#
# Runtime: ~20 min on a single GPU node for 5k val studies.
#
#SBATCH -p dev
#SBATCH --nodelist=ip-10-0-50-146
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH -t 0-02:00:00
#SBATCH -J echoset_target_difficulty
#SBATCH -o /opt/dlami/nvme/logs/echoset_target_difficulty-%j.out
#SBATCH -e /opt/dlami/nvme/logs/echoset_target_difficulty-%j.err
#SBATCH --chdir=/opt/dlami/nvme

set -euxo pipefail

export AWS_DEFAULT_REGION="us-west-2"
export AWS_REGION="us-west-2"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export HOME="/home/ubuntu"
unset AWS_SHARED_CREDENTIALS_FILE 2>/dev/null || true
unset AWS_CONFIG_FILE 2>/dev/null || true
export PATH="/opt/vjepa2-312/bin:$PATH"
mkdir -p /opt/dlami/nvme/logs /opt/dlami/nvme/tmp
export TMPDIR="/opt/dlami/nvme/tmp"

ART_BUCKET="sagemaker-hyperpod-lifecycle-495467399120-usw2"
S3_BASE="s3://${ART_BUCKET}/vjepa2-artifacts"
SRC_TARBALL_S3="${S3_BASE}/setup/vjepa2-src-echoset-v1.tar.gz"

MANIFEST_S3="${S3_BASE}/echoset_jepa/study_clip_manifest_dedup.parquet"
ELEM_S3="${S3_BASE}/echoset_jepa/study_element_manifest.parquet"
CACHE_S3="${S3_BASE}/echoset_jepa/cclip"

CACHE_LOCAL="/opt/dlami/nvme/echoset_cache"
WORKDIR="/opt/dlami/nvme/echoset_audit_${SLURM_JOB_ID}"
mkdir -p "${WORKDIR}" "${CACHE_LOCAL}"

# --- Deploy source ---------------------------------------------------------
REPO_DIR="${WORKDIR}/code"
rm -rf "$REPO_DIR"; mkdir -p "$REPO_DIR"
aws s3 cp "${SRC_TARBALL_S3}" "${TMPDIR}/src.tar.gz" --only-show-errors
tar xzf "${TMPDIR}/src.tar.gz" -C "$REPO_DIR"
cd "$REPO_DIR"
pip install -e . --no-deps --no-build-isolation 2>/dev/null || true
export PYTHONPATH="$PWD:/opt/dlami/nvme/pyarrow_site:${PYTHONPATH:-}"
if ! python -c "import pyarrow" 2>/dev/null; then
  pip install --target=/opt/dlami/nvme/pyarrow_site --no-cache-dir pyarrow 2>&1 | tail -3
fi

# --- Pull inputs -----------------------------------------------------------
MANIFEST_LOCAL="${WORKDIR}/study_clip_manifest_dedup.parquet"
ELEM_LOCAL="${WORKDIR}/study_element_manifest.parquet"
aws s3 cp "${MANIFEST_S3}" "${MANIFEST_LOCAL}" --quiet
aws s3 cp "${ELEM_S3}" "${ELEM_LOCAL}" --quiet
aws s3 sync "${CACHE_S3}" "${CACHE_LOCAL}" --only-show-errors --exclude "_index/*"

# --- Run audit -------------------------------------------------------------
OUT_DIR="${WORKDIR}/report"
mkdir -p "${OUT_DIR}"

echo "[INFO] === Target-difficulty audit ==="
python -m experiments.echoset_jepa.target_difficulty_audit \
  --clip_manifest "${MANIFEST_LOCAL}" \
  --element_manifest "${ELEM_LOCAL}" \
  --cache_prefix "${CACHE_LOCAL}" \
  --out_dir "${OUT_DIR}" \
  --audit_split val \
  --proto_split train \
  --num_studies 5000 \
  --seed 0
GATE_EXIT=$?

# --- Upload report ---------------------------------------------------------
aws s3 sync "${OUT_DIR}" "${S3_BASE}/echoset_jepa/reports/target_difficulty/" --only-show-errors

echo "[INFO] === audit gate exit code: ${GATE_EXIT} ==="
echo "   report: ${S3_BASE}/echoset_jepa/reports/target_difficulty/"
exit $GATE_EXIT
