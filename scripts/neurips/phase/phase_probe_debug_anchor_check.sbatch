#!/bin/bash
# Debug: load 100 samples through the patched anchor-aware VideoDataset and
# verify each clip's indices contain the labeled anchor + target matches
# the parquet ground truth. Fails non-zero if any sample mismatches.
#SBATCH -p ml-p5-48xlarge
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -t 0-00:30:00
#SBATCH -J phase_debug_anchor
#SBATCH -o /opt/dlami/nvme/logs/phase_debug_anchor-%j.out
#SBATCH -e /opt/dlami/nvme/logs/phase_debug_anchor-%j.err
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
unset AWS_EC2_METADATA_DISABLED 2>/dev/null || true
export AWS_METADATA_SERVICE_TIMEOUT=10
export AWS_METADATA_SERVICE_NUM_ATTEMPTS=5

ART_BUCKET="sagemaker-hyperpod-lifecycle-495467399120-usw2"
S3_BASE="s3://${ART_BUCKET}/vjepa2-artifacts"
ENV_S3="${S3_BASE}/env/vjepa2-312-pack-env.tar"
SRC_TARBALL_S3="${S3_BASE}/setup/vjepa2-src.tar.gz"
RUN_S3="${S3_BASE}/runs/phase_debug_anchor_${SLURM_JOB_ID}"

upload_logs() {
  local out="/opt/dlami/nvme/logs/phase_debug_anchor-${SLURM_JOB_ID}.out"
  local err="/opt/dlami/nvme/logs/phase_debug_anchor-${SLURM_JOB_ID}.err"
  [ -f "${out}" ] && aws s3 cp "${out}" "${RUN_S3}/logs/job.out" --only-show-errors || true
  [ -f "${err}" ] && aws s3 cp "${err}" "${RUN_S3}/logs/job.err" --only-show-errors || true
}
trap 'upload_logs' EXIT

aws sts get-caller-identity || exit 1
nvidia-smi -L || true
mkdir -p /opt/dlami/nvme/logs

SCRATCH="/opt/dlami/nvme"
WORKDIR="${SCRATCH}/phase_debug_anchor_${SLURM_JOB_ID}"
mkdir -p "$WORKDIR"

ENV_DIR="${SCRATCH}/env/vjepa2-312-pack"
if [ ! -f "${ENV_DIR}/bin/activate" ]; then
    mkdir -p "$ENV_DIR"
    aws s3 cp "$ENV_S3" "${WORKDIR}/env.tar" --only-show-errors
    tar -xf "${WORKDIR}/env.tar" -C "$ENV_DIR"
fi
set +u && source "${ENV_DIR}/bin/activate" && set -u
"${ENV_DIR}/bin/conda-unpack" || true
export TMPDIR="${WORKDIR}/tmp" && mkdir -p "$TMPDIR"
pip install boto3 pyarrow --no-cache-dir --quiet

CODE_DIR="${WORKDIR}/code"
mkdir -p "$CODE_DIR"
aws s3 cp "${SRC_TARBALL_S3}" "${WORKDIR}/vjepa2-src.tar.gz" --quiet
tar xzf "${WORKDIR}/vjepa2-src.tar.gz" -C "$CODE_DIR"
cd "$CODE_DIR"
pip install -e . --no-deps --no-build-isolation 2>/dev/null || true
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# Download phase probe train CSV + parquet
DATA_DIR="${WORKDIR}/data"
mkdir -p "$DATA_DIR/csv" "$DATA_DIR/parquet"
aws s3 cp "${S3_BASE}/data/csv/mimic_phase_sin_train_10k.csv" "${DATA_DIR}/csv/" --quiet
aws s3 cp "${S3_BASE}/data/phase_annotations/phase_annotations.parquet" "${DATA_DIR}/parquet/" --quiet

echo "[INFO] Running anchor-probe debug check (100 samples, sin target)..."
python3 classifier/phase/debug_anchor_probe_loader.py \
  --csv "${DATA_DIR}/csv/mimic_phase_sin_train_10k.csv" \
  --parquet "${DATA_DIR}/parquet/phase_annotations.parquet" \
  --n 100 \
  --target sin

echo "[INFO] Debug check passed"
upload_logs
