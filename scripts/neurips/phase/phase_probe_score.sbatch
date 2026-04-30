#!/bin/bash
# Post-process phase-probe sin/cos predictions → phase metrics.
# Queued afterany:574 (last of the phase test jobs).
# Needs PRED_JOB_IDS env var mapping arm+target → test SLURM id for CSV lookup.
#
# Required env vars (set via --export when submitting):
#   PHASE_TEST_JIDS: comma-separated string of 6 test job IDs in canonical order
#                    e100_sin,e100_cos,phase_542_sin,phase_542_cos,sv_548_sin,sv_548_cos
#                    example: "569,570,571,572,573,574"
#   PHASE_TRAIN_JIDS: same order for train jobs (used for best-epoch lookup)
#                     "563,564,565,566,567,568"
#SBATCH -p ml-p5-48xlarge
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -t 0-00:30:00
#SBATCH -J phase_probe_score
#SBATCH -o /opt/dlami/nvme/logs/phase_probe_score-%j.out
#SBATCH -e /opt/dlami/nvme/logs/phase_probe_score-%j.err
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

: "${PHASE_TEST_JIDS:?Required: comma-separated 6 test SLURM ids}"
: "${PHASE_TRAIN_JIDS:?Required: comma-separated 6 train SLURM ids}"

ART_BUCKET="sagemaker-hyperpod-lifecycle-495467399120-usw2"
S3_BASE="s3://${ART_BUCKET}/vjepa2-artifacts"
ENV_S3="${S3_BASE}/env/vjepa2-312-pack-env.tar"
SRC_TARBALL_S3="${S3_BASE}/setup/vjepa2-src.tar.gz"
RUN_S3="${S3_BASE}/runs/phase_probe_score_${SLURM_JOB_ID}"

upload_logs() {
  local out="/opt/dlami/nvme/logs/phase_probe_score-${SLURM_JOB_ID}.out"
  local err="/opt/dlami/nvme/logs/phase_probe_score-${SLURM_JOB_ID}.err"
  [ -f "${out}" ] && aws s3 cp "${out}" "${RUN_S3}/logs/job.out" --only-show-errors || true
  [ -f "${err}" ] && aws s3 cp "${err}" "${RUN_S3}/logs/job.err" --only-show-errors || true
}
trap 'upload_logs' EXIT

aws sts get-caller-identity || exit 1
mkdir -p /opt/dlami/nvme/logs

SCRATCH="/opt/dlami/nvme"
WORKDIR="${SCRATCH}/phase_probe_score_${SLURM_JOB_ID}"
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
pip install boto3 --no-cache-dir --quiet

CODE_DIR="${WORKDIR}/code"
mkdir -p "$CODE_DIR"
aws s3 cp "${SRC_TARBALL_S3}" "${WORKDIR}/vjepa2-src.tar.gz" --quiet
tar xzf "${WORKDIR}/vjepa2-src.tar.gz" -C "$CODE_DIR"
cd "$CODE_DIR"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# ---- Download the 6 prediction CSVs + anchors CSV + 6 train logs ----
PRED_DIR="${WORKDIR}/predictions"
LOGS_DIR="${WORKDIR}/train_logs"
ANCHORS_CSV="${WORKDIR}/anchors.csv"
mkdir -p "$PRED_DIR" "$LOGS_DIR"

aws s3 cp "${S3_BASE}/data/csv/mimic_phase_anchors_10k.csv" "$ANCHORS_CSV" --quiet

# Unpack ordered arrays
IFS=',' read -ra TEST_JIDS  <<< "$PHASE_TEST_JIDS"
IFS=',' read -ra TRAIN_JIDS <<< "$PHASE_TRAIN_JIDS"
ARMS=(e100 e100 phase_542 phase_542 sv_548 sv_548)
TGTS=(sin  cos  sin       cos       sin    cos)

for i in "${!ARMS[@]}"; do
    ARM="${ARMS[$i]}"; TGT="${TGTS[$i]}"
    TEST_JID="${TEST_JIDS[$i]}"; TRAIN_JID="${TRAIN_JIDS[$i]}"
    PILOT="phase_${ARM}_${TGT}"
    PRED_S3="${S3_BASE}/runs/${PILOT}_test_${TEST_JID}/predictions/${PILOT}_test.csv"
    aws s3 cp "$PRED_S3" "${PRED_DIR}/${PILOT}_test.csv" --quiet || {
        echo "[ERROR] missing prediction CSV for ${PILOT} (test_jid=${TEST_JID})"; exit 2; }

    # Train log for best-epoch report
    TAG="neurips-phase-${ARM}-${TGT}"
    LOG_S3="${S3_BASE}/runs/${PILOT}_${TRAIN_JID}/probe/video_classification_frozen/${TAG}/log_r0.csv"
    mkdir -p "${LOGS_DIR}/${PILOT}"
    aws s3 cp "$LOG_S3" "${LOGS_DIR}/${PILOT}/log_r0.csv" --quiet || true
done

echo "[INFO] Running phase-probe scorer..."
OUT_DIR="${WORKDIR}/scored"
python3 classifier/phase/score_phase_probes.py \
    --pred-dir "$PRED_DIR" \
    --anchors-csv "$ANCHORS_CSV" \
    --train-logs-dir "$LOGS_DIR" \
    --arms e100 phase_542 sv_548 \
    --out-dir "$OUT_DIR"

echo "[INFO] Uploading scored artifacts to ${RUN_S3}/..."
aws s3 sync "$OUT_DIR" "${RUN_S3}/scored/" --only-show-errors
echo "[INFO] Done. Key artifacts:"
echo "  ${RUN_S3}/scored/summary.csv"
echo "  ${RUN_S3}/scored/per_bin.csv"
echo "  ${RUN_S3}/scored/decision.txt"
upload_logs
