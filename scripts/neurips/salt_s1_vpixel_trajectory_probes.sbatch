#!/bin/bash
#SBATCH -p dev
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 6:00:00
#SBATCH -J s1vp_traj
#SBATCH -o /opt/dlami/nvme/logs/salt_s1_vpixel_traj_probe_%j.out
#SBATCH -e /opt/dlami/nvme/logs/salt_s1_vpixel_traj_probe_%j.err

set -euxo pipefail

# =============================================================================
# SALT S1 V-Pixel teacher (job 329): trajectory probe at a single epoch.
#
# Env var EPOCH selects which teacher checkpoint to probe. Submit 4x for
# e24/49/74/99 to build the {25,50,75,100} trajectory.
#
#   EPOCH=24  sbatch salt_s1_vpixel_trajectory_probes.sbatch
#   EPOCH=49  sbatch ...
#   EPOCH=74  sbatch ...
#   EPOCH=99  sbatch ...
#
# Phase 1: Train d=4 attentive probe on EchoNet-Dynamic (6-head HP grid)
# Phase 2: Clean test inference
# Phase 3: Matched-frame test inference (RoPE remap)
# =============================================================================

EPOCH="${EPOCH:?EPOCH env var required (e.g. EPOCH=24)}"

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

# S1 V-Pixel teacher checkpoints (trained to e100 via resume).
S1_CKPT_S3="${S3_BASE}/runs/salt_s1_e100_resume_329/checkpoints/e${EPOCH}.pt"
RUN_S3="${S3_BASE}/runs/salt_s1_vpixel_traj_e${EPOCH}_${SLURM_JOB_ID}"

# Unique per-epoch tags to avoid probe-folder collisions.
TAG_BASE="neurips-salt-s1-vpixel-e${EPOCH}-echonet-lvef-d4"
PROBE_CFG_REL="configs/eval/vitl/neurips/salt_s1_vpixel_e${EPOCH}_echonet_lvef_d4.yaml"
INFER_CFG_REL="configs/inference/vitl/neurips/echonet-dynamic/salt_s1_vpixel_e${EPOCH}_end_test.yaml"
INFER_OUT_DIR="/opt/dlami/nvme/evals/neurips/salt_s1_vpixel_e${EPOCH}_end_test"

upload_logs() {
  local out="/opt/dlami/nvme/logs/salt_s1_vpixel_traj_probe_${SLURM_JOB_ID}.out"
  local err="/opt/dlami/nvme/logs/salt_s1_vpixel_traj_probe_${SLURM_JOB_ID}.err"
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
WORKDIR="${SCRATCH}/salt_s1_traj_e${EPOCH}_${SLURM_JOB_ID}"
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

# =========================
# DOWNLOAD TEACHER CHECKPOINT
# =========================
S1_LOCAL="/opt/dlami/nvme/checkpoints/pretrain/mimic/salt_s1_vpixel_e100_resume/e${EPOCH}.pt"
mkdir -p "$(dirname $S1_LOCAL)"
if [ ! -f "$S1_LOCAL" ]; then
  echo "[INFO] Downloading S1 teacher checkpoint e${EPOCH} from S3..."
  aws s3 cp "$S1_CKPT_S3" "$S1_LOCAL"
fi
ls -lh "$S1_LOCAL"

# =========================
# INJECT + PATCH CONFIGS
# =========================
# We reuse the existing S2 V-Pixel probe YAMLs and sed in (a) the epoch-
# specific checkpoint path, (b) unique folder/tag, and (c) inference paths.
mkdir -p configs/eval/vitl/neurips configs/inference/vitl/neurips/echonet-dynamic
aws s3 cp "${S3_BASE}/setup/salt_s2_vpixel_e99t_echonet_lvef_d4.yaml" \
  "${PROBE_CFG_REL}" --quiet
aws s3 cp "${S3_BASE}/setup/salt_s2_vpixel_e99t_end_test.yaml" \
  "${INFER_CFG_REL}" --quiet

# Patch probe config: checkpoint path, folder, tag.
sed -i \
  -e "s|/opt/dlami/nvme/checkpoints/pretrain/mimic/salt_s2_vpixel_e99_teacher_v1/latest.pt|${S1_LOCAL}|g" \
  -e "s|evals/vitl/neurips/salt_s2_vpixel_e99t_echonet_lvef_d4|evals/vitl/neurips/salt_s1_vpixel_e${EPOCH}_echonet_lvef_d4|g" \
  -e "s|neurips-salt-s2-vpixel-e99t-echonet-lvef-d4|${TAG_BASE}|g" \
  "${PROBE_CFG_REL}"

# Patch inference config: checkpoint, probe_checkpoint, folder, tag, output paths.
sed -i \
  -e "s|/opt/dlami/nvme/checkpoints/pretrain/mimic/salt_s2_vpixel_e99_teacher_v1/latest.pt|${S1_LOCAL}|g" \
  -e "s|/opt/dlami/nvme/probes/salt_vpixel_e99t_end/best.pt|/opt/dlami/nvme/probes/salt_s1_vpixel_e${EPOCH}/best.pt|g" \
  -e "s|/opt/dlami/nvme/evals/neurips/salt_vpixel_e99t_end_test|${INFER_OUT_DIR}|g" \
  -e "s|neurips-salt-s2-vpixel-e99t-end-test|neurips-salt-s1-vpixel-e${EPOCH}-end-test|g" \
  "${INFER_CFG_REL}"

echo "[INFO] Patched probe config:"
grep -E "folder:|tag:|checkpoint:|probe_checkpoint:" "${PROBE_CFG_REL}" | head -10
echo "[INFO] Patched inference config:"
grep -E "folder:|tag:|checkpoint:|probe_checkpoint:" "${INFER_CFG_REL}" | head -10

# =========================
# STAGE DATA
# =========================
mkdir -p /opt/dlami/nvme/data/csv
aws s3 cp "${S3_BASE}/data/csv/echonet_dynamic_train_s3_raw.csv" /opt/dlami/nvme/data/csv/ --quiet
aws s3 cp "${S3_BASE}/data/csv/echonet_dynamic_val_s3_raw.csv"   /opt/dlami/nvme/data/csv/ --quiet
aws s3 cp "${S3_BASE}/data/csv/echonet_dynamic_test_s3_raw.csv"  /opt/dlami/nvme/data/csv/ --quiet

# =========================
# PHASE 1: PROBE TRAINING
# =========================
echo "[INFO] Phase 1: Training d=4 attentive probe on S1 V-Pixel e${EPOCH}..."
python3 -m evals.main --fname "${PROBE_CFG_REL}" \
  --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7
echo "[INFO] Probe training complete"

# Find best probe checkpoint.
PROBE_DIR="${CODE_DIR}/evals/vitl/neurips/salt_s1_vpixel_e${EPOCH}_echonet_lvef_d4/video_classification_frozen/${TAG_BASE}"
BEST_PROBE="${PROBE_DIR}/best.pt"
if [ ! -f "$BEST_PROBE" ]; then
  echo "[WARN] best.pt not at expected path, searching..."
  BEST_PROBE=$(find "${CODE_DIR}/evals" -name "best.pt" -path "*s1_vpixel_e${EPOCH}*" | head -1)
fi
echo "[INFO] Best probe: $BEST_PROBE"

PROBE_DEST="/opt/dlami/nvme/probes/salt_s1_vpixel_e${EPOCH}"
mkdir -p "$PROBE_DEST"
cp "$BEST_PROBE" "$PROBE_DEST/best.pt"
aws s3 cp "$BEST_PROBE" "${RUN_S3}/probe/best.pt" --only-show-errors || true

# =========================
# PHASE 2: CLEAN INFERENCE
# =========================
echo "[INFO] Phase 2: Clean test inference..."
python3 -m evals.main --fname "${INFER_CFG_REL}" \
  --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7
echo "[INFO] Clean inference complete"

aws s3 sync "${INFER_OUT_DIR}/" \
  "${RUN_S3}/clean/" --only-show-errors || true

# =========================
# PHASE 3: MATCHED-FRAME INFERENCE
# =========================
echo "[INFO] Phase 3: Matched-frame inference..."
FRAME_SHUFFLE=100 FRAME_SHUFFLE_TYPE=matched_frame \
OVERRIDE_TAG="neurips-salt-s1-vpixel-e${EPOCH}-end-test-mf" \
OVERRIDE_PRED_PATH="${INFER_OUT_DIR}_mf/predictions.csv" \
python3 -m evals.main --fname "${INFER_CFG_REL}" \
  --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7
echo "[INFO] Matched-frame inference complete"

aws s3 sync "${INFER_OUT_DIR}_mf/" \
  "${RUN_S3}/matched_frame/" --only-show-errors || true

upload_logs
echo "[INFO] All phases complete for SALT S1 V-Pixel e${EPOCH} trajectory probe"
