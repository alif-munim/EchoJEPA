#!/bin/bash
# Frozen-encoder phase probe TEST-set inference (sin, finalbudget single-view +25 (job 548)).
# Requires PROBE_JOB env var.
#SBATCH -p ml-p5-48xlarge
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -t 0-01:00:00
#SBATCH -J phase_sv_548_sin_test
#SBATCH -o /opt/dlami/nvme/logs/phase_sv_548_sin_test-%j.out
#SBATCH -e /opt/dlami/nvme/logs/phase_sv_548_sin_test-%j.err
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

PILOT_NAME="phase_sv_548_sin"
PRETRAIN_CKPT_S3="s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts/runs/final_sv_25_pairs24_548/checkpoints/latest.pt"
: "${PROBE_JOB:?PROBE_JOB env var required — SLURM id of matching phase train probe}"
PROBE_TAG="neurips-phase-sv_548-sin"

ART_BUCKET="sagemaker-hyperpod-lifecycle-495467399120-usw2"
S3_BASE="s3://${ART_BUCKET}/vjepa2-artifacts"
ENV_S3="${S3_BASE}/env/vjepa2-312-pack-env.tar"
SRC_TARBALL_S3="${S3_BASE}/setup/vjepa2-src.tar.gz"
PROBE_CKPT_S3="${S3_BASE}/runs/${PILOT_NAME}_${PROBE_JOB}/probe/video_classification_frozen/${PROBE_TAG}/best.pt"
RUN_S3="${S3_BASE}/runs/${PILOT_NAME}_test_${SLURM_JOB_ID}"

upload_logs() {
  local out="/opt/dlami/nvme/logs/${PILOT_NAME}_test-${SLURM_JOB_ID}.out"
  local err="/opt/dlami/nvme/logs/${PILOT_NAME}_test-${SLURM_JOB_ID}.err"
  [ -f "${out}" ] && aws s3 cp "${out}" "${RUN_S3}/logs/job.out" --only-show-errors || true
  [ -f "${err}" ] && aws s3 cp "${err}" "${RUN_S3}/logs/job.err" --only-show-errors || true
}
trap 'upload_logs' EXIT

aws sts get-caller-identity || exit 1
nvidia-smi -L || true
mkdir -p /opt/dlami/nvme/logs

SCRATCH="/opt/dlami/nvme"
WORKDIR="${SCRATCH}/${PILOT_NAME}_test_${SLURM_JOB_ID}"
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
pip install -e . --no-deps --no-build-isolation 2>/dev/null || true
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

DATA_DIR="${WORKDIR}/data/csv"
mkdir -p "$DATA_DIR"
aws s3 cp "${S3_BASE}/data/csv/mimic_phase_sin_test_10k.csv" "$DATA_DIR/" --quiet

CKPT_DIR="${WORKDIR}/checkpoints"
PROBE_DIR="${WORKDIR}/probes"
PRED_DIR="${WORKDIR}/predictions"
mkdir -p "$CKPT_DIR" "$PROBE_DIR" "$PRED_DIR"

echo "[INFO] Downloading pretrain ckpt + probe best.pt..."
aws s3 cp "$PRETRAIN_CKPT_S3" "${CKPT_DIR}/pretrain.pt" --quiet
aws s3 cp "$PROBE_CKPT_S3"    "${PROBE_DIR}/probe_best.pt" --quiet
ls -lh "${CKPT_DIR}/pretrain.pt" "${PROBE_DIR}/probe_best.pt"

FOLDER="evals/vitl/neurips/phase_probe/${PILOT_NAME}_test"
PRED_CSV="${PRED_DIR}/${PILOT_NAME}_test.csv"
EVAL_CFG="configs/inference/vitl/neurips/${PILOT_NAME}_test.yaml"
mkdir -p "$(dirname $EVAL_CFG)" "$FOLDER"

cat > "$EVAL_CFG" <<YAMLEOF
app: vjepa
cpus_per_task: 16
folder: ${FOLDER}
mem_per_gpu: 80G
nodes: 1
tasks_per_node: 1
num_workers: 8
eval_name: video_classification_frozen
resume_checkpoint: true
tag: ${PROBE_TAG}-test
val_only: true
predictions_save_path: ${PRED_CSV}
probe_checkpoint: ${PROBE_DIR}/probe_best.pt
experiment:
  classifier:
    task_type: regression
    num_heads: 16
    num_probe_blocks: 4
    num_targets: 1
  data:
    dataset_type: VideoDataset
    dataset_train: ${DATA_DIR}/mimic_phase_sin_test_10k.csv
    dataset_val:   ${DATA_DIR}/mimic_phase_sin_test_10k.csv
    num_classes: 1
    target_mean: -0.0153
    target_std: 0.6618
    resolution: 224
    frames_per_clip: 16
    frame_step: 2
    num_segments: 1
    num_views_per_segment: 1
  optimization:
    batch_size: 1
    num_epochs: 1
    use_bfloat16: true
    use_pos_embed: false
    multihead_kwargs:
    - {final_lr: 0.0, final_weight_decay: 0.1, lr: 0.0001, start_lr: 0.0001, warmup: 0.0, weight_decay: 0.1}
model_kwargs:
  checkpoint: ${CKPT_DIR}/pretrain.pt
  module_name: evals.video_classification_frozen.modelcustom.vit_encoder_multiclip
  pretrain_kwargs:
    encoder:
      checkpoint_key: target_encoder
      model_name: vit_large
      patch_size: 16
      tubelet_size: 2
      uniform_power: true
      use_rope: true
  wrapper_kwargs:
    max_frames: 128
    use_pos_embed: false
YAMLEOF

echo "[INFO] Running ${PILOT_NAME} test inference..."
python3 -m evals.main --fname "$EVAL_CFG" --devices cuda:0 --val_only

aws s3 sync "${FOLDER}/" "${RUN_S3}/eval/" --only-show-errors || true
[ -f "$PRED_CSV" ] && aws s3 cp "$PRED_CSV" "${RUN_S3}/predictions/$(basename $PRED_CSV)" --only-show-errors || true
upload_logs
echo "[INFO] ${PILOT_NAME} test inference complete"
