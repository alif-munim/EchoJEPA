#!/bin/bash
#SBATCH -p dev
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 04:00:00
#SBATCH -J pvl_lvef_test
#SBATCH -o /opt/dlami/nvme/logs/pvl_lvef_test_%j.out
#SBATCH -e /opt/dlami/nvme/logs/pvl_lvef_test_%j.err

set -euxo pipefail

# =============================================================================
# Test inference for phase_mask + localblock LVEF probes on EchoNet-Dynamic test.
# Picks up probes from two different training runs:
#   phase_mask e25, e50 and localblock e25 -> PM_TRAIN_RUN_S3 (job 396)
#   localblock e50 -> LB_TRAIN_RUN_S3 (new lb-only job)
# Requires --export=PM_TRAIN_RUN_S3=s3://..., LB_TRAIN_RUN_S3=s3://...
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

PM_RUN_S3="${S3_BASE}/runs/phase_mask_389/checkpoints"
LB_RUN_S3="${S3_BASE}/runs/localblock_393/checkpoints"

: "${PM_TRAIN_RUN_S3:?Must pass PM_TRAIN_RUN_S3 via --export}"
: "${LB_TRAIN_RUN_S3:?Must pass LB_TRAIN_RUN_S3 via --export}"

RUN_S3="${S3_BASE}/runs/phase_vs_localblock_lvef_test_${SLURM_JOB_ID}"

# Jobs: "<model>:<epoch>:<train_run_key>"  (train_run_key = pm|lb)
JOBS=(
  "phase_mask:25:pm"
  "phase_mask:50:pm"
  "localblock:25:lb"
  "localblock:50:lb"
)

upload_logs() {
  local out="/opt/dlami/nvme/logs/pvl_lvef_test_${SLURM_JOB_ID}.out"
  local err="/opt/dlami/nvme/logs/pvl_lvef_test_${SLURM_JOB_ID}.err"
  [ -f "${out}" ] && aws s3 cp "${out}" "${RUN_S3}/logs/job.out" --only-show-errors || true
  [ -f "${err}" ] && aws s3 cp "${err}" "${RUN_S3}/logs/job.err" --only-show-errors || true
}
trap 'upload_logs' EXIT

nvidia-smi -L || true
mkdir -p /opt/dlami/nvme/logs

SCRATCH="/opt/dlami/nvme"
WORKDIR="${SCRATCH}/pvl_lvef_test_${SLURM_JOB_ID}"
mkdir -p "$WORKDIR"

aws s3 cp "$ENV_S3" "${WORKDIR}/env.tar"
mkdir -p "${WORKDIR}/env" && tar -xf "${WORKDIR}/env.tar" -C "${WORKDIR}/env"
set +u && source "${WORKDIR}/env/bin/activate" && set -u
"${WORKDIR}/env/bin/conda-unpack" || true
export TMPDIR="${WORKDIR}/tmp" && mkdir -p "$TMPDIR"
pip install boto3 --no-cache-dir --quiet

CODE_DIR="${WORKDIR}/code"
mkdir -p "$CODE_DIR"
aws s3 cp "${S3_BASE}/setup/vjepa2-src.tar.gz" "${WORKDIR}/vjepa2-src.tar.gz" --quiet
tar xzf "${WORKDIR}/vjepa2-src.tar.gz" -C "$CODE_DIR"
cd "$CODE_DIR"
pip install -e . --no-deps --no-build-isolation 2>/dev/null || true
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

DATA_DIR="${WORKDIR}/data/csv"
mkdir -p "$DATA_DIR"
aws s3 cp "${S3_BASE}/data/csv/echonet_dynamic_test_s3_raw.csv" "$DATA_DIR/" --quiet

CKPT_DIR="${WORKDIR}/checkpoints"
mkdir -p "$CKPT_DIR"
PROBE_DIR="${WORKDIR}/probes"
mkdir -p "$PROBE_DIR"
PRED_DIR="${WORKDIR}/predictions"
mkdir -p "$PRED_DIR"

for J in "${JOBS[@]}"; do
  MODEL="${J%%:*}"
  REST="${J#*:}"
  EP="${REST%%:*}"
  RUN_KEY="${REST##*:}"

  if [ "$MODEL" = "phase_mask" ]; then
    BACKBONE_SRC="${PM_RUN_S3}/e${EP}.pt"
  else
    BACKBONE_SRC="${LB_RUN_S3}/e${EP}.pt"
  fi
  BACKBONE_LOCAL="${CKPT_DIR}/${MODEL}_e${EP}.pt"
  echo "[INFO] Downloading backbone ${MODEL} e${EP}..."
  aws s3 cp "$BACKBONE_SRC" "$BACKBONE_LOCAL" --quiet

  TAG="neurips-${MODEL//_/-}-e${EP}-end-lvef-d4"
  if [ "$RUN_KEY" = "pm" ]; then
    PROBE_S3="${PM_TRAIN_RUN_S3}/${MODEL}_e${EP}_lvef/video_classification_frozen/${TAG}/best.pt"
  else
    PROBE_S3="${LB_TRAIN_RUN_S3}/${MODEL}_e${EP}_lvef/video_classification_frozen/${TAG}/best.pt"
  fi
  PROBE_LOCAL="${PROBE_DIR}/${MODEL}_e${EP}_best.pt"
  echo "[INFO] Downloading probe ${MODEL} e${EP} from ${PROBE_S3}..."
  aws s3 cp "$PROBE_S3" "$PROBE_LOCAL" --quiet
  ls -lh "$PROBE_LOCAL"
done

for J in "${JOBS[@]}"; do
  MODEL="${J%%:*}"
  REST="${J#*:}"
  EP="${REST%%:*}"

  TAG="neurips-${MODEL//_/-}-e${EP}-end-lvef-d4-test"
  FOLDER="evals/vitl/neurips/${MODEL}_e${EP}_end_lvef_test"
  EVAL_CFG="configs/inference/vitl/neurips/${MODEL}_e${EP}_end_lvef_test.yaml"
  PRED_CSV="${PRED_DIR}/${MODEL}_e${EP}_end_lvef_test.csv"
  mkdir -p "$(dirname $EVAL_CFG)" "$FOLDER"

  cat > "$EVAL_CFG" <<YAMLEOF
app: vjepa
cpus_per_task: 32
folder: ${FOLDER}
mem_per_gpu: 80G
nodes: 1
tasks_per_node: 1
num_workers: 8
eval_name: video_classification_frozen
resume_checkpoint: true
tag: ${TAG}
val_only: true
predictions_save_path: ${PRED_CSV}
probe_checkpoint: ${PROBE_DIR}/${MODEL}_e${EP}_best.pt
experiment:
  classifier:
    task_type: regression
    num_heads: 16
    num_probe_blocks: 4
    num_targets: 1
  data:
    dataset_type: VideoDataset
    dataset_train: ${DATA_DIR}/echonet_dynamic_test_s3_raw.csv
    dataset_val: ${DATA_DIR}/echonet_dynamic_test_s3_raw.csv
    num_classes: 1
    resolution: 224
    frames_per_clip: 16
    frame_step: 2
    num_segments: 2
    num_views_per_segment: 1
    target_mean: 55.7776
    target_std: 12.4072
  optimization:
    batch_size: 8
    num_epochs: 1
    use_bfloat16: true
    use_pos_embed: false
    multihead_kwargs:
    - {final_lr: 0.0, final_weight_decay: 0.01, lr: 0.0, start_lr: 0.0, warmup: 0.0, weight_decay: 0.01}
    - {final_lr: 0.0, final_weight_decay: 0.1, lr: 0.0, start_lr: 0.0, warmup: 0.0, weight_decay: 0.1}
    - {final_lr: 0.0, final_weight_decay: 0.4, lr: 0.0, start_lr: 0.0, warmup: 0.0, weight_decay: 0.4}
    - {final_lr: 0.0, final_weight_decay: 0.01, lr: 0.0, start_lr: 0.0, warmup: 0.0, weight_decay: 0.01}
    - {final_lr: 0.0, final_weight_decay: 0.1, lr: 0.0, start_lr: 0.0, warmup: 0.0, weight_decay: 0.1}
    - {final_lr: 0.0, final_weight_decay: 0.4, lr: 0.0, start_lr: 0.0, warmup: 0.0, weight_decay: 0.4}
model_kwargs:
  checkpoint: ${CKPT_DIR}/${MODEL}_e${EP}.pt
  module_name: evals.video_classification_frozen.modelcustom.vit_encoder_multiclip
  pretrain_kwargs:
    encoder:
      checkpoint_key: target_encoder
      img_temporal_dim_size: null
      model_name: vit_large
      patch_size: 16
      tubelet_size: 2
      uniform_power: true
      use_rope: true
  wrapper_kwargs:
    max_frames: 128
    use_pos_embed: false
YAMLEOF

  echo "============================================"
  echo "[INFO] Test inference for ${MODEL} e${EP}..."
  echo "============================================"
  python3 -m evals.main --fname "$EVAL_CFG" \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7

  aws s3 sync "$FOLDER/" "${RUN_S3}/${MODEL}_e${EP}_test/" --only-show-errors || true
  [ -f "$PRED_CSV" ] && aws s3 cp "$PRED_CSV" "${RUN_S3}/predictions/${MODEL}_e${EP}_end_lvef_test.csv" --only-show-errors || true
  upload_logs
done

echo "[INFO] All 4-way phase_vs_localblock LVEF test inference complete"
