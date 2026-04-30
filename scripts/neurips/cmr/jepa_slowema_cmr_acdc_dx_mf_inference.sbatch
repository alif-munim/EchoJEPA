#!/bin/bash
#SBATCH -p dev
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 2:00:00
#SBATCH -J cmr_j_dxmf
#SBATCH -o /opt/dlami/nvme/logs/cmr_j_dxmf_%j.out
#SBATCH -e /opt/dlami/nvme/logs/cmr_j_dxmf_%j.err

# Root fs is 100% full on this cluster; redirect all temp/cache dirs to NVMe.
mkdir -p /opt/dlami/nvme/tmp /opt/dlami/nvme/cache/pip
export TMPDIR=/opt/dlami/nvme/tmp
export TEMP=/opt/dlami/nvme/tmp
export TMP=/opt/dlami/nvme/tmp
export PIP_CACHE_DIR=/opt/dlami/nvme/cache/pip
export XDG_CACHE_HOME=/opt/dlami/nvme/cache

set -euxo pipefail

# =============================================================================
# CMR JEPA slow-EMA ViT-S (job 346) -> ACDC DX: clean + matched_frame inference
# Uses probes from job 376 (cmr_jepa_dx_traj_slowema_376).
# 5 checkpoints: e30, e55, e100, e200, e295 (covers up-to-e300 request).
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
CKPT_S3="${S3_BASE}/runs/jepa_cmr_vits_slowema_346/training_folder"
PROBE_S3="${S3_BASE}/runs/cmr_jepa_dx_traj_slowema_376"
RUN_S3="${S3_BASE}/runs/cmr_j_dxmf_${SLURM_JOB_ID}"

EPOCHS="30 55 100 200 295"

upload_logs() {
  local out="/opt/dlami/nvme/logs/cmr_j_dxmf_${SLURM_JOB_ID}.out"
  local err="/opt/dlami/nvme/logs/cmr_j_dxmf_${SLURM_JOB_ID}.err"
  [ -f "${out}" ] && aws s3 cp "${out}" "${RUN_S3}/logs/job.out" --only-show-errors || true
  [ -f "${err}" ] && aws s3 cp "${err}" "${RUN_S3}/logs/job.err" --only-show-errors || true
}
trap 'upload_logs' EXIT

nvidia-smi -L || true
pkill -9 -f "evals.main" 2>/dev/null || true; sleep 2
mkdir -p /opt/dlami/nvme/logs

SCRATCH="/opt/dlami/nvme"

aws s3 cp "$ENV_S3" "${SCRATCH}/env.tar"
mkdir -p "${SCRATCH}/env" && tar -xf "${SCRATCH}/env.tar" -C "${SCRATCH}/env"
set +u && source "${SCRATCH}/env/bin/activate" && set -u
"${SCRATCH}/env/bin/conda-unpack" || true
pip install boto3 scikit-learn --no-cache-dir --quiet

# Use a job-specific code dir to avoid stale extractions from prior jobs.
CODE_DIR="${SCRATCH}/code_j_dxmf_${SLURM_JOB_ID}"
mkdir -p "$CODE_DIR"
aws s3 cp "${S3_BASE}/setup/vjepa2-src.tar.gz" "${SCRATCH}/vjepa2-src-j_dxmf.tar.gz" --quiet
tar xzf "${SCRATCH}/vjepa2-src-j_dxmf.tar.gz" -C "$CODE_DIR"
cd "$CODE_DIR"
pip install -e . --no-deps --no-build-isolation 2>/dev/null || true
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# Download ACDC clips locally
mkdir -p "${SCRATCH}/data/csv" "${SCRATCH}/data/acdc_clips" "${SCRATCH}/checkpoints" "${SCRATCH}/probes"
echo "[INFO] Downloading ACDC clips..."
aws s3 sync "${S3_BASE}/data/cmr/clips/" "${SCRATCH}/data/acdc_clips/" \
  --exclude "*" --include "acdc_*" --only-show-errors
echo "[INFO] Downloaded $(ls ${SCRATCH}/data/acdc_clips/acdc_*.mp4 | wc -l) clips"

# Create local DX CSVs
aws s3 cp "${S3_BASE}/data/csv/acdc_dx_train_s3.csv" "${SCRATCH}/data/csv/tmp_s3.csv" --quiet
aws s3 cp "${S3_BASE}/data/csv/acdc_dx_test_s3.csv"  "${SCRATCH}/data/csv/tmp_test_s3.csv" --quiet
sed "s|s3://.*clips/|${SCRATCH}/data/acdc_clips/|g" "${SCRATCH}/data/csv/tmp_s3.csv"       > "${SCRATCH}/data/csv/acdc_dx_train_local.csv"
sed "s|s3://.*clips/|${SCRATCH}/data/acdc_clips/|g" "${SCRATCH}/data/csv/tmp_test_s3.csv" > "${SCRATCH}/data/csv/acdc_dx_test_local.csv"
echo "[INFO] Local DX test CSV: $(wc -l < ${SCRATCH}/data/csv/acdc_dx_test_local.csv) clips"

# Download DX probes from job 376
echo "[INFO] Downloading DX probes..."
for EP in ${EPOCHS}; do
  mkdir -p "${SCRATCH}/probes/cmr_jepa_slowema_e${EP}_dx"
  PROBE_KEY="runs/cmr_jepa_dx_traj_slowema_376/jepa_slowema_e${EP}_acdc_dx/video_classification_frozen/cmr-jepa-slowema-vits-e${EP}-acdc-dx-d4/best.pt"
  aws s3 cp "s3://${ART_BUCKET}/vjepa2-artifacts/${PROBE_KEY}" "${SCRATCH}/probes/cmr_jepa_slowema_e${EP}_dx/best.pt" --quiet
  echo "  e${EP}: OK"
done

# Download & run clean + matched_frame inference per epoch
for EP in ${EPOCHS}; do
  CKPT_LOCAL="${SCRATCH}/checkpoints/jepa_cmr_slowema_e${EP}.pt"
  aws s3 cp "${CKPT_S3}/e${EP}.pt" "${CKPT_LOCAL}" --quiet

  CONFIG="${CODE_DIR}/configs/inference/vits16/cmr/jepa_slowema_e${EP}_acdc_dx_test.yaml"
  mkdir -p "$(dirname $CONFIG)"
  cat > "$CONFIG" <<YAMLEOF
app: vjepa
cpus_per_task: 32
folder: /opt/dlami/nvme/evals/cmr/jepa_slowema_e${EP}_acdc_dx_mf
mem_per_gpu: 80G
nodes: 1
tasks_per_node: 1
num_workers: 8

eval_name: video_classification_frozen
resume_checkpoint: true
tag: cmr-jepa-slowema-vits-e${EP}-acdc-dx-test
val_only: true
predictions_save_path: /opt/dlami/nvme/evals/cmr/jepa_slowema_e${EP}_acdc_dx_mf/predictions.csv
probe_checkpoint: ${SCRATCH}/probes/cmr_jepa_slowema_e${EP}_dx/best.pt

experiment:
  classifier:
    num_heads: 6
    num_probe_blocks: 4
    num_classes: 5
  data:
    dataset_type: VideoDataset
    dataset_train: ${SCRATCH}/data/csv/acdc_dx_test_local.csv
    dataset_val: ${SCRATCH}/data/csv/acdc_dx_test_local.csv
    num_classes: 5
    resolution: 224
    frames_per_clip: 16
    frame_step: 1
    num_segments: 1
    num_views_per_segment: 1
  optimization:
    batch_size: 4
    num_epochs: 1
    use_bfloat16: true
    use_pos_embed: false
    multihead_kwargs:
    - {final_lr: 0.0, final_weight_decay: 0.01, lr: 0.0, start_lr: 0.0, warmup: 0.0, weight_decay: 0.01}
    - {final_lr: 0.0, final_weight_decay: 0.1,  lr: 0.0, start_lr: 0.0, warmup: 0.0, weight_decay: 0.1}
    - {final_lr: 0.0, final_weight_decay: 0.4,  lr: 0.0, start_lr: 0.0, warmup: 0.0, weight_decay: 0.4}
    - {final_lr: 0.0, final_weight_decay: 0.01, lr: 0.0, start_lr: 0.0, warmup: 0.0, weight_decay: 0.01}
    - {final_lr: 0.0, final_weight_decay: 0.1,  lr: 0.0, start_lr: 0.0, warmup: 0.0, weight_decay: 0.1}
    - {final_lr: 0.0, final_weight_decay: 0.4,  lr: 0.0, start_lr: 0.0, warmup: 0.0, weight_decay: 0.4}
model_kwargs:
  checkpoint: ${CKPT_LOCAL}
  module_name: evals.video_classification_frozen.modelcustom.vit_encoder_multiclip
  pretrain_kwargs:
    encoder:
      checkpoint_key: target_encoder
      model_name: vit_small
      patch_size: 16
      tubelet_size: 2
      uniform_power: true
      use_rope: true
  wrapper_kwargs:
    max_frames: 128
    use_pos_embed: false
YAMLEOF

  # CLEAN
  echo "[INFO] CLEAN DX: jepa-slowema e${EP}"
  OVERRIDE_TAG="cmr-jepa-slowema-vits-e${EP}-acdc-dx-clean" \
  OVERRIDE_PRED_PATH="/opt/dlami/nvme/evals/cmr/jepa_slowema_e${EP}_acdc_dx_clean/predictions.csv" \
  python3 -m evals.main --fname "${CONFIG}" \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7
  echo "[INFO] jepa-slowema e${EP} DX clean done"

  # MATCHED_FRAME
  echo "[INFO] MATCHED_FRAME DX: jepa-slowema e${EP}"
  FRAME_SHUFFLE=100 FRAME_SHUFFLE_TYPE=matched_frame \
  OVERRIDE_TAG="cmr-jepa-slowema-vits-e${EP}-acdc-dx-mf" \
  OVERRIDE_PRED_PATH="/opt/dlami/nvme/evals/cmr/jepa_slowema_e${EP}_acdc_dx_mf/predictions.csv" \
  python3 -m evals.main --fname "${CONFIG}" \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7
  echo "[INFO] jepa-slowema e${EP} DX matched_frame done"

  # Sync per-epoch output folder to S3
  aws s3 sync "/opt/dlami/nvme/evals/cmr/jepa_slowema_e${EP}_acdc_dx_mf/" \
    "${RUN_S3}/jepa_slowema_e${EP}_acdc_dx_mf/" --only-show-errors || true
  aws s3 sync "/opt/dlami/nvme/evals/cmr/jepa_slowema_e${EP}_acdc_dx_clean/" \
    "${RUN_S3}/jepa_slowema_e${EP}_acdc_dx_clean/" --only-show-errors || true

  upload_logs
  rm -f "${CKPT_LOCAL}"
done

echo "[INFO] All CMR JEPA slow-EMA DX clean + matched_frame inference complete"
