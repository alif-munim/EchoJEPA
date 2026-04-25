#!/bin/bash
#SBATCH -p dev
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 1-00:00:00
#SBATCH -J cmr_jdx_sema
#SBATCH -o /opt/dlami/nvme/logs/cmr_jepa_dx_sema_%j.out
#SBATCH -e /opt/dlami/nvme/logs/cmr_jepa_dx_sema_%j.err
#SBATCH --signal=B:USR1@300

set -euxo pipefail

# =============================================================================
# V-JEPA ViT-S CMR slow-EMA trajectory -> ACDC 5-class diagnosis probes
# =============================================================================
# Trains d=4 attentive probes at e30, e55, e100, e200, e295 on the ACDC holdout.
# Checkpoints are from job 346 (ema=[0.99925, 0.99925], cancelled at e300).
# Matches the MAE DX trajectory protocol (job 210) up to e300.
# ACDC classes: DCM=0, HCM=1, MINF=2, NOR=3, RV=4 (balanced).
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
ENV_S3="${S3_BASE}/env/vjepa2-312-pack-env.tar"
CKPT_S3="${S3_BASE}/runs/jepa_cmr_vits_slowema_346/training_folder"
RUN_S3="${S3_BASE}/runs/cmr_jepa_dx_traj_slowema_${SLURM_JOB_ID}"

DISPLAY_EPOCHS="30 55 100 200 295"

# =========================
# LOGGING / SYNC
# =========================
mkdir -p /opt/dlami/nvme/logs

upload_logs() {
  local out="/opt/dlami/nvme/logs/cmr_jepa_dx_sema_${SLURM_JOB_ID}.out"
  local err="/opt/dlami/nvme/logs/cmr_jepa_dx_sema_${SLURM_JOB_ID}.err"
  [ -f "${out}" ] && aws s3 cp "${out}" "${RUN_S3}/logs/job.out" --only-show-errors || true
  [ -f "${err}" ] && aws s3 cp "${err}" "${RUN_S3}/logs/job.err" --only-show-errors || true
}

sync_results() {
  for ep in ${DISPLAY_EPOCHS}; do
    local outdir="/opt/dlami/nvme/evals/cmr/jepa_slowema_e${ep}_acdc_dx"
    if [ -d "${outdir}" ]; then
      aws s3 sync "${outdir}" "${RUN_S3}/jepa_slowema_e${ep}_acdc_dx/" \
        --exclude "*" \
        --include "*.csv" --include "*.json" --include "*.pt" --include "*.npz" \
        --only-show-errors || true
    fi
  done
}

on_exit() { rc=$?; upload_logs; sync_results; exit $rc; }
trap on_exit EXIT
trap 'sync_results; upload_logs' USR1

# =========================
# SETUP
# =========================
nvidia-smi -L || true
pkill -9 -f "evals.main" 2>/dev/null || true; sleep 2

SCRATCH="/opt/dlami/nvme"
WORKDIR="${SCRATCH}/cmr_jepa_dx_sema_${SLURM_JOB_ID}"
mkdir -p "$WORKDIR"

# Env
aws s3 cp "$ENV_S3" "${WORKDIR}/env.tar"
mkdir -p "${WORKDIR}/env" && tar -xf "${WORKDIR}/env.tar" -C "${WORKDIR}/env"
set +u && source "${WORKDIR}/env/bin/activate" && set -u
"${WORKDIR}/env/bin/conda-unpack" || true
export TMPDIR="${WORKDIR}/tmp" && mkdir -p "$TMPDIR"
pip install boto3 scikit-learn --no-cache-dir --quiet

# Code
CODE_DIR="${WORKDIR}/code"
mkdir -p "$CODE_DIR"
aws s3 cp "${S3_BASE}/setup/vjepa2-src.tar.gz" "${WORKDIR}/vjepa2-src.tar.gz" --quiet
tar xzf "${WORKDIR}/vjepa2-src.tar.gz" -C "$CODE_DIR"
cd "$CODE_DIR"
pip install -e . --no-deps --no-build-isolation 2>/dev/null || true
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# =========================
# DOWNLOAD ACDC CLIPS LOCALLY (avoids S3 credential expiration during training)
# =========================
mkdir -p "${SCRATCH}/data/csv" "${SCRATCH}/data/acdc_clips" "${SCRATCH}/checkpoints"
echo "[INFO] Downloading ACDC clips to local NVMe..."
aws s3 sync "${S3_BASE}/data/cmr/clips/" "${SCRATCH}/data/acdc_clips/" \
  --exclude "*" --include "acdc_*" --only-show-errors
echo "[INFO] Downloaded $(ls ${SCRATCH}/data/acdc_clips/acdc_*.mp4 | wc -l) ACDC clips"

# Rewrite S3 paths to local paths
aws s3 cp "${S3_BASE}/data/csv/acdc_dx_train_s3.csv" "${SCRATCH}/data/csv/acdc_dx_s3.csv" --quiet
aws s3 cp "${S3_BASE}/data/csv/acdc_dx_test_s3.csv" "${SCRATCH}/data/csv/acdc_dx_test_s3.csv" --quiet
sed "s|s3://.*clips/|${SCRATCH}/data/acdc_clips/|g" "${SCRATCH}/data/csv/acdc_dx_s3.csv" > "${SCRATCH}/data/csv/acdc_dx_train_local.csv"
sed "s|s3://.*clips/|${SCRATCH}/data/acdc_clips/|g" "${SCRATCH}/data/csv/acdc_dx_test_s3.csv" > "${SCRATCH}/data/csv/acdc_dx_test_local.csv"
echo "[INFO] Local train CSV: $(wc -l < ${SCRATCH}/data/csv/acdc_dx_train_local.csv) clips"
echo "[INFO] Local test CSV: $(wc -l < ${SCRATCH}/data/csv/acdc_dx_test_local.csv) clips"

# =========================
# GENERATE CONFIGS AND RUN PROBES
# =========================
for EP in ${DISPLAY_EPOCHS}; do
  echo "============================================================"
  echo "[INFO] Probing CMR JEPA slow-EMA ViT-S e${EP} (ACDC DX)"
  echo "============================================================"

  CKPT_LOCAL="${SCRATCH}/checkpoints/jepa_cmr_slowema_e${EP}.pt"
  aws s3 cp "${CKPT_S3}/e${EP}.pt" "${CKPT_LOCAL}" --quiet
  echo "[INFO] Checkpoint: $(ls -lh ${CKPT_LOCAL})"

  CONFIG="configs/eval/vits16/cmr/jepa_cmr_slowema_e${EP}_acdc_dx_d4.yaml"
  mkdir -p "$(dirname $CONFIG)"
  cat > "$CONFIG" <<YAMLEOF
app: vjepa
cpus_per_task: 96
folder: /opt/dlami/nvme/evals/cmr/jepa_slowema_e${EP}_acdc_dx
mem_per_gpu: 80G
nodes: 1
tasks_per_node: 1
num_workers: 8
eval_name: video_classification_frozen
resume_checkpoint: false
tag: cmr-jepa-slowema-vits-e${EP}-acdc-dx-d4
experiment:
  classifier:
    num_heads: 6
    num_probe_blocks: 4
    num_classes: 5
  data:
    dataset_type: VideoDataset
    dataset_train: ${SCRATCH}/data/csv/acdc_dx_train_local.csv
    dataset_val: ${SCRATCH}/data/csv/acdc_dx_test_local.csv
    num_classes: 5
    resolution: 224
    frames_per_clip: 16
    frame_step: 1
    num_segments: 2
    num_views_per_segment: 1
  optimization:
    batch_size: 1
    num_epochs: 30
    use_bfloat16: true
    use_pos_embed: false
    multihead_kwargs:
    - {lr: 0.0001, start_lr: 0.0001, final_lr: 0.0, warmup: 0.0, weight_decay: 0.01, final_weight_decay: 0.01}
    - {lr: 0.0001, start_lr: 0.0001, final_lr: 0.0, warmup: 0.0, weight_decay: 0.1, final_weight_decay: 0.1}
    - {lr: 0.0001, start_lr: 0.0001, final_lr: 0.0, warmup: 0.0, weight_decay: 0.4, final_weight_decay: 0.4}
    - {lr: 0.00005, start_lr: 0.00005, final_lr: 0.0, warmup: 0.0, weight_decay: 0.01, final_weight_decay: 0.01}
    - {lr: 0.00005, start_lr: 0.00005, final_lr: 0.0, warmup: 0.0, weight_decay: 0.1, final_weight_decay: 0.1}
    - {lr: 0.00005, start_lr: 0.00005, final_lr: 0.0, warmup: 0.0, weight_decay: 0.4, final_weight_decay: 0.4}
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

  echo "[INFO] Config written: $CONFIG"

  python3 -m evals.main \
    --fname "${CONFIG}" \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7
  echo "[INFO] e${EP} probe finished"

  sync_results
  upload_logs
  rm -f "${CKPT_LOCAL}"
done

echo "[INFO] All CMR JEPA slow-EMA DX trajectory probes complete"
