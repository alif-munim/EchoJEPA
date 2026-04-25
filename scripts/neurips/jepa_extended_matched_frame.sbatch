#!/bin/bash
#SBATCH -p dev
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 6:00:00
#SBATCH -J jepa_ext_mf
#SBATCH -o /opt/dlami/nvme/logs/jepa_ext_mf_%j.out
#SBATCH -e /opt/dlami/nvme/logs/jepa_ext_mf_%j.err

set -euxo pipefail

# =============================================================================
# JEPA IN21K extended: matched_frame inference on EchoNet-Dynamic LVEF test set
# for e125/e150/e175/e200. Probes already trained (runs/jepa_ext_probes_332/).
#
# Produces paired clean + matched_frame inference per checkpoint -> Δ R² trend
# through extended JEPA training (the missing half of the trajectory figure).
# =============================================================================

export AWS_DEFAULT_REGION="us-west-2"
export AWS_REGION="us-west-2"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export AWS_SDK_LOAD_CONFIG=1
export HOME="/home/ubuntu"
export AWS_SHARED_CREDENTIALS_FILE="${HOME}/.aws/credentials"
export AWS_CONFIG_FILE="${HOME}/.aws/config"
unset AWS_EC2_METADATA_DISABLED 2>/dev/null || true
export AWS_METADATA_SERVICE_TIMEOUT=10
export AWS_METADATA_SERVICE_NUM_ATTEMPTS=5

aws sts get-caller-identity || exit 1

ART_BUCKET="sagemaker-hyperpod-lifecycle-495467399120-usw2"
S3_BASE="s3://${ART_BUCKET}/vjepa2-artifacts"
ENV_S3="${S3_BASE}/env/vjepa2-312-pack-env.tar"

# Encoder source: JEPA IN21K extended-training run (280) with ckpts every 25ep.
JEPA_RUN_S3="${S3_BASE}/runs/jepa_in21k_e200_280/training_folder"
# Probe source: extended probes from job 332 on e125/150/175/200.
PROBE_RUN_S3="${S3_BASE}/runs/jepa_ext_probes_332"
RUN_S3="${S3_BASE}/runs/jepa_ext_mf_${SLURM_JOB_ID}"

EPOCHS="125 150 175 200"

upload_logs() {
  local out="/opt/dlami/nvme/logs/jepa_ext_mf_${SLURM_JOB_ID}.out"
  local err="/opt/dlami/nvme/logs/jepa_ext_mf_${SLURM_JOB_ID}.err"
  [ -f "${out}" ] && aws s3 cp "${out}" "${RUN_S3}/logs/job.out" --only-show-errors || true
  [ -f "${err}" ] && aws s3 cp "${err}" "${RUN_S3}/logs/job.err" --only-show-errors || true
}
trap 'upload_logs' EXIT

# =========================
# SETUP
# =========================
nvidia-smi -L || true
mkdir -p /opt/dlami/nvme/logs
pkill -9 -f "evals.main" 2>/dev/null || true; sleep 2

SCRATCH="/opt/dlami/nvme"

aws s3 cp "$ENV_S3" "${SCRATCH}/env.tar"
mkdir -p "${SCRATCH}/env" && tar -xf "${SCRATCH}/env.tar" -C "${SCRATCH}/env"
set +u && source "${SCRATCH}/env/bin/activate" && set -u
"${SCRATCH}/env/bin/conda-unpack" || true
# /tmp is not always writable on compute nodes after preemption; use NVMe.
export TMPDIR="${SCRATCH}/tmp_jepa_ext_mf_${SLURM_JOB_ID}" && mkdir -p "$TMPDIR"
pip install boto3 --no-cache-dir --quiet

CODE_DIR="${SCRATCH}/code_jepa_ext_mf"
mkdir -p "$CODE_DIR"
aws s3 cp "${S3_BASE}/setup/vjepa2-src.tar.gz" "${SCRATCH}/vjepa2-src.tar.gz" --quiet
tar xzf "${SCRATCH}/vjepa2-src.tar.gz" -C "$CODE_DIR"
cd "$CODE_DIR"
pip install -e . --no-deps --no-build-isolation 2>/dev/null || true
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

mkdir -p "${SCRATCH}/data/csv" "${SCRATCH}/checkpoints" "${SCRATCH}/probes"
aws s3 cp "${S3_BASE}/data/csv/echonet_dynamic_train_s3_raw.csv" "${SCRATCH}/data/csv/" --quiet
aws s3 cp "${S3_BASE}/data/csv/echonet_dynamic_test_s3_raw.csv" "${SCRATCH}/data/csv/" --quiet

# Download all probes
for EP in ${EPOCHS}; do
  mkdir -p "${SCRATCH}/probes/jepa_in21k_e${EP}_end"
  aws s3 cp "${PROBE_RUN_S3}/jepa_e${EP}_lvef/video_classification_frozen/neurips-jepa-in21k-e${EP}-end-lvef-d4/best.pt" \
    "${SCRATCH}/probes/jepa_in21k_e${EP}_end/best.pt" --quiet
  ls -lh "${SCRATCH}/probes/jepa_in21k_e${EP}_end/best.pt"
done

# =========================
# GENERATE INFERENCE CONFIGS
# =========================
CFG_DIR="${CODE_DIR}/configs/inference/vitl/neurips/echonet-dynamic"
mkdir -p "$CFG_DIR"
for EP in ${EPOCHS}; do
  CFG="${CFG_DIR}/jepa_in21k_e${EP}_end_test.yaml"
  cat > "$CFG" <<YAMLEOF
app: vjepa
folder: /opt/dlami/nvme/evals/neurips/jepa_e${EP}_end_test
eval_name: video_classification_frozen
resume_checkpoint: true
val_only: true
tag: neurips-jepa-in21k-e${EP}-end-test
probe_checkpoint: /opt/dlami/nvme/probes/jepa_in21k_e${EP}_end/best.pt
predictions_save_path: /opt/dlami/nvme/evals/neurips/jepa_e${EP}_end_test/predictions.csv
experiment:
  classifier:
    task_type: regression
    num_heads: 16
    num_probe_blocks: 4
    num_targets: 1
  data:
    dataset_type: VideoDataset
    dataset_train: /opt/dlami/nvme/data/csv/echonet_dynamic_train_s3_raw.csv
    dataset_val: /opt/dlami/nvme/data/csv/echonet_dynamic_test_s3_raw.csv
    target_mean: 55.7776
    target_std: 12.4072
    resolution: 224
    frames_per_clip: 16
    frame_step: 2
    num_segments: 2
    num_views_per_segment: 1
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
  checkpoint: /opt/dlami/nvme/checkpoints/jepa_in21k_vitl_e${EP}.pt
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
  echo "[INFO] Wrote $CFG"
done

# =========================
# RUN CLEAN + MATCHED_FRAME
# =========================
for EP in ${EPOCHS}; do
  echo "============================================================"
  echo "[INFO] Processing JEPA IN21K e${EP}"
  echo "============================================================"

  # Resolve source key. e200 is saved as latest.pt in the 280 run.
  if [ "$EP" = "200" ]; then
    SRC="${JEPA_RUN_S3}/latest.pt"
  else
    SRC="${JEPA_RUN_S3}/e${EP}.pt"
  fi
  CKPT_LOCAL="${SCRATCH}/checkpoints/jepa_in21k_vitl_e${EP}.pt"
  aws s3 cp "${SRC}" "${CKPT_LOCAL}" --quiet
  ls -lh "${CKPT_LOCAL}"

  CONFIG="${CFG_DIR}/jepa_in21k_e${EP}_end_test.yaml"

  # --- CLEAN ---
  echo "[INFO] CLEAN: e${EP}"
  python3 -m evals.main --fname "${CONFIG}" \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7
  echo "[INFO] e${EP} clean done"

  # Stage clean predictions
  aws s3 cp "/opt/dlami/nvme/evals/neurips/jepa_e${EP}_end_test/predictions.csv" \
    "${RUN_S3}/jepa_e${EP}/clean/predictions.csv" --only-show-errors || true

  # --- MATCHED_FRAME (override tag + out path so the CSV doesn't clobber) ---
  echo "[INFO] MATCHED_FRAME: e${EP}"
  FRAME_SHUFFLE=100 FRAME_SHUFFLE_TYPE=matched_frame \
  OVERRIDE_TAG="neurips-jepa-in21k-e${EP}-end-test-mf" \
  OVERRIDE_PRED_PATH="/opt/dlami/nvme/evals/neurips/jepa_e${EP}_end_test_mf/predictions.csv" \
  python3 -m evals.main --fname "${CONFIG}" \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7
  echo "[INFO] e${EP} matched_frame done"

  aws s3 cp "/opt/dlami/nvme/evals/neurips/jepa_e${EP}_end_test_mf/predictions.csv" \
    "${RUN_S3}/jepa_e${EP}/matched_frame/predictions.csv" --only-show-errors || true

  upload_logs
  rm -f "${CKPT_LOCAL}"
done

echo "[INFO] All 4 JEPA extended checkpoints × 2 conditions complete"
