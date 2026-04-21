#!/bin/bash
#SBATCH -p dev
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 6:00:00
#SBATCH -J mae_eshuf
#SBATCH -o /tmp/mae_eshuf_%j.out
#SBATCH -e /tmp/mae_eshuf_%j.err
#SBATCH --signal=B:USR1@300

set -euxo pipefail

# =============================================================================
# EchoMAE-L early trajectory → EchoNet-Dynamic LVEF test: clean + shuffled
# =============================================================================
# e25, e50, e75, e99 — complements the e124-e194 run (job 210).
# Uses pre-trained probes from ICML runs (already on S3).
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
export AWS_METADATA_SERVICE_TIMEOUT=5
export AWS_METADATA_SERVICE_NUM_ATTEMPTS=3

echo "[PREFLIGHT] Checking AWS credentials..."
aws sts get-caller-identity || { echo "[FATAL] AWS CLI credentials check failed"; exit 1; }

ART_BUCKET="sagemaker-hyperpod-lifecycle-495467399120-usw2"
S3_BASE="s3://${ART_BUCKET}/vjepa2-artifacts"
ENV_S3="${S3_BASE}/env/vjepa2-312-pack-env.tar"
RUN_S3="${S3_BASE}/runs/mae_eshuf_test_${SLURM_JOB_ID}"

EPOCHS="25 50 75 99"

# Encoder checkpoint names (0-indexed)
declare -A CKPT_NAMES=( [25]="echomae_l_mimic_ep24" [50]="echomae_l_mimic_ep50" [75]="echomae_l_mimic_ep74" [99]="echomae_l_mimic_ep99" )

# Probe S3 paths
declare -A PROBE_S3=(
  [25]="${S3_BASE}/probes/echomae_e25_end_lvef/best.pt"
  [50]="${S3_BASE}/probes/echomae_e50_end_lvef/best.pt"
  [75]="${S3_BASE}/probes/echomae_e75_end_lvef/best.pt"
  [99]="${S3_BASE}/probes/echomae_e99_end_lvef/best.pt"
)

upload_logs() {
  local out="/tmp/mae_eshuf_${SLURM_JOB_ID}.out"
  local err="/tmp/mae_eshuf_${SLURM_JOB_ID}.err"
  [ -f "${out}" ] && aws s3 cp "${out}" "${RUN_S3}/logs/job.out" --only-show-errors || true
  [ -f "${err}" ] && aws s3 cp "${err}" "${RUN_S3}/logs/job.err" --only-show-errors || true
}

sync_results() {
  for ep in ${EPOCHS}; do
    for cond in clean shuffled; do
      local outdir="/opt/dlami/nvme/evals/neurips/echomae_e${ep}_end_${cond}"
      if [ -d "${outdir}" ]; then
        aws s3 sync "${outdir}" "${RUN_S3}/echomae_e${ep}_end_${cond}/" \
          --exclude "*" --include "*.csv" --include "*.json" --only-show-errors || true
      fi
    done
  done
}

on_exit() { rc=$?; upload_logs; sync_results; exit $rc; }
trap on_exit EXIT
trap 'sync_results; upload_logs' USR1

# =========================
# SETUP
# =========================
nvidia-smi -L || true
pkill -9 -f "evals.main" 2>/dev/null || true
sleep 2

SCRATCH="/opt/dlami/nvme"

aws s3 cp "$ENV_S3" "${SCRATCH}/env.tar"
mkdir -p "${SCRATCH}/env" && tar -xf "${SCRATCH}/env.tar" -C "${SCRATCH}/env"
set +u && source "${SCRATCH}/env/bin/activate" && set -u
"${SCRATCH}/env/bin/conda-unpack" || true
pip install boto3 --no-cache-dir --quiet

CODE_DIR="${SCRATCH}/code"
mkdir -p "$CODE_DIR"
aws s3 cp "${S3_BASE}/setup/vjepa2-src.tar.gz" "${SCRATCH}/vjepa2-src.tar.gz" --quiet
tar xzf "${SCRATCH}/vjepa2-src.tar.gz" -C "$CODE_DIR"
cd "$CODE_DIR"
pip install -e . --no-deps --no-build-isolation 2>/dev/null || true

mkdir -p "${SCRATCH}/data/csv"
aws s3 cp "${S3_BASE}/data/csv/echonet_dynamic_train_s3_raw.csv" "${SCRATCH}/data/csv/"
aws s3 cp "${S3_BASE}/data/csv/echonet_dynamic_test_s3_raw.csv" "${SCRATCH}/data/csv/"

mkdir -p "${SCRATCH}/checkpoints" "${SCRATCH}/probes"

# Download all probes
for EP in ${EPOCHS}; do
  mkdir -p "${SCRATCH}/probes/echomae_e${EP}_end"
  aws s3 cp "${PROBE_S3[$EP]}" "${SCRATCH}/probes/echomae_e${EP}_end/best.pt" --quiet
done

# =========================
# CREATE INFERENCE CONFIGS ON THE FLY
# =========================
for EP in ${EPOCHS}; do
  CKPT="${CKPT_NAMES[$EP]}"
  mkdir -p "${CODE_DIR}/configs/inference/vitl/neurips/echonet-dynamic"
  cat > "${CODE_DIR}/configs/inference/vitl/neurips/echonet-dynamic/echomae_e${EP}_end_test.yaml" << EOF
app: vjepa
folder: /opt/dlami/nvme/evals/neurips/echomae_e${EP}_end_test
eval_name: video_classification_frozen
resume_checkpoint: true
val_only: true
tag: neurips-echomae-l-e${EP}-end-test
probe_checkpoint: /opt/dlami/nvme/probes/echomae_e${EP}_end/best.pt
predictions_save_path: /opt/dlami/nvme/evals/neurips/echomae_e${EP}_end_test/predictions.csv
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
    batch_size: 4
    num_epochs: 1
    use_bfloat16: true
    use_pos_embed: false
    multihead_kwargs:
    - {lr: 0.0, start_lr: 0.0, final_lr: 0.0, warmup: 0.0, weight_decay: 0.0, final_weight_decay: 0.0}
model_kwargs:
  checkpoint: /opt/dlami/nvme/checkpoints/${CKPT}.pth
  module_name: evals.video_classification_frozen.modelcustom.videomae_encoder
  pretrain_kwargs:
    encoder:
      model_name: vit_large_patch16_224
      tubelet_size: 2
  wrapper_kwargs: {}
EOF
done

# =========================
# RUN CLEAN + SHUFFLED
# =========================
for EP in ${EPOCHS}; do
  CKPT="${CKPT_NAMES[$EP]}"
  CKPT_LOCAL="${SCRATCH}/checkpoints/${CKPT}.pth"

  echo "[INFO] Downloading encoder e${EP}..."
  aws s3 cp "${S3_BASE}/checkpoints/${CKPT}.pth" "${CKPT_LOCAL}" --quiet

  CONFIG="${CODE_DIR}/configs/inference/vitl/neurips/echonet-dynamic/echomae_e${EP}_end_test.yaml"

  # CLEAN
  echo "============================================================"
  echo "[INFO] CLEAN test: EchoMAE-L e${EP}"
  echo "============================================================"
  OVERRIDE_TAG="neurips-echomae-l-e${EP}-end-clean" \
  OVERRIDE_PRED_PATH="/opt/dlami/nvme/evals/neurips/echomae_e${EP}_end_clean/predictions.csv" \
  python3 -m evals.main --fname "${CONFIG}" \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7
  echo "[INFO] e${EP} clean done (rc=$?)"

  # SHUFFLED
  echo "============================================================"
  echo "[INFO] SHUFFLED test: EchoMAE-L e${EP}"
  echo "============================================================"
  FRAME_SHUFFLE=100 FRAME_SHUFFLE_TYPE=frame \
  OVERRIDE_TAG="neurips-echomae-l-e${EP}-end-shuffled" \
  OVERRIDE_PRED_PATH="/opt/dlami/nvme/evals/neurips/echomae_e${EP}_end_shuffled/predictions.csv" \
  python3 -m evals.main --fname "${CONFIG}" \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7
  echo "[INFO] e${EP} shuffled done (rc=$?)"

  sync_results; upload_logs
  rm -f "${CKPT_LOCAL}"
done

echo "[INFO] All early trajectory clean+shuffled tests complete"
