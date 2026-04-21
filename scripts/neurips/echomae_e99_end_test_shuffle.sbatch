#!/bin/bash
#SBATCH -p dev
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 1:00:00
#SBATCH -J mae99_shf
#SBATCH -o /tmp/mae99_shf_%j.out
#SBATCH -e /tmp/mae99_shf_%j.err

set -euxo pipefail

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

aws sts get-caller-identity || exit 1

ART_BUCKET="sagemaker-hyperpod-lifecycle-495467399120-usw2"
S3_BASE="s3://${ART_BUCKET}/vjepa2-artifacts"
ENV_S3="${S3_BASE}/env/vjepa2-312-pack-env.tar"
RUN_S3="${S3_BASE}/runs/mae99_shuf_${SLURM_JOB_ID}"

upload_logs() {
  [ -f "/tmp/mae99_shf_${SLURM_JOB_ID}.out" ] && aws s3 cp "/tmp/mae99_shf_${SLURM_JOB_ID}.out" "${RUN_S3}/logs/job.out" --only-show-errors || true
  [ -f "/tmp/mae99_shf_${SLURM_JOB_ID}.err" ] && aws s3 cp "/tmp/mae99_shf_${SLURM_JOB_ID}.err" "${RUN_S3}/logs/job.err" --only-show-errors || true
}
sync_results() {
  for cond in clean shuffled; do
    local d="/opt/dlami/nvme/evals/neurips/echomae_e99_end_${cond}"
    [ -d "$d" ] && aws s3 sync "$d" "${RUN_S3}/echomae_e99_end_${cond}/" --exclude "*" --include "*.csv" --only-show-errors || true
  done
}
trap 'upload_logs; sync_results' EXIT

SCRATCH="/opt/dlami/nvme"
pkill -9 -f "evals.main" 2>/dev/null || true; sleep 2

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

mkdir -p "${SCRATCH}/data/csv" "${SCRATCH}/checkpoints" "${SCRATCH}/probes/echomae_e99_end"
aws s3 cp "${S3_BASE}/data/csv/echonet_dynamic_train_s3_raw.csv" "${SCRATCH}/data/csv/"
aws s3 cp "${S3_BASE}/data/csv/echonet_dynamic_test_s3_raw.csv" "${SCRATCH}/data/csv/"
aws s3 cp "${S3_BASE}/checkpoints/echomae_l_mimic_ep99.pth" "${SCRATCH}/checkpoints/"
aws s3 cp "${S3_BASE}/probes/echomae_e99_end_lvef/best.pt" "${SCRATCH}/probes/echomae_e99_end/best.pt"

# Create config
cat > "${CODE_DIR}/e99_test.yaml" << EOF
app: vjepa
folder: /opt/dlami/nvme/evals/neurips/echomae_e99_end_test
eval_name: video_classification_frozen
resume_checkpoint: true
val_only: true
tag: neurips-echomae-l-e99-end-test
probe_checkpoint: /opt/dlami/nvme/probes/echomae_e99_end/best.pt
predictions_save_path: /opt/dlami/nvme/evals/neurips/echomae_e99_end_test/predictions.csv
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
  checkpoint: /opt/dlami/nvme/checkpoints/echomae_l_mimic_ep99.pth
  module_name: evals.video_classification_frozen.modelcustom.videomae_encoder
  pretrain_kwargs:
    encoder:
      model_name: vit_large_patch16_224
      tubelet_size: 2
  wrapper_kwargs: {}
EOF

# CLEAN
echo "[INFO] CLEAN test: e99"
OVERRIDE_TAG="neurips-echomae-l-e99-end-clean" \
OVERRIDE_PRED_PATH="/opt/dlami/nvme/evals/neurips/echomae_e99_end_clean/predictions.csv" \
python3 -m evals.main --fname "${CODE_DIR}/e99_test.yaml" \
  --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7
echo "[INFO] e99 clean done"

# SHUFFLED
echo "[INFO] SHUFFLED test: e99"
FRAME_SHUFFLE=100 FRAME_SHUFFLE_TYPE=frame \
OVERRIDE_TAG="neurips-echomae-l-e99-end-shuffled" \
OVERRIDE_PRED_PATH="/opt/dlami/nvme/evals/neurips/echomae_e99_end_shuffled/predictions.csv" \
python3 -m evals.main --fname "${CODE_DIR}/e99_test.yaml" \
  --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7
echo "[INFO] e99 shuffled done"
