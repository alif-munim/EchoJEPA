#!/bin/bash
# EchoNet-Dynamic LVEF probe — V-JEPA†-e120 (jepa_in21k_e200_280/training_folder/e120.pt).
# Stratified 50% train (3,731 clips, 4-bin clinical stratification preserved).
# Same protocol as other LVEF probes: 20 ep, 6-HP grid, d=4 attentive, encoder_pool.
#
#SBATCH -p dev
#SBATCH --nodelist=ip-10-0-50-56
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 0-06:00:00
#SBATCH -J echonet_lvef_base_e120_strat50
#SBATCH -o /opt/dlami/nvme/logs/echonet_lvef_base_e120_strat50-%j.out
#SBATCH -e /opt/dlami/nvme/logs/echonet_lvef_base_e120_strat50-%j.err
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
export PATH="/opt/vjepa2-312/bin:$PATH"
export LD_LIBRARY_PATH="/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cublas/lib:/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
mkdir -p /opt/dlami/nvme/logs /opt/dlami/nvme/tmp
export TMPDIR="/opt/dlami/nvme/tmp"

PILOT_NAME="base_e120_strat50"
TAG="neurips-${PILOT_NAME}-echonet-lvef"
FOLDER_REL="evals/vitl/neurips/${PILOT_NAME}_echonet_lvef_224"
ART_BUCKET="sagemaker-hyperpod-lifecycle-495467399120-usw2"
S3_BASE="s3://${ART_BUCKET}/vjepa2-artifacts"
SRC_TARBALL_S3="${S3_BASE}/setup/vjepa2-src.tar.gz"
CKPT_S3="${S3_BASE}/runs/jepa_in21k_e200_280/training_folder/e120.pt"
RUN_S3="${S3_BASE}/runs/${PILOT_NAME}_echonet_lvef_${SLURM_JOB_ID}"
WORKDIR="/opt/dlami/nvme/${PILOT_NAME}_echonet_lvef_${SLURM_JOB_ID}"
mkdir -p "$WORKDIR"

upload_artifacts() {
  [ -d "$FOLDER_REL" ] && aws s3 sync "$FOLDER_REL" "${RUN_S3}/probe/" --only-show-errors 2>/dev/null || true
  for f in /opt/dlami/nvme/logs/echonet_lvef_base_e120_strat50-${SLURM_JOB_ID}.{out,err}; do
    [ -f "$f" ] && aws s3 cp "$f" "${RUN_S3}/logs/$(basename "$f")" --only-show-errors 2>/dev/null || true
  done
}
trap 'upload_artifacts' EXIT

nvidia-smi -L || true

REPO_DIR="${WORKDIR}/code"
rm -rf "$REPO_DIR"; mkdir -p "$REPO_DIR"
aws s3 cp "${SRC_TARBALL_S3}" "${TMPDIR}/src.tar.gz" --only-show-errors
tar xzf "${TMPDIR}/src.tar.gz" -C "$REPO_DIR"
cd "$REPO_DIR"
pip install -e . --no-deps --no-build-isolation 2>/dev/null || true
export PYTHONPATH="$PWD:/opt/dlami/nvme/pyarrow_site:${PYTHONPATH:-}"

if ! python -c "import pyarrow" 2>/dev/null; then
  pip install --target=/opt/dlami/nvme/pyarrow_site --no-cache-dir pyarrow 2>&1 | tail -3
fi
pip install boto3 --no-cache-dir --quiet 2>&1 | tail -3

DATA_DIR="${WORKDIR}/data/csv"; mkdir -p "$DATA_DIR"
aws s3 cp "${S3_BASE}/data/csv/echonet_dynamic_train_s3_raw_strat50.csv" "$DATA_DIR/" --quiet
aws s3 cp "${S3_BASE}/data/csv/echonet_dynamic_val_s3_raw.csv"           "$DATA_DIR/" --quiet

CKPT_DIR="${WORKDIR}/checkpoints"; mkdir -p "$CKPT_DIR"
aws s3 cp "$CKPT_S3" "${CKPT_DIR}/pretrain.pt" --quiet
ls -lh "${CKPT_DIR}/pretrain.pt"

EVAL_CFG="configs/eval/vitl/neurips/${PILOT_NAME}_echonet_lvef_d4.yaml"
mkdir -p "$(dirname $EVAL_CFG)"
cat > "$EVAL_CFG" <<YAMLEOF
app: vjepa
cpus_per_task: 32
folder: ${FOLDER_REL}
mem_per_gpu: 80G
nodes: 1
tasks_per_node: 8
num_workers: 8
eval_name: video_classification_frozen
resume_checkpoint: false
tag: ${TAG}
experiment:
  classifier:
    task_type: regression
    num_heads: 16
    num_probe_blocks: 4
    num_targets: 1
  data:
    dataset_type: VideoDataset
    dataset_train: ${DATA_DIR}/echonet_dynamic_train_s3_raw_strat50.csv
    dataset_val: ${DATA_DIR}/echonet_dynamic_val_s3_raw.csv
    target_mean: 55.7776
    target_std: 12.4072
    resolution: 224
    frames_per_clip: 16
    frame_step: 2
    num_segments: 2
    num_views_per_segment: 1
  optimization:
    batch_size: 1
    num_epochs: 20
    use_bfloat16: true
    use_pos_embed: false
    multihead_kwargs:
    - {final_lr: 0.0, final_weight_decay: 0.01, lr: 0.0001, start_lr: 0.0001, warmup: 0.0, weight_decay: 0.01}
    - {final_lr: 0.0, final_weight_decay: 0.1,  lr: 0.0001, start_lr: 0.0001, warmup: 0.0, weight_decay: 0.1}
    - {final_lr: 0.0, final_weight_decay: 0.4,  lr: 0.0001, start_lr: 0.0001, warmup: 0.0, weight_decay: 0.4}
    - {final_lr: 0.0, final_weight_decay: 0.01, lr: 0.00005, start_lr: 0.00005, warmup: 0.0, weight_decay: 0.01}
    - {final_lr: 0.0, final_weight_decay: 0.1,  lr: 0.00005, start_lr: 0.00005, warmup: 0.0, weight_decay: 0.1}
    - {final_lr: 0.0, final_weight_decay: 0.4,  lr: 0.00005, start_lr: 0.00005, warmup: 0.0, weight_decay: 0.4}
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

echo "[INFO] Training ${PILOT_NAME} EchoNet-Dynamic LVEF probe..."
python3 -m evals.main --fname "$EVAL_CFG" \
  --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7

aws s3 sync "${FOLDER_REL}/" "${RUN_S3}/probe/" --only-show-errors || true
echo "[INFO] ${PILOT_NAME} LVEF probe complete"
