#!/bin/bash
# Nature Medicine: UHN pericardial_effusion 5-class, EchoJEPA-G probe training.
# d=1 attentive, DistributedStudySampler, 20-head HP grid, 35 epochs, view-filtered CSVs.
#
#SBATCH -p dev
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 1-00:00:00
#SBATCH -J nmed_uhn_peff_lk
#SBATCH -o /opt/dlami/nvme/logs/nmed_uhn_peff_lk-%j.out
#SBATCH -e /opt/dlami/nvme/logs/nmed_uhn_peff_lk-%j.err
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
export TORCH_HOME="/opt/dlami/nvme/torch_cache"

PILOT_NAME="nmed_uhn_peff_lk"
TAG="nmed-echojepa-l-k-uhn-pericardial-effusion"
FOLDER_REL="evals/vitl/nature_medicine/uhn/${PILOT_NAME}"
ART_BUCKET="sagemaker-hyperpod-lifecycle-495467399120-usw2"
S3_BASE="s3://${ART_BUCKET}/vjepa2-artifacts"
SRC_TARBALL_S3="${S3_BASE}/setup/vjepa2-src.tar.gz"
ENCODER_S3="${S3_BASE}/checkpoints/anneal/keep/vitl-kinetics-pt220-an55.pt"
CSV_S3="s3://echodata25/nature_medicine/uhn/probe_csvs/pericardial_effusion"
RUN_S3="${S3_BASE}/runs/${PILOT_NAME}_${SLURM_JOB_ID}"
WORKDIR="/opt/dlami/nvme/${PILOT_NAME}_${SLURM_JOB_ID}"
mkdir -p "$WORKDIR"

upload_artifacts() {
  [ -d "$FOLDER_REL" ] && aws s3 sync "$FOLDER_REL" "${RUN_S3}/probe/" --only-show-errors 2>/dev/null || true
  for f in /opt/dlami/nvme/logs/${PILOT_NAME}-${SLURM_JOB_ID}.{out,err}; do
    [ -f "$f" ] && aws s3 cp "$f" "${RUN_S3}/logs/$(basename "$f")" --only-show-errors 2>/dev/null || true
  done
}
trap 'upload_artifacts' EXIT

(
  while true; do
    sleep 600
    [ -d "$FOLDER_REL" ] && aws s3 sync "$FOLDER_REL" "${RUN_S3}/probe/" \
      --include "*.pt" --include "*.csv" --include "*.log" --include "*.json" \
      --exclude "*.pth" --only-show-errors 2>/dev/null || true
  done
) &
SYNC_PID=$!

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
aws s3 cp "${CSV_S3}/train_vf.csv" "${DATA_DIR}/uhn_peff_train_vf.csv" --quiet
aws s3 cp "${CSV_S3}/val_vf.csv"   "${DATA_DIR}/uhn_peff_val_vf.csv"   --quiet

CKPT_DIR="${WORKDIR}/checkpoints"; mkdir -p "$CKPT_DIR"
aws s3 cp "$ENCODER_S3" "${CKPT_DIR}/encoder.pt" --quiet
ls -lh "${CKPT_DIR}/"

EVAL_CFG="configs/eval/vitl/nature_medicine/${PILOT_NAME}.yaml"
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
    num_heads: 16
    num_probe_blocks: 1
  data:
    dataset_type: VideoDataset
    dataset_train: ${DATA_DIR}/uhn_peff_train_vf.csv
    dataset_val:   ${DATA_DIR}/uhn_peff_val_vf.csv
    num_classes: 5
    resolution: 224
    frames_per_clip: 16
    frame_step: 2
    num_segments: 2
    num_views_per_segment: 1
    study_sampling: true
  optimization:
    batch_size: 1
    num_epochs: 15
    use_bfloat16: true
    use_pos_embed: false
    multihead_kwargs:
    - {lr: 0.001,   start_lr: 0.0, warmup: 3.0, final_lr: 0.0, weight_decay: 0.001, final_weight_decay: 0.001}
    - {lr: 0.001,   start_lr: 0.0, warmup: 3.0, final_lr: 0.0, weight_decay: 0.01,  final_weight_decay: 0.01}
    - {lr: 0.001,   start_lr: 0.0, warmup: 3.0, final_lr: 0.0, weight_decay: 0.1,   final_weight_decay: 0.1}
    - {lr: 0.001,   start_lr: 0.0, warmup: 3.0, final_lr: 0.0, weight_decay: 0.4,   final_weight_decay: 0.4}
    - {lr: 0.0005,  start_lr: 0.0, warmup: 3.0, final_lr: 0.0, weight_decay: 0.001, final_weight_decay: 0.001}
    - {lr: 0.0005,  start_lr: 0.0, warmup: 3.0, final_lr: 0.0, weight_decay: 0.01,  final_weight_decay: 0.01}
    - {lr: 0.0005,  start_lr: 0.0, warmup: 3.0, final_lr: 0.0, weight_decay: 0.1,   final_weight_decay: 0.1}
    - {lr: 0.0005,  start_lr: 0.0, warmup: 3.0, final_lr: 0.0, weight_decay: 0.4,   final_weight_decay: 0.4}
    - {lr: 0.0001,  start_lr: 0.0, warmup: 3.0, final_lr: 0.0, weight_decay: 0.001, final_weight_decay: 0.001}
    - {lr: 0.0001,  start_lr: 0.0, warmup: 3.0, final_lr: 0.0, weight_decay: 0.01,  final_weight_decay: 0.01}
    - {lr: 0.0001,  start_lr: 0.0, warmup: 3.0, final_lr: 0.0, weight_decay: 0.1,   final_weight_decay: 0.1}
    - {lr: 0.0001,  start_lr: 0.0, warmup: 3.0, final_lr: 0.0, weight_decay: 0.4,   final_weight_decay: 0.4}
    - {lr: 0.00005, start_lr: 0.0, warmup: 3.0, final_lr: 0.0, weight_decay: 0.001, final_weight_decay: 0.001}
    - {lr: 0.00005, start_lr: 0.0, warmup: 3.0, final_lr: 0.0, weight_decay: 0.01,  final_weight_decay: 0.01}
    - {lr: 0.00005, start_lr: 0.0, warmup: 3.0, final_lr: 0.0, weight_decay: 0.1,   final_weight_decay: 0.1}
    - {lr: 0.00005, start_lr: 0.0, warmup: 3.0, final_lr: 0.0, weight_decay: 0.4,   final_weight_decay: 0.4}
    - {lr: 0.00001, start_lr: 0.0, warmup: 3.0, final_lr: 0.0, weight_decay: 0.001, final_weight_decay: 0.001}
    - {lr: 0.00001, start_lr: 0.0, warmup: 3.0, final_lr: 0.0, weight_decay: 0.01,  final_weight_decay: 0.01}
    - {lr: 0.00001, start_lr: 0.0, warmup: 3.0, final_lr: 0.0, weight_decay: 0.1,   final_weight_decay: 0.1}
    - {lr: 0.00001, start_lr: 0.0, warmup: 3.0, final_lr: 0.0, weight_decay: 0.4,   final_weight_decay: 0.4}
model_kwargs:
  checkpoint: ${CKPT_DIR}/encoder.pt
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

echo "[INFO] === Probe training: ${PILOT_NAME} ==="
python3 -m evals.main --fname "$EVAL_CFG" \
  --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
  2>&1 | tee "${WORKDIR}/train.log"

kill $SYNC_PID 2>/dev/null || true; wait $SYNC_PID 2>/dev/null || true
aws s3 sync "${FOLDER_REL}/" "${RUN_S3}/probe/" --only-show-errors || true
echo "[INFO] === ${PILOT_NAME} complete ==="
