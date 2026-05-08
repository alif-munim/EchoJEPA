#!/bin/bash
# EchoNet-Dynamic LVEF matched-frame inference (clean + matched_frame)
# for the JEPA e125 ordinary +25-epoch continuation checkpoint.
#
# Inference-only — reuses existing probe from:
#   s3://.../runs/jepa_ext_probes_332/jepa_e125_lvef/.../best.pt
#
#SBATCH -p dev
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 0-00:45:00
#SBATCH -J lvef_mf_base_e125
#SBATCH -o /opt/dlami/nvme/logs/lvef_mf_base_e125-%j.out
#SBATCH -e /opt/dlami/nvme/logs/lvef_mf_base_e125-%j.err
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

PILOT_NAME="base_e125_mf"
TAG="neurips-${PILOT_NAME}-echonet-lvef"
ART_BUCKET="sagemaker-hyperpod-lifecycle-495467399120-usw2"
S3_BASE="s3://${ART_BUCKET}/vjepa2-artifacts"
SRC_TARBALL_S3="${S3_BASE}/setup/vjepa2-src.tar.gz"
CKPT_S3="${S3_BASE}/runs/jepa_in21k_e200_280/training_folder/e125.pt"
PROBE_S3="${S3_BASE}/runs/jepa_ext_probes_332/jepa_e125_lvef/video_classification_frozen/neurips-jepa-in21k-e125-end-lvef-d4/best.pt"
RUN_S3="${S3_BASE}/runs/${PILOT_NAME}_${SLURM_JOB_ID}"
WORKDIR="/opt/dlami/nvme/${PILOT_NAME}_${SLURM_JOB_ID}"
mkdir -p "$WORKDIR"

upload_artifacts() {
  TEST_CLEAN_DIR="evals/vitl/neurips/${PILOT_NAME}_test_clean"
  TEST_MF_DIR="evals/vitl/neurips/${PILOT_NAME}_test_mf"
  [ -d "$TEST_CLEAN_DIR" ] && aws s3 sync "$TEST_CLEAN_DIR" "${RUN_S3}/test_clean/" --only-show-errors 2>/dev/null || true
  [ -d "$TEST_MF_DIR"    ] && aws s3 sync "$TEST_MF_DIR"    "${RUN_S3}/test_mf/"    --only-show-errors 2>/dev/null || true
  for p in "${WORKDIR}/predictions/"*.csv; do
    [ -f "$p" ] && aws s3 cp "$p" "${RUN_S3}/predictions/$(basename "$p")" --only-show-errors 2>/dev/null || true
  done
  for f in /opt/dlami/nvme/logs/lvef_mf_base_e125-${SLURM_JOB_ID}.{out,err}; do
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
aws s3 cp "${S3_BASE}/data/csv/echonet_dynamic_train_s3_raw.csv" "$DATA_DIR/" --quiet
aws s3 cp "${S3_BASE}/data/csv/echonet_dynamic_test_s3_raw.csv"  "$DATA_DIR/" --quiet

CKPT_DIR="${WORKDIR}/checkpoints"; mkdir -p "$CKPT_DIR"
aws s3 cp "$CKPT_S3"  "${CKPT_DIR}/pretrain.pt"   --quiet
aws s3 cp "$PROBE_S3" "${CKPT_DIR}/probe_best.pt" --quiet
ls -lh "${CKPT_DIR}/"

# =============================================================================
# Shared inference config (reused for clean + matched_frame)
# =============================================================================
TEST_CFG="configs/inference/vitl/neurips/${PILOT_NAME}_test.yaml"
TEST_CLEAN_DIR="evals/vitl/neurips/${PILOT_NAME}_test_clean"
TEST_MF_DIR="evals/vitl/neurips/${PILOT_NAME}_test_mf"
PRED_CLEAN="${WORKDIR}/predictions/${PILOT_NAME}_clean.csv"
PRED_MF="${WORKDIR}/predictions/${PILOT_NAME}_mf.csv"
mkdir -p "$(dirname $TEST_CFG)" "$TEST_CLEAN_DIR" "$TEST_MF_DIR" "$(dirname $PRED_CLEAN)"

cat > "$TEST_CFG" <<YAMLEOF
app: vjepa
cpus_per_task: 32
folder: ${TEST_CLEAN_DIR}
mem_per_gpu: 80G
nodes: 1
tasks_per_node: 1
num_workers: 8
eval_name: video_classification_frozen
resume_checkpoint: true
tag: ${TAG}-test-clean
val_only: true
predictions_save_path: ${PRED_CLEAN}
probe_checkpoint: ${CKPT_DIR}/probe_best.pt
experiment:
  classifier:
    task_type: regression
    num_heads: 16
    num_probe_blocks: 4
    num_targets: 1
  data:
    dataset_type: VideoDataset
    dataset_train: ${DATA_DIR}/echonet_dynamic_train_s3_raw.csv
    dataset_val:   ${DATA_DIR}/echonet_dynamic_test_s3_raw.csv
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
    - {final_lr: 0.0, final_weight_decay: 0.1,  lr: 0.0, start_lr: 0.0, warmup: 0.0, weight_decay: 0.1}
    - {final_lr: 0.0, final_weight_decay: 0.4,  lr: 0.0, start_lr: 0.0, warmup: 0.0, weight_decay: 0.4}
    - {final_lr: 0.0, final_weight_decay: 0.01, lr: 0.0, start_lr: 0.0, warmup: 0.0, weight_decay: 0.01}
    - {final_lr: 0.0, final_weight_decay: 0.1,  lr: 0.0, start_lr: 0.0, warmup: 0.0, weight_decay: 0.1}
    - {final_lr: 0.0, final_weight_decay: 0.4,  lr: 0.0, start_lr: 0.0, warmup: 0.0, weight_decay: 0.4}
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

# =============================================================================
# CLEAN inference
# =============================================================================
echo "[INFO] === CLEAN inference: ${PILOT_NAME} ==="
python3 -m evals.main --fname "$TEST_CFG" \
  --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
  --val_only 2>&1 | tee "${WORKDIR}/test_clean.log"
aws s3 sync "${TEST_CLEAN_DIR}/" "${RUN_S3}/test_clean/" --only-show-errors || true
[ -f "$PRED_CLEAN" ] && aws s3 cp "$PRED_CLEAN" "${RUN_S3}/predictions/$(basename "$PRED_CLEAN")" --only-show-errors || true

# =============================================================================
# MATCHED_FRAME inference (override tag + pred path so clean isn't clobbered)
# =============================================================================
echo "[INFO] === MATCHED_FRAME inference: ${PILOT_NAME} ==="
FRAME_SHUFFLE=100 FRAME_SHUFFLE_TYPE=matched_frame \
OVERRIDE_TAG="${TAG}-test-mf" \
OVERRIDE_PRED_PATH="${PRED_MF}" \
python3 -m evals.main --fname "$TEST_CFG" \
  --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
  --val_only 2>&1 | tee "${WORKDIR}/test_mf.log"
aws s3 sync "${TEST_MF_DIR}/" "${RUN_S3}/test_mf/" --only-show-errors || true
[ -f "$PRED_MF" ] && aws s3 cp "$PRED_MF" "${RUN_S3}/predictions/$(basename "$PRED_MF")" --only-show-errors || true

echo "[INFO] === ${PILOT_NAME} complete ==="
