#!/bin/bash
# LVEDD VAL-inference for both encoders (N=1,141).
# Writes per-clip val predictions so we can paired-bootstrap val ΔR² and
# compare it directly to the test ΔR² bootstrap CI already in
# flops-matched-probe-results.md.
#
# This is a sanity check on the val/test disagreement: val is 3.4× larger
# than test (1,141 vs 340), so its Δ estimate is ~1.8× less noisy. If the
# val Δ-CI excludes zero in the opposite direction from test, the
# disagreement is real (task-specific). If val Δ-CI crosses zero, the
# test "win" is likely small-N sampling noise.
#
# Uses each probe run's best.pt (same as the headline test numbers).
#
#SBATCH -p dev
#SBATCH --nodelist=ip-10-0-50-146
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -t 0-01:30:00
#SBATCH -J lvedd_val_infer
#SBATCH -o /opt/dlami/nvme/logs/lvedd_val_infer-%j.out
#SBATCH -e /opt/dlami/nvme/logs/lvedd_val_infer-%j.err
#SBATCH --chdir=/opt/dlami/nvme

set -euxo pipefail
export AWS_DEFAULT_REGION="us-west-2"; export AWS_REGION="us-west-2"
export OMP_NUM_THREADS=1; export MKL_NUM_THREADS=1; export PYTHONUNBUFFERED=1
export HOME="/home/ubuntu"
unset AWS_SHARED_CREDENTIALS_FILE 2>/dev/null || true
unset AWS_CONFIG_FILE 2>/dev/null || true
export PATH="/opt/vjepa2-312/bin:$PATH"
export LD_LIBRARY_PATH="/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cublas/lib:/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
mkdir -p /opt/dlami/nvme/logs /opt/dlami/nvme/tmp
export TMPDIR="/opt/dlami/nvme/tmp"

ART_BUCKET="sagemaker-hyperpod-lifecycle-495467399120-usw2"
S3_BASE="s3://${ART_BUCKET}/vjepa2-artifacts"
SRC_TARBALL_S3="${S3_BASE}/setup/vjepa2-src-tokenrel-motion-v1.tar.gz"

RUN_S3_BASE="${S3_BASE}/runs/lvedd_val_infer_${SLURM_JOB_ID}"
WORKDIR="/opt/dlami/nvme/lvedd_val_infer_${SLURM_JOB_ID}"
mkdir -p "$WORKDIR"

REPO_DIR="${WORKDIR}/code"
rm -rf "$REPO_DIR"; mkdir -p "$REPO_DIR"
aws s3 cp "$SRC_TARBALL_S3" "${TMPDIR}/src.tar.gz" --only-show-errors
tar xzf "${TMPDIR}/src.tar.gz" -C "$REPO_DIR"
cd "$REPO_DIR"
pip install -e . --no-deps --no-build-isolation 2>/dev/null || true
export PYTHONPATH="$PWD:/opt/dlami/nvme/pyarrow_site:${PYTHONPATH:-}"
if ! python -c "import pyarrow" 2>/dev/null; then
  pip install --target=/opt/dlami/nvme/pyarrow_site --no-cache-dir pyarrow 2>&1 | tail -3
fi
pip install boto3 --no-cache-dir --quiet 2>&1 | tail -3

DATA_DIR="${WORKDIR}/data/csv"; mkdir -p "$DATA_DIR"
aws s3 cp "${S3_BASE}/data/csv/echonet_lvh_lvedd_val.csv" "${DATA_DIR}/" --quiet

run_val_inference() {
  local MODEL_TAG="$1"      # base_e125 | v4_e25
  local BACKBONE_S3="$2"    # encoder pretrain checkpoint
  local PROBE_RUN_ID="$3"   # original probe run (724 / 725)
  local CKPT_DIR="${WORKDIR}/${MODEL_TAG}/ckpts"
  mkdir -p "$CKPT_DIR"

  aws s3 cp "$BACKBONE_S3" "${CKPT_DIR}/pretrain.pt" --quiet
  aws s3 cp "${S3_BASE}/runs/lvh_lvedd_${MODEL_TAG}_${PROBE_RUN_ID}/echonet_lvh_lvedd_encoder_pool/probe/video_classification_frozen/${MODEL_TAG}-echonet_lvh_lvedd-encoder-pool/best.pt" "${CKPT_DIR}/probe_best.pt" --quiet
  ls -lh "${CKPT_DIR}/"

  local VAL_CFG="configs/inference/vitl/neurips/${MODEL_TAG}_echonet_lvh_lvedd_val_infer.yaml"
  local FOLDER_REL="evals/vitl/neurips/${MODEL_TAG}_echonet_lvh_lvedd_val_infer"
  local PRED_CSV="${WORKDIR}/${MODEL_TAG}/predictions/${MODEL_TAG}_echonet_lvh_lvedd_val.csv"
  mkdir -p "$(dirname $VAL_CFG)" "$FOLDER_REL" "$(dirname $PRED_CSV)"

  cat > "$VAL_CFG" <<YAMLEOF
app: vjepa
cpus_per_task: 16
folder: ${FOLDER_REL}
mem_per_gpu: 80G
nodes: 1
tasks_per_node: 1
num_workers: 8
eval_name: video_classification_frozen
resume_checkpoint: true
tag: ${MODEL_TAG}-echonet_lvh_lvedd-val-infer
val_only: true
predictions_save_path: ${PRED_CSV}
probe_checkpoint: ${CKPT_DIR}/probe_best.pt
experiment:
  classifier:
    task_type: regression
    num_heads: 16
    num_probe_blocks: 4
    num_targets: 1
  data:
    dataset_type: VideoDataset
    dataset_train: ${DATA_DIR}/echonet_lvh_lvedd_val.csv
    dataset_val: ${DATA_DIR}/echonet_lvh_lvedd_val.csv
    num_classes: 1
    target_mean: 4.705
    target_std: 0.8335
    resolution: 224
    frames_per_clip: 16
    frame_step: 2
    num_segments: 2
    num_views_per_segment: 1
  optimization:
    batch_size: 1
    num_epochs: 1
    use_bfloat16: true
    use_pos_embed: false
    multihead_kwargs:
    - {final_lr: 0.0, final_weight_decay: 0.01, lr: 0.0, start_lr: 0.0, warmup: 0.0, weight_decay: 0.01}
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

  echo "[INFO] === ${MODEL_TAG} best.pt val inference (N=1141) ==="
  python3 -m evals.main --fname "$VAL_CFG" --devices cuda:0 --val_only 2>&1 | tee "${WORKDIR}/${MODEL_TAG}_val_infer.log"
  aws s3 cp "$PRED_CSV" "${RUN_S3_BASE}/${MODEL_TAG}/predictions/$(basename $PRED_CSV)" --only-show-errors || true
  aws s3 sync "${FOLDER_REL}/" "${RUN_S3_BASE}/${MODEL_TAG}/val_infer/" --only-show-errors || true
}

nvidia-smi -L || true

# 1) base_e125
run_val_inference \
  "base_e125" \
  "${S3_BASE}/runs/jepa_in21k_e200_280/training_folder/e125.pt" \
  "724"

# 2) V4-e25
run_val_inference \
  "v4_e25" \
  "${S3_BASE}/runs/final_phase_rel25_paper_593/checkpoints/latest.pt" \
  "725"

# Upload logs.
for f in /opt/dlami/nvme/logs/lvedd_val_infer-${SLURM_JOB_ID}.{out,err}; do
  [ -f "$f" ] && aws s3 cp "$f" "${RUN_S3_BASE}/logs/$(basename "$f")" --only-show-errors || true
done

echo "[INFO] lvedd val inference complete"
