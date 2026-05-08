#!/bin/bash
# Extend MR classification reruns to V-JEPA†-e130 (job 849), MCC-via-848,
# and TokenRel-Motion-e25 (job 853). Missing from 862 because those runs
# completed before the per-class-prob patch landed. Runs in sklearn-enabled
# env with patched eval.py — saves full prob_class_0..3 matrix for true
# multi-class OVR AUROC + bootstrap CIs.
#
#SBATCH -p dev
#SBATCH --nodelist=ip-10-0-50-56
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -t 0-01:00:00
#SBATCH -J classif_rerun_base_mcc
#SBATCH -o /opt/dlami/nvme/logs/classif_rerun_base_mcc-%j.out
#SBATCH -e /opt/dlami/nvme/logs/classif_rerun_base_mcc-%j.err
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
SRC_TARBALL_S3="${S3_BASE}/setup/vjepa2-src.tar.gz"  # has the prob_class_* patch
RUN_S3_BASE="${S3_BASE}/runs/classif_rerun_base_mcc_${SLURM_JOB_ID}"
WORKDIR="/opt/dlami/nvme/classif_rerun_base_mcc_${SLURM_JOB_ID}"
mkdir -p "$WORKDIR"

REPO_DIR="${WORKDIR}/code"; rm -rf "$REPO_DIR"; mkdir -p "$REPO_DIR"
aws s3 cp "$SRC_TARBALL_S3" "${TMPDIR}/src.tar.gz" --only-show-errors
tar xzf "${TMPDIR}/src.tar.gz" -C "$REPO_DIR"
cd "$REPO_DIR"
pip install -e . --no-deps --no-build-isolation 2>/dev/null || true
export PYTHONPATH="$PWD:/opt/dlami/nvme/pyarrow_site:${PYTHONPATH:-}"
if ! python -c "import pyarrow" 2>/dev/null; then
  pip install --target=/opt/dlami/nvme/pyarrow_site --no-cache-dir pyarrow 2>&1 | tail -3
fi
pip install boto3 scikit-learn --no-cache-dir --quiet 2>&1 | tail -3
python -c "import sklearn; print('sklearn OK', sklearn.__version__)"

DATA_DIR="${WORKDIR}/data/csv"; mkdir -p "$DATA_DIR"
aws s3 cp "${S3_BASE}/data/csv/mimic_mr_a4c_test.csv" "${DATA_DIR}/" --quiet

run_rerun() {
  local TAG="$1"
  local BACKBONE_S3="$2"
  local PROBE_BEST_S3="$3"
  local CKPT_DIR="${WORKDIR}/${TAG}/ckpts"
  mkdir -p "$CKPT_DIR"
  aws s3 cp "$BACKBONE_S3" "${CKPT_DIR}/pretrain.pt" --quiet
  aws s3 cp "$PROBE_BEST_S3" "${CKPT_DIR}/probe_best.pt" --quiet

  local TEST_CFG="configs/inference/vitl/neurips/${TAG}_mr_rerun.yaml"
  local FOLDER_REL="evals/vitl/neurips/${TAG}_mr_rerun"
  local PRED_CSV="${WORKDIR}/${TAG}/predictions/${TAG}_mr_test_rerun.csv"
  mkdir -p "$(dirname $TEST_CFG)" "$FOLDER_REL" "$(dirname $PRED_CSV)"
  cat > "$TEST_CFG" <<YAMLEOF
app: vjepa
cpus_per_task: 16
folder: ${FOLDER_REL}
mem_per_gpu: 80G
nodes: 1
tasks_per_node: 1
num_workers: 8
eval_name: video_classification_frozen
resume_checkpoint: true
tag: ${TAG}-mr-rerun
val_only: true
predictions_save_path: ${PRED_CSV}
probe_checkpoint: ${CKPT_DIR}/probe_best.pt
experiment:
  classifier:
    task_type: classification
    num_heads: 16
    num_probe_blocks: 4
  data:
    dataset_type: VideoDataset
    dataset_train: ${DATA_DIR}/mimic_mr_a4c_test.csv
    dataset_val: ${DATA_DIR}/mimic_mr_a4c_test.csv
    num_classes: 4
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
  python3 -m evals.main --fname "$TEST_CFG" --devices cuda:0 --val_only 2>&1 | tee "${WORKDIR}/${TAG}_rerun.log"
  aws s3 cp "$PRED_CSV" "${RUN_S3_BASE}/${TAG}/predictions/$(basename $PRED_CSV)" --only-show-errors || true
  aws s3 sync "${FOLDER_REL}/" "${RUN_S3_BASE}/${TAG}/test/" --only-show-errors || true
}

nvidia-smi -L || true

# V-JEPA†-e130 MR (from job 849)
run_rerun "base_e130_mr" \
  "${S3_BASE}/runs/jepa_in21k_e200_280/training_folder/e130.pt" \
  "${S3_BASE}/runs/mr_a4c_probes_base_e130_849/mr_a4c_encoder_pool/probe/video_classification_frozen/base_e130-mr_a4c-encoder-pool/best.pt"

# MCC via job 848 MR (restart's probe)
run_rerun "mcc_e25_mr_848" \
  "${S3_BASE}/runs/mcc_target_anchored_25of100_762/checkpoints/e25.pt" \
  "${S3_BASE}/runs/mr_a4c_probes_mcc_e25_848/mr_a4c_encoder_pool/probe/video_classification_frozen/mcc_e25-mr_a4c-encoder-pool/best.pt"

# TokenRel-Motion-e25 MR (from job 853). Encoder: echojepa_tokenrel_delta_run2_5to25_703/ckpt/e25.pt
# (this is the e5→e25 resume of run 695, matches 853's CKPT_S3 exactly)
run_rerun "tokenrel_r2_e25_mr_853" \
  "${S3_BASE}/runs/echojepa_tokenrel_delta_run2_5to25_703/ckpt/e25.pt" \
  "${S3_BASE}/runs/mr_a4c_probes_tokenrel_r2_e25_853/mr_a4c_encoder_pool/probe/video_classification_frozen/tokenrel_r2_e25-mr_a4c-encoder-pool/best.pt"

for f in /opt/dlami/nvme/logs/classif_rerun_base_mcc-${SLURM_JOB_ID}.{out,err}; do
  [ -f "$f" ] && aws s3 cp "$f" "${RUN_S3_BASE}/logs/$(basename "$f")" --only-show-errors || true
done
echo "[INFO] base + MCC MR reruns complete"
