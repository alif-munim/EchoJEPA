#!/bin/bash
# Option A classification re-runs. Uses patched eval.py (saves full per-class
# probability matrix as prob_class_0..N-1 columns). Each re-run loads the
# original best.pt probe head + correct pretrain encoder, runs test inference
# in the current sklearn-enabled environment, and writes new predictions with
# real per-class probs.
#
# Reruns:
#   1. MR V4 (from run 609's best.pt) — replaces top-1 approx with true OVR
#   2. MR MV-PairedIntra (from run 611's best.pt) — same
#   3. MR TokenRel-Motion-e5 (from run 699's best.pt)
#   4. HCM A4C V-JEPA†-e125 (from run 705's best.pt)
#   5. HCM A4C TokenRel-Motion-e5 (from run 704's best.pt)
#
#SBATCH -p dev
#SBATCH --nodelist=ip-10-0-50-146
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -t 0-02:00:00
#SBATCH -J classif_rerun_optA
#SBATCH -o /opt/dlami/nvme/logs/classif_rerun_optA-%j.out
#SBATCH -e /opt/dlami/nvme/logs/classif_rerun_optA-%j.err
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
SRC_TARBALL_S3="${S3_BASE}/setup/vjepa2-src.tar.gz"  # has the prob_class_* patch (line 1078 in eval.py)
RUN_S3_BASE="${S3_BASE}/runs/classif_rerun_optA_${SLURM_JOB_ID}"
WORKDIR="/opt/dlami/nvme/classif_rerun_optA_${SLURM_JOB_ID}"
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
for csv in mimic_mr_a4c_test.csv mimic_hcm_a4c_test_10k.csv; do
  aws s3 cp "${S3_BASE}/data/csv/${csv}" "${DATA_DIR}/" --quiet
done

run_rerun() {
  local TAG="$1"        # a label for output dir + logging
  local BACKBONE_S3="$2"
  local PROBE_BEST_S3="$3"
  local TEST_CSV="$4"
  local NUM_CLASSES="$5"
  local MODULE_NAME="$6" # vit_encoder_multiclip
  local CKPT_DIR="${WORKDIR}/${TAG}/ckpts"
  mkdir -p "$CKPT_DIR"

  # Backbone: if this is FJ, we'd need adapter wrapping — not included here (test-inference path
  # is meant for already-trained probes on vanilla V-JEPA style encoders).
  aws s3 cp "$BACKBONE_S3" "${CKPT_DIR}/pretrain.pt" --quiet
  aws s3 cp "$PROBE_BEST_S3" "${CKPT_DIR}/probe_best.pt" --quiet
  ls -lh "${CKPT_DIR}/"

  local TEST_CFG="configs/inference/vitl/neurips/${TAG}_rerun.yaml"
  local FOLDER_REL="evals/vitl/neurips/${TAG}_rerun"
  local PRED_CSV="${WORKDIR}/${TAG}/predictions/${TAG}_test_rerun.csv"
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
tag: ${TAG}-rerun
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
    dataset_train: ${TEST_CSV}
    dataset_val: ${TEST_CSV}
    num_classes: ${NUM_CLASSES}
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
  module_name: ${MODULE_NAME}
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

  echo "[INFO] === rerun: ${TAG} ==="
  python3 -m evals.main --fname "$TEST_CFG" --devices cuda:0 --val_only 2>&1 | tee "${WORKDIR}/${TAG}_rerun.log" || echo "[WARN] ${TAG} rerun failed"
  aws s3 cp "$PRED_CSV" "${RUN_S3_BASE}/${TAG}/predictions/$(basename $PRED_CSV)" --only-show-errors || true
  aws s3 sync "${FOLDER_REL}/" "${RUN_S3_BASE}/${TAG}/test/" --only-show-errors || true
}

nvidia-smi -L || true

MODULE="evals.video_classification_frozen.modelcustom.vit_encoder_multiclip"

# 1. MR V4
run_rerun "v4_e25_mr_a4c" \
  "${S3_BASE}/runs/final_phase_rel25_paper_593/checkpoints/latest.pt" \
  "${S3_BASE}/runs/final_phase_rel25_mr_a4c_609/probe/video_classification_frozen/neurips-final-phase-rel25-mr-a4c/best.pt" \
  "${DATA_DIR}/mimic_mr_a4c_test.csv" \
  "4" \
  "${MODULE}"

# 2. MR MV-PairedIntra
run_rerun "paired_iv25_mr_a4c" \
  "${S3_BASE}/runs/final_paired_iv25_paper_608/checkpoints/latest.pt" \
  "${S3_BASE}/runs/final_paired_iv25_mr_a4c_611/probe/video_classification_frozen/neurips-final-paired-iv25-mr-a4c/best.pt" \
  "${DATA_DIR}/mimic_mr_a4c_test.csv" \
  "4" \
  "${MODULE}"

# 3. MR TokenRel-Motion-e5 (probe best.pt from 699)
run_rerun "tokenrel_r2_e5_mr_a4c" \
  "${S3_BASE}/runs/echojepa_tokenrel_delta_run2_695/ckpt/e5.pt" \
  "${S3_BASE}/runs/tokenrel_r2_probes_699/mr_encoder_pool/probe/video_classification_frozen/tokenrel-r2-695-e5-mr-encoder-pool/best.pt" \
  "${DATA_DIR}/mimic_mr_a4c_test.csv" \
  "4" \
  "${MODULE}"

# 4. HCM A4C V-JEPA†-e125 (probe best.pt from 705). CSV from original 704/705 setup.
HCM_TEST="${DATA_DIR}/mimic_hcm_a4c_test_10k.csv"
if [ -f "$HCM_TEST" ]; then
  run_rerun "base_e125_hcm_a4c" \
    "${S3_BASE}/runs/jepa_in21k_e200_280/training_folder/e125.pt" \
    "${S3_BASE}/runs/a4c_probes_base_e125_705/hcm_encoder_pool/probe/video_classification_frozen/base_e125-hcm-a4c-encoder-pool/best.pt" \
    "$HCM_TEST" \
    "2" \
    "${MODULE}"
  # 5. HCM A4C TokenRel-Motion-e5
  run_rerun "tokenrel_r2_e5_hcm_a4c" \
    "${S3_BASE}/runs/echojepa_tokenrel_delta_run2_695/ckpt/e5.pt" \
    "${S3_BASE}/runs/a4c_probes_tokenrel_r2_695_e5_704/hcm_encoder_pool/probe/video_classification_frozen/tokenrel_r2_695_e5-hcm-a4c-encoder-pool/best.pt" \
    "$HCM_TEST" \
    "2" \
    "${MODULE}"
else
  echo "[FATAL] HCM test CSV $HCM_TEST missing; HCM reruns skipped"
fi

for f in /opt/dlami/nvme/logs/classif_rerun_optA-${SLURM_JOB_ID}.{out,err}; do
  [ -f "$f" ] && aws s3 cp "$f" "${RUN_S3_BASE}/logs/$(basename "$f")" --only-show-errors || true
done
echo "[INFO] classification rerun option A complete"
