#!/bin/bash
# Re-run test inference for V4 (job 609) and MV-PairedIntra (job 611) MR probes
# in the current sklearn-available environment to fix the nan AUROC/bal_acc/kappa
# logged by those earlier runs.
#
# Each probe's best.pt still exists on S3. This sbatch loads the original
# encoder + best.pt probe head, runs test inference on the same mr_a4c_test.csv,
# and writes a fresh log_r0.csv with true multi-class OVR AUROC (from eval.py's
# sklearn call) + predictions CSV. Runtime ~5-10 min each.
#SBATCH -p dev
#SBATCH --nodelist=ip-10-0-50-146
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -t 0-01:00:00
#SBATCH -J mr_test_rerun_v4_pi
#SBATCH -o /opt/dlami/nvme/logs/mr_test_rerun_v4_pi-%j.out
#SBATCH -e /opt/dlami/nvme/logs/mr_test_rerun_v4_pi-%j.err
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
RUN_S3_BASE="${S3_BASE}/runs/mr_test_rerun_${SLURM_JOB_ID}"
WORKDIR="/opt/dlami/nvme/mr_test_rerun_${SLURM_JOB_ID}"
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
  local MODEL_TAG="$1"   # v4_e25 | paired_iv25
  local BACKBONE_S3="$2" # encoder pretrain checkpoint
  local PROBE_RUN_ID="$3" # original run id (609 / 611)
  local PROBE_TAG="$4"   # old probe tag used in the original run path
  local CKPT_DIR="${WORKDIR}/${MODEL_TAG}/ckpts"
  mkdir -p "$CKPT_DIR"

  aws s3 cp "$BACKBONE_S3" "${CKPT_DIR}/pretrain.pt" --quiet
  aws s3 cp "${S3_BASE}/runs/${PROBE_RUN_ID}/probe/video_classification_frozen/${PROBE_TAG}/best.pt" "${CKPT_DIR}/probe_best.pt" --quiet
  ls -lh "${CKPT_DIR}/"

  local TEST_CFG="configs/inference/vitl/neurips/${MODEL_TAG}_mr_a4c_test_rerun.yaml"
  local FOLDER_REL="evals/vitl/neurips/${MODEL_TAG}_mr_a4c_test_rerun"
  local PRED_CSV="${WORKDIR}/${MODEL_TAG}/predictions/${MODEL_TAG}_mr_a4c_test_rerun.csv"
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
tag: ${MODEL_TAG}-mr-test-rerun
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

  echo "[INFO] === MR test rerun: ${MODEL_TAG} ==="
  python3 -m evals.main --fname "$TEST_CFG" --devices cuda:0 --val_only 2>&1 | tee "${WORKDIR}/${MODEL_TAG}_rerun.log"
  aws s3 cp "$PRED_CSV" "${RUN_S3_BASE}/${MODEL_TAG}/predictions/$(basename $PRED_CSV)" --only-show-errors || true
  aws s3 sync "${FOLDER_REL}/" "${RUN_S3_BASE}/${MODEL_TAG}/test/" --only-show-errors || true
}

nvidia-smi -L || true

# V4 (MV-PhaseRel) — job 609
run_rerun \
  "v4_e25" \
  "${S3_BASE}/runs/final_phase_rel25_paper_593/checkpoints/latest.pt" \
  "final_phase_rel25_mr_a4c_609" \
  "neurips-final-phase-rel25-mr-a4c"

# MV-PairedIntra — job 611
run_rerun \
  "paired_iv25" \
  "${S3_BASE}/runs/final_paired_iv25_paper_608/checkpoints/latest.pt" \
  "final_paired_iv25_mr_a4c_611" \
  "neurips-final-paired-iv25-mr-a4c"

for f in /opt/dlami/nvme/logs/mr_test_rerun_v4_pi-${SLURM_JOB_ID}.{out,err}; do
  [ -f "$f" ] && aws s3 cp "$f" "${RUN_S3_BASE}/logs/$(basename "$f")" --only-show-errors || true
done
echo "[INFO] MR test reruns complete"
