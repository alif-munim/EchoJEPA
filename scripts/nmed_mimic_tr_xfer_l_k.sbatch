#!/bin/bash
# Nature Medicine: UHN-trained EchoJEPA-L-K MR probe → MIMIC test (transfer).
# Inference only — loads existing probe from S3, runs on MIMIC MR test.
#
#SBATCH -p dev
#SBATCH --nodelist=ip-10-0-50-56
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 0-01:00:00
#SBATCH -J nmed_mimic_tr_xfer_lk
#SBATCH -o /opt/dlami/nvme/logs/nmed_mimic_tr_xfer_lk-%j.out
#SBATCH -e /opt/dlami/nvme/logs/nmed_mimic_tr_xfer_lk-%j.err
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

MODEL_TAG="echojepa-l-k"
PILOT_NAME="nmed_mimic_tr_xfer_lk"
TAG="tricuspid-regurg-xfer-${MODEL_TAG}"
FOLDER_REL="evals/vitl/nature_medicine/mimic"
ART_BUCKET="sagemaker-hyperpod-lifecycle-495467399120-usw2"
S3_BASE="s3://${ART_BUCKET}/vjepa2-artifacts"
SRC_TARBALL_S3="${S3_BASE}/setup/vjepa2-src.tar.gz"
ENCODER_S3="${S3_BASE}/checkpoints/anneal/keep/vitl-kinetics-pt220-an55.pt"
PROBE_S3="${S3_BASE}/checkpoints/probes/tr_severity/${MODEL_TAG}/best.pt"
CSV_S3="s3://echodata25/nature_medicine/mimic/probe_csvs/tricuspid_regurg"
RUN_S3="${S3_BASE}/runs/${PILOT_NAME}_${SLURM_JOB_ID}"
WORKDIR="/opt/dlami/nvme/${PILOT_NAME}_${SLURM_JOB_ID}"
mkdir -p "$WORKDIR"

upload_artifacts() {
  [ -d "$FOLDER_REL" ] && aws s3 sync "$FOLDER_REL" "${RUN_S3}/out/" --only-show-errors 2>/dev/null || true
  for f in /opt/dlami/nvme/logs/${PILOT_NAME}-${SLURM_JOB_ID}.{out,err}; do
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
aws s3 cp "${CSV_S3}/train.csv" "${DATA_DIR}/mimic_tricuspid_regurg_train.csv" --quiet
aws s3 cp "${CSV_S3}/test.csv"  "${DATA_DIR}/mimic_tricuspid_regurg_test.csv"  --quiet

CKPT_DIR="${WORKDIR}/checkpoints"; mkdir -p "$CKPT_DIR"
aws s3 cp "$ENCODER_S3" "${CKPT_DIR}/encoder.pt" --quiet
aws s3 cp "$PROBE_S3"   "${CKPT_DIR}/probe_best.pt" --quiet
ls -lh "${CKPT_DIR}/"

PRED_CSV="${WORKDIR}/predictions/tricuspid_regurg-xfer-${MODEL_TAG}.csv"
mkdir -p "$(dirname $PRED_CSV)"

EVAL_CFG="configs/inference/vitl/nature_medicine/${PILOT_NAME}.yaml"
mkdir -p "$(dirname $EVAL_CFG)"
cat > "$EVAL_CFG" <<YAMLEOF
app: vjepa
cpus_per_task: 32
folder: ${FOLDER_REL}
mem_per_gpu: 80G
nodes: 1
tasks_per_node: 8
num_workers: 4
eval_name: video_classification_frozen
val_only: true
resume_checkpoint: true
probe_checkpoint: ${CKPT_DIR}/probe_best.pt
predictions_save_path: ${PRED_CSV}
tag: ${TAG}
experiment:
  classifier:
    task_type: classification
    num_heads: 16
    num_probe_blocks: 1
    num_targets: 5
  data:
    dataset_type: VideoDataset
    dataset_train: ${DATA_DIR}/mimic_tricuspid_regurg_train.csv
    dataset_val:   ${DATA_DIR}/mimic_tricuspid_regurg_test.csv
    num_classes: 5
    resolution: 224
    frames_per_clip: 16
    frame_step: 2
    num_segments: 2
    num_views_per_segment: 1
    study_sampling: false
  optimization:
    batch_size: 2
    val_batch_size: 128
    num_epochs: 1
    use_bfloat16: true
    use_pos_embed: false
    multihead_kwargs:
    # 12-head grid (4 LR x 3 WD) — MUST match training grid used for the
    # UHN MR probe checkpoint (best.pt); values only affect bookkeeping
    # since val_only: true skips optimiser stepping.
    - {lr: 0.0005,  start_lr: 0.0, warmup: 2.0, final_lr: 0.0, weight_decay: 0.001, final_weight_decay: 0.001}
    - {lr: 0.0005,  start_lr: 0.0, warmup: 2.0, final_lr: 0.0, weight_decay: 0.01,  final_weight_decay: 0.01}
    - {lr: 0.0005,  start_lr: 0.0, warmup: 2.0, final_lr: 0.0, weight_decay: 0.1,   final_weight_decay: 0.1}
    - {lr: 0.0001,  start_lr: 0.0, warmup: 2.0, final_lr: 0.0, weight_decay: 0.001, final_weight_decay: 0.001}
    - {lr: 0.0001,  start_lr: 0.0, warmup: 2.0, final_lr: 0.0, weight_decay: 0.01,  final_weight_decay: 0.01}
    - {lr: 0.0001,  start_lr: 0.0, warmup: 2.0, final_lr: 0.0, weight_decay: 0.1,   final_weight_decay: 0.1}
    - {lr: 0.00005, start_lr: 0.0, warmup: 2.0, final_lr: 0.0, weight_decay: 0.001, final_weight_decay: 0.001}
    - {lr: 0.00005, start_lr: 0.0, warmup: 2.0, final_lr: 0.0, weight_decay: 0.01,  final_weight_decay: 0.01}
    - {lr: 0.00005, start_lr: 0.0, warmup: 2.0, final_lr: 0.0, weight_decay: 0.1,   final_weight_decay: 0.1}
    - {lr: 0.00001, start_lr: 0.0, warmup: 2.0, final_lr: 0.0, weight_decay: 0.001, final_weight_decay: 0.001}
    - {lr: 0.00001, start_lr: 0.0, warmup: 2.0, final_lr: 0.0, weight_decay: 0.01,  final_weight_decay: 0.01}
    - {lr: 0.00001, start_lr: 0.0, warmup: 2.0, final_lr: 0.0, weight_decay: 0.1,   final_weight_decay: 0.1}
model_kwargs:
  checkpoint: ${CKPT_DIR}/encoder.pt
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

echo "[INFO] === MIMIC MR test inference: ${MODEL_TAG} ==="
python3 -m evals.main --fname "$EVAL_CFG" \
  --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
  --val_only 2>&1 | tee "${WORKDIR}/infer.log"

aws s3 sync "${FOLDER_REL}/" "${RUN_S3}/out/" --only-show-errors || true
[ -f "$PRED_CSV" ] && aws s3 cp "$PRED_CSV" "${RUN_S3}/predictions/$(basename "$PRED_CSV")" --only-show-errors || true

echo "[INFO] === ${PILOT_NAME} complete ==="
