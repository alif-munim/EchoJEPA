#!/bin/bash
# MR A4C 4-class probe — MCC-Anchored e25 (mcc_target_anchored_25of100_762/e25.pt).
# Matched config to final_phase_rel25_mr_a4c_609 / final_paired_iv25_mr_a4c_611.
#
#SBATCH -p dev
#SBATCH --nodelist=ip-10-0-50-56
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 0-08:00:00
#SBATCH -J mr_a4c_mcc_e25_restart
#SBATCH -o /opt/dlami/nvme/logs/mr_a4c_mcc_e25-%j.out
#SBATCH -e /opt/dlami/nvme/logs/mr_a4c_mcc_e25-%j.err
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

MODEL_TAG="mcc_e25"
TASK="mr_a4c"
ART_BUCKET="sagemaker-hyperpod-lifecycle-495467399120-usw2"
S3_BASE="s3://${ART_BUCKET}/vjepa2-artifacts"
SRC_TARBALL_S3="${S3_BASE}/setup/vjepa2-src-tokenrel-motion-v1.tar.gz"
CKPT_S3="${S3_BASE}/runs/mcc_target_anchored_25of100_762/checkpoints/e25.pt"
RUN_S3="${S3_BASE}/runs/${TASK}_probes_${MODEL_TAG}_${SLURM_JOB_ID}"
WORKDIR="/opt/dlami/nvme/${TASK}_probes_${MODEL_TAG}_${SLURM_JOB_ID}"
mkdir -p "$WORKDIR"

upload_artifacts() {
  FOLDER="evals/vitl/neurips/${MODEL_TAG}_${TASK}_encoder_pool_224"
  [ -d "$FOLDER" ] && aws s3 sync "$FOLDER" "${RUN_S3}/${TASK}_encoder_pool/probe/" --only-show-errors 2>/dev/null || true
  FOLDER_TEST="evals/vitl/neurips/${MODEL_TAG}_${TASK}_encoder_pool_test"
  [ -d "$FOLDER_TEST" ] && aws s3 sync "$FOLDER_TEST" "${RUN_S3}/${TASK}_encoder_pool/test/" --only-show-errors 2>/dev/null || true
  PRED="${WORKDIR}/predictions/${MODEL_TAG}_${TASK}_test.csv"
  [ -f "$PRED" ] && aws s3 cp "$PRED" "${RUN_S3}/${TASK}_encoder_pool/predictions/$(basename $PRED)" --only-show-errors || true
  for f in /opt/dlami/nvme/logs/mr_a4c_mcc_e25-${SLURM_JOB_ID}.{out,err}; do
    [ -f "$f" ] && aws s3 cp "$f" "${RUN_S3}/logs/$(basename "$f")" --only-show-errors || true
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

CKPT_DIR="${WORKDIR}/checkpoints"; mkdir -p "$CKPT_DIR"
aws s3 cp "$CKPT_S3" "${CKPT_DIR}/pretrain.pt" --quiet

DATA_DIR="${WORKDIR}/data/csv"; mkdir -p "$DATA_DIR"
for f in mimic_mr_a4c_train_10k.csv mimic_mr_a4c_val.csv mimic_mr_a4c_test.csv; do
  aws s3 cp "${S3_BASE}/data/csv/${f}" "${DATA_DIR}/" --quiet
done

tag="${MODEL_TAG}-${TASK}-encoder-pool"
folder="evals/vitl/neurips/${MODEL_TAG}_${TASK}_encoder_pool_224"
eval_cfg="configs/eval/vitl/neurips/${MODEL_TAG}_${TASK}_encoder_pool.yaml"
mkdir -p "$(dirname $eval_cfg)" "$folder"

cat > "$eval_cfg" <<YAMLEOF
app: vjepa
cpus_per_task: 32
folder: ${folder}
mem_per_gpu: 80G
nodes: 1
tasks_per_node: 8
num_workers: 8
eval_name: video_classification_frozen
resume_checkpoint: false
tag: ${tag}
experiment:
  classifier:
    task_type: classification
    num_heads: 16
    num_probe_blocks: 4
  data:
    dataset_type: VideoDataset
    dataset_train: ${DATA_DIR}/mimic_mr_a4c_train_10k.csv
    dataset_val: ${DATA_DIR}/mimic_mr_a4c_val.csv
    num_classes: 4
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
    - {final_lr: 0.0, final_weight_decay: 0.01, lr: 0.0001,  start_lr: 0.0001,  warmup: 0.0, weight_decay: 0.01}
    - {final_lr: 0.0, final_weight_decay: 0.1,  lr: 0.0001,  start_lr: 0.0001,  warmup: 0.0, weight_decay: 0.1}
    - {final_lr: 0.0, final_weight_decay: 0.4,  lr: 0.0001,  start_lr: 0.0001,  warmup: 0.0, weight_decay: 0.4}
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

echo "[INFO] === Training ${MODEL_TAG}: ${TASK} / encoder_pool ==="
python3 -m evals.main --fname "$eval_cfg" \
  --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
  2>&1 | tee "${WORKDIR}/${TASK}_train.log" || echo "[WARN] ${TASK} failed"

aws s3 sync "${folder}/" "${RUN_S3}/${TASK}_encoder_pool/probe/" --only-show-errors || true

probe_best="${folder}/video_classification_frozen/${tag}/best.pt"
if [ -f "$probe_best" ]; then
  test_tag="${tag}-test"
  test_folder="evals/vitl/neurips/${MODEL_TAG}_${TASK}_encoder_pool_test"
  test_cfg="configs/inference/vitl/neurips/${MODEL_TAG}_${TASK}_test.yaml"
  pred_csv="${WORKDIR}/predictions/${MODEL_TAG}_${TASK}_test.csv"
  mkdir -p "$(dirname $test_cfg)" "$test_folder" "$(dirname $pred_csv)"

  cat > "$test_cfg" <<YAMLEOF
app: vjepa
cpus_per_task: 16
folder: ${test_folder}
mem_per_gpu: 80G
nodes: 1
tasks_per_node: 1
num_workers: 8
eval_name: video_classification_frozen
resume_checkpoint: true
tag: ${test_tag}
val_only: true
predictions_save_path: ${pred_csv}
probe_checkpoint: ${probe_best}
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
  python3 -m evals.main --fname "$test_cfg" --devices cuda:0 --val_only \
    2>&1 | tee "${WORKDIR}/${TASK}_test.log" || echo "[WARN] ${TASK} test failed"
  aws s3 sync "${test_folder}/" "${RUN_S3}/${TASK}_encoder_pool/test/" --only-show-errors || true
  [ -f "$pred_csv" ] && aws s3 cp "$pred_csv" "${RUN_S3}/${TASK}_encoder_pool/predictions/$(basename $pred_csv)" --only-show-errors || true
fi

echo "[INFO] === ${MODEL_TAG} ${TASK} complete ==="
