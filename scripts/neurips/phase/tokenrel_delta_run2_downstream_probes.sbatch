#!/bin/bash
# Downstream probes for EchoJEPA-TokenRel + MotionDelta Run 2 (695, e5 ckpt).
# 4 probes on encoder_pool (target_encoder), 20-ep attentive d=4 protocol,
# 6-HP grid, matching V4 / Ctrl 608 / pilot 655 finalbudget probes:
#   LVEF encoder_pool
#   RVSP SV encoder_pool
#   MR A4C 4-class encoder_pool
#   TAPSE A4C encoder_pool  (primary endpoint for Run 2 hypothesis; also probed here)
#
#SBATCH -p dev
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 0-14:00:00
#SBATCH -J tokenrel_r2_probes
#SBATCH -o /opt/dlami/nvme/logs/tokenrel_r2_probes-%j.out
#SBATCH -e /opt/dlami/nvme/logs/tokenrel_r2_probes-%j.err
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

ART_BUCKET="sagemaker-hyperpod-lifecycle-495467399120-usw2"
S3_BASE="s3://${ART_BUCKET}/vjepa2-artifacts"
SRC_TARBALL_S3="${S3_BASE}/setup/vjepa2-src-tokenrel-motion-v1.tar.gz"
TOKENREL_CKPT_S3="${S3_BASE}/runs/echojepa_tokenrel_delta_run2_695/ckpt/e5.pt"
RUN_S3="${S3_BASE}/runs/tokenrel_r2_probes_${SLURM_JOB_ID}"
WORKDIR="/opt/dlami/nvme/tokenrel_r2_probes_${SLURM_JOB_ID}"
mkdir -p "$WORKDIR"

upload_artifacts() {
  for task in lvef rvsp mr tapse; do
    FOLDER="evals/vitl/neurips/tokenrel_r2_695_${task}_encoder_pool_224"
    [ -d "$FOLDER" ] && aws s3 sync "$FOLDER" "${RUN_S3}/${task}_encoder_pool/probe/" --only-show-errors 2>/dev/null || true
    FOLDER_TEST="evals/vitl/neurips/tokenrel_r2_695_${task}_encoder_pool_test"
    [ -d "$FOLDER_TEST" ] && aws s3 sync "$FOLDER_TEST" "${RUN_S3}/${task}_encoder_pool/test/" --only-show-errors 2>/dev/null || true
    PRED="${WORKDIR}/predictions/tokenrel_r2_695_e5_${task}_test.csv"
    [ -f "$PRED" ] && aws s3 cp "$PRED" "${RUN_S3}/${task}_encoder_pool/predictions/$(basename $PRED)" --only-show-errors || true
  done
  for f in /opt/dlami/nvme/logs/tokenrel_r2_probes-${SLURM_JOB_ID}.{out,err}; do
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

CKPT_DIR="${WORKDIR}/checkpoints"; mkdir -p "$CKPT_DIR"
aws s3 cp "$TOKENREL_CKPT_S3" "${CKPT_DIR}/pretrain.pt" --quiet
ls -lh "${CKPT_DIR}/pretrain.pt"

DATA_DIR="${WORKDIR}/data/csv"; mkdir -p "$DATA_DIR"
for f in \
    echonet_dynamic_train_s3_raw.csv echonet_dynamic_val_s3_raw.csv echonet_dynamic_test_s3_raw.csv \
    mimic_rvsp_sv_train_10k.csv mimic_rvsp_sv_val_10k.csv mimic_rvsp_sv_test_10k.csv \
    mimic_mr_a4c_train_10k.csv mimic_mr_a4c_val.csv mimic_mr_a4c_test.csv \
    mimic_tapse_a4c_train.csv mimic_tapse_a4c_val.csv mimic_tapse_a4c_test.csv; do
  aws s3 cp "${S3_BASE}/data/csv/${f}" "${DATA_DIR}/" --quiet || true
done

run_probe() {
  local task="$1"
  local tag="tokenrel-r2-695-e5-${task}-encoder-pool"
  local folder="evals/vitl/neurips/tokenrel_r2_695_${task}_encoder_pool_224"
  local eval_cfg="configs/eval/vitl/neurips/tokenrel_r2_${task}_encoder_pool.yaml"
  mkdir -p "$(dirname $eval_cfg)" "$folder"

  local task_block=""; local data_block=""
  case "$task" in
    lvef)
      task_block="task_type: regression
    num_heads: 16
    num_probe_blocks: 4
    num_targets: 1"
      data_block="    dataset_type: VideoDataset
    dataset_train: ${DATA_DIR}/echonet_dynamic_train_s3_raw.csv
    dataset_val: ${DATA_DIR}/echonet_dynamic_val_s3_raw.csv
    num_classes: 1
    target_mean: 55.7776
    target_std: 12.4072"
      ;;
    rvsp)
      task_block="task_type: regression
    num_heads: 16
    num_probe_blocks: 4
    num_targets: 1"
      data_block="    dataset_type: VideoDataset
    dataset_train: ${DATA_DIR}/mimic_rvsp_sv_train_10k.csv
    dataset_val: ${DATA_DIR}/mimic_rvsp_sv_val_10k.csv
    num_classes: 1
    target_mean: 30.0959
    target_std: 12.2321"
      ;;
    mr)
      task_block="task_type: classification
    num_heads: 16
    num_probe_blocks: 4"
      data_block="    dataset_type: VideoDataset
    dataset_train: ${DATA_DIR}/mimic_mr_a4c_train_10k.csv
    dataset_val: ${DATA_DIR}/mimic_mr_a4c_val.csv
    num_classes: 4"
      ;;
    tapse)
      task_block="task_type: regression
    num_heads: 16
    num_probe_blocks: 4
    num_targets: 1"
      data_block="    dataset_type: VideoDataset
    dataset_train: ${DATA_DIR}/mimic_tapse_a4c_train.csv
    dataset_val: ${DATA_DIR}/mimic_tapse_a4c_val.csv
    num_classes: 1
    target_mean: 1.974
    target_std: 0.51"
      ;;
  esac

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
    ${task_block}
  data:
${data_block}
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

  echo "[INFO] === Training TokenRel + MotionDelta Run 2 695 e5 probe: ${task} / encoder_pool ==="
  python3 -m evals.main --fname "$eval_cfg" \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
    2>&1 | tee "${WORKDIR}/${task}_train.log" || echo "[WARN] ${task} failed"

  aws s3 sync "${folder}/" "${RUN_S3}/${task}_encoder_pool/probe/" --only-show-errors || true

  # Inline test inference
  local probe_best="${folder}/video_classification_frozen/${tag}/best.pt"
  if [ -f "$probe_best" ]; then
    local test_tag="${tag}-test"
    local test_folder="evals/vitl/neurips/tokenrel_r2_695_${task}_encoder_pool_test"
    local test_cfg="configs/inference/vitl/neurips/tokenrel_r2_${task}_test.yaml"
    local pred_csv="${WORKDIR}/predictions/tokenrel_r2_695_e5_${task}_test.csv"
    mkdir -p "$(dirname $test_cfg)" "$test_folder" "$(dirname $pred_csv)"
    local test_data_block=""
    case "$task" in
      lvef)
        test_data_block="    dataset_type: VideoDataset
    dataset_train: ${DATA_DIR}/echonet_dynamic_test_s3_raw.csv
    dataset_val: ${DATA_DIR}/echonet_dynamic_test_s3_raw.csv
    num_classes: 1
    target_mean: 55.7776
    target_std: 12.4072"
        ;;
      rvsp)
        test_data_block="    dataset_type: VideoDataset
    dataset_train: ${DATA_DIR}/mimic_rvsp_sv_test_10k.csv
    dataset_val: ${DATA_DIR}/mimic_rvsp_sv_test_10k.csv
    num_classes: 1
    target_mean: 30.0959
    target_std: 12.2321"
        ;;
      mr)
        test_data_block="    dataset_type: VideoDataset
    dataset_train: ${DATA_DIR}/mimic_mr_a4c_test.csv
    dataset_val: ${DATA_DIR}/mimic_mr_a4c_test.csv
    num_classes: 4"
        ;;
      tapse)
        test_data_block="    dataset_type: VideoDataset
    dataset_train: ${DATA_DIR}/mimic_tapse_a4c_test.csv
    dataset_val: ${DATA_DIR}/mimic_tapse_a4c_test.csv
    num_classes: 1
    target_mean: 1.974
    target_std: 0.51"
        ;;
    esac

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
    ${task_block}
  data:
${test_data_block}
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
    echo "[INFO] === Test inference TokenRel + MotionDelta Run 2 695 e5: ${task} ==="
    python3 -m evals.main --fname "$test_cfg" --devices cuda:0 --val_only \
      2>&1 | tee "${WORKDIR}/${task}_test.log" || echo "[WARN] ${task} test failed"
    aws s3 sync "${test_folder}/" "${RUN_S3}/${task}_encoder_pool/test/" --only-show-errors || true
    [ -f "$pred_csv" ] && aws s3 cp "$pred_csv" "${RUN_S3}/${task}_encoder_pool/predictions/$(basename $pred_csv)" --only-show-errors || true
  else
    echo "[WARN] probe best.pt missing for ${task}; skipping test"
  fi
}

for task in lvef rvsp mr tapse; do
  run_probe "$task"
done

echo "[INFO] === TokenRel + MotionDelta Run 2 695 e5 downstream probes complete ==="
