#!/bin/bash
#SBATCH -p dev
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 06:00:00
#SBATCH -J fm_extern_lvef
#SBATCH -o /opt/dlami/nvme/logs/fm_extern_lvef_%j.out
#SBATCH -e /opt/dlami/nvme/logs/fm_extern_lvef_%j.err

set -euxo pipefail

# =============================================================================
# Four-model LVEF probe training (2 of 4): external encoders.
# Trains d=4 attentive probes on:
#   echoprime (MViT, 512-dim)  - uses echo_prime_encoder module
#   panecho   (ConvNeXt+Xfmr)  - uses panecho_encoder module, torch.hub cache
# Protocol matches phase_vs_localblock_lvef_train: 20 epochs, 6-HP grid.
# =============================================================================

export AWS_DEFAULT_REGION="us-west-2"
export AWS_REGION="us-west-2"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export HOME="/home/ubuntu"
unset AWS_SHARED_CREDENTIALS_FILE 2>/dev/null || true
unset AWS_CONFIG_FILE 2>/dev/null || true
unset AWS_EC2_METADATA_DISABLED 2>/dev/null || true
export AWS_METADATA_SERVICE_TIMEOUT=10
export AWS_METADATA_SERVICE_NUM_ATTEMPTS=5

aws sts get-caller-identity || exit 1

ART_BUCKET="sagemaker-hyperpod-lifecycle-495467399120-usw2"
S3_BASE="s3://${ART_BUCKET}/vjepa2-artifacts"
ENV_S3="${S3_BASE}/env/vjepa2-312-pack-env.tar"
RUN_S3="${S3_BASE}/runs/fourmodel_extern_lvef_${SLURM_JOB_ID}"

upload_logs() {
  local out="/opt/dlami/nvme/logs/fm_extern_lvef_${SLURM_JOB_ID}.out"
  local err="/opt/dlami/nvme/logs/fm_extern_lvef_${SLURM_JOB_ID}.err"
  [ -f "${out}" ] && aws s3 cp "${out}" "${RUN_S3}/logs/job.out" --only-show-errors || true
  [ -f "${err}" ] && aws s3 cp "${err}" "${RUN_S3}/logs/job.err" --only-show-errors || true
}
trap 'upload_logs' EXIT

nvidia-smi -L || true
mkdir -p /opt/dlami/nvme/logs

SCRATCH="/opt/dlami/nvme"
WORKDIR="${SCRATCH}/fm_extern_lvef_${SLURM_JOB_ID}"
mkdir -p "$WORKDIR"

aws s3 cp "$ENV_S3" "${WORKDIR}/env.tar"
mkdir -p "${WORKDIR}/env" && tar -xf "${WORKDIR}/env.tar" -C "${WORKDIR}/env"
set +u && source "${WORKDIR}/env/bin/activate" && set -u
"${WORKDIR}/env/bin/conda-unpack" || true
export TMPDIR="${WORKDIR}/tmp" && mkdir -p "$TMPDIR"
pip install boto3 --no-cache-dir --quiet

CODE_DIR="${WORKDIR}/code"
mkdir -p "$CODE_DIR"
aws s3 cp "${S3_BASE}/setup/vjepa2-src.tar.gz" "${WORKDIR}/vjepa2-src.tar.gz" --quiet
tar xzf "${WORKDIR}/vjepa2-src.tar.gz" -C "$CODE_DIR"
cd "$CODE_DIR"
pip install -e . --no-deps --no-build-isolation 2>/dev/null || true
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

DATA_DIR="${WORKDIR}/data/csv"
mkdir -p "$DATA_DIR"
aws s3 cp "${S3_BASE}/data/csv/echonet_dynamic_train_s3_raw.csv" "$DATA_DIR/" --quiet
aws s3 cp "${S3_BASE}/data/csv/echonet_dynamic_val_s3_raw.csv" "$DATA_DIR/" --quiet

# EchoPrime encoder weight (single .pt file, loaded via "checkpoint" kwarg)
CKPT_DIR="${WORKDIR}/checkpoints"
mkdir -p "$CKPT_DIR"
aws s3 cp "${S3_BASE}/checkpoints/echoprime/echo_prime_encoder.pt" "${CKPT_DIR}/echo_prime_encoder.pt" --quiet
ls -lh "${CKPT_DIR}/echo_prime_encoder.pt"

# PanEcho torch.hub cache (hubconf downloads panecho.pt from github, avoid via cache preload)
export TORCH_HOME="${WORKDIR}/torch_cache"
PE_CACHE="${TORCH_HOME}/hub/checkpoints"
mkdir -p "$PE_CACHE"
aws s3 cp "${S3_BASE}/checkpoints/panecho/panecho.pt" "${PE_CACHE}/panecho.pt" --quiet
ls -lh "${PE_CACHE}/panecho.pt"

# --- EchoPrime config ---
EP_CFG="configs/eval/vitl/neurips/echoprime_end_lvef_d4.yaml"
EP_TAG="neurips-echoprime-end-lvef-d4"
EP_FOLDER="evals/vitl/neurips/echoprime_end_lvef_224"
mkdir -p "$(dirname $EP_CFG)"
cat > "$EP_CFG" <<YAMLEOF
app: vjepa
cpus_per_task: 32
folder: ${EP_FOLDER}
mem_per_gpu: 80G
nodes: 1
tasks_per_node: 8
num_workers: 8
eval_name: video_classification_frozen
resume_checkpoint: false
tag: ${EP_TAG}
experiment:
  classifier:
    task_type: regression
    num_heads: 16
    num_probe_blocks: 4
    num_targets: 1
  data:
    dataset_type: VideoDataset
    dataset_train: ${DATA_DIR}/echonet_dynamic_train_s3_raw.csv
    dataset_val: ${DATA_DIR}/echonet_dynamic_val_s3_raw.csv
    target_mean: 55.7776
    target_std: 12.4072
    resolution: 224
    frames_per_clip: 16
    frame_step: 2
    num_segments: 2
    num_views_per_segment: 1
    normalization:
      - [0.0, 0.0, 0.0]
      - [1.0, 1.0, 1.0]
  optimization:
    batch_size: 1
    num_epochs: 20
    use_bfloat16: true
    use_pos_embed: false
    multihead_kwargs:
    - {final_lr: 0.0, final_weight_decay: 0.01, lr: 0.0001, start_lr: 0.0001, warmup: 0.0, weight_decay: 0.01}
    - {final_lr: 0.0, final_weight_decay: 0.1, lr: 0.0001, start_lr: 0.0001, warmup: 0.0, weight_decay: 0.1}
    - {final_lr: 0.0, final_weight_decay: 0.4, lr: 0.0001, start_lr: 0.0001, warmup: 0.0, weight_decay: 0.4}
    - {final_lr: 0.0, final_weight_decay: 0.01, lr: 0.00005, start_lr: 0.00005, warmup: 0.0, weight_decay: 0.01}
    - {final_lr: 0.0, final_weight_decay: 0.1, lr: 0.00005, start_lr: 0.00005, warmup: 0.0, weight_decay: 0.1}
    - {final_lr: 0.0, final_weight_decay: 0.4, lr: 0.00005, start_lr: 0.00005, warmup: 0.0, weight_decay: 0.4}
model_kwargs:
  checkpoint: ${CKPT_DIR}/echo_prime_encoder.pt
  module_name: evals.video_classification_frozen.modelcustom.echo_prime_encoder
  pretrain_kwargs: {}
  wrapper_kwargs:
    force_fp32: true
    bin_size: 50
YAMLEOF

# --- PanEcho config ---
PN_CFG="configs/eval/vitl/neurips/panecho_end_lvef_d4.yaml"
PN_TAG="neurips-panecho-end-lvef-d4"
PN_FOLDER="evals/vitl/neurips/panecho_end_lvef_224"
mkdir -p "$(dirname $PN_CFG)"
cat > "$PN_CFG" <<YAMLEOF
app: vjepa
cpus_per_task: 32
folder: ${PN_FOLDER}
mem_per_gpu: 80G
nodes: 1
tasks_per_node: 8
num_workers: 8
eval_name: video_classification_frozen
resume_checkpoint: false
tag: ${PN_TAG}
experiment:
  classifier:
    task_type: regression
    num_heads: 16
    num_probe_blocks: 4
    num_targets: 1
  data:
    dataset_type: VideoDataset
    dataset_train: ${DATA_DIR}/echonet_dynamic_train_s3_raw.csv
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
    - {final_lr: 0.0, final_weight_decay: 0.1, lr: 0.0001, start_lr: 0.0001, warmup: 0.0, weight_decay: 0.1}
    - {final_lr: 0.0, final_weight_decay: 0.4, lr: 0.0001, start_lr: 0.0001, warmup: 0.0, weight_decay: 0.4}
    - {final_lr: 0.0, final_weight_decay: 0.01, lr: 0.00005, start_lr: 0.00005, warmup: 0.0, weight_decay: 0.01}
    - {final_lr: 0.0, final_weight_decay: 0.1, lr: 0.00005, start_lr: 0.00005, warmup: 0.0, weight_decay: 0.1}
    - {final_lr: 0.0, final_weight_decay: 0.4, lr: 0.00005, start_lr: 0.00005, warmup: 0.0, weight_decay: 0.4}
model_kwargs:
  checkpoint: null
  module_name: evals.video_classification_frozen.modelcustom.panecho_encoder
  pretrain_kwargs: {}
  wrapper_kwargs: {}
YAMLEOF

for SPEC in "echoprime:${EP_CFG}:${EP_FOLDER}" "panecho:${PN_CFG}:${PN_FOLDER}"; do
  NAME="${SPEC%%:*}"
  REST="${SPEC#*:}"
  CFG="${REST%%:*}"
  FOLDER="${REST##*:}"
  echo "============================================"
  echo "[INFO] Training ${NAME} LVEF probe..."
  echo "============================================"
  python3 -m evals.main --fname "$CFG" \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7

  aws s3 sync "$FOLDER/" "${RUN_S3}/${NAME}_lvef/" --only-show-errors || true
  upload_logs
done

echo "[INFO] fourmodel external LVEF probes complete"
