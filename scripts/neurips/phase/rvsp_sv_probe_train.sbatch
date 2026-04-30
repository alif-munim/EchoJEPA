#!/bin/bash
#SBATCH -p dev
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 10:00:00
#SBATCH -J rvsp_sv
#SBATCH -o /opt/dlami/nvme/logs/rvsp_sv_%j.out
#SBATCH -e /opt/dlami/nvme/logs/rvsp_sv_%j.err

set -euxo pipefail

# =============================================================================
# Single-view MIMIC RVSP probe — 9-way model comparison on a 10K/2K/2K subset.
#
#   4 IN21K-init pretrains (e100):  jepa, mae, byol, salt
#   3 echo specialists:             echojepa-l-k, echoprime, panecho
#   2 e200 extensions:              jepa_in21k_e200, mae_e200
#
# Single-view: one clip per video at a time, d=4 attentive probe,
# 20 epochs, 6-HP grid. Uses mimic_rvsp_sv_{train,val}_10k.csv (study-disjoint
# subset of the full MIMIC RVSP CSVs at
# experiments/nature_medicine/mimic/probe_csvs/rvsp/).
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
RUN_S3="${S3_BASE}/runs/rvsp_sv_${SLURM_JOB_ID}"

# MIMIC RVSP z-score params (from experiments/nature_medicine/mimic/probe_csvs/rvsp/zscore_params.json)
TARGET_MEAN="30.0959"
TARGET_STD="12.2321"

# One entry per model:
#   NAME|BACKBONE|CKPT_S3|CHECKPOINT_KEY
# Delimiter is "|" because CKPT_S3 values contain ":" (s3://...).
# BACKBONE determines the encoder wrapper family:
#   vit    -> vit_encoder_multiclip (JEPA, BYOL, SALT, EchoJEPA-L-K)
#   mae    -> videomae_encoder
#   eprime -> echo_prime_encoder
#   panecho-> panecho_encoder
JOBS=(
  # --- 4 IN21K e100 pretrains ---
  "jepa_in21k_e100|vit|${S3_BASE}/checkpoints/jepa_in21k/jepa_in21k_vitl_e100.pt|target_encoder"
  "mae_in21k_e100|mae|${S3_BASE}/checkpoints/echomae_l_mimic_ep99.pth|"
  "byol_in21k_e100|vit|${S3_BASE}/checkpoints/byol/byol_vitl_imagenet_v2_e100.pt|target_encoder"
  "salt_in21k_e79|vit|${S3_BASE}/checkpoints/salt_v1/salt_s2v1_e79.pt|target_encoder"
  # --- 3 echo specialists ---
  "echojepa_l_k|vit|${S3_BASE}/checkpoints/anneal/keep/vitl-kinetics-pt220-an55.pt|target_encoder"
  "echoprime|eprime|${S3_BASE}/checkpoints/echoprime/echo_prime_encoder.pt|"
  "panecho|panecho|${S3_BASE}/checkpoints/panecho/panecho.pt|"
  # --- 2 e200 extensions ---
  "jepa_in21k_e200|vit|${S3_BASE}/runs/jepa_in21k_e200_280/training_folder/latest.pt|target_encoder"
  "mae_in21k_e200|mae|${S3_BASE}/runs/videomae_e200b_178/training_folder/checkpoint-184.pth|"
)

upload_logs() {
  local out="/opt/dlami/nvme/logs/rvsp_sv_${SLURM_JOB_ID}.out"
  local err="/opt/dlami/nvme/logs/rvsp_sv_${SLURM_JOB_ID}.err"
  [ -f "${out}" ] && aws s3 cp "${out}" "${RUN_S3}/logs/job.out" --only-show-errors || true
  [ -f "${err}" ] && aws s3 cp "${err}" "${RUN_S3}/logs/job.err" --only-show-errors || true
}
trap 'upload_logs' EXIT

nvidia-smi -L || true
mkdir -p /opt/dlami/nvme/logs

SCRATCH="/opt/dlami/nvme"
WORKDIR="${SCRATCH}/rvsp_sv_${SLURM_JOB_ID}"
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
aws s3 cp "${S3_BASE}/data/csv/mimic_rvsp_sv_train_10k.csv" "${DATA_DIR}/" --quiet
aws s3 cp "${S3_BASE}/data/csv/mimic_rvsp_sv_val_10k.csv"   "${DATA_DIR}/" --quiet

CKPT_DIR="${WORKDIR}/checkpoints"
mkdir -p "$CKPT_DIR"

# ---------------------------------------------------------------------------
# Download all backbones (pre-stage so network I/O doesn't interleave with
# training). Keep extension to preserve MAE's .pth convention.
# ---------------------------------------------------------------------------
for J in "${JOBS[@]}"; do
  IFS='|' read -r NAME BACKBONE CKPT_SRC CKPT_KEY <<< "$J"
  EXT="${CKPT_SRC##*.}"
  DEST="${CKPT_DIR}/${NAME}.${EXT}"
  echo "[INFO] staging ${NAME} from ${CKPT_SRC}"
  aws s3 cp "$CKPT_SRC" "$DEST" --quiet
  ls -lh "$DEST"
done

# ---------------------------------------------------------------------------
# Emit per-model YAML configs, then train sequentially.
# ---------------------------------------------------------------------------
for J in "${JOBS[@]}"; do
  IFS='|' read -r NAME BACKBONE CKPT_SRC CKPT_KEY <<< "$J"
  EXT="${CKPT_SRC##*.}"
  CKPT_LOCAL="${CKPT_DIR}/${NAME}.${EXT}"
  CFG="configs/eval/vitl/neurips/rvsp_sv/${NAME}_rvsp_sv_d4.yaml"
  FOLDER="evals/vitl/neurips/rvsp_sv/${NAME}_rvsp_sv_224"
  TAG="neurips-${NAME//_/-}-rvsp-sv-d4"
  mkdir -p "$(dirname "$CFG")"

  # Common YAML header shared across model families.
  cat > "$CFG" <<YAMLEOF
app: vjepa
cpus_per_task: 32
folder: ${FOLDER}
mem_per_gpu: 80G
nodes: 1
tasks_per_node: 8
num_workers: 8
eval_name: video_classification_frozen
resume_checkpoint: false
tag: ${TAG}
experiment:
  classifier:
    task_type: regression
    num_heads: 16
    num_probe_blocks: 4
    num_targets: 1
  data:
    dataset_type: VideoDataset
    dataset_train: ${DATA_DIR}/mimic_rvsp_sv_train_10k.csv
    dataset_val:   ${DATA_DIR}/mimic_rvsp_sv_val_10k.csv
    num_classes: 1
    target_mean: ${TARGET_MEAN}
    target_std: ${TARGET_STD}
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
YAMLEOF

  # Model-family-specific model_kwargs block.
  case "$BACKBONE" in
    vit)
      cat >> "$CFG" <<YAMLEOF
model_kwargs:
  checkpoint: ${CKPT_LOCAL}
  module_name: evals.video_classification_frozen.modelcustom.vit_encoder_multiclip
  pretrain_kwargs:
    encoder:
      checkpoint_key: ${CKPT_KEY}
      model_name: vit_large
      patch_size: 16
      tubelet_size: 2
      uniform_power: true
      use_rope: true
  wrapper_kwargs:
    max_frames: 128
    use_pos_embed: false
YAMLEOF
      ;;
    mae)
      cat >> "$CFG" <<YAMLEOF
model_kwargs:
  checkpoint: ${CKPT_LOCAL}
  module_name: evals.video_classification_frozen.modelcustom.videomae_encoder
  pretrain_kwargs:
    encoder:
      model_name: vit_large_patch16_224
      tubelet_size: 2
  wrapper_kwargs:
    return_per_clip: true
YAMLEOF
      ;;
    eprime)
      cat >> "$CFG" <<YAMLEOF
model_kwargs:
  checkpoint: ${CKPT_LOCAL}
  module_name: evals.video_classification_frozen.modelcustom.echo_prime_encoder
  pretrain_kwargs: {}
  wrapper_kwargs: {}
YAMLEOF
      ;;
    panecho)
      cat >> "$CFG" <<YAMLEOF
model_kwargs:
  checkpoint: ${CKPT_LOCAL}
  module_name: evals.video_classification_frozen.modelcustom.panecho_encoder
  pretrain_kwargs: {}
  wrapper_kwargs: {}
YAMLEOF
      ;;
    *)
      echo "[ERROR] unknown backbone family: $BACKBONE"; exit 2;;
  esac

  echo "[INFO] wrote config $CFG"
done

# ---------------------------------------------------------------------------
# Train each model. On failure, log it and keep going — we want to finish the
# rest of the comparison rather than bail on the first broken backbone.
# ---------------------------------------------------------------------------
for J in "${JOBS[@]}"; do
  IFS='|' read -r NAME _ _ _ <<< "$J"
  CFG="configs/eval/vitl/neurips/rvsp_sv/${NAME}_rvsp_sv_d4.yaml"
  FOLDER="evals/vitl/neurips/rvsp_sv/${NAME}_rvsp_sv_224"

  echo "============================================"
  echo "[INFO] training ${NAME} — RVSP single-view probe"
  echo "============================================"
  if python3 -m evals.main --fname "$CFG" \
       --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 ; then
    echo "[INFO] ${NAME} done"
  else
    echo "[WARN] ${NAME} failed — continuing with remaining models"
  fi

  aws s3 sync "$FOLDER/" "${RUN_S3}/${NAME}_sv/" --only-show-errors || true
  upload_logs
  # Free disk for subsequent backbones. Glob to catch either .pt or .pth.
  rm -f "${CKPT_DIR}/${NAME}".* 2>/dev/null || true
done

echo "[INFO] rvsp_sv 9-way probe comparison complete"
