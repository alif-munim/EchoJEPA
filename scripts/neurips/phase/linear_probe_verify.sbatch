#!/bin/bash
#SBATCH -p dev
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 02:00:00
#SBATCH -J lin_verify
#SBATCH -o /opt/dlami/nvme/logs/lin_verify_%j.out
#SBATCH -e /opt/dlami/nvme/logs/lin_verify_%j.err
#SBATCH --priority=10000

set -euxo pipefail

# =============================================================================
# Linear-probe re-run for JEPA e100+e200 and MAE e99+e194 on EchoNet-Dynamic.
# Pipeline:
#   1. Extract pre-pool features for 4 (ckpt, split) pairs we don't have:
#        - JEPA e200 train + test
#        - MAE  e194 train + test
#      (JEPA e100 and MAE e99 caches already on S3.)
#   2. Train linear-A-raw probes (seeds 0..4, WD sweep 1e-4/1e-2) on all 4 ckpts.
#   3. Report mean±std test R² vs the published table.
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
FEATURES_S3="${S3_BASE}/features/diff_probe"
RUN_S3="${S3_BASE}/runs/linear_probe_verify_${SLURM_JOB_ID}"

upload_logs() {
  local out="/opt/dlami/nvme/logs/lin_verify_${SLURM_JOB_ID}.out"
  local err="/opt/dlami/nvme/logs/lin_verify_${SLURM_JOB_ID}.err"
  [ -f "${out}" ] && aws s3 cp "${out}" "${RUN_S3}/logs/job.out" --only-show-errors || true
  [ -f "${err}" ] && aws s3 cp "${err}" "${RUN_S3}/logs/job.err" --only-show-errors || true
}
trap 'upload_logs' EXIT

nvidia-smi -L || true
mkdir -p /opt/dlami/nvme/logs

SCRATCH="/opt/dlami/nvme"
WORKDIR="${SCRATCH}/lin_verify_${SLURM_JOB_ID}"
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

# EchoNet CSVs
DATA_DIR="${WORKDIR}/data/csv"
mkdir -p "$DATA_DIR"
aws s3 cp "${S3_BASE}/data/csv/echonet_dynamic_train_s3_raw.csv" "${DATA_DIR}/" --quiet
aws s3 cp "${S3_BASE}/data/csv/echonet_dynamic_test_s3_raw.csv" "${DATA_DIR}/" --quiet

# Backbone checkpoints
CKPT_DIR="${WORKDIR}/checkpoints"
mkdir -p "$CKPT_DIR"
aws s3 cp "${S3_BASE}/runs/jepa_in21k_e200_280/training_folder/latest.pt" "${CKPT_DIR}/jepa_in21k_vitl_e200.pt" --quiet
aws s3 cp "${S3_BASE}/runs/videomae_e200b_179/training_folder/checkpoint-194.pth" "${CKPT_DIR}/echomae_l_mimic_ep194.pth" --quiet

# Local feature cache dir
FEAT_DIR="${WORKDIR}/features/diff_probe"
mkdir -p "$FEAT_DIR"

# Pull existing JEPA e100 + MAE e99 caches (already on S3) — saves us extraction
for pair in "jepa_e100_train" "jepa_e100_test" "mae_e99_train" "mae_e99_test"; do
  aws s3 cp "${FEATURES_S3}/${pair}.pt" "${FEAT_DIR}/${pair}.pt" --quiet
done

# ---------------- Extraction ----------------
extract() {
  local MODEL="$1"      # label, e.g. jepa_e200
  local SPLIT="$2"      # train|test
  local CKPT_PATH="$3"  # full path to backbone .pt
  local MODULE="$4"     # evals.video_classification_frozen.modelcustom.vit_encoder_multiclip | videomae_encoder

  local CSV="${DATA_DIR}/echonet_dynamic_${SPLIT}_s3_raw.csv"
  local OUT_FEATURES="${FEAT_DIR}/${MODEL}_${SPLIT}.pt"

  # Create a minimal yaml on the fly
  local CFG="${WORKDIR}/cfg_${MODEL}_${SPLIT}.yaml"
  if [[ "${MODULE}" == *"videomae"* ]]; then
    cat > "$CFG" <<YAML
app: vjepa
folder: ${WORKDIR}/evals/${MODEL}_${SPLIT}_prepool
eval_name: feature_extraction_pre_pool
tag: lin-verify-${MODEL}-${SPLIT}
experiment:
  data:
    dataset_type: VideoDataset
    dataset_val: ${CSV}
    resolution: 224
    frames_per_clip: 16
    frame_step: 2
    num_segments: 2
    num_views_per_segment: 1
  optimization:
    batch_size: 8
    use_bfloat16: true
model_kwargs:
  checkpoint: ${CKPT_PATH}
  module_name: ${MODULE}
  pretrain_kwargs:
    encoder:
      model_name: vit_large_patch16_224
      tubelet_size: 2
  wrapper_kwargs: {}
YAML
  else
    cat > "$CFG" <<YAML
app: vjepa
folder: ${WORKDIR}/evals/${MODEL}_${SPLIT}_prepool
eval_name: feature_extraction_pre_pool
tag: lin-verify-${MODEL}-${SPLIT}
experiment:
  data:
    dataset_type: VideoDataset
    dataset_val: ${CSV}
    resolution: 224
    frames_per_clip: 16
    frame_step: 2
    num_segments: 2
    num_views_per_segment: 1
  optimization:
    batch_size: 8
    use_bfloat16: true
model_kwargs:
  checkpoint: ${CKPT_PATH}
  module_name: ${MODULE}
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
YAML
  fi

  echo "[EXTRACT] ${MODEL} ${SPLIT} -> ${OUT_FEATURES}"
  FEATURES_OUTPUT="${OUT_FEATURES}" \
  python3 -m evals.main --fname "${CFG}" \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7

  # Upload to S3 so future runs can reuse
  aws s3 cp "${OUT_FEATURES}" "${FEATURES_S3}/$(basename ${OUT_FEATURES})" --only-show-errors
}

# JEPA e200
extract "jepa_e200" "train" "${CKPT_DIR}/jepa_in21k_vitl_e200.pt" \
  "evals.video_classification_frozen.modelcustom.vit_encoder_multiclip"
extract "jepa_e200" "test"  "${CKPT_DIR}/jepa_in21k_vitl_e200.pt" \
  "evals.video_classification_frozen.modelcustom.vit_encoder_multiclip"

# MAE e194
extract "mae_e194" "train" "${CKPT_DIR}/echomae_l_mimic_ep194.pth" \
  "evals.video_classification_frozen.modelcustom.videomae_encoder"
extract "mae_e194" "test"  "${CKPT_DIR}/echomae_l_mimic_ep194.pth" \
  "evals.video_classification_frozen.modelcustom.videomae_encoder"

# ---------------- Linear probes ----------------
RES_DIR="${WORKDIR}/results"
mkdir -p "$RES_DIR"

run_probe() {
  local M="$1"
  python3 -u scripts/neurips/diff_probe_train.py \
    --train-cache "${FEAT_DIR}/${M}_train.pt" \
    --test-cache  "${FEAT_DIR}/${M}_test.pt" \
    --model "$M" \
    --out-json "${RES_DIR}/${M}.json" \
    --archs linear-A --inputs raw \
    --seeds 0 1 2 3 4 --wd-sweep 1e-4 1e-2
}

for M in jepa_e100 jepa_e200 mae_e99 mae_e194; do
  run_probe "$M"
done

# Sync everything
aws s3 sync "${RES_DIR}/" "${RUN_S3}/results/" --only-show-errors || true

# Print a clean summary for the logs
python3 -u <<'PYEOF'
import json, glob
import numpy as np
print("\n========== LINEAR-A RAW CLEAN TEST R² =========")
print(f"{'model':<12} {'mean':>8} {'std':>8} {'n_runs':>8}")
rows = {}
for f in sorted(glob.glob('/opt/dlami/nvme/lin_verify_${SLURM_JOB_ID}/results/*.json')):
    model = f.split('/')[-1].replace('.json', '')
    d = json.load(open(f))
    # Filter to linear-A raw only (no diff, no shuffle control, no inference-mf)
    rows[model] = []
    for r in d.get('results', []):
        if r.get('arch') == 'linear-A' and r.get('input') == 'raw':
            rows[model].append(r['test_r2'])
    if rows[model]:
        arr = np.array(rows[model])
        print(f"{model:<12} {arr.mean():>8.3f} {arr.std():>8.3f} {len(arr):>8}")
PYEOF

echo "[INFO] linear-probe verify complete"
