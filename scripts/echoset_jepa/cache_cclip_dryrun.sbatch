#!/bin/bash
# Cache frozen V-JEPA c_clip for the whole EchoSet-JEPA manifest.
#
# Prerequisites:
#   - /tmp/echoset_pr_n2/study_clip_manifest_dedup.parquet exists (PR-N2 output)
#   - encoder checkpoint accessible (set CKPT_S3 below)
#   - V-JEPA source tarball on S3 (matches existing deployment pattern)
#
# Runtime: ~2h on 1 × 8×H100 node for 214k clips at 14 clips/s/GPU.
# Output:  s3://${ART_BUCKET}/vjepa2-artifacts/echoset_jepa/cclip/{study_id}/{clip_id}.npy
#          + s3://${ART_BUCKET}/vjepa2-artifacts/echoset_jepa/cache_index.parquet
#
#SBATCH -p dev
#SBATCH --nodelist=ip-10-0-50-146
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 0-00:30:00
#SBATCH -J echoset_cache_dryrun
#SBATCH -o /opt/dlami/nvme/logs/echoset_cache_dryrun-%j.out
#SBATCH -e /opt/dlami/nvme/logs/echoset_cache_dryrun-%j.err
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
SRC_TARBALL_S3="${S3_BASE}/setup/vjepa2-src-echoset-v1.tar.gz"
# EchoJEPA-Base e125 is the Stage-1 encoder for EchoSet-JEPA v1.
CKPT_S3="${S3_BASE}/runs/jepa_in21k_e200_280/training_folder/e125.pt"
# Manifest copied from the SageMaker head node (PR-N2 output).
MANIFEST_S3="${S3_BASE}/echoset_jepa/study_clip_manifest_dedup.parquet"

CACHE_S3="${S3_BASE}/echoset_jepa/cclip"
CACHE_LOCAL="/opt/dlami/nvme/echoset_cache"
INDEX_LOCAL="${CACHE_LOCAL}/_index"
WORKDIR="/opt/dlami/nvme/echoset_cclip_dryrun_${SLURM_JOB_ID}"
mkdir -p "${WORKDIR}" "${CACHE_LOCAL}" "${INDEX_LOCAL}"

nvidia-smi -L || true

# --- Deploy source ---------------------------------------------------------
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

# --- Pull inputs -----------------------------------------------------------
CKPT_DIR="${WORKDIR}/checkpoints"; mkdir -p "$CKPT_DIR"
aws s3 cp "$CKPT_S3" "${CKPT_DIR}/pretrain.pt" --quiet
ls -lh "${CKPT_DIR}/pretrain.pt"

MANIFEST_LOCAL="${WORKDIR}/study_clip_manifest_dedup.parquet"
aws s3 cp "${MANIFEST_S3}" "${MANIFEST_LOCAL}" --quiet
python -c "import pandas as pd; df=pd.read_parquet('${MANIFEST_LOCAL}'); print('manifest rows:', len(df))"

# --- Build a minimal inference config pointing at this checkpoint ----------
CONFIG="${WORKDIR}/echoset_cache_config.yaml"
cat > "$CONFIG" <<YAMLEOF
app: vjepa
experiment:
  data:
    resolution: 224
    frames_per_clip: 16
    frame_step: 2
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

# --- Sync existing cache locally to enable resume across re-runs -----------
# For the first run this is empty. Later re-runs pull back any partial work.
aws s3 sync "${CACHE_S3}" "${CACHE_LOCAL}" --only-show-errors --exclude "_index/*" || true

# --- Run multi-GPU cache ---------------------------------------------------
echo "[INFO] === Caching c_clip with 8 GPUs ==="
python -m experiments.echoset_jepa.cache_cclip \
  --manifest "${MANIFEST_LOCAL}" \
  --config "${CONFIG}" \
  --cache_prefix "${CACHE_LOCAL}" \
  --index_dir "${INDEX_LOCAL}" \
  --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
  --batch_size 4 --num_workers 4 --dry_run 8 \
  2>&1 | tee "${WORKDIR}/cache_cclip.log" || echo "[WARN] cache_cclip non-zero exit"

# --- Upload results to S3 --------------------------------------------------
echo "[INFO] === Uploading cache + index to S3 ==="
aws s3 sync "${CACHE_LOCAL}" "${CACHE_S3}" --only-show-errors --exclude "_index/*"
aws s3 cp "${INDEX_LOCAL}/cache_index.parquet" "${S3_BASE}/echoset_jepa/cache_index.parquet" --only-show-errors

echo "[INFO] === c_clip cache complete ==="
echo "   local:  ${CACHE_LOCAL}"
echo "   s3:     ${CACHE_S3}"
echo "   index:  ${S3_BASE}/echoset_jepa/cache_index.parquet"
