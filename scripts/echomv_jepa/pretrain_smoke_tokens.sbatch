#!/bin/bash
# EchoMV-JEPA Stage-1 Option A smoke pretrain (token-level, no c_clip cache).
#
# Runs ~225 steps of Stage-1 with on-the-fly V-JEPA encoding:
#   - no aws s3 sync of cclip/ (not used)
#   - raw MP4s decoded per step from S3 via VideoDataset + Decord
#   - frozen V-JEPA ViT-L produces (T=1568, 1024) tokens per clip; wrapper
#     applies a 2x2 spatial pool → (392, 1024)
#   - TokenStudyTransformer attends at token granularity (M × T = ~6 × 392 = 2352)
#
# Gate: §20.2.a contextualization probes —
#   z_cosine_vs_isolated < 0.90   (pooled Stage-1 showed 0.998)
#   z_cosine_vs_peer_drop < 0.98  (pooled Stage-1 showed 0.9999)
#   var_t > 0.3, cov_off within bounds, no NaNs
#
# Duration: ~15 min wall (225 steps x ~3s + ~3min startup + ckpt sync).
#
#SBATCH -p dev
#SBATCH --nodelist=ip-10-0-50-146
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH -t 0-00:45:00
#SBATCH -J echomv_smoke_tok
#SBATCH -o /opt/dlami/nvme/logs/echomv_smoke_tok-%j.out
#SBATCH -e /opt/dlami/nvme/logs/echomv_smoke_tok-%j.err
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
SRC_TARBALL_S3="${S3_BASE}/setup/vjepa2-src-echomv-v1.tar.gz"

K_SAMPLE_S3="${S3_BASE}/echoset_jepa/study_clip_sample_K8_seed0_train.parquet"
# Encoder checkpoint — the same e125.pt that the v1 cache used, so the token
# representation is encoder-matched with Stage-0 / pooled Stage-1.
CKPT_S3="${S3_BASE}/runs/jepa_in21k_e200_280/training_folder/e125.pt"

# Local paths (referenced by the generated encoder YAML).
CKPT_LOCAL="/opt/dlami/nvme/echomv_cache/e125.pt"
ENC_YAML_LOCAL="/opt/dlami/nvme/echomv_cache/encoder.yaml"
K_SAMPLE_LOCAL="/opt/dlami/nvme/echomv_cache/k8_train.parquet"
WORKDIR="/opt/dlami/nvme/echomv_smoke_tok_${SLURM_JOB_ID}"
mkdir -p "${WORKDIR}" "$(dirname ${K_SAMPLE_LOCAL})"

nvidia-smi -L || true

# --- Deploy source --------------------------------------------------------
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

# --- Pull inputs ----------------------------------------------------------
aws s3 cp "${K_SAMPLE_S3}" "${K_SAMPLE_LOCAL}" --quiet
# Encoder checkpoint is only 4.8 GB — pull to NVMe once.
if [ ! -f "${CKPT_LOCAL}" ]; then
  aws s3 cp "${CKPT_S3}" "${CKPT_LOCAL}" --quiet
fi
ls -lh "${CKPT_LOCAL}"

python -c "import pandas as pd; df=pd.read_parquet('${K_SAMPLE_LOCAL}'); print('K-sample rows:', len(df), 'studies:', df.study_id.nunique())"

# --- Write encoder inference YAML ----------------------------------------
# Matches the one cache_cclip.sbatch generates — identical model kwargs.
cat > "${ENC_YAML_LOCAL}" <<YAMLEOF
app: vjepa
experiment:
  data:
    resolution: 224
    frames_per_clip: 16
    frame_step: 2
model_kwargs:
  checkpoint: ${CKPT_LOCAL}
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

# --- Compile training config ---------------------------------------------
# The checked-in stage1_token_online.yaml references this exact encoder YAML
# path; copy it into the workdir so folder overrides don't break it.
RUN_FOLDER="${WORKDIR}/run"
mkdir -p "${RUN_FOLDER}"
CONFIG="${WORKDIR}/stage1_token_online.yaml"
cp "${REPO_DIR}/configs/train/echomv_jepa/stage1_token_online.yaml" "${CONFIG}"
# Redirect sampler manifest and run folder to local paths.
python - <<PY
import yaml
path = "${CONFIG}"
with open(path) as f:
    cfg = yaml.safe_load(f)
cfg["folder"] = "${RUN_FOLDER}"
cfg["experiment"]["sampler"]["sample_manifest"] = "${K_SAMPLE_LOCAL}"
cfg["experiment"]["clip_encoder"]["config_path"] = "${ENC_YAML_LOCAL}"
with open(path, "w") as f:
    yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
print("wrote", path)
PY

# --- Launch ---------------------------------------------------------------
# srun-based DDP; app.main_srun reads SLURM_PROCID / SLURM_LOCALID.
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -1)
export MASTER_PORT=37129

echo "[INFO] === EchoMV-JEPA Stage-1 Option A smoke (800 steps, token-level) ==="
echo "[INFO] MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
srun --ntasks-per-node=8 python -m app.main_srun --fname "$CONFIG" 2>&1 | tee "${WORKDIR}/smoke_tok.log" \
  || echo "[WARN] training exited non-zero"

# --- Upload artifacts -----------------------------------------------------
aws s3 sync "${RUN_FOLDER}" "${S3_BASE}/echomv_jepa/smoke_token_runs/${SLURM_JOB_ID}/" --only-show-errors

echo "[INFO] === token-level smoke complete ==="
echo "   run folder: ${RUN_FOLDER}"
echo "   s3:         ${S3_BASE}/echomv_jepa/smoke_token_runs/${SLURM_JOB_ID}/"
