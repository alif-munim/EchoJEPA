#!/bin/bash
# EchoSet-JEPA Stage-2 smoke pretrain (PR-N5 gate run).
#
# Prerequisites:
#   - PR-N3 c_clip cache at s3://.../echoset_jepa/cclip/
#   - PR-N3 target-difficulty audit passed
#   - K-sample manifest at s3://.../echoset_jepa/study_clip_sample_K8_seed0_train.parquet
#
# Duration: 5k steps × 256 studies/step ≈ ~30 min on 8×H100 (cached features).
# Gate:     L_regress decreasing, var_t > 0.3 sustained, no NaN.
#
#SBATCH -p dev
#SBATCH --nodelist=ip-10-0-50-146
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH -t 0-01:30:00
#SBATCH -J echoset_smoke
#SBATCH -o /opt/dlami/nvme/logs/echoset_smoke-%j.out
#SBATCH -e /opt/dlami/nvme/logs/echoset_smoke-%j.err
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

CACHE_S3="${S3_BASE}/echoset_jepa/cclip"
K_SAMPLE_S3="${S3_BASE}/echoset_jepa/study_clip_sample_K8_seed0_train.parquet"

CACHE_LOCAL="/opt/dlami/nvme/echoset_cache"
WORKDIR="/opt/dlami/nvme/echoset_smoke_${SLURM_JOB_ID}"
mkdir -p "${WORKDIR}" "${CACHE_LOCAL}"

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
K_SAMPLE_LOCAL="${WORKDIR}/k8.parquet"
aws s3 cp "${K_SAMPLE_S3}" "${K_SAMPLE_LOCAL}" --quiet
aws s3 sync "${CACHE_S3}" "${CACHE_LOCAL}" --only-show-errors --exclude "_index/*"
python -c "import pandas as pd; df=pd.read_parquet('${K_SAMPLE_LOCAL}'); print('K-sample rows:', len(df), 'studies:', df.study_id.nunique())"

# --- Compile smoke config -------------------------------------------------
RUN_FOLDER="${WORKDIR}/run"
mkdir -p "${RUN_FOLDER}"
CONFIG="${WORKDIR}/echoset_smoke.yaml"
cat > "$CONFIG" <<YAMLEOF
app: echoset_jepa
folder: ${RUN_FOLDER}
num_workers: 4
checkpoint_every_steps: 2500
seed: 0

experiment:
  study_transformer:
    d_clip: 1024
    d_model: 512
    n_layers: 4
    n_heads: 8
    ffn_mult: 4
    dropout_ffn: 0.1
    dropout_attn: 0.0
    max_M: 64

  elements:
    element_agg: mean

  meta_dropout: {view: 0.15, modality: 0.10, phase: 0.30, quality: 0.30}
  target_meta: {include_target_phase: true, target_quality_token_ablation: false}
  target_mode: element_target

  projector: {d_hidden: 1024, d_proj: 256}

  masking:
    strategy_weights:
      random_element: 0.55
      whole_view_family: 0.25
      whole_modality: 0.10
      apical_holdout: 0.05
      doppler_holdout: 0.025
      bmode_holdout: 0.025
    mask_ratio_cap: 0.6
    min_context: 1
    min_target: 1

  sampler:
    K_clips: 8
    seed: 0
    policy: mixed_modality
    sample_manifest: ${K_SAMPLE_LOCAL}

  loss: {lambda_nce: 0.03, tau_nce: 0.1}
  ema:  {tau_start: 0.996, tau_end: 0.9999, schedule: cosine}

  clip_encoder:
    source: cached
    frozen: true
    cache_local_prefix: ${CACHE_LOCAL}

  optim:
    lr: 5.0e-4
    weight_decay: 0.05
    betas: [0.9, 0.95]
    warmup_steps: 500
    main_steps: 4000
    cooldown_steps: 500
    batch_studies_per_gpu: 32

  logging:
    log_every_steps: 50

  collapse_monitor:
    var_t_floor: 0.3
    halt_if_below_for_steps: 500
YAMLEOF

# --- Launch ---------------------------------------------------------------
echo "[INFO] === Stage-2 smoke pretrain (5k steps) ==="
python -m app.main_distributed --fname "$CONFIG" 2>&1 | tee "${WORKDIR}/smoke.log" \
  || echo "[WARN] training exited non-zero"

# --- Upload artifacts -----------------------------------------------------
aws s3 sync "${RUN_FOLDER}" "${S3_BASE}/echoset_jepa/smoke_runs/${SLURM_JOB_ID}/" --only-show-errors

echo "[INFO] === smoke complete ==="
echo "   run folder: ${RUN_FOLDER}"
echo "   s3:         ${S3_BASE}/echoset_jepa/smoke_runs/${SLURM_JOB_ID}/"
