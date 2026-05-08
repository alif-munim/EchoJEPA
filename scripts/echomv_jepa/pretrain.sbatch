#!/bin/bash
# EchoMV-JEPA Stage-1 / 1b / 1m full pretrain.
#
# Which stage runs is controlled by the STAGE env var set by the caller:
#   - STAGE=stage1  (Stage-1 MVP: lambda_nce=0, num_modalities=1)
#   - STAGE=stage1b (Stage-1b: lambda_nce=0.005)
#   - STAGE=stage1m (Stage-1m: per-modality projector, num_modalities=8)
#
# Submit:
#   sbatch --export=STAGE=stage1  scripts/echomv_jepa/pretrain.sbatch
#   sbatch --export=STAGE=stage1b scripts/echomv_jepa/pretrain.sbatch
#   sbatch --export=STAGE=stage1m scripts/echomv_jepa/pretrain.sbatch
#
# Or use scripts/echomv_jepa/launch_breadth.sh to submit all three at once.
#
# Duration: ~3-4 h for 27k steps × 256 studies/step on 8×H100 (cached features).
# All three share the same K=8 sampler-matched manifest + cclip cache, so
# wall-clock is dominated by compute not I/O.
#
#SBATCH -p dev
#SBATCH --nodelist=ip-10-0-50-146
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH -t 0-06:00:00
#SBATCH -J echomv_pretrain
#SBATCH -o /opt/dlami/nvme/logs/echomv_pretrain-%j.out
#SBATCH -e /opt/dlami/nvme/logs/echomv_pretrain-%j.err
#SBATCH --chdir=/opt/dlami/nvme

set -euxo pipefail

# --- Resolve stage --------------------------------------------------------
STAGE="${STAGE:-stage1}"
case "${STAGE}" in
  stage1)
    NUM_MODALITIES=1
    LAMBDA_NCE=0.0
    ;;
  stage1b)
    NUM_MODALITIES=1
    LAMBDA_NCE=0.005
    ;;
  stage1m)
    NUM_MODALITIES=8
    LAMBDA_NCE=0.0
    ;;
  *)
    echo "ERROR: unknown STAGE='${STAGE}' (expected: stage1 | stage1b | stage1m)" >&2
    exit 1
    ;;
esac

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

CACHE_S3="${S3_BASE}/echoset_jepa/cclip"
K_SAMPLE_S3="${S3_BASE}/echoset_jepa/study_clip_sample_K8_seed0_train.parquet"

CACHE_LOCAL="/opt/dlami/nvme/echoset_cache"
K_SAMPLE_LOCAL="/opt/dlami/nvme/echomv_cache/k8_train.parquet"
WORKDIR="/opt/dlami/nvme/echomv_${STAGE}_${SLURM_JOB_ID}"
mkdir -p "${WORKDIR}" "${CACHE_LOCAL}" "$(dirname ${K_SAMPLE_LOCAL})"

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
aws s3 sync "${CACHE_S3}" "${CACHE_LOCAL}" --only-show-errors --exclude "_index/*"
python -c "import pandas as pd; df=pd.read_parquet('${K_SAMPLE_LOCAL}'); print('K-sample rows:', len(df), 'studies:', df.study_id.nunique())"

# --- Compile config -------------------------------------------------------
# We generate the config at submit time rather than checking in three near-
# identical files. This keeps the three stages byte-for-byte comparable
# (same seed, sampler, optimizer) and lets us bump step counts without
# touching 3 YAMLs.
RUN_FOLDER="${WORKDIR}/run"
mkdir -p "${RUN_FOLDER}"
CONFIG="${WORKDIR}/echomv_${STAGE}.yaml"
cat > "$CONFIG" <<YAMLEOF
app: echomv_jepa
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
  target_meta:  {include_target_phase: true, target_quality_token_ablation: false}

  projector:
    num_modalities: ${NUM_MODALITIES}
    d_hidden: 1024
    d_proj: 256

  modality_cohort: ["b_mode", "color_doppler"]

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
    bmode_budget: 6
    color_budget: 2
    spectral_budget: 0
    sample_manifest: ${K_SAMPLE_LOCAL}

  loss: {lambda_nce: ${LAMBDA_NCE}, tau_nce: 0.1}
  ema:  {tau_start: 0.996, tau_end: 0.9999, schedule: cosine}

  clip_encoder:
    source: cached
    frozen: true
    cache_local_prefix: ${CACHE_LOCAL}

  optim:
    lr: 5.0e-4
    weight_decay: 0.05
    betas: [0.9, 0.95]
    warmup_steps: 5000
    main_steps: 20000
    cooldown_steps: 2000
    batch_studies_per_gpu: 32

  diagnostics:
    peer_drop_every_n_steps: 50
    halt_z_cosine_vs_v1_threshold: 0.98
    halt_z_cosine_vs_v1_consec_steps: 5000

  logging:
    log_every_steps: 50
    csv_schema:
      - step
      - loss
      - loss_regress
      - loss_nce
      - var_t
      - cov_off
      - z_cosine_vs_v1
      - z_cosine_vs_isolated
      - z_cosine_vs_peer_drop
      - valid_neg_count_same_view_mean
      - valid_neg_count_same_view_min
      - valid_neg_count_same_modality_mean
      - fallback_fraction
      - mask_strategy
      - M_elements_mean

  collapse_monitor:
    var_t_floor: 0.3
    halt_if_below_for_steps: 500
YAMLEOF

# --- Launch ---------------------------------------------------------------
# srun-based DDP: one task per GPU; app.main_srun reads SLURM_PROCID / SLURM_LOCALID.
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -1)
export MASTER_PORT=37129

echo "[INFO] === EchoMV-JEPA ${STAGE} pretrain (27k steps, lambda_nce=${LAMBDA_NCE}, num_modalities=${NUM_MODALITIES}) ==="
echo "[INFO] MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
srun --ntasks-per-node=8 python -m app.main_srun --fname "$CONFIG" 2>&1 | tee "${WORKDIR}/pretrain.log" \
  || echo "[WARN] training exited non-zero"

# --- Upload artifacts -----------------------------------------------------
aws s3 sync "${RUN_FOLDER}" "${S3_BASE}/echomv_jepa/${STAGE}_runs/${SLURM_JOB_ID}/" --only-show-errors

echo "[INFO] === ${STAGE} complete ==="
echo "   run folder: ${RUN_FOLDER}"
echo "   s3:         ${S3_BASE}/echomv_jepa/${STAGE}_runs/${SLURM_JOB_ID}/"
