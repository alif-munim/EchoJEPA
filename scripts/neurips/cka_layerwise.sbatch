#!/bin/bash
#SBATCH -p dev
#SBATCH -N 2
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 2:00:00
#SBATCH -J cka_layerwise
#SBATCH -o /opt/dlami/nvme/cka_layerwise_%j.out
#SBATCH -e /opt/dlami/nvme/cka_layerwise_%j.err
#SBATCH --signal=B:USR1@300

set -euxo pipefail

# =============================================================================
# Layer-wise CKA: JEPA vs BYOL vs MAE vs SALT (all IN21K-init, ~ep100, ViT-L).
# 4 models × 4 shards across 2 nodes × 8 GPUs = 16 GPUs.
# Node mapping:
#   node 0 GPUs 0-3 → JEPA-IN21K-e100 shards 0-3
#   node 0 GPUs 4-7 → BYOL-L-e100     shards 0-3
#   node 1 GPUs 0-3 → MAE-L-e99       shards 0-3
#   node 1 GPUs 4-7 → SALT-S2v1-e79   shards 0-3
# Rendezvous is via S3 (no shared FS across compute nodes).
# After all shards finish, merge + compute CKA + plot runs on node 0.
# =============================================================================

export AWS_DEFAULT_REGION="us-west-2"
export AWS_REGION="us-west-2"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export AWS_SDK_LOAD_CONFIG=1
export HOME="/home/ubuntu"
export AWS_SHARED_CREDENTIALS_FILE="${HOME}/.aws/credentials"
export AWS_CONFIG_FILE="${HOME}/.aws/config"
unset AWS_EC2_METADATA_DISABLED 2>/dev/null || true

aws sts get-caller-identity || exit 1

ART_BUCKET="sagemaker-hyperpod-lifecycle-495467399120-usw2"
S3_BASE="s3://${ART_BUCKET}/vjepa2-artifacts"
ENV_S3="${S3_BASE}/env/vjepa2-312-pack-env.tar"
RUN_S3="${S3_BASE}/runs/cka_layerwise_${SLURM_JOB_ID}"
FEATURES_S3="${RUN_S3}/features"

# Models are defined as a space-separated string so they can survive
# `export` across srun-spawned subshells (bash can't export arrays).
NUM_SHARDS=4
MODELS_STR="JEPA-IN21K-e100 BYOL-L-e100 MAE-L-e99 SALT-S2v1-e79"

# =========================
# Env/code setup (runs on each node in per_node_runner; and once more on
# node 0 in the aggregator step — it's idempotent and fast once cached).
# =========================
setup_env_and_code() {
  local SCRATCH="/opt/dlami/nvme"
  if [ ! -f "${SCRATCH}/env/bin/activate" ]; then
    aws s3 cp "${ENV_S3}" "${SCRATCH}/env.tar"
    mkdir -p "${SCRATCH}/env"
    tar -xf "${SCRATCH}/env.tar" -C "${SCRATCH}/env"
  fi
  set +u && source "${SCRATCH}/env/bin/activate" && set -u
  "${SCRATCH}/env/bin/conda-unpack" 2>/dev/null || true
  pip install boto3 matplotlib --no-cache-dir --quiet 2>/dev/null || true

  # Use a job-specific code dir so we don't collide with stale prior extractions.
  local CODE_DIR="${SCRATCH}/code_cka_${SLURM_JOB_ID}"
  if [ ! -f "${CODE_DIR}/scripts/neurips/cka_extract.py" ]; then
    rm -rf "${CODE_DIR}"
    mkdir -p "${CODE_DIR}"
    aws s3 cp "${S3_BASE}/setup/vjepa2-src.tar.gz" "${SCRATCH}/vjepa2-src-cka-${SLURM_JOB_ID}.tar.gz" --quiet
    tar xzf "${SCRATCH}/vjepa2-src-cka-${SLURM_JOB_ID}.tar.gz" -C "${CODE_DIR}"
    cd "${CODE_DIR}"
    pip install -e . --no-deps --no-build-isolation 2>/dev/null || true
  fi
  cd "${CODE_DIR}"
}
export -f setup_env_and_code

# =========================
# Per-node runner: 8 parallel shard extractions (2 models × 4 GPUs each).
# =========================
per_node_runner() {
  local NID="${SLURM_NODEID}"
  local SCRATCH="/opt/dlami/nvme"
  local LOCAL_FEAT="${SCRATCH}/cka/features"
  mkdir -p "${LOCAL_FEAT}"

  nvidia-smi -L || true
  pkill -9 -f "cka_extract" 2>/dev/null || true; sleep 2

  setup_env_and_code
  local CODE_DIR="${SCRATCH}/code_cka_${SLURM_JOB_ID}"

  mkdir -p "${SCRATCH}/data/csv"
  aws s3 cp "${S3_BASE}/data/csv/echonet_dynamic_test_s3_raw.csv" "${SCRATCH}/data/csv/" --quiet

  # Pull canonical encoders from echodata25 per canonical-checkpoints.md.
  # Hardcoded per-node assignments (cannot use bash arrays — don't export).
  local CANON_S3="s3://echodata25/neurips/encoders"
  local MODEL_A MODEL_B S3_A S3_B LOCAL_A LOCAL_B
  if [ "${NID}" = "0" ]; then
    MODEL_A="JEPA-IN21K-e100"
    MODEL_B="BYOL-L-e100"
    S3_A="${CANON_S3}/jepa_in21k_vitl_e100.pt"
    S3_B="${CANON_S3}/byol_vitl_e100.pt"
    LOCAL_A="checkpoints/jepa_in21k_vitl_e100.pt"
    LOCAL_B="checkpoints/byol_vitl_imagenet_v2_e100.pt"
  else
    MODEL_A="MAE-L-e99"
    MODEL_B="SALT-S2v1-e79"
    S3_A="${CANON_S3}/mae_vitl_e99.pth"
    S3_B="${CANON_S3}/salt_s2_vitl_e79.pt"
    LOCAL_A="checkpoints/videomae_l_mimic_ep99.pth"
    LOCAL_B="checkpoints/salt_s2v1_e79.pt"
  fi

  mkdir -p "${CODE_DIR}/checkpoints"
  aws s3 cp "${S3_A}" "${CODE_DIR}/${LOCAL_A}" --quiet
  aws s3 cp "${S3_B}" "${CODE_DIR}/${LOCAL_B}" --quiet

  local CSV="${SCRATCH}/data/csv/echonet_dynamic_test_s3_raw.csv"

  # Launch 4 shards per model × 2 models = 8 procs, one per GPU.
  local pids=()
  for SH in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=${SH} \
    python3 -m scripts.neurips.cka_extract \
      --model "${MODEL_A}" --device cuda:0 \
      --csv "${CSV}" \
      --shard_id ${SH} --num_shards ${NUM_SHARDS} \
      --output_dir "${LOCAL_FEAT}" \
      > "/opt/dlami/nvme/cka_${MODEL_A}_s${SH}.log" 2>&1 &
    pids+=($!)
  done
  for SH in 0 1 2 3; do
    local GPU=$((4 + SH))
    CUDA_VISIBLE_DEVICES=${GPU} \
    python3 -m scripts.neurips.cka_extract \
      --model "${MODEL_B}" --device cuda:0 \
      --csv "${CSV}" \
      --shard_id ${SH} --num_shards ${NUM_SHARDS} \
      --output_dir "${LOCAL_FEAT}" \
      > "/opt/dlami/nvme/cka_${MODEL_B}_s${SH}.log" 2>&1 &
    pids+=($!)
  done

  local rc=0
  for pid in "${pids[@]}"; do
    wait "${pid}" || rc=$?
  done

  # Upload shard NPZs to S3 rendezvous.
  for MODEL in "${MODEL_A}" "${MODEL_B}"; do
    for SH in 0 1 2 3; do
      local F="${LOCAL_FEAT}/${MODEL}.shard${SH}of${NUM_SHARDS}.npz"
      if [ -f "${F}" ]; then
        aws s3 cp "${F}" "${FEATURES_S3}/$(basename "${F}")" --only-show-errors
      else
        echo "[warn] node ${NID} missing shard: ${F}"
      fi
    done
  done

  # Upload per-shard logs.
  for f in /opt/dlami/nvme/cka_*.log; do
    [ -f "${f}" ] || continue
    aws s3 cp "${f}" "${RUN_S3}/logs/node${NID}/$(basename "${f}")" --only-show-errors || true
  done

  return ${rc}
}
export -f per_node_runner

# =========================
# Aggregator (runs on node 0 via srun after extraction srun returns).
# =========================
aggregator_runner() {
  local SCRATCH="/opt/dlami/nvme"
  local LOCAL_FEAT="${SCRATCH}/cka/features"
  local OUT_DIR="${SCRATCH}/cka/out"
  local FIG_DIR="${SCRATCH}/cka/figures"
  mkdir -p "${LOCAL_FEAT}" "${OUT_DIR}" "${FIG_DIR}"

  setup_env_and_code

  # Rebuild array locally (can't export bash arrays).
  local MODELS_LIST=(${MODELS_STR})

  # Pull ALL 16 shard NPZs from S3 rendezvous.
  aws s3 cp "${FEATURES_S3}/" "${LOCAL_FEAT}/" --recursive --only-show-errors

  local CODE_DIR="${SCRATCH}/code_cka_${SLURM_JOB_ID}"
  cd "${CODE_DIR}"

  python3 -m scripts.neurips.cka_merge_shards \
    --models "${MODELS_LIST[@]}" \
    --num_shards ${NUM_SHARDS} \
    --feat_dir "${LOCAL_FEAT}"

  python3 -m scripts.neurips.cka_compute_plot \
    --models "${MODELS_LIST[@]}" \
    --feat_dir "${LOCAL_FEAT}" \
    --out_npz "${OUT_DIR}/cka_matrices.npz" \
    --fig_dir "${FIG_DIR}" \
    --fig_name cka_layerwise_pertoken \
    --vmin 0.3 --vmax 1.0 \
    --title_suffix " — per-token K=32, LN" \
    --device cuda:0

  # Upload merged NPZs, CKA matrix, figures.
  for M in "${MODELS_LIST[@]}"; do
    aws s3 cp "${LOCAL_FEAT}/${M}.npz" "${RUN_S3}/features_merged/${M}.npz" --only-show-errors || true
  done
  aws s3 cp "${OUT_DIR}/cka_matrices.npz" "${RUN_S3}/cka_matrices.npz" --only-show-errors
  aws s3 cp "${FIG_DIR}/cka_layerwise_pertoken.pdf" "${RUN_S3}/cka_layerwise_pertoken.pdf" --only-show-errors
  aws s3 cp "${FIG_DIR}/cka_layerwise_pertoken.png" "${RUN_S3}/cka_layerwise_pertoken.png" --only-show-errors

  echo "[done] aggregator"
}
export -f aggregator_runner

# =========================
# Log uploader.
# =========================
upload_job_logs() {
  [ -f "/opt/dlami/nvme/cka_layerwise_${SLURM_JOB_ID}.out" ] && \
    aws s3 cp "/opt/dlami/nvme/cka_layerwise_${SLURM_JOB_ID}.out" \
    "${RUN_S3}/logs/job.out" --only-show-errors || true
  [ -f "/opt/dlami/nvme/cka_layerwise_${SLURM_JOB_ID}.err" ] && \
    aws s3 cp "/opt/dlami/nvme/cka_layerwise_${SLURM_JOB_ID}.err" \
    "${RUN_S3}/logs/job.err" --only-show-errors || true
}
trap 'upload_job_logs' EXIT

# Propagate env vars into srun tasks (scalars only — bash can't export arrays).
export MODELS_STR NUM_SHARDS S3_BASE ENV_S3 RUN_S3 FEATURES_S3 SLURM_JOB_ID

# Stage 1: extract on all nodes (1 task/node, 8 parallel shards each).
# Use `set +u` inside srun tasks so exported function bodies that reference
# env vars don't trip on unbound-at-parse-time quirks.
srun --ntasks=2 --ntasks-per-node=1 bash -c 'set +u; per_node_runner'

# Stage 2: aggregate on node 0 only.
srun --ntasks=1 --nodes=1 bash -c 'set +u; aggregator_runner'

echo "[done] CKA job ${SLURM_JOB_ID}"
