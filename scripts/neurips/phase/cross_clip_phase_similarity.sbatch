#!/bin/bash
# Cross-clip phase-aligned cosine similarity on MIMIC-IV-Echo.
#
# For each (A, B) same-study clip pair, pool per-tubelet features and
# compute cosine as a function of phase offset Δφ in [-0.5, 0.5). 8-GPU
# sharded, single node. All 6 encoders (trajectory: JEPA e100/e125/V4;
# cross-encoder: MAE/BYOL/SALT) run sequentially per rank on the same
# pair shard.
#
#SBATCH -p dev
#SBATCH --nodelist=ip-10-0-50-56
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 0-01:30:00
#SBATCH -J cross_clip_phase_sim
#SBATCH -o /opt/dlami/nvme/logs/cross_clip_phase_sim-%j.out
#SBATCH -e /opt/dlami/nvme/logs/cross_clip_phase_sim-%j.err
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

RUN_NAME="cross_clip_phase_sim"
ART_BUCKET="sagemaker-hyperpod-lifecycle-495467399120-usw2"
S3_BASE="s3://${ART_BUCKET}/vjepa2-artifacts"
SRC_TARBALL_S3="${S3_BASE}/setup/vjepa2-src.tar.gz"
RUN_S3="${S3_BASE}/runs/${RUN_NAME}_${SLURM_JOB_ID}"
WORKDIR="/opt/dlami/nvme/${RUN_NAME}_${SLURM_JOB_ID}"
mkdir -p "$WORKDIR"

# Encoders
JEPA_E100_S3="s3://echodata25/neurips/encoders/jepa_in21k_vitl_e100.pt"
JEPA_E125_S3="${S3_BASE}/runs/jepa_in21k_e200_280/training_folder/e125.pt"
V4_E25_S3="${S3_BASE}/runs/final_phase_rel25_paper_593/checkpoints/latest.pt"
MAE_S3="s3://echodata25/neurips/encoders/mae_vitl_e99.pth"
BYOL_S3="s3://echodata25/neurips/encoders/byol_vitl_e100.pt"
SALT_S3="s3://echodata25/neurips/encoders/salt_s2_vitl_e79.pt"

PHASE_PARQUET_S3="${S3_BASE}/data/phase_annotations/phase_annotations.parquet"

N_STUDIES=400
PAIRS_PER_STUDY=2
BATCH_SIZE=4
WORLD_SIZE=8
N_BINS=16

upload_artifacts() {
  [ -d "${WORKDIR}/out" ] && aws s3 sync "${WORKDIR}/out" "${RUN_S3}/out/" --only-show-errors 2>/dev/null || true
  for f in /opt/dlami/nvme/logs/${RUN_NAME}-${SLURM_JOB_ID}.{out,err}; do
    [ -f "$f" ] && aws s3 cp "$f" "${RUN_S3}/logs/$(basename "$f")" --only-show-errors 2>/dev/null || true
  done
}
trap 'upload_artifacts' EXIT

nvidia-smi -L || true

# --- Code deploy -------------------------------------------------------------
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
python -c "import timm" 2>/dev/null || pip install --no-cache-dir timm==0.4.12 2>&1 | tail -3

# --- Data + checkpoint downloads (shared) -----------------------------------
DATA_DIR="${WORKDIR}/data"; mkdir -p "$DATA_DIR"
aws s3 cp "${PHASE_PARQUET_S3}" "${DATA_DIR}/phase_annotations.parquet" --quiet

CKPT_DIR="${WORKDIR}/checkpoints"; mkdir -p "$CKPT_DIR"
aws s3 cp "${JEPA_E100_S3}" "${CKPT_DIR}/jepa_e100.pt"  --quiet &
aws s3 cp "${JEPA_E125_S3}" "${CKPT_DIR}/jepa_e125.pt"  --quiet &
aws s3 cp "${V4_E25_S3}"    "${CKPT_DIR}/v4_e25.pt"     --quiet &
aws s3 cp "${MAE_S3}"       "${CKPT_DIR}/mae.pth"       --quiet &
aws s3 cp "${BYOL_S3}"      "${CKPT_DIR}/byol.pt"       --quiet &
aws s3 cp "${SALT_S3}"      "${CKPT_DIR}/salt.pt"       --quiet &
wait
ls -lh "${CKPT_DIR}/"

# --- Launch 8 rank processes -------------------------------------------------
OUT_DIR="${WORKDIR}/out"; mkdir -p "$OUT_DIR"
echo "[INFO] === Cross-clip phase similarity (n_studies=${N_STUDIES}, pairs/study=${PAIRS_PER_STUDY}, WS=${WORLD_SIZE}) ==="

pids=()
for RANK in $(seq 0 $((WORLD_SIZE - 1))); do
  CUDA_VISIBLE_DEVICES=${RANK} \
  python3 scripts/neurips/phase/compute_cross_clip_phase_similarity.py \
    --phase-parquet "${DATA_DIR}/phase_annotations.parquet" \
    --jepa-e100-ckpt "${CKPT_DIR}/jepa_e100.pt" \
    --jepa-e125-ckpt "${CKPT_DIR}/jepa_e125.pt" \
    --v4-e25-ckpt    "${CKPT_DIR}/v4_e25.pt" \
    --mae-ckpt       "${CKPT_DIR}/mae.pth" \
    --byol-ckpt      "${CKPT_DIR}/byol.pt" \
    --salt-ckpt      "${CKPT_DIR}/salt.pt" \
    --out-dir        "${OUT_DIR}" \
    --n-studies "${N_STUDIES}" --pairs-per-study "${PAIRS_PER_STUDY}" \
    --batch-size "${BATCH_SIZE}" --n-bins "${N_BINS}" \
    --rank "${RANK}" --world-size "${WORLD_SIZE}" --seed 0 \
    > "${WORKDIR}/rank${RANK}.log" 2>&1 &
  pids+=($!)
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then status=1; fi
done

echo "[INFO] === shard phase complete (status=${status}) ==="
for RANK in $(seq 0 $((WORLD_SIZE - 1))); do
  echo "--- rank${RANK}.log tail ---"
  tail -10 "${WORKDIR}/rank${RANK}.log" || true
done

if [ "$status" -ne 0 ]; then
  echo "[ERROR] at least one rank failed"; exit "$status"
fi

# --- Aggregate partials ------------------------------------------------------
echo "[INFO] === aggregating ==="
python3 scripts/neurips/phase/compute_cross_clip_phase_similarity.py \
  --out-dir "${OUT_DIR}" --world-size "${WORLD_SIZE}" --n-bins "${N_BINS}" --aggregate \
  2>&1 | tee "${WORKDIR}/aggregate.log"

aws s3 sync "${OUT_DIR}" "${RUN_S3}/out/" --only-show-errors

echo "[INFO] === ${RUN_NAME} complete ==="
echo "[INFO] Aggregated curves at: ${RUN_S3}/out/{jepa_e100,jepa_e125,v4_e25,mae,byol,salt}_phase_similarity.npy"
