#!/bin/bash
# Interpretability figure: JEPA IN21K e100 vs MAE IN21K e99 tubelet-to-
# tubelet cosine similarity on EchoNet-Dynamic test clips.
#
# 8-GPU sharded on a single node. Each rank processes
# clips[rank::world_size] for both encoders, writes a partial-sum .npy +
# per-encoder clip count; a final aggregate pass averages the partials
# into the two 8x8 matrices (jepa_tubelet_cos.npy, mae_tubelet_cos.npy).
#
#SBATCH -p dev
#SBATCH --nodelist=ip-10-0-50-146
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 0-00:30:00
#SBATCH -J jepa_vs_mae_tubelet_cos
#SBATCH -o /opt/dlami/nvme/logs/jepa_vs_mae_tubelet_cos-%j.out
#SBATCH -e /opt/dlami/nvme/logs/jepa_vs_mae_tubelet_cos-%j.err
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

RUN_NAME="jepa_vs_mae_tubelet_cos"
ART_BUCKET="sagemaker-hyperpod-lifecycle-495467399120-usw2"
S3_BASE="s3://${ART_BUCKET}/vjepa2-artifacts"
SRC_TARBALL_S3="${S3_BASE}/setup/vjepa2-src.tar.gz"
RUN_S3="${S3_BASE}/runs/${RUN_NAME}_${SLURM_JOB_ID}"
WORKDIR="/opt/dlami/nvme/${RUN_NAME}_${SLURM_JOB_ID}"
mkdir -p "$WORKDIR"

JEPA_CKPT_S3="s3://echodata25/neurips/encoders/jepa_in21k_vitl_e100.pt"
MAE_CKPT_S3="s3://echodata25/neurips/encoders/mae_vitl_e99.pth"
BYOL_CKPT_S3="s3://echodata25/neurips/encoders/byol_vitl_e100.pt"
SALT_CKPT_S3="s3://echodata25/neurips/encoders/salt_s2_vitl_e79.pt"

N_CLIPS=2000
BATCH_SIZE=8
WORLD_SIZE=8

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

# --- Data & checkpoint downloads (once, shared by all ranks) -----------------
DATA_DIR="${WORKDIR}/data/csv"; mkdir -p "$DATA_DIR"
aws s3 cp "${S3_BASE}/data/csv/echonet_dynamic_test_s3_raw.csv" "$DATA_DIR/" --quiet

CKPT_DIR="${WORKDIR}/checkpoints"; mkdir -p "$CKPT_DIR"
aws s3 cp "${JEPA_CKPT_S3}" "${CKPT_DIR}/jepa_in21k_vitl_e100.pt" --quiet &
aws s3 cp "${MAE_CKPT_S3}"  "${CKPT_DIR}/mae_vitl_e99.pth"        --quiet &
aws s3 cp "${BYOL_CKPT_S3}" "${CKPT_DIR}/byol_vitl_e100.pt"       --quiet &
aws s3 cp "${SALT_CKPT_S3}" "${CKPT_DIR}/salt_s2_vitl_e79.pt"     --quiet &
wait
ls -lh "${CKPT_DIR}/"

# --- Launch 8 rank processes in parallel ------------------------------------
OUT_DIR="${WORKDIR}/out"; mkdir -p "$OUT_DIR"
echo "[INFO] === Sharded 4-encoder tubelet cosine sim (N=${N_CLIPS}, WS=${WORLD_SIZE}) ==="

pids=()
for RANK in $(seq 0 $((WORLD_SIZE - 1))); do
  CUDA_VISIBLE_DEVICES=${RANK} \
  python3 scripts/neurips/phase/compute_tubelet_cosine_sim.py \
    --jepa-ckpt "${CKPT_DIR}/jepa_in21k_vitl_e100.pt" \
    --mae-ckpt  "${CKPT_DIR}/mae_vitl_e99.pth" \
    --byol-ckpt "${CKPT_DIR}/byol_vitl_e100.pt" \
    --salt-ckpt "${CKPT_DIR}/salt_s2_vitl_e79.pt" \
    --test-csv  "${DATA_DIR}/echonet_dynamic_test_s3_raw.csv" \
    --out-dir   "${OUT_DIR}" \
    --n-clips "${N_CLIPS}" --batch-size "${BATCH_SIZE}" \
    --rank "${RANK}" --world-size "${WORLD_SIZE}" --seed 0 \
    > "${WORKDIR}/rank${RANK}.log" 2>&1 &
  pids+=($!)
done

# Wait on all ranks; fail the job if any rank errored.
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

# --- Aggregate partials into final 8x8 matrices ----------------------------
echo "[INFO] === aggregating ==="
python3 scripts/neurips/phase/compute_tubelet_cosine_sim.py \
  --out-dir "${OUT_DIR}" --world-size "${WORLD_SIZE}" --aggregate \
  2>&1 | tee "${WORKDIR}/aggregate.log"

aws s3 sync "${OUT_DIR}" "${RUN_S3}/out/" --only-show-errors

echo "[INFO] === ${RUN_NAME} complete ==="
echo "[INFO] Matrices at: ${RUN_S3}/out/{jepa,mae}_tubelet_cos.npy"
