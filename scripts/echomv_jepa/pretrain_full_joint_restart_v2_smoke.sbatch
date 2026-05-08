#!/bin/bash
# Full-Joint Global Study-Token EchoMV-JEPA — smoke (225 steps).
# Trainable V-JEPA clip encoder + EMA teacher + frozen e100 anchor +
# student/teacher study transformer + study projector + single-view-to-
# study branch. Reads raw MP4s (pixel dataset), NOT the c_clip cache.

#SBATCH -p dev
#SBATCH --nodelist=ip-10-0-50-146
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH -t 0-01:30:00
#SBATCH -J fj_v2_sm
#SBATCH -o /opt/dlami/nvme/logs/fj_v2_sm-%j.out
#SBATCH -e /opt/dlami/nvme/logs/fj_v2_sm-%j.err
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
SRC_TARBALL_S3="${S3_BASE}/setup/vjepa2-src.tar.gz"
E100_CKPT_S3="${S3_BASE}/checkpoints/jepa_in21k/jepa_in21k_vitl_e100.pt"
K_SAMPLE_S3="${S3_BASE}/echoset_jepa/study_clip_sample_K8_seed0_train.parquet"

E100_LOCAL="/opt/dlami/nvme/checkpoints/jepa_in21k_vitl_e100.pt"
K_SAMPLE_LOCAL="/opt/dlami/nvme/echomv_cache/k8_train.parquet"
WORKDIR="/opt/dlami/nvme/fj_v2_sm_${SLURM_JOB_ID}"
mkdir -p "${WORKDIR}" "$(dirname ${E100_LOCAL})" "$(dirname ${K_SAMPLE_LOCAL})"

nvidia-smi -L || true

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

# --- Pull required inputs ------------------------------------------------
[ ! -f "${E100_LOCAL}" ] && aws s3 cp "${E100_CKPT_S3}" "${E100_LOCAL}" --only-show-errors
[ ! -f "${K_SAMPLE_LOCAL}" ] && aws s3 cp "${K_SAMPLE_S3}" "${K_SAMPLE_LOCAL}" --quiet
python -c "import pandas as pd; df=pd.read_parquet('${K_SAMPLE_LOCAL}'); print('K-sample rows:', len(df), 'studies:', df.study_id.nunique())"
ls -lh "${E100_LOCAL}"

# --- Config: copy from repo, point ``folder`` at per-job run dir ---------
RUN_FOLDER="${WORKDIR}/run"
mkdir -p "${RUN_FOLDER}"
CONFIG="${WORKDIR}/full_joint_global_study_restart_v2_smoke.yaml"
cp "${REPO_DIR}/configs/train/echomv_jepa/full_joint_global_study_restart_v2_smoke.yaml" "${CONFIG}"
python - <<PY
import yaml
path = "${CONFIG}"
with open(path) as f:
    cfg = yaml.safe_load(f)
cfg["folder"] = "${RUN_FOLDER}"
cfg["experiment"]["data"]["k_sample_manifest"] = "${K_SAMPLE_LOCAL}"
cfg["experiment"]["clip_encoder"]["init_ckpt"] = "${E100_LOCAL}"
with open(path, "w") as f:
    yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
print("wrote", path)
PY

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -1)
export MASTER_PORT=37163

echo "[INFO] === Full-Joint smoke (225 steps, K=4, batch=4, bf16) ==="
echo "[INFO] MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
srun --ntasks-per-node=8 python -m app.main_srun --fname "$CONFIG" 2>&1 | tee "${WORKDIR}/fj_v2_sm.log" \
  || echo "[WARN] training exited non-zero"

aws s3 sync "${RUN_FOLDER}" "${S3_BASE}/echomv_jepa/full_joint_restart_v2_smoke_runs/${SLURM_JOB_ID}/" --only-show-errors
echo "[INFO] === Full-Joint smoke complete ==="
echo "   run folder: ${RUN_FOLDER}"
echo "   s3:         ${S3_BASE}/echomv_jepa/full_joint_restart_v2_smoke_runs/${SLURM_JOB_ID}/"
