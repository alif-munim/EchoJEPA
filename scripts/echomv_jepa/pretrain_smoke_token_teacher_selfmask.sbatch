#!/bin/bash
# Arm B — token-online + teacher target self-masking smoke.
# Identical launch mechanics to pretrain_smoke_tokens.sbatch; only the training
# YAML differs.

#SBATCH -p dev
#SBATCH --nodelist=ip-10-0-50-146
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH -t 0-00:45:00
#SBATCH -J arm_b_smoke
#SBATCH -o /opt/dlami/nvme/logs/arm_b_smoke-%j.out
#SBATCH -e /opt/dlami/nvme/logs/arm_b_smoke-%j.err
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
CKPT_S3="${S3_BASE}/runs/jepa_in21k_e200_280/training_folder/e125.pt"

CKPT_LOCAL="/opt/dlami/nvme/echomv_cache/e125.pt"
ENC_YAML_LOCAL="/opt/dlami/nvme/echomv_cache/encoder.yaml"
K_SAMPLE_LOCAL="/opt/dlami/nvme/echomv_cache/k8_train.parquet"
WORKDIR="/opt/dlami/nvme/arm_b_smoke_${SLURM_JOB_ID}"
mkdir -p "${WORKDIR}" "$(dirname ${K_SAMPLE_LOCAL})"

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

aws s3 cp "${K_SAMPLE_S3}" "${K_SAMPLE_LOCAL}" --quiet
if [ ! -f "${CKPT_LOCAL}" ]; then
  aws s3 cp "${CKPT_S3}" "${CKPT_LOCAL}" --quiet
fi
ls -lh "${CKPT_LOCAL}"
python -c "import pandas as pd; df=pd.read_parquet('${K_SAMPLE_LOCAL}'); print('K-sample rows:', len(df), 'studies:', df.study_id.nunique())"

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

RUN_FOLDER="${WORKDIR}/run"
mkdir -p "${RUN_FOLDER}"
CONFIG="${WORKDIR}/stage1_token_teacher_selfmask_smoke.yaml"
cp "${REPO_DIR}/configs/train/echomv_jepa/stage1_token_teacher_selfmask_smoke.yaml" "${CONFIG}"
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

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -1)
export MASTER_PORT=37132

echo "[INFO] === Arm B smoke (225 steps; token + teacher self-mask) ==="
echo "[INFO] MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
srun --ntasks-per-node=8 python -m app.main_srun --fname "$CONFIG" 2>&1 | tee "${WORKDIR}/arm_b.log" \
  || echo "[WARN] training exited non-zero"

aws s3 sync "${RUN_FOLDER}" "${S3_BASE}/echomv_jepa/arm_b_smoke_runs/${SLURM_JOB_ID}/" --only-show-errors
echo "[INFO] === Arm B smoke complete ==="
echo "   run folder: ${RUN_FOLDER}"
echo "   s3:         ${S3_BASE}/echomv_jepa/arm_b_smoke_runs/${SLURM_JOB_ID}/"
