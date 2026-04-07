#!/bin/bash
#SBATCH -p ml-p5-48xlarge
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 2-00:00:00
#SBATCH -J ejg_mr
#SBATCH -o /tmp/ejg_mr-%j.out
#SBATCH -e /tmp/ejg_mr-%j.err
#SBATCH --signal=B:USR1@300
#SBATCH --nodelist=ip-10-0-50-83

set -euxo pipefail

# =========================
# GLOBAL CONFIG
# =========================
export AWS_DEFAULT_REGION="us-west-2"
export AWS_REGION="us-west-2"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export MASTER_PORT=$((12000 + $RANDOM % 20000))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

export AWS_SDK_LOAD_CONFIG=1
export HOME="/home/ubuntu"
export AWS_SHARED_CREDENTIALS_FILE="${HOME}/.aws/credentials"
export AWS_CONFIG_FILE="${HOME}/.aws/config"
unset AWS_EC2_METADATA_DISABLED 2>/dev/null || true
export AWS_METADATA_SERVICE_TIMEOUT=5
export AWS_METADATA_SERVICE_NUM_ATTEMPTS=3

export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export FI_EFA_USE_DEVICE_RDMA=1
export FI_PROVIDER=efa
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# cuBLAS fix for H100 bf16
export LD_LIBRARY_PATH=/opt/amazon/efa/lib:/opt/aws-ofi-nccl/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

# S3 paths
ART_BUCKET="sagemaker-hyperpod-lifecycle-495467399120-usw2"
ART_PREFIX="vjepa2-artifacts"
S3_BASE="s3://${ART_BUCKET}/${ART_PREFIX}"
RUN_S3="${S3_BASE}/runs/echojepa_g_mitral_regurg_${SLURM_JOB_ID}"
PREV_RUN_S3="${S3_BASE}/runs/echojepa_g_mitral_regurg_415"

# =========================
# HOST SETUP
# =========================
echo "[INFO] host=$(hostname) job=${SLURM_JOB_ID}"
nvidia-smi -L || true

SCRATCH="/opt/dlami/nvme"
WORKDIR="${SCRATCH}/ejg_mr_${SLURM_JOB_ID}"
mkdir -p "$WORKDIR"
RUNLOG="${WORKDIR}/probe_train_${SLURM_JOB_ID}.log"

# =========================
# TRAPS
# =========================
CFG_FOLDER=""
upload_logs() {
  echo "[INFO] Uploading logs..."
  if [ -f "$RUNLOG" ]; then
    aws s3 cp "$RUNLOG" "${RUN_S3}/logs/job.out" --region us-west-2 2>/dev/null || true
  fi
  if [ -n "$CFG_FOLDER" ] && [ -d "$CFG_FOLDER" ]; then
    aws s3 sync "$CFG_FOLDER" "${RUN_S3}/training_folder/" \
      --include "*.pt" --include "*.csv" --include "*.log" --include "*.json" \
      --exclude "*.pth" --region us-west-2 --only-show-errors 2>/dev/null || true
  fi
}
trap 'rc=$?; upload_logs; exit $rc' EXIT

# =========================
# ENVIRONMENT
# =========================
ENV_S3="${S3_BASE}/env/vjepa2-312-pack-env.tar"
BOTO3_WHL_S3="${S3_BASE}/wheels/boto3-wheelhouse.tar"

aws s3 cp "$ENV_S3" "${WORKDIR}/env.tar" --region us-west-2
mkdir -p "${WORKDIR}/env"
tar -xf "${WORKDIR}/env.tar" -C "${WORKDIR}/env"
set +u; source "${WORKDIR}/env/bin/activate"; set -u
"${WORKDIR}/env/bin/conda-unpack" 2>/dev/null || true

aws s3 cp "$BOTO3_WHL_S3" "${WORKDIR}/boto3-wheelhouse.tar" --region us-west-2
mkdir -p "${WORKDIR}/wheelhouse"
tar -xf "${WORKDIR}/boto3-wheelhouse.tar" -C "${WORKDIR}/wheelhouse"
python -m pip install --no-index --find-links "${WORKDIR}/wheelhouse" boto3

# =========================
# CODE
# =========================
REPO_DIR="/opt/vjepa2"
cd "$REPO_DIR"
pip install -e . --no-deps --no-build-isolation 2>/dev/null || true
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# =========================
# DOWNLOAD INPUTS
# =========================
TRAIN_CFG="configs/eval/vitg-384/nature_medicine/echojepa_g_mitral_regurg_hp.yaml"
PREDAVG_CFG="configs/eval/vitg-384/nature_medicine/echojepa_g_mitral_regurg_predavg.yaml"

echo "[INFO] Downloading CSVs (train + val + test)..."
mkdir -p data/csv
aws s3 cp s3://echodata25/nature_medicine/mimic/probe_csvs/mitral_regurg/train.csv data/csv/mimic_mitral_regurg_train.csv --region us-west-2
aws s3 cp s3://echodata25/nature_medicine/mimic/probe_csvs/mitral_regurg/val.csv data/csv/mimic_mitral_regurg_val.csv --region us-west-2
aws s3 cp s3://echodata25/nature_medicine/mimic/probe_csvs/mitral_regurg/test.csv data/csv/mimic_mitral_regurg_test.csv --region us-west-2

echo "[INFO] Downloading EchoJEPA-G checkpoint (16 GB) to NVMe..."
CKPT_DIR="/opt/dlami/nvme/checkpoints/anneal/keep"
mkdir -p "$CKPT_DIR"
if [ ! -f "${CKPT_DIR}/pt-280-an81.pt" ]; then
  aws s3 cp "${S3_BASE}/checkpoints/anneal/keep/pt-280-an81.pt" "${CKPT_DIR}/pt-280-an81.pt" --region us-west-2
fi
mkdir -p checkpoints/anneal/keep
ln -sf "${CKPT_DIR}/pt-280-an81.pt" checkpoints/anneal/keep/pt-280-an81.pt

echo "[INFO] Downloading resume checkpoint from job 415..."
RESUME_DIR="/opt/dlami/nvme/evals/vitg-384/nature_medicine/video_classification_frozen/echojepa-g-mitral-regurg"
mkdir -p "$RESUME_DIR"
aws s3 cp "${PREV_RUN_S3}/training_folder/video_classification_frozen/echojepa-g-mitral-regurg/latest.pt" "${RESUME_DIR}/latest.pt" --region us-west-2
aws s3 cp "${PREV_RUN_S3}/training_folder/video_classification_frozen/echojepa-g-mitral-regurg/best.pt" "${RESUME_DIR}/best.pt" --region us-west-2
aws s3 cp "${PREV_RUN_S3}/training_folder/video_classification_frozen/echojepa-g-mitral-regurg/log_r0.csv" "${RESUME_DIR}/log_r0.csv" --region us-west-2 || true

# =========================
# PARSE CONFIG
# =========================
export CFG_REL="$TRAIN_CFG"
eval $(python - <<'PY'
import re, os
from pathlib import Path
cfg_rel = os.environ["CFG_REL"]
folder, tag = "", ""
if os.path.exists(cfg_rel):
    t = Path(cfg_rel).read_text()
    m = re.search(r'^\s*folder:\s*(.+)\s*$', t, re.M)
    if m: folder = m.group(1).strip().strip('"').strip("'")
    m = re.search(r'^\s*tag:\s*(.+)\s*$', t, re.M)
    if m: tag = m.group(1).strip().strip('"').strip("'")
print(f"export CFG_FOLDER='{folder}'")
print(f"export CFG_TAG='{tag}'")
PY
)
mkdir -p "$CFG_FOLDER"
echo "[INFO] folder=$CFG_FOLDER tag=$CFG_TAG"

# =========================
# CREDENTIALS
# =========================
unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN 2>/dev/null || true

# =========================
# BACKGROUND SYNC
# =========================
(
  while true; do
    sleep 600
    aws s3 sync "$CFG_FOLDER" "${RUN_S3}/training_folder/" \
      --include "*.pt" --include "*.csv" --include "*.log" --include "*.json" \
      --exclude "*.pth" --region us-west-2 --only-show-errors 2>/dev/null || true
    [ -f "$RUNLOG" ] && aws s3 cp "$RUNLOG" "${RUN_S3}/logs/job_latest.log" --region us-west-2 2>/dev/null || true
  done
) &
SYNC_PID=$!

# =========================
# PHASE 1: RESUME PROBE TRAINING (e16 → e35)
# =========================
GPU_COUNT=$(nvidia-smi -L | wc -l)
DEVICES=()
for ((i=0; i<GPU_COUNT; i++)); do DEVICES+=("cuda:${i}"); done
echo "[INFO] ${GPU_COUNT} GPUs: ${DEVICES[*]}"
echo "[INFO] === PHASE 1: Resume MR probe training ==="

set +e
python -m evals.main \
  --fname "$TRAIN_CFG" \
  --devices "${DEVICES[@]}" \
  2>&1 | tee "$RUNLOG"
TRAIN_RC=${PIPESTATUS[0]}
set -e

echo "[INFO] Training exit code: ${TRAIN_RC}"

# Upload training results immediately
aws s3 sync "$CFG_FOLDER" "${RUN_S3}/training_folder/" \
  --include "*.pt" --include "*.csv" --include "*.log" --include "*.json" \
  --exclude "*.pth" --region us-west-2 --only-show-errors || true

if [ "$TRAIN_RC" -ne 0 ]; then
  echo "[ERROR] Training failed with exit code ${TRAIN_RC}"
  kill $SYNC_PID 2>/dev/null || true; wait $SYNC_PID 2>/dev/null || true
  exit "$TRAIN_RC"
fi

# =========================
# PHASE 2: SINGLE-CLIP TEST EVAL (val_only on test.csv, no study averaging)
# =========================
echo "[INFO] === PHASE 2: Single-clip test evaluation ==="
SINGLECLIP_LOG="${WORKDIR}/singleclip_test_${SLURM_JOB_ID}.log"

# Create single-clip test config: use test.csv as val, val_only, no study_sampling
cat > /tmp/mr_singleclip_test.yaml <<YAML
app: vjepa
cpus_per_task: 32
folder: /opt/dlami/nvme/evals/vitg-384/nature_medicine
mem_per_gpu: 80G
nodes: 1
tasks_per_node: 8
num_workers: 8

eval_name: video_classification_frozen
val_only: true
resume_checkpoint: true
probe_checkpoint: ${RESUME_DIR}/best.pt
tag: echojepa-g-mitral-regurg-singleclip-test

experiment:
  classifier:
    num_heads: 16
    num_probe_blocks: 1

  data:
    dataset_type: VideoDataset
    dataset_train: data/csv/mimic_mitral_regurg_train.csv
    dataset_val: data/csv/mimic_mitral_regurg_test.csv
    num_classes: 4
    resolution: 224
    frames_per_clip: 16
    frame_step: 2
    num_segments: 2
    num_views_per_segment: 1
    study_sampling: false

  optimization:
    batch_size: 1
    val_batch_size: 128
    num_epochs: 1
    use_bfloat16: true
    use_pos_embed: false
    multihead_kwargs:
    - {lr: 0.001, start_lr: 0.0, warmup: 3.0, final_lr: 0.0, weight_decay: 0.001, final_weight_decay: 0.001}
    - {lr: 0.001, start_lr: 0.0, warmup: 3.0, final_lr: 0.0, weight_decay: 0.01,  final_weight_decay: 0.01}
    - {lr: 0.001, start_lr: 0.0, warmup: 3.0, final_lr: 0.0, weight_decay: 0.1,   final_weight_decay: 0.1}
    - {lr: 0.001, start_lr: 0.0, warmup: 3.0, final_lr: 0.0, weight_decay: 0.4,   final_weight_decay: 0.4}
    - {lr: 0.0005, start_lr: 0.0, warmup: 3.0, final_lr: 0.0, weight_decay: 0.001, final_weight_decay: 0.001}
    - {lr: 0.0005, start_lr: 0.0, warmup: 3.0, final_lr: 0.0, weight_decay: 0.01,  final_weight_decay: 0.01}
    - {lr: 0.0005, start_lr: 0.0, warmup: 3.0, final_lr: 0.0, weight_decay: 0.1,   final_weight_decay: 0.1}
    - {lr: 0.0005, start_lr: 0.0, warmup: 3.0, final_lr: 0.0, weight_decay: 0.4,   final_weight_decay: 0.4}
    - {lr: 0.0001, start_lr: 0.0, warmup: 3.0, final_lr: 0.0, weight_decay: 0.001, final_weight_decay: 0.001}
    - {lr: 0.0001, start_lr: 0.0, warmup: 3.0, final_lr: 0.0, weight_decay: 0.01,  final_weight_decay: 0.01}
    - {lr: 0.0001, start_lr: 0.0, warmup: 3.0, final_lr: 0.0, weight_decay: 0.1,   final_weight_decay: 0.1}
    - {lr: 0.0001, start_lr: 0.0, warmup: 3.0, final_lr: 0.0, weight_decay: 0.4,   final_weight_decay: 0.4}
    - {lr: 0.00005, start_lr: 0.0, warmup: 3.0, final_lr: 0.0, weight_decay: 0.001, final_weight_decay: 0.001}
    - {lr: 0.00005, start_lr: 0.0, warmup: 3.0, final_lr: 0.0, weight_decay: 0.01,  final_weight_decay: 0.01}
    - {lr: 0.00005, start_lr: 0.0, warmup: 3.0, final_lr: 0.0, weight_decay: 0.1,   final_weight_decay: 0.1}
    - {lr: 0.00005, start_lr: 0.0, warmup: 3.0, final_lr: 0.0, weight_decay: 0.4,   final_weight_decay: 0.4}
    - {lr: 0.00001, start_lr: 0.0, warmup: 3.0, final_lr: 0.0, weight_decay: 0.001, final_weight_decay: 0.001}
    - {lr: 0.00001, start_lr: 0.0, warmup: 3.0, final_lr: 0.0, weight_decay: 0.01,  final_weight_decay: 0.01}
    - {lr: 0.00001, start_lr: 0.0, warmup: 3.0, final_lr: 0.0, weight_decay: 0.1,   final_weight_decay: 0.1}
    - {lr: 0.00001, start_lr: 0.0, warmup: 3.0, final_lr: 0.0, weight_decay: 0.4,   final_weight_decay: 0.4}

model_kwargs:
  checkpoint: checkpoints/anneal/keep/pt-280-an81.pt
  module_name: evals.video_classification_frozen.modelcustom.vit_encoder_multiclip
  pretrain_kwargs:
    encoder:
      checkpoint_key: target_encoder
      img_temporal_dim_size: null
      model_name: vit_giant_xformers
      patch_size: 16
      tubelet_size: 2
      uniform_power: true
      use_rope: true
  wrapper_kwargs:
    max_frames: 128
    use_pos_embed: false
YAML

export MASTER_PORT=$((MASTER_PORT + 1))
set +e
python -m evals.main \
  --fname /tmp/mr_singleclip_test.yaml \
  --devices "${DEVICES[@]}" \
  2>&1 | tee "$SINGLECLIP_LOG"
SC_RC=${PIPESTATUS[0]}
set -e
echo "[INFO] Single-clip test exit code: ${SC_RC}"
aws s3 cp "$SINGLECLIP_LOG" "${RUN_S3}/logs/singleclip_test.log" --region us-west-2 || true

# =========================
# PHASE 3: PRED-AVG TEST EVAL (study-level averaging on test.csv)
# =========================
echo "[INFO] === PHASE 3: Pred-avg test evaluation ==="
PREDAVG_LOG="${WORKDIR}/predavg_test_${SLURM_JOB_ID}.log"

# Copy best.pt to where predavg config expects it
mkdir -p checkpoints/probes/echojepa-g-mitral-regurg
cp "${RESUME_DIR}/best.pt" checkpoints/probes/echojepa-g-mitral-regurg/best.pt

export MASTER_PORT=$((MASTER_PORT + 1))
set +e
python -m evals.main \
  --fname "$PREDAVG_CFG" \
  --devices "${DEVICES[@]}" \
  2>&1 | tee "$PREDAVG_LOG"
PA_RC=${PIPESTATUS[0]}
set -e
echo "[INFO] Pred-avg test exit code: ${PA_RC}"
aws s3 cp "$PREDAVG_LOG" "${RUN_S3}/logs/predavg_test.log" --region us-west-2 || true

# =========================
# FINAL UPLOAD
# =========================
kill $SYNC_PID 2>/dev/null || true; wait $SYNC_PID 2>/dev/null || true

echo "[INFO] Uploading all final artifacts..."
aws s3 cp "$RUNLOG" "${RUN_S3}/logs/job_final.log" --region us-west-2 || true
aws s3 sync "/opt/dlami/nvme/evals/vitg-384/nature_medicine/video_classification_frozen/" "${RUN_S3}/training_folder/video_classification_frozen/" \
  --include "*.pt" --include "*.csv" --include "*.log" --include "*.json" \
  --exclude "*.pth" --region us-west-2 --only-show-errors || true

echo "[INFO] DONE. Results: ${RUN_S3}"
echo "[INFO] Training RC=${TRAIN_RC}, SingleClip RC=${SC_RC}, PredAvg RC=${PA_RC}"
