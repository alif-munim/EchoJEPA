#!/bin/bash
#SBATCH -p ml-p5-48xlarge
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 1-00:00:00
#SBATCH -J byl_a4c
#SBATCH -o /tmp/byl_a4c-%j.out
#SBATCH -e /tmp/byl_a4c-%j.err
#SBATCH --chdir=/tmp
#SBATCH --nodelist=ip-10-0-50-184

set -euxo pipefail
export PATH="/opt/vjepa2-312/bin:$PATH"

export AWS_DEFAULT_REGION="us-west-2"
export AWS_REGION="us-west-2"
export AWS_SDK_LOAD_CONFIG=1
export HOME="/home/ubuntu"
export AWS_SHARED_CREDENTIALS_FILE="${HOME}/.aws/credentials"
export AWS_CONFIG_FILE="${HOME}/.aws/config"
unset AWS_EC2_METADATA_DISABLED 2>/dev/null || true
export AWS_METADATA_SERVICE_TIMEOUT=5
export AWS_METADATA_SERVICE_NUM_ATTEMPTS=3

ART_BUCKET="sagemaker-hyperpod-lifecycle-495467399120-usw2"
S3_BASE="s3://${ART_BUCKET}/vjepa2-artifacts"
ENV_S3="${S3_BASE}/env/vjepa2-312-pack-env.tar"

CFG_REL="configs/eval/vitl/icml/echobyol_l_pt50_rvsp_a4c_d4.yaml"
TRAIN_CSV_S3="${S3_BASE}/data/csv/rvsp_train_a4c.csv"
VAL_CSV_S3="${S3_BASE}/data/csv/rvsp_val_a4c.csv"
ENC_CKPT_S3="${S3_BASE}/checkpoints/echobyol_l_mimic_ep50.pt"
ENC_CKPT_LOCAL="/opt/dlami/nvme/checkpoints/echobyol_l_mimic_ep50.pt"
RUN_S3="${S3_BASE}/runs/echobyol_pt50_rvsp_a4c_${SLURM_JOB_ID}"

upload_logs() {
  aws s3 cp "/tmp/byl_a4c-${SLURM_JOB_ID}.out" "${RUN_S3}/logs/job.out" --only-show-errors 2>/dev/null || true
  aws s3 cp "/tmp/byl_a4c-${SLURM_JOB_ID}.err" "${RUN_S3}/logs/job.err" --only-show-errors 2>/dev/null || true
}
trap 'rc=$?; upload_logs; exit $rc' EXIT

echo "[INFO] host=$(hostname) job=${SLURM_JOB_ID}"
nvidia-smi -L || true

SCRATCH="/opt/dlami/nvme"
[ ! -d "$SCRATCH" ] && SCRATCH="/tmp"
WORKDIR="${SCRATCH}/byl_a4c_${SLURM_JOB_ID}"
mkdir -p "$WORKDIR" && cd "$WORKDIR"

echo "[INFO] Setting up environment..."
if [ ! -d "env/bin" ]; then
    aws s3 cp "$ENV_S3" env.tar
    mkdir -p env && tar -xf env.tar -C env
    set +u && source env/bin/activate && set -u
    env/bin/conda-unpack || true
    pip install "numpy<2.0" boto3 --no-cache-dir
    rm -f env.tar
else
    set +u && source env/bin/activate && set -u
fi

REPO_DIR="/opt/vjepa2"
cd "$REPO_DIR"
if [ ! -f "$CFG_REL" ]; then echo "[FATAL] Config $CFG_REL not found"; exit 1; fi
pip install -e . --no-deps --no-build-isolation 2>/dev/null || true
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

echo "[INFO] Downloading data CSVs..."
mkdir -p /opt/dlami/nvme/data/csv
aws s3 cp "$TRAIN_CSV_S3" /opt/dlami/nvme/data/csv/rvsp_train_a4c.csv
aws s3 cp "$VAL_CSV_S3" /opt/dlami/nvme/data/csv/rvsp_val_a4c.csv

echo "[INFO] Downloading backbone checkpoint..."
mkdir -p "$(dirname "$ENC_CKPT_LOCAL")"
aws s3 cp "$ENC_CKPT_S3" "$ENC_CKPT_LOCAL"

CFG_FOLDER="/opt/dlami/nvme/evals/icml/echobyol_pt50_rvsp_a4c"
mkdir -p "$CFG_FOLDER"

GPU_COUNT=$(nvidia-smi -L | wc -l)
DEVICES=()
for ((i=0; i<GPU_COUNT; i++)); do DEVICES+=("cuda:${i}"); done

echo "[INFO] Launching single-view RVSP probe on ${GPU_COUNT} GPUs..."
(while true; do sleep 600; aws s3 sync "$CFG_FOLDER" "${RUN_S3}/training_folder/" --exclude "*.pth" --include "*.pt" --include "*.csv" --include "*.log" --include "*.json" --only-show-errors || true; done) &
SYNC_PID=$!

set +e
python -m evals.main --fname "$CFG_REL" --devices "${DEVICES[@]}" 2>&1 | tee "probe_train_${SLURM_JOB_ID}.log"
RC=${PIPESTATUS[0]}
set -e

kill $SYNC_PID 2>/dev/null || true; wait $SYNC_PID 2>/dev/null || true
aws s3 sync "$CFG_FOLDER" "${RUN_S3}/training_folder/" --exclude "*.pth" --include "*.pt" --include "*.csv" --include "*.log" --include "*.json" --only-show-errors || true
aws s3 cp "probe_train_${SLURM_JOB_ID}.log" "${RUN_S3}/logs/probe_train.log" --only-show-errors || true
[ "$RC" -ne 0 ] && echo "[ERROR] Failed RC=${RC}" && exit "$RC"
echo "[INFO] DONE. Results at: ${RUN_S3}"
