#!/bin/bash
# LVEF A4C-only probe on base e125 (echojepa_l_mimic_v2_run125_e100.pt).
# d=1 attentive probe, 16-head LR/WD sweep, 35 epochs, bfloat16.
# Nature Medicine Strategy E: DistributedStudySampler picks 1 random clip/study
# per epoch; test-time predictions averaged across all clips per study
# (emits study_predictions.csv).

#SBATCH -p dev
#SBATCH --nodelist=ip-10-0-50-146
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 0-12:00:00
#SBATCH -J probe_e125_k8
#SBATCH -o /opt/dlami/nvme/logs/probe_e125_k8-%j.out
#SBATCH -e /opt/dlami/nvme/logs/probe_e125_k8-%j.err
#SBATCH --chdir=/opt/dlami/nvme

set -euxo pipefail

export PATH="/opt/vjepa2-312/bin:$PATH"
export TMPDIR="/opt/dlami/nvme/tmp"
mkdir -p "$TMPDIR" /opt/dlami/nvme/logs

export AWS_DEFAULT_REGION="us-west-2"
export AWS_REGION="us-west-2"
export AWS_SDK_LOAD_CONFIG=1
export HOME="/home/ubuntu"
export AWS_SHARED_CREDENTIALS_FILE="${HOME}/.aws/credentials"
export AWS_CONFIG_FILE="${HOME}/.aws/config"
unset AWS_EC2_METADATA_DISABLED 2>/dev/null || true
export LD_LIBRARY_PATH="/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cublas/lib:/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"

ART_BUCKET="sagemaker-hyperpod-lifecycle-495467399120-usw2"
S3_BASE="s3://${ART_BUCKET}/vjepa2-artifacts"
SRC_TARBALL_S3="${S3_BASE}/setup/vjepa2-src.tar.gz"
BASE_E125_S3="${S3_BASE}/checkpoints/echojepa_l_mimic_v2_run125_e100.pt"
PROBE_CSV_S3="${S3_BASE}/data/nature_medicine_mimic_probe_csvs/lvef_structured"

REPO_DIR="/opt/dlami/nvme/src/vjepa2"
CFG_REL="configs/eval/vitl/mcc_fj/base_e125_lvef_k8.yaml"
ENC_CKPT_S3="${S3_BASE}/checkpoints/echojepa_l_mimic_v2_run125_e100.pt"
ENC_CKPT_LOCAL="/opt/dlami/nvme/checkpoints/echojepa_l_mimic_v2_run125_e100.pt"
RUN_S3="${S3_BASE}/runs/probe_base_e125_lvef_k8_${SLURM_JOB_ID}"
EVAL_OUTDIR="/opt/dlami/nvme/evals/vitl/mcc_fj/base-e125-lvef-a4c"
LOG_OUT="/opt/dlami/nvme/logs/probe_e125_k8-${SLURM_JOB_ID}.out"
LOG_ERR="/opt/dlami/nvme/logs/probe_e125_k8-${SLURM_JOB_ID}.err"
JOB_LOG="/opt/dlami/nvme/logs/probe_e125_k8_${SLURM_JOB_ID}.log"

sync_evals() {
  [ -d "$EVAL_OUTDIR" ] && aws s3 sync "$EVAL_OUTDIR" "${RUN_S3}/eval/" --only-show-errors || true
}
upload_logs() {
  [ -f "$LOG_OUT" ] && aws s3 cp "$LOG_OUT" "${RUN_S3}/logs/job.out" --only-show-errors 2>/dev/null || true
  [ -f "$LOG_ERR" ] && aws s3 cp "$LOG_ERR" "${RUN_S3}/logs/job.err" --only-show-errors 2>/dev/null || true
  [ -f "$JOB_LOG" ] && aws s3 cp "$JOB_LOG" "${RUN_S3}/logs/job.log" --only-show-errors 2>/dev/null || true
}
on_exit() { rc=$?; sync_evals; upload_logs; exit $rc; }
trap on_exit EXIT

echo "[INFO] host=$(hostname) job=${SLURM_JOB_ID}"
nvidia-smi -L || true

mkdir -p "$REPO_DIR"
rm -rf "$REPO_DIR"/*
aws s3 cp "$SRC_TARBALL_S3" /opt/dlami/nvme/tmp/vjepa2-src.tar.gz --only-show-errors
tar xzf /opt/dlami/nvme/tmp/vjepa2-src.tar.gz -C "$REPO_DIR"
cd "$REPO_DIR"
[ -f "$CFG_REL" ] || { echo "[FATAL] $CFG_REL missing in tarball"; exit 1; }
export PYTHONPATH="$REPO_DIR:/opt/dlami/nvme/pyarrow_site:${PYTHONPATH:-}"

# Pull the encoder checkpoint from S3 (cached on NVMe between runs).
mkdir -p "$(dirname "$ENC_CKPT_LOCAL")"
if [ ! -f "$ENC_CKPT_LOCAL" ]; then
  echo "[INFO] Fetching encoder checkpoint from S3..."
  aws s3 cp "$ENC_CKPT_S3" "$ENC_CKPT_LOCAL" --only-show-errors
fi
ls -lh "$ENC_CKPT_LOCAL"

# Patch the config's checkpoint: path in-place to point at the local copy.
# (The config's EFS path is unreachable from compute nodes; the tarball does
# have the yaml, so we edit the extracted copy before launching.)
python - <<PY
import yaml
path = "${REPO_DIR}/${CFG_REL}"
with open(path) as f:
    cfg = yaml.safe_load(f)
cfg["model_kwargs"]["checkpoint"] = "${ENC_CKPT_LOCAL}"
with open(path, "w") as f:
    yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
print(f"patched checkpoint -> ${ENC_CKPT_LOCAL}")
PY

# The probe CSVs live on EFS and are not reachable from the compute node. We
# assume they were pre-copied to NVMe by the launcher; fall back to S3 if not.
PROBE_CSV_DIR="/opt/dlami/nvme/probe_csvs/lvef_structured"
mkdir -p "$PROBE_CSV_DIR"
for split in train val test; do
  if [ ! -f "$PROBE_CSV_DIR/${split}.csv" ]; then
    aws s3 cp "${S3_BASE}/data/nature_medicine_mimic_probe_csvs/lvef_structured/${split}.csv" "$PROBE_CSV_DIR/${split}.csv" --only-show-errors
  fi
done
ls -lh "$PROBE_CSV_DIR/"

# Patch the config's CSV paths to point at the NVMe-local copy.
python - <<PY
import yaml
path = "${REPO_DIR}/${CFG_REL}"
with open(path) as f:
    cfg = yaml.safe_load(f)
cfg["experiment"]["data"]["dataset_train"] = "${PROBE_CSV_DIR}/train.csv"
cfg["experiment"]["data"]["dataset_val"] = "${PROBE_CSV_DIR}/val.csv"
with open(path, "w") as f:
    yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
print("patched CSV paths")
PY

df -h /opt/dlami/nvme

if [ -d "$EVAL_OUTDIR" ]; then
  echo "[INFO] Clearing stale EVAL_OUTDIR=$EVAL_OUTDIR"
  rm -rf "$EVAL_OUTDIR"/*
fi
mkdir -p "$EVAL_OUTDIR"

(
  while true; do
    sleep 600
    sync_evals; upload_logs
  done
) &
SYNC_PID=$!

echo "[INFO] Launching probe: base e125 LVEF K=8..."
python -m evals.main \
  --fname "$CFG_REL" \
  --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
  2>&1 | tee "$JOB_LOG"
RC=$?

kill $SYNC_PID 2>/dev/null || true
wait $SYNC_PID 2>/dev/null || true
sync_evals; upload_logs
exit $RC
