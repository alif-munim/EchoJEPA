#!/bin/bash
# EchoJEPA-L-K pretrain resume: e130 -> e155
# (Kinetics-400 -> MIMIC continuation; runs/11 lineage).
#
# Purpose: extend the L-K pretrain arc by 25 epochs so the L-K snapshot
# at e155 matches V-JEPA†-e155 (~35.5 EFLOPs post-e100) for a clean
# FLOPs-matched comparison across init lineages.
#
# Source checkpoint:
#   s3://.../checkpoints/echojepa_l_k_pretrain_k400_to_mimic/e130.pt
#   (byte-identical to runs/11/training_folder/e130.pt; see
#    claude/architecture/echojepa-l-k-pretrain-checkpoints.md)
# Config:
#   configs/train/vitl16/pretrain-mimic-224px-16f-resume-lk-e130-to-e155.yaml
#   (load_checkpoint=true, force_load_pretrain=false,
#    scheduler_total_epochs=240 preserved, stop_after_epochs=155).
#SBATCH -p dev
#SBATCH --nodelist=ip-10-0-50-56
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH -t 1-12:00:00
#SBATCH -J echojepa_lk_resume_e155
#SBATCH -o /opt/dlami/nvme/logs/echojepa_lk_resume_e155-%j.out
#SBATCH -e /opt/dlami/nvme/logs/echojepa_lk_resume_e155-%j.err
#SBATCH --chdir=/opt/dlami/nvme
#SBATCH --signal=B:USR1@300

set -euxo pipefail

export PATH="/opt/vjepa2-312/bin:$PATH"
export TMPDIR="/opt/dlami/nvme/tmp"
mkdir -p "$TMPDIR" /opt/dlami/nvme/logs

export AWS_DEFAULT_REGION="us-west-2"
export AWS_REGION="us-west-2"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export MASTER_PORT=$((12000 + $RANDOM % 20000))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

export HOME="/home/ubuntu"
# IMDS-only AWS auth (stale shared-creds files break srun workers).
unset AWS_SHARED_CREDENTIALS_FILE 2>/dev/null || true
unset AWS_CONFIG_FILE 2>/dev/null || true
unset AWS_EC2_METADATA_DISABLED 2>/dev/null || true
export AWS_METADATA_SERVICE_TIMEOUT=10
export AWS_METADATA_SERVICE_NUM_ATTEMPTS=5
export LD_LIBRARY_PATH="/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cublas/lib:/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"

export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1

ART_BUCKET="sagemaker-hyperpod-lifecycle-495467399120-usw2"
S3_BASE="s3://${ART_BUCKET}/vjepa2-artifacts"
# Source e130.pt (canonical L-K lineage, byte-identical to runs/11 e130).
RESUME_CKPT_S3="${S3_BASE}/checkpoints/echojepa_l_k_pretrain_k400_to_mimic/e130.pt"
CSV_S3="${S3_BASE}/data/csv/mimic_annotations_s3.csv"
SRC_TARBALL_S3="${S3_BASE}/setup/vjepa2-src.tar.gz"
RUN_S3="${S3_BASE}/runs/echojepa_lk_resume_e155_${SLURM_JOB_ID}"

SCRATCH="/opt/dlami/nvme"
CODE_DIR="${SCRATCH}/src/vjepa2"
CKPT_DIR="${SCRATCH}/echojepa_l_k_resume_e155"
DATA_DIR="${SCRATCH}/data/csv"
CFG_REL="configs/train/vitl16/pretrain-mimic-224px-16f-resume-lk-e130-to-e155.yaml"
LOG_OUT="/opt/dlami/nvme/logs/echojepa_lk_resume_e155-${SLURM_JOB_ID}.out"
LOG_ERR="/opt/dlami/nvme/logs/echojepa_lk_resume_e155-${SLURM_JOB_ID}.err"

sync_ckpts() {
  [ -d "$CKPT_DIR" ] && aws s3 sync "$CKPT_DIR" "${RUN_S3}/training_folder/" \
    --exclude "*" --include "e*.pt" --include "latest.pt" --include "*.yaml" --include "*.csv" \
    --only-show-errors || true
}
upload_logs() {
  [ -f "$LOG_OUT" ] && aws s3 cp "$LOG_OUT" "${RUN_S3}/logs/job.out" --only-show-errors 2>/dev/null || true
  [ -f "$LOG_ERR" ] && aws s3 cp "$LOG_ERR" "${RUN_S3}/logs/job.err" --only-show-errors 2>/dev/null || true
}
on_exit() { rc=$?; sync_ckpts; upload_logs; exit $rc; }
trap on_exit EXIT
trap 'sync_ckpts; upload_logs' USR1

echo "[INFO] host=$(hostname) job=${SLURM_JOB_ID}"
nvidia-smi -L || true
pkill -9 -f "app.main" 2>/dev/null || true
sleep 2

# ---- Code ---------------------------------------------------------------
mkdir -p "$CODE_DIR"
rm -rf "$CODE_DIR"/*
aws s3 cp "$SRC_TARBALL_S3" "${SCRATCH}/tmp/vjepa2-src.tar.gz" --only-show-errors
tar xzf "${SCRATCH}/tmp/vjepa2-src.tar.gz" -C "$CODE_DIR"
cd "$CODE_DIR"
[ -f "$CFG_REL" ] || { echo "[FATAL] $CFG_REL missing"; exit 1; }
pip install -e . --no-deps --no-build-isolation 2>/dev/null || true
export PYTHONPATH="$CODE_DIR:/opt/dlami/nvme/pyarrow_site:${PYTHONPATH:-}"

# ---- Data ---------------------------------------------------------------
mkdir -p "$DATA_DIR"
[ ! -f "${DATA_DIR}/mimic_annotations_s3.csv" ] && aws s3 cp "$CSV_S3" "${DATA_DIR}/mimic_annotations_s3.csv" --only-show-errors
ls -lh "${DATA_DIR}/mimic_annotations_s3.csv"
df -h /opt/dlami/nvme

# ---- Stage resume checkpoint as <CKPT_DIR>/latest.pt --------------------
mkdir -p "$CKPT_DIR"
# Wipe any stale contents from a prior aborted run; DO NOT skip this step
# (a leftover latest.pt with a different epoch would hijack the resume).
find "$CKPT_DIR" -mindepth 1 -maxdepth 1 ! -name '.keep' -exec rm -rf {} +
echo "[INFO] Staging L-K e130 resume checkpoint from ${RESUME_CKPT_S3} -> ${CKPT_DIR}/latest.pt"
aws s3 cp "$RESUME_CKPT_S3" "${CKPT_DIR}/latest.pt" --only-show-errors
ls -lh "${CKPT_DIR}/latest.pt"

# ---- Sanity: verify optimizer state + epoch counter on the staged ckpt ---
python3 - <<'PY'
import torch
p = "/opt/dlami/nvme/echojepa_l_k_resume_e155/latest.pt"
c = torch.load(p, map_location="cpu", weights_only=False)
# Required keys for a clean resume via app/vjepa/utils.py:load_checkpoint
for k in ("encoder", "predictor", "target_encoder", "opt", "epoch"):
    assert k in c, f"missing required resume key: {k}"
epoch = int(c["epoch"])
assert epoch == 130, f"expected epoch=130, got {epoch}"
# Optimizer state must contain AdamW momentum buffers to resume cleanly.
opt = c["opt"]
assert "state" in opt and "param_groups" in opt, "malformed optimizer state"
n_state = len(opt["state"])
assert n_state > 0, "optimizer state dict is empty — cannot resume AdamW momenta"
n_groups = len(opt["param_groups"])
# scaler may be None for bf16 runs; just print it for the record.
has_scaler = (c.get("scaler") is not None)
print(f"[INFO] L-K resume-ckpt OK: epoch={epoch} "
      f"opt_state_entries={n_state} param_groups={n_groups} has_scaler={has_scaler} "
      f"itr={c.get('itr', 0)}")
PY

# ---- Periodic uploads ---------------------------------------------------
(
  while true; do
    sleep 600
    sync_ckpts; upload_logs
  done
) &
SYNC_PID=$!

echo "[INFO] Launching echojepa_lk_resume_e155 (resume e130 -> e155 of 240)..."
# Use srun-compatible entry point; this training loop is app/vjepa/train.py
# (single-view V-JEPA), matching the original runs/11 launcher.
srun --kill-on-bad-exit=1 \
  python3 -m app.main_srun \
  --fname "${CFG_REL}"
RC=$?

kill $SYNC_PID 2>/dev/null || true
wait $SYNC_PID 2>/dev/null || true
sync_ckpts; upload_logs
exit $RC
