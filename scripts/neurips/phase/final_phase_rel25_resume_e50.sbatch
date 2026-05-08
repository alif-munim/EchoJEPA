#!/bin/bash
# Phase-relational JEPA method — resume V4 e25 -> e50 of continuation
# (global e125 -> e150). Extends the paper-ready hardneg25 run for a
# FLOPs-matched comparison with V-JEPA†-e155 (~35.5 EFLOPs post-e100).
#
# Source checkpoint: s3://.../runs/final_phase_rel25_paper_593/checkpoints/e25.pt
# Config: configs/train/vitl16/pretrain-multiview-phase-relational-hardneg-e25to50-paper.yaml
#   (sets load_checkpoint=true, force_load_pretrain=false, stop_after_epochs=50,
#    scheduler_total_epochs=100 — same cosine horizon as the original run).
#SBATCH -p dev
#SBATCH --nodelist=ip-10-0-50-56
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 1-12:00:00
#SBATCH -J final_phase_rel25_resume_e50
#SBATCH -o /opt/dlami/nvme/logs/final_phase_rel25_resume_e50-%j.out
#SBATCH -e /opt/dlami/nvme/logs/final_phase_rel25_resume_e50-%j.err
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
export AWS_METADATA_SERVICE_TIMEOUT=5
export AWS_METADATA_SERVICE_NUM_ATTEMPTS=3
export LD_LIBRARY_PATH="/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cublas/lib:/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"

ART_BUCKET="sagemaker-hyperpod-lifecycle-495467399120-usw2"
S3_BASE="s3://${ART_BUCKET}/vjepa2-artifacts"
# Resume source: V4 e25 from the original paper run (job 593).
RESUME_CKPT_S3="${S3_BASE}/runs/final_phase_rel25_paper_593/checkpoints/e25.pt"
PARQUET_S3="${S3_BASE}/data/phase_annotations/phase_annotations.parquet"
VIEW_CSV_S3="${S3_BASE}/data/view_labels/mimic_view_predictions.csv"
SRC_TARBALL_S3="${S3_BASE}/setup/vjepa2-src.tar.gz"

REPO_DIR="/opt/dlami/nvme/src/vjepa2"
CFG_REL="configs/train/vitl16/pretrain-multiview-phase-relational-hardneg-e25to50-paper.yaml"
RUN_S3="${S3_BASE}/runs/final_phase_rel25_resume_e50_${SLURM_JOB_ID}"
# Reuse the original CKPT_DIR so `folder:` in the config matches and the
# train loop's `folder/latest.pt` resume lookup finds our staged file.
CKPT_DIR="/opt/dlami/nvme/checkpoints/final_phase_rel_hardneg25_paper"
PROBE_DIR="/opt/dlami/nvme/probe"
VIEW_DIR="/opt/dlami/nvme/data/view_labels"
LOG_OUT="/opt/dlami/nvme/logs/final_phase_rel25_resume_e50-${SLURM_JOB_ID}.out"
LOG_ERR="/opt/dlami/nvme/logs/final_phase_rel25_resume_e50-${SLURM_JOB_ID}.err"
JOB_LOG="/opt/dlami/nvme/logs/final_phase_rel25_resume_e50_${SLURM_JOB_ID}.log"

sync_ckpts() {
  [ -d "$CKPT_DIR" ] && aws s3 sync "$CKPT_DIR" "${RUN_S3}/checkpoints/" \
    --exclude "*" --include "e*.pt" --include "latest.pt" --include "*.csv" --include "*.yaml" \
    --only-show-errors || true
}
upload_logs() {
  [ -f "$LOG_OUT" ] && aws s3 cp "$LOG_OUT" "${RUN_S3}/logs/job.out" --only-show-errors 2>/dev/null || true
  [ -f "$LOG_ERR" ] && aws s3 cp "$LOG_ERR" "${RUN_S3}/logs/job.err" --only-show-errors 2>/dev/null || true
  [ -f "$JOB_LOG" ] && aws s3 cp "$JOB_LOG" "${RUN_S3}/logs/job.log" --only-show-errors 2>/dev/null || true
}
on_exit() { rc=$?; sync_ckpts; upload_logs; exit $rc; }
trap on_exit EXIT

echo "[INFO] host=$(hostname) job=${SLURM_JOB_ID}"
nvidia-smi -L || true

mkdir -p "$REPO_DIR"
rm -rf "$REPO_DIR"/*
aws s3 cp "$SRC_TARBALL_S3" /opt/dlami/nvme/tmp/vjepa2-src.tar.gz --only-show-errors
tar xzf /opt/dlami/nvme/tmp/vjepa2-src.tar.gz -C "$REPO_DIR"
cd "$REPO_DIR"
[ -f "$CFG_REL" ] || { echo "[FATAL] $CFG_REL missing"; exit 1; }
export PYTHONPATH="$REPO_DIR:/opt/dlami/nvme/pyarrow_site:${PYTHONPATH:-}"

# ---- Stage the V4 e25 checkpoint as <CKPT_DIR>/latest.pt so the train
# loop's resume branch picks it up (cfg sets load_checkpoint: true and
# read_checkpoint: latest.pt).
mkdir -p "$PROBE_DIR" "$VIEW_DIR" "$CKPT_DIR"
# Clear any stale per-rank logs / prior-run state from NVMe, but DO NOT
# remove the staged latest.pt (so the flow is: clear, then download).
find "$CKPT_DIR" -mindepth 1 -maxdepth 1 ! -name '.keep' -exec rm -rf {} +
[ ! -f "$PROBE_DIR/phase_annotations.parquet" ] && aws s3 cp "$PARQUET_S3" "$PROBE_DIR/phase_annotations.parquet" --only-show-errors
[ ! -f "$VIEW_DIR/mimic_view_predictions.csv" ] && aws s3 cp "$VIEW_CSV_S3" "$VIEW_DIR/mimic_view_predictions.csv" --only-show-errors

echo "[INFO] Staging V4 e25 resume checkpoint from ${RESUME_CKPT_S3} -> ${CKPT_DIR}/latest.pt"
aws s3 cp "$RESUME_CKPT_S3" "${CKPT_DIR}/latest.pt" --only-show-errors
ls -lh "${CKPT_DIR}/latest.pt" "$PROBE_DIR/phase_annotations.parquet" "$VIEW_DIR/mimic_view_predictions.csv"
df -h /opt/dlami/nvme

# Sanity: confirm the staged checkpoint has epoch=25 and a relational_head.
python3 - <<'PY'
import os, torch
p = "/opt/dlami/nvme/checkpoints/final_phase_rel_hardneg25_paper/latest.pt"
c = torch.load(p, map_location="cpu", weights_only=False)
assert c.get("epoch") == 25, f"expected epoch=25, got {c.get('epoch')}"
assert "relational_head" in c, "relational_head missing from resume checkpoint"
assert c.get("multiview_objective") == "phase_relational", c.get("multiview_objective")
print(f"[INFO] resume-ckpt OK: epoch={c['epoch']} itr={c.get('itr')} "
      f"objective={c['multiview_objective']} has_rel_head=True")
PY

(
  while true; do
    sleep 600
    sync_ckpts; upload_logs
  done
) &
SYNC_PID=$!

echo "[INFO] Launching final_phase_rel25_resume_e50 (resume e25 -> e50 of 100)..."
python -m app.main \
  --fname "$CFG_REL" \
  --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
  2>&1 | tee "$JOB_LOG"
RC=$?

kill $SYNC_PID 2>/dev/null || true
wait $SYNC_PID 2>/dev/null || true
sync_ckpts; upload_logs
exit $RC
