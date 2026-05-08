#!/bin/bash
# Adapter job: after MCC 762 and FJ 776 complete, pull their final
# checkpoints from S3, run the eval-adapter (wraps clip_target_encoder →
# target_encoder + injects epoch), and push back to the canonical S3 paths
# that the probe sbatches expect.
#
# Dependency: afterany:762:776 so it runs when both pretrains complete,
# regardless of exit code (so even partial runs can be probed).

#SBATCH -p dev
#SBATCH --nodelist=ip-10-0-50-146
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -t 0-01:00:00
#SBATCH -J adapt_final_ckpts
#SBATCH -o /opt/dlami/nvme/logs/adapt_final-%j.out
#SBATCH -e /opt/dlami/nvme/logs/adapt_final-%j.err
#SBATCH --chdir=/opt/dlami/nvme

set -euxo pipefail

export PATH="/opt/vjepa2-312/bin:$PATH"
export AWS_DEFAULT_REGION="us-west-2"
export AWS_REGION="us-west-2"
export HOME="/home/ubuntu"
mkdir -p /opt/dlami/nvme/logs /opt/dlami/nvme/tmp

ART_BUCKET="sagemaker-hyperpod-lifecycle-495467399120-usw2"
S3_BASE="s3://${ART_BUCKET}/vjepa2-artifacts"
SRC_TARBALL_S3="${S3_BASE}/setup/vjepa2-src.tar.gz"

REPO_DIR="/opt/dlami/nvme/src/vjepa2"
mkdir -p "$REPO_DIR"
rm -rf "$REPO_DIR"/*
aws s3 cp "$SRC_TARBALL_S3" /opt/dlami/nvme/tmp/vjepa2-src.tar.gz --only-show-errors
tar xzf /opt/dlami/nvme/tmp/vjepa2-src.tar.gz -C "$REPO_DIR"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"

WORK="/opt/dlami/nvme/adapt_final_${SLURM_JOB_ID}"
mkdir -p "$WORK"

# ---- MCC: prefer e25.pt; fall back to latest e*.pt ----
MCC_S3="${S3_BASE}/runs/mcc_target_anchored_25of100_762/checkpoints"
if aws s3 ls "${MCC_S3}/e25.pt" > /dev/null 2>&1; then
  MCC_SRC="${MCC_S3}/e25.pt"
else
  latest=$(aws s3 ls "${MCC_S3}/" | awk '/e[0-9]+\.pt$/{print $4}' | sort -t e -k2 -n | tail -1)
  MCC_SRC="${MCC_S3}/${latest}"
fi
echo "[ADAPT] MCC source: $MCC_SRC"
aws s3 cp "$MCC_SRC" "$WORK/mcc_src.pt" --only-show-errors
python scripts/mcc_jepa/adapt_checkpoint_for_eval.py "$WORK/mcc_src.pt"
aws s3 cp "$WORK/mcc_src_for_eval.pt" "${S3_BASE}/checkpoints/probe_inputs/mcc_e25_for_eval.pt" --only-show-errors

# ---- FJ: prefer latest.pt; fall back to highest step_*.pt ----
FJ_S3="${S3_BASE}/echomv_jepa/full_joint_restart_v2_30k_runs/776"
if aws s3 ls "${FJ_S3}/latest.pt" > /dev/null 2>&1; then
  FJ_SRC="${FJ_S3}/latest.pt"
else
  latest=$(aws s3 ls "${FJ_S3}/" | awk '/step_[0-9]+\.pt$/{print $4}' | sort -t _ -k2 -n | tail -1)
  FJ_SRC="${FJ_S3}/${latest}"
fi
echo "[ADAPT] FJ source: $FJ_SRC"
aws s3 cp "$FJ_SRC" "$WORK/fj_src.pt" --only-show-errors
python scripts/mcc_jepa/adapt_checkpoint_for_eval.py "$WORK/fj_src.pt"
aws s3 cp "$WORK/fj_src_for_eval.pt" "${S3_BASE}/checkpoints/probe_inputs/fj_v2_latest_for_eval.pt" --only-show-errors

# ---- Cleanup ----
rm -rf "$WORK"
echo "[ADAPT] DONE"
