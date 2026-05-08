#!/bin/bash
# Phase-relational JEPA (paper-ready), L-K-init variant.
# Init from EchoJEPA-L-K e100 (K400 -> MIMIC), train 50 continuation
# epochs on the phase_relational objective (batch 32, hardneg sampler).
# FLOPs-matched to V4-IN21K-e50 (i.e., IN21K-init V4 resumed to global e150).
#SBATCH -p dev
#SBATCH --nodelist=ip-10-0-50-146
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 2-00:00:00
#SBATCH -J final_phase_rel_lkinit50
#SBATCH -o /opt/dlami/nvme/logs/final_phase_rel_lkinit50-%j.out
#SBATCH -e /opt/dlami/nvme/logs/final_phase_rel_lkinit50-%j.err
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
# L-K e100 (K400 -> MIMIC). Byte-identical to runs/11/e100.pt.
LK_E100_CKPT_S3="${S3_BASE}/checkpoints/echojepa_l_k_pretrain_k400_to_mimic/e100.pt"
PARQUET_S3="${S3_BASE}/data/phase_annotations/phase_annotations.parquet"
VIEW_CSV_S3="${S3_BASE}/data/view_labels/mimic_view_predictions.csv"
SRC_TARBALL_S3="${S3_BASE}/setup/vjepa2-src.tar.gz"

REPO_DIR="/opt/dlami/nvme/src/vjepa2"
CFG_REL="configs/train/vitl16/pretrain-multiview-phase-relational-hardneg-lkinit-50of100-paper.yaml"
RUN_S3="${S3_BASE}/runs/final_phase_rel_lkinit50_paper_${SLURM_JOB_ID}"
CKPT_DIR="/opt/dlami/nvme/checkpoints/final_phase_rel_hardneg_lkinit50_paper"
PROBE_DIR="/opt/dlami/nvme/probe"
VIEW_DIR="/opt/dlami/nvme/data/view_labels"
# Local path the config points at via optimization.anneal_ckpt.
LK_E100_LOCAL="/opt/dlami/nvme/checkpoints/echojepa_l_k_vitl_e100.pt"
LOG_OUT="/opt/dlami/nvme/logs/final_phase_rel_lkinit50-${SLURM_JOB_ID}.out"
LOG_ERR="/opt/dlami/nvme/logs/final_phase_rel_lkinit50-${SLURM_JOB_ID}.err"
JOB_LOG="/opt/dlami/nvme/logs/final_phase_rel_lkinit50_${SLURM_JOB_ID}.log"

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

mkdir -p "$PROBE_DIR" "$VIEW_DIR" /opt/dlami/nvme/checkpoints
[ ! -f "$PROBE_DIR/phase_annotations.parquet" ] && aws s3 cp "$PARQUET_S3" "$PROBE_DIR/phase_annotations.parquet" --only-show-errors
[ ! -f "$VIEW_DIR/mimic_view_predictions.csv" ] && aws s3 cp "$VIEW_CSV_S3" "$VIEW_DIR/mimic_view_predictions.csv" --only-show-errors
[ ! -f "$LK_E100_LOCAL" ] && aws s3 cp "$LK_E100_CKPT_S3" "$LK_E100_LOCAL" --only-show-errors
ls -lh "$LK_E100_LOCAL" "$PROBE_DIR/phase_annotations.parquet" "$VIEW_DIR/mimic_view_predictions.csv"
df -h /opt/dlami/nvme

# ===== Init-checkpoint preflight =====
# First run: allowlist expected_sha256_prefix is TBD. We run NON-strict so
# the prefix is computed and printed for the operator to commit back into
# verify_init_checkpoint.py. Subsequent jobs can flip to --strict.
# We also enforce our own minimum sanity (file size + tensor shape of
# patch_embed.proj.weight, which is the K400-vs-IN21K smoking gun per
# claude/architecture/echojepa-l-k-pretrain-checkpoints.md).
echo "[INFO] Running init-checkpoint preflight (L-K e100)..."
python scripts/neurips/phase/verify_init_checkpoint.py \
  --ckpt "$LK_E100_LOCAL" \
  --expected-tag mimic_lk_jepa_e100

python3 - <<'PY'
import os, torch
p = os.environ.get("LK_E100_LOCAL", "/opt/dlami/nvme/checkpoints/echojepa_l_k_vitl_e100.pt")
sz = os.path.getsize(p)
# 5,127,746,365 bytes is the canonical L-K e100 size.
assert 5_127_000_000 <= sz <= 5_128_500_000, f"L-K e100 unexpected size: {sz}"
c = torch.load(p, map_location="cpu", weights_only=False)
assert c.get("epoch") == 100, f"expected epoch=100, got {c.get('epoch')}"
# K400 lineage: patch_embed.proj.weight must be 5-D [1024,3,2,16,16]
# (tubelet-2 3D conv). IN21K would be 4-D.
for key_root in ("encoder", "target_encoder"):
    sd = c[key_root]
    # state_dicts are saved with "module." prefix
    pe = None
    for k, v in sd.items():
        if k.endswith("patch_embed.proj.weight"):
            pe = v
            break
    assert pe is not None, f"patch_embed.proj.weight not found in {key_root}"
    assert pe.ndim == 5 and tuple(pe.shape) == (1024, 3, 2, 16, 16), \
        f"{key_root}: patch_embed shape {tuple(pe.shape)} — expected [1024,3,2,16,16] (K400 tubelet-2)"
print(f"[INFO] L-K e100 init OK: size={sz:,} epoch={c['epoch']} patch_embed=5D tubelet-2 (K400 lineage)")
PY

# Fresh CKPT_DIR per launch-invariant contract.
if [ -d "$CKPT_DIR" ]; then
  echo "[INFO] Clearing stale CKPT_DIR=$CKPT_DIR"
  rm -rf "$CKPT_DIR"/*
fi
mkdir -p "$CKPT_DIR"

(
  while true; do
    sleep 600
    sync_ckpts; upload_logs
  done
) &
SYNC_PID=$!

echo "[INFO] Launching final_phase_rel_lkinit50_paper (L-K e100 -> e150, phase_relational 50 ep)..."
python -m app.main \
  --fname "$CFG_REL" \
  --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
  2>&1 | tee "$JOB_LOG"
RC=$?

kill $SYNC_PID 2>/dev/null || true
wait $SYNC_PID 2>/dev/null || true
sync_ckpts; upload_logs
exit $RC
