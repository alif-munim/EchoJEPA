#!/bin/bash
#SBATCH --job-name=phase_anno
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --output=/opt/dlami/nvme/phase_anno/logs/shard_%a.out
#SBATCH --array=0-255%16
# The `%16` throttle runs at most 16 array tasks concurrently — one per
# ~4-8 vCPU slice of the 96-core compute node. Raise if the node has
# capacity; lower if S3 throttles (symptom: SlowDown errors in stderr).

# Build per-clip phase annotations for a single shard of the MIMIC-IV-Echo
# record list. Pairs with build_phase_annotations.py.
#
# Submit as an array job (`--array=0-255`). Each task processes every
# 256th record, so 256 tasks in parallel cover the full corpus.
#
# Usage on the controller (after deploying the phase code via S3 tarball):
#   sbatch /tmp/vjepa2-ctrl/scripts/phase_annotations.sbatch
# Then when all shards finish:
#   sbatch /tmp/vjepa2-ctrl/scripts/phase_annotations_aggregate.sbatch
# (or run the aggregate step locally from any shard output directory)

set -x

# Canonical HyperPod env (torch, pandas, pydicom, neurokit2 available; see
# claude/dev/hyperpod-deployment.md).
export PATH="/opt/vjepa2-312/bin:$PATH"
source /opt/vjepa2-312/bin/activate

# Keep all caches / HOME on NVMe; root fs is near-full.
export TMPDIR=/opt/dlami/nvme/tmp
export MPLCONFIGDIR=/opt/dlami/nvme/mplcache
export PIP_CACHE_DIR=/opt/dlami/nvme/pip-cache
export HOME=/opt/dlami/nvme/phase_anno/home
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$PIP_CACHE_DIR" "$HOME"

WORKDIR=/opt/dlami/nvme/phase
OUT_DIR=/opt/dlami/nvme/phase_anno/shards
LOGS=/opt/dlami/nvme/phase_anno/logs
mkdir -p "$OUT_DIR" "$LOGS"

# Install any missing deps once (idempotent).
python -m pip install -q --cache-dir "$PIP_CACHE_DIR" \
  neurokit2 wfdb pyarrow pandas 2>&1 | tail -3

# Each array task gets a private tmpdir for S3 staging.
TASK=${SLURM_ARRAY_TASK_ID}
SHARD_TMP=/opt/dlami/nvme/tmp/phase_shard_${TASK}
rm -rf "$SHARD_TMP"
mkdir -p "$SHARD_TMP"

cd "$WORKDIR"
time python build_phase_annotations.py \
  --shard-id "$TASK" \
  --n-shards 256 \
  --record-list "$WORKDIR/echo-record-list.csv" \
  --out-dir "$OUT_DIR" \
  --tmpdir "$SHARD_TMP"

# Upload this shard's parquet to S3 so the aggregate step can pick it up
# from any node.
BUCKET=sagemaker-echojepa-h100-neurips-f85ad7df-bucket
SHARD_FILE="$OUT_DIR/phase_annotations_shard_$(printf '%05d' "$TASK")_of_00256.parquet"
if [ -f "$SHARD_FILE" ]; then
  aws s3 cp "$SHARD_FILE" \
    "s3://$BUCKET/phase_anno/shards/$(basename "$SHARD_FILE")" --quiet
fi

rm -rf "$SHARD_TMP"
echo "[shard $TASK] done"
