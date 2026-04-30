#!/bin/bash
#SBATCH --job-name=phase_anno_agg
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=/opt/dlami/nvme/phase_anno/logs/aggregate.out

# Concatenate all shard parquets produced by phase_annotations.sbatch into
# one table. Pulls from S3 so this can run on any node once the shards are
# uploaded.

set -x
export PATH="/opt/vjepa2-312/bin:$PATH"
source /opt/vjepa2-312/bin/activate

export TMPDIR=/opt/dlami/nvme/tmp
export HOME=/opt/dlami/nvme/phase_anno/home
mkdir -p "$TMPDIR" "$HOME"

BUCKET=sagemaker-echojepa-h100-neurips-f85ad7df-bucket
OUT_DIR=/opt/dlami/nvme/phase_anno/shards
AGG=/opt/dlami/nvme/phase_anno/phase_annotations.parquet
mkdir -p "$OUT_DIR"

aws s3 sync "s3://$BUCKET/phase_anno/shards/" "$OUT_DIR" --quiet
ls "$OUT_DIR" | wc -l

cd /opt/dlami/nvme/phase
python build_phase_annotations.py \
  --aggregate \
  --out-dir "$OUT_DIR" \
  --aggregated-out "$AGG"

aws s3 cp "$AGG" "s3://$BUCKET/phase_anno/phase_annotations.parquet" --quiet
echo "[agg] done -> s3://$BUCKET/phase_anno/phase_annotations.parquet"
