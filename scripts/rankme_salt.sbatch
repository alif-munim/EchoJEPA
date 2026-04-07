#!/bin/bash
#SBATCH --job-name=rankme
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/tmp/rankme-%j.out
#SBATCH --error=/tmp/rankme-%j.out
#SBATCH -w ip-10-0-50-184

# RankMe effective dimensionality: SALT v1 e79
# Tests dimensionality-collapse hypothesis: does SALT inherit MAE's low eff_dim?
# Reference values (from prior analysis): BYOL 209, JEPA 197, MAE 63

set -euo pipefail

REPO_DIR="/opt/vjepa2"
S3_BASE="s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts"

cd "$REPO_DIR"
export PYTHONUNBUFFERED=1
export PATH="/opt/vjepa2-312/bin:$PATH"

# LD_LIBRARY_PATH fix for H100 bf16
export LD_LIBRARY_PATH="/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:/usr/local/cuda-12.6/compat:$LD_LIBRARY_PATH"

# Clean up failed JEPA download that filled disk
rm -f "$REPO_DIR/checkpoints/jepa_in21k_vitl_e95.pt"

echo "=== Setting up SALT v1 e79 checkpoint ==="

# SALT v1 e79 — should already be on node from probe job 465
if [ ! -f "$REPO_DIR/checkpoints/pretrain/mimic/salt_s2v1_e79.pt" ]; then
    echo "Downloading SALT v1 e79..."
    aws s3 cp "$S3_BASE/runs/salt_s2_pretrain_388/checkpoints/e79.pt" \
        "$REPO_DIR/checkpoints/pretrain/mimic/salt_s2v1_e79.pt"
fi

ls -lh "$REPO_DIR/checkpoints/pretrain/mimic/salt_s2v1_e79.pt"

# Use EchoNet test set (S3 paths) — 1277 videos, sample 500
CSV="$REPO_DIR/data/csv/echonet_dynamic_test_s3_raw.csv"
if [ ! -f "$CSV" ]; then
    echo "Downloading EchoNet test CSV..."
    aws s3 cp "$S3_BASE/data/csv/echonet_dynamic_test_s3_raw.csv" "$CSV"
fi

echo "=== Running RankMe on SALT v1 ==="
echo "CSV: $CSV"
wc -l "$CSV"

python scripts/rebuttal/rankme.py \
    --models SALT-S2v1-e79 \
    --csv "$CSV" \
    --device cuda:0 \
    --n_videos 500 \
    --output scripts/rebuttal/samples/rankme_salt.csv

echo "=== Done ==="

# Upload results to S3
aws s3 cp scripts/rebuttal/samples/rankme_salt.csv \
    "$S3_BASE/results/rankme_salt.csv" 2>/dev/null || true
