#!/bin/bash
#SBATCH --job-name=rankme
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/tmp/rankme-%j.out
#SBATCH --error=/tmp/rankme-%j.out
#SBATCH -w ip-10-0-50-184

# RankMe effective dimensionality: all 4 models run sequentially
# Download checkpoint → run RankMe → delete checkpoint → next model
# Tests dimensionality-collapse hypothesis for SALT

set -euo pipefail

REPO_DIR="/opt/vjepa2"
S3_BASE="s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts"
CKPT_DIR="$REPO_DIR/checkpoints"
NVME_TMP="/opt/dlami/nvme/rankme_tmp"
OUT="$REPO_DIR/scripts/rebuttal/samples/rankme_all.csv"

cd "$REPO_DIR"
export PYTHONUNBUFFERED=1
export PATH="/opt/vjepa2-312/bin:$PATH"
export LD_LIBRARY_PATH="/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cublas/lib:/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cudnn/lib:/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/nccl/lib:/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:$LD_LIBRARY_PATH"

# Use NVMe for temp checkpoint downloads (26TB free vs 2GB on root)
mkdir -p "$NVME_TMP"

# Clean up any leftover large checkpoints from failed runs on root
rm -f "$CKPT_DIR/jepa_in21k_vitl_e95.pt"
rm -f "$CKPT_DIR/byol_vitl_imagenet_v2_e100.pt"
rm -f "$CKPT_DIR/videomae_l_mimic_ep99.pth"

# EchoNet test CSV
CSV="$REPO_DIR/data/csv/echonet_dynamic_test_s3_raw.csv"
if [ ! -f "$CSV" ]; then
    aws s3 cp "$S3_BASE/data/csv/echonet_dynamic_test_s3_raw.csv" "$CSV"
fi

echo "=== RankMe: 4-model sequential run ==="
echo "CSV: $CSV ($(wc -l < "$CSV") videos)"
echo ""

# Write CSV header
mkdir -p "$(dirname "$OUT")"
echo "model,eff_dim_rankme,participation_ratio,embed_dim,usage_pct,top10_energy_pct,top50_energy_pct,n_videos" > "$OUT"

run_model() {
    local NAME="$1" S3_PATH="$2" LOCAL_PATH="$3"
    local NVME_PATH="$NVME_TMP/$(basename "$LOCAL_PATH")"

    echo "============================================================"
    echo "[$NAME] Downloading checkpoint to NVMe..."
    echo "============================================================"
    aws s3 cp "$S3_PATH" "$NVME_PATH" --quiet
    ls -lh "$NVME_PATH"

    # Symlink into expected location
    mkdir -p "$(dirname "$LOCAL_PATH")"
    ln -sf "$NVME_PATH" "$LOCAL_PATH"

    echo "[$NAME] Running RankMe..."
    python scripts/rebuttal/rankme.py \
        --models "$NAME" \
        --csv "$CSV" \
        --device cuda:0 \
        --n_videos 500 \
        --output "/tmp/rankme_${NAME}.csv"

    # Append result (skip header) to combined output
    tail -1 "/tmp/rankme_${NAME}.csv" >> "$OUT"

    echo "[$NAME] Cleaning up checkpoint..."
    rm -f "$NVME_PATH" "$LOCAL_PATH"
    echo "[$NAME] Done."
    echo ""
}

# 1-3 already completed from job 510. Copy results into combined CSV.
for prev in SALT-S2v1-e79 JEPA-IN21K-e100 BYOL-L-e100; do
    if [ -f "/tmp/rankme_${prev}.csv" ]; then
        echo "[$prev] Using cached result from job 510"
        tail -1 "/tmp/rankme_${prev}.csv" >> "$OUT"
    fi
done

# 4. MAE e99 (failed in job 510 with cuBLAS error — now fixed with float32)
run_model "MAE-L-e99" \
    "$S3_BASE/runs/videomae_resume_e54_354/training_folder/checkpoint-99.pth" \
    "$CKPT_DIR/videomae_l_mimic_ep99.pth"

echo "============================================================"
echo "FINAL COMBINED RESULTS"
echo "============================================================"
cat "$OUT"

# Upload to S3
aws s3 cp "$OUT" "$S3_BASE/results/rankme_all.csv" 2>/dev/null || true

echo "=== All done ==="
