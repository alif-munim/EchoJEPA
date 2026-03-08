#!/bin/bash
# Re-extract MIMIC embeddings for PanEcho, EchoPrime, EchoFM
# These 3 models had normalization bugs (bug 002) that are now fixed.
# Run with: bash scripts/reextract_mimic_3models.sh

set -e

PYTHON=/home/sagemaker-user/.conda/envs/vjepa2-312/bin/python
VJEPA=/mnt/custom-file-systems/efs/fs-0049217cdf69186d7_fsap-0fa7145b64eaa046b/vjepa2
DATA=$VJEPA/data/csv/nature_medicine/mimic/mortality_1yr.csv
MIMIC=$VJEPA/experiments/nature_medicine/mimic
DEVICES="cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7"

cd $VJEPA

echo "============================================================"
echo "MIMIC Re-extraction: PanEcho, EchoPrime, EchoFM"
echo "Data: $DATA (525K clips)"
echo "Devices: $DEVICES"
echo "============================================================"

# --- PanEcho ---
echo ""
echo "[$(date)] Starting PanEcho extraction..."
PYTHONUNBUFFERED=1 $PYTHON -m evals.extract_embeddings \
    --config configs/inference/vitg-384/view/panecho_224px.yaml \
    --data "$DATA" \
    --output "$MIMIC/panecho_mimic_embeddings.npz" \
    --devices $DEVICES \
    --batch_size 32 \
    --num_workers 8
echo "[$(date)] PanEcho DONE."

# --- EchoPrime ---
echo ""
echo "[$(date)] Starting EchoPrime extraction..."
PYTHONUNBUFFERED=1 $PYTHON -m evals.extract_embeddings \
    --config configs/inference/vitg-384/view/echoprime_224px.yaml \
    --data "$DATA" \
    --output "$MIMIC/echoprime_mimic_embeddings.npz" \
    --devices $DEVICES \
    --batch_size 32 \
    --num_workers 8
echo "[$(date)] EchoPrime DONE."

# --- EchoFM ---
echo ""
echo "[$(date)] Starting EchoFM extraction..."
PYTHONUNBUFFERED=1 $PYTHON -m evals.extract_embeddings \
    --config configs/inference/vitg-384/view/echofm_224px.yaml \
    --data "$DATA" \
    --output "$MIMIC/echofm_mimic_embeddings.npz" \
    --devices $DEVICES \
    --batch_size 32 \
    --num_workers 8
echo "[$(date)] EchoFM DONE."

echo ""
echo "============================================================"
echo "[$(date)] All 3 re-extractions complete."
echo "Next: run pool_embeddings.py and regenerate splits."
echo "============================================================"
