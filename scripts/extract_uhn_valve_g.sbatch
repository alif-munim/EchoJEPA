#!/bin/bash
#SBATCH --job-name=uhn_mr_g
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --time=10:00:00
#SBATCH --output=/tmp/uhn_mr_g.out
#SBATCH --error=/tmp/uhn_mr_g.err

# UHN view-filtered MR severity clip-level embedding extraction for
# EchoJEPA-G, intended for CY-style sklearn probe reproduction.
#
# Input CSVs live under experiments/nature_medicine/uhn/probe_csvs/mr_severity/
# but the clip-level master NPZ only needs ONE deduplicated S3 URI per row.
# We build that CSV inline at the top of this script by taking the union of
# train_vf + val_vf + test_vf across the four valve tasks — so the resulting
# NPZ is reusable for MR, TR, AR, AS probes without re-extraction.
#
# Output: $MASTER_NPZ (clip-level embeddings, paths, placeholder labels).
# Per-task sklearn probes later map labels by study_id via the same
# pipeline used in scripts/cy_reproduction/cy_repro_task.py.

set -euo pipefail

PYTHON=/home/sagemaker-user/.conda/envs/vjepa2-312/bin/python
VJEPA=/opt/vjepa2
PROBE_CSVS=$VJEPA/experiments/nature_medicine/uhn/probe_csvs
OUT_DIR=$VJEPA/experiments/nature_medicine/uhn/valve_extractions
MASTER_CSV=$OUT_DIR/uhn_valve_union_vf.csv
MASTER_NPZ=$OUT_DIR/echojepa_g_uhn_valve_vf_embeddings.npz

mkdir -p $OUT_DIR

cd $VJEPA

# Build the deduplicated union CSV once (skip if it already exists)
if [ ! -f "$MASTER_CSV" ]; then
    echo "[$(date)] Building deduplicated union CSV from valve VF splits..."
    cat \
        $PROBE_CSVS/mr_severity/train_vf.csv \
        $PROBE_CSVS/mr_severity/val_vf.csv \
        $PROBE_CSVS/mr_severity/test_vf.csv \
        $PROBE_CSVS/tr_severity/train_vf.csv \
        $PROBE_CSVS/tr_severity/val_vf.csv \
        $PROBE_CSVS/tr_severity/test_vf.csv \
        $PROBE_CSVS/ar_severity/train_vf.csv \
        $PROBE_CSVS/ar_severity/val_vf.csv \
        $PROBE_CSVS/ar_severity/test_vf.csv \
        $PROBE_CSVS/as_severity/train_vf.csv \
        $PROBE_CSVS/as_severity/val_vf.csv \
        $PROBE_CSVS/as_severity/test_vf.csv \
        | awk '{print $1" 0"}' \
        | sort -u > $MASTER_CSV
    echo "[$(date)] Union CSV size: $(wc -l < $MASTER_CSV) clips"
else
    echo "[$(date)] Reusing existing union CSV: $MASTER_CSV ($(wc -l < $MASTER_CSV) clips)"
fi

echo ""
echo "============================================================"
echo "UHN valve extraction: EchoJEPA-G"
echo "Data: $MASTER_CSV"
echo "Output: $MASTER_NPZ"
echo "Devices: cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7"
echo "============================================================"
echo ""

PYTHONUNBUFFERED=1 $PYTHON -m evals.extract_embeddings \
    --config configs/inference/vitg-384/view/echojepa_224px.yaml \
    --data "$MASTER_CSV" \
    --output "$MASTER_NPZ" \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
    --batch_size 32 \
    --num_workers 8

echo ""
echo "[$(date)] EchoJEPA-G UHN valve extraction DONE"
echo "Output size: $(ls -la $MASTER_NPZ | awk '{print $5}')"
