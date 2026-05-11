#!/bin/bash
#SBATCH --job-name=uhn_mr_g
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --time=12:00:00
#SBATCH --output=/tmp/uhn_mr_g.out
#SBATCH --error=/tmp/uhn_mr_g.err

# UHN MR-severity view-filtered clip-level embedding extraction for
# EchoJEPA-G. Scoped down from the 4-valve union (which was too big for the
# 10h limit) to just MR, which is what we actually need for the
# CY-sklearn-vs-Strategy-E comparison that directly parallels the MIMIC MR
# result.
#
# Compute node has neither EFS nor the encoder checkpoint locally; both
# pulled from S3 at job start.

set -eo pipefail

PYTHON=/opt/vjepa2-312/bin/python
VJEPA=/opt/vjepa2
S3_ARTIFACTS=s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts

# Compute-node-local working directory (NVMe; 26 TB free on /opt/dlami/nvme)
WORK=/opt/dlami/nvme/uhn_mr_g
CKPT_DIR=$WORK/checkpoints
# Concat all 3 MR splits (train+val+test) into one master CSV for extraction
MR_TRAIN=$WORK/uhn_mr_train.csv
MR_VAL=$WORK/uhn_mr_val.csv
MR_TEST=$WORK/uhn_mr_test.csv
MASTER_CSV=$WORK/uhn_mr_all.csv
MASTER_NPZ=$WORK/echojepa_g_uhn_mr_embeddings.npz

ORIG_CONFIG=configs/inference/vitg-384/view/echojepa_224px.yaml
LOCAL_CONFIG=$WORK/echojepa_224px_mr.yaml
ENCODER_CKPT=$CKPT_DIR/pt-280-an81.pt

S3_OUTPUT=$S3_ARTIFACTS/valve_extractions/echojepa_g_uhn_mr_embeddings.npz

mkdir -p $WORK $CKPT_DIR

cd $VJEPA

echo "[$(date)] Downloading MR CSVs from S3..."
aws s3 cp s3://echodata25/nature_medicine/uhn/probe_csvs/mr_severity_4class/train.csv $MR_TRAIN --quiet
aws s3 cp s3://echodata25/nature_medicine/uhn/probe_csvs/mr_severity_4class/val.csv   $MR_VAL   --quiet
aws s3 cp s3://echodata25/nature_medicine/uhn/probe_csvs/mr_severity_4class/test.csv  $MR_TEST  --quiet
cat $MR_TRAIN $MR_VAL $MR_TEST > $MASTER_CSV
echo "[$(date)] Master CSV: $(wc -l < $MASTER_CSV) clips, $(ls -lh $MASTER_CSV | awk '{print $5}')"

echo ""
echo "[$(date)] Downloading encoder checkpoint (~16 GB) from S3..."
aws s3 cp $S3_ARTIFACTS/checkpoints/anneal/keep/pt-280-an81.pt $ENCODER_CKPT --quiet
echo "[$(date)] Checkpoint: $(ls -lh $ENCODER_CKPT | awk '{print $5}')"

echo ""
echo "[$(date)] Patching inference config to use local checkpoint path..."
sed "s|/mnt/custom-file-systems/efs/fs-0049217cdf69186d7_fsap-0fa7145b64eaa046b/vjepa2/checkpoints/anneal/keep/pt-280-an81.pt|$ENCODER_CKPT|g" $ORIG_CONFIG > $LOCAL_CONFIG
echo "[$(date)] Patched config: $LOCAL_CONFIG"
grep "checkpoint:" $LOCAL_CONFIG | head -3

echo ""
echo "============================================================"
echo "UHN MR severity extraction: EchoJEPA-G"
echo "Data: $MASTER_CSV"
echo "Output: $MASTER_NPZ"
echo "Config: $LOCAL_CONFIG"
echo "Devices: cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7"
echo "============================================================"
echo ""

PYTHONUNBUFFERED=1 $PYTHON -m evals.extract_embeddings \
    --config "$LOCAL_CONFIG" \
    --data "$MASTER_CSV" \
    --output "$MASTER_NPZ" \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
    --batch_size 32 \
    --num_workers 8

echo ""
echo "[$(date)] Extraction done. NPZ size: $(ls -lh $MASTER_NPZ | awk '{print $5}')"

echo ""
echo "[$(date)] Syncing NPZ to S3 for SageMaker-side retrieval..."
aws s3 cp $MASTER_NPZ $S3_OUTPUT --quiet
echo "[$(date)] Uploaded to $S3_OUTPUT"
