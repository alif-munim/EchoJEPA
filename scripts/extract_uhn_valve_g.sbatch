#!/bin/bash
#SBATCH --job-name=uhn_valve_g
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --time=10:00:00
#SBATCH --output=/tmp/uhn_valve_g.out
#SBATCH --error=/tmp/uhn_valve_g.err

# UHN view-filtered valve severity clip-level embedding extraction for
# EchoJEPA-G (MR + TR + AR + AS union — one NPZ covers all 4 valve tasks).
#
# Compute node has neither EFS nor the encoder checkpoint locally; both
# are pulled from S3 at job start.

set -eo pipefail

PYTHON=/opt/vjepa2-312/bin/python
VJEPA=/opt/vjepa2
S3_ARTIFACTS=s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts

# Compute-node-local working directory (NVMe; 26 TB free on /opt/dlami/nvme)
WORK=/opt/dlami/nvme/uhn_valve_extract
CKPT_DIR=$WORK/checkpoints
MASTER_CSV=$WORK/uhn_valve_union_vf.csv
MASTER_NPZ=$WORK/echojepa_g_uhn_valve_vf_embeddings.npz

# The inference config has an EFS path hardcoded for the encoder
# checkpoint. Copy the config to a local copy and patch the path.
ORIG_CONFIG=configs/inference/vitg-384/view/echojepa_224px.yaml
LOCAL_CONFIG=$WORK/echojepa_224px_valve.yaml
ENCODER_CKPT=$CKPT_DIR/pt-280-an81.pt

S3_OUTPUT=$S3_ARTIFACTS/valve_extractions/echojepa_g_uhn_valve_vf_embeddings.npz

mkdir -p $WORK $CKPT_DIR

cd $VJEPA

echo "[$(date)] Downloading union CSV from S3..."
aws s3 cp $S3_ARTIFACTS/setup/uhn_valve_union_vf.csv $MASTER_CSV --quiet
echo "[$(date)] Union CSV: $(wc -l < $MASTER_CSV) clips, $(ls -lh $MASTER_CSV | awk '{print $5}')"

echo ""
echo "[$(date)] Downloading encoder checkpoint (~16 GB) from S3..."
aws s3 cp $S3_ARTIFACTS/checkpoints/anneal/keep/pt-280-an81.pt $ENCODER_CKPT --quiet
echo "[$(date)] Checkpoint: $(ls -lh $ENCODER_CKPT | awk '{print $5}')"

echo ""
echo "[$(date)] Patching inference config to use local checkpoint path..."
# Replace the EFS checkpoint path with the local one
sed "s|/mnt/custom-file-systems/efs/fs-0049217cdf69186d7_fsap-0fa7145b64eaa046b/vjepa2/checkpoints/anneal/keep/pt-280-an81.pt|$ENCODER_CKPT|g" $ORIG_CONFIG > $LOCAL_CONFIG
echo "[$(date)] Patched config: $LOCAL_CONFIG"
grep "checkpoint:" $LOCAL_CONFIG | head -3

echo ""
echo "============================================================"
echo "UHN valve extraction: EchoJEPA-G"
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
