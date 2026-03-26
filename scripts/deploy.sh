#!/bin/bash
# Deploy latest code from controller repo to compute node.
# Usage: ~/deploy.sh [node]   (default: ip-10-0-50-184)
set -e

NODE="${1:-ip-10-0-50-184}"
REPO="$HOME/EchoJEPA-repo"
REMOTE="/opt/vjepa2"

cd "$REPO"

# Show what's being deployed
echo "Deploying $(git log --oneline -1) to $NODE:$REMOTE"

# Pack source (exclude large/generated files)
tar czf /tmp/vjepa2-deploy.tar.gz \
    --exclude='.git' \
    --exclude='*.pt' --exclude='*.pth' --exclude='*.pkl' --exclude='*.zip' \
    --exclude='*.dcm' --exclude='*.csv' --exclude='*.npz' --exclude='*.parquet' \
    --exclude='checkpoints' --exclude='experiments' --exclude='predictions' \
    --exclude='results' --exclude='logs' --exclude='figures' \
    --exclude='evals/video_classification_frozen/*/output*' \
    --exclude='evals/video_classification_frozen/modelcustom/EchoPrime/model_data*' \
    --exclude='data/samsung' --exclude='data/data' \
    --exclude='uhn_echo' --exclude='notebooks' \
    --exclude='__pycache__' --exclude='*.egg-info' \
    .

SIZE=$(du -sh /tmp/vjepa2-deploy.tar.gz | cut -f1)
echo "Archive: $SIZE"

# Deploy to compute node
srun -N1 -w "$NODE" --ntasks=1 bash -c "
    tar xzf /tmp/vjepa2-deploy.tar.gz -C $REMOTE
    echo 'Deployed to $NODE:$REMOTE'
    ls -la $REMOTE/app/vjepa_2_1/train.py | awk '{print \"  train.py:\", \$6, \$7, \$8}'
"

echo "Done."
