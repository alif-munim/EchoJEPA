#!/bin/bash
# Deploy latest code from controller repo to compute nodes.
# Usage: ~/deploy.sh [node ...]   (default: both compute nodes)
set -e

REPO="$HOME/EchoJEPA-repo"
REMOTE="/opt/vjepa2"
ALL_NODES=("ip-10-0-50-83" "ip-10-0-50-184")

# Use args if provided, otherwise deploy to all nodes
if [ $# -gt 0 ]; then
    NODES=("$@")
else
    NODES=("${ALL_NODES[@]}")
fi

cd "$REPO"

echo "Deploying $(git log --oneline -1) to ${NODES[*]}"

# Pack source once (exclude large/generated files)
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

# Deploy to each node
for NODE in "${NODES[@]}"; do
    echo "-> $NODE"
    srun -N1 -w "$NODE" --ntasks=1 bash -c "cat > /tmp/vjepa2-deploy.tar.gz" < /tmp/vjepa2-deploy.tar.gz
    srun -N1 -w "$NODE" --ntasks=1 bash -c "
        mkdir -p $REMOTE
        tar xzf /tmp/vjepa2-deploy.tar.gz -C $REMOTE
        rm -f /tmp/vjepa2-deploy.tar.gz
        echo '  Deployed to $NODE:$REMOTE'
    "
done

echo "Done."
