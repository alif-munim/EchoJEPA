#!/bin/bash
#SBATCH --job-name=byol-2node
#SBATCH --partition=ml-p5-48xlarge
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --output=/tmp/byol_2node-%j.out
#SBATCH --error=/tmp/byol_2node-%j.err
#SBATCH --time=7-00:00:00
#SBATCH --nodelist=ip-10-0-50-83,ip-10-0-50-184

# --- Environment (runs on first node: ip-10-0-50-83) ---
export PATH="/opt/vjepa2-312/bin:$PATH"
source /opt/vjepa2-312/bin/activate
cd /opt/vjepa2

# Force PyTorch's bundled CUDA 12.8 libs (cuBLAS 12.9.1.4 has bf16 bug on H100)
export LD_LIBRARY_PATH="/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cublas/lib:/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cudnn/lib:/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH"

# --- Multi-node distributed setup ---
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -1)
export MASTER_PORT=37129
echo "MASTER_ADDR=$MASTER_ADDR, nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST | tr '\n' ',')"

# --- Sync checkpoint from first node to other nodes ---
CKPT_DIR=/opt/vjepa2/checkpoints/pretrain/mimic/byol_vitl_224px_16f_imagenet
OTHER_NODES=$(scontrol show hostnames $SLURM_JOB_NODELIST | tail -n +2)

for node in $OTHER_NODES; do
    echo "Syncing checkpoint to $node..."
    # Ensure checkpoint directory exists on other node
    srun --nodes=1 --nodelist=$node --ntasks=1 bash -c "mkdir -p $CKPT_DIR" 2>/dev/null
    for f in latest.pt latest_opt.pt; do
        if [ -f "$CKPT_DIR/$f" ]; then
            SIZE=$(stat -c%s "$CKPT_DIR/$f")
            echo "  Copying $f ($(( SIZE / 1024 / 1024 )) MB) to $node..."
            srun --nodes=1 --nodelist=$node --ntasks=1 \
                bash -c "cat > $CKPT_DIR/$f" < "$CKPT_DIR/$f"
            echo "  Done."
        fi
    done
done

# Clean shared memory on all nodes
srun --ntasks-per-node=1 bash -c "rm -rf /dev/shm/* 2>/dev/null; echo \$(hostname): shm cleared"

echo "Launching 2-node BYOL training (16 GPUs)..."

# --- Launch: 8 tasks per node, each task = 1 GPU ---
srun python -m app.main_srun \
    --fname configs/train/vitl16/pretrain-byol-mimic-224px-16f-h100.yaml
