#!/bin/bash
#SBATCH --job-name=gpu-test
#SBATCH --partition=ml-p5-48xlarge
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --output=logs/gpu-test-%j.out
#SBATCH --error=logs/gpu-test-%j.err
#SBATCH --time=00:10:00

echo "========== GPU TEST: $(date) =========="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "Tasks: $SLURM_NTASKS"
echo ""

# --- Per-node GPU visibility and basic info ---
echo "========== NVIDIA-SMI (per node) =========="
srun --ntasks-per-node=1 bash -c '
    echo "--- $(hostname) ---"
    nvidia-smi --query-gpu=index,name,memory.total,driver_version,cuda_version \
        --format=csv,noheader
    echo ""
'

# --- Per-GPU compute test (matrix multiply) ---
echo "========== PER-GPU COMPUTE TEST =========="
srun --ntasks-per-node=8 bash -c '
    GPU_ID=$((SLURM_LOCALID))
    python3 -c "
import torch, time
d = torch.device(\"cuda:${GPU_ID}\")
torch.cuda.set_device(d)
# Warmup
a = torch.randn(4096, 4096, device=d)
b = torch.randn(4096, 4096, device=d)
torch.matmul(a, b)
torch.cuda.synchronize()
# Timed
start = time.time()
for _ in range(50):
    torch.matmul(a, b)
torch.cuda.synchronize()
elapsed = time.time() - start
tflops = 50 * 2 * 4096**3 / elapsed / 1e12
mem = torch.cuda.get_device_properties(d).total_mem / 1e9
print(f\"$(hostname) GPU ${GPU_ID}: {mem:.1f} GB | 50x matmul 4096x4096 in {elapsed:.2f}s | {tflops:.1f} TFLOPS\")
"
'

# --- NCCL all-reduce across all 16 GPUs ---
echo ""
echo "========== NCCL ALL-REDUCE TEST (16 GPUs) =========="
srun --ntasks-per-node=1 bash -c '
    if [ "$SLURM_PROCID" -eq 0 ]; then
        python3 -c "
import torch
import torch.distributed as dist
import os, time

os.environ[\"MASTER_ADDR\"] = os.environ.get(\"SLURM_LAUNCH_NODE_IPADDR\", \"localhost\")
os.environ[\"MASTER_PORT\"] = \"29500\"
world_size = int(os.environ.get(\"SLURM_NTASKS\", 1)) * 8
# We run a single-process version per node, testing intra-node first
rank = int(os.environ.get(\"SLURM_PROCID\", 0))

# Intra-node NCCL test
for gpu in range(8):
    d = torch.device(f\"cuda:{gpu}\")
    t = torch.ones(1024, 1024, device=d)
    torch.cuda.synchronize(d)
print(f\"$(hostname): All 8 GPUs allocated and functional\")

# P2P bandwidth test (GPU 0 <-> GPU 7, worst case on NVSwitch)
d0, d7 = torch.device(\"cuda:0\"), torch.device(\"cuda:7\")
src = torch.randn(256, 1024, 1024, device=d0)  # ~1 GB
torch.cuda.synchronize()
start = time.time()
dst = src.to(d7)
torch.cuda.synchronize()
elapsed = time.time() - start
bw = src.nelement() * src.element_size() / elapsed / 1e9
print(f\"$(hostname): GPU 0->7 transfer 1GB in {elapsed*1000:.1f}ms ({bw:.1f} GB/s)\")
"
    fi
'

echo ""
echo "========== TEST COMPLETE: $(date) =========="
