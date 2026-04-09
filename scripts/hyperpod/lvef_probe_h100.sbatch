#!/bin/bash
#SBATCH --job-name=lvef-probe
#SBATCH --partition=ml-p5-48xlarge
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --output=/tmp/lvef_probe-%j.out
#SBATCH --error=/tmp/lvef_probe-%j.err
#SBATCH --time=1-00:00:00
#SBATCH --nodelist=ip-10-0-50-184

export PATH="/opt/vjepa2-312/bin:$PATH"
source /opt/vjepa2-312/bin/activate
cd /opt/vjepa2

# Force PyTorch's bundled CUDA 12.8 libs instead of system CUDA 12.9
# (cuBLAS 12.9.1.4 has a bf16 gemm bug on H100)
export LD_LIBRARY_PATH="/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cublas/lib:/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cudnn/lib:/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH"

rm -rf /dev/shm/* 2>/dev/null || true

python -m evals.main \
    --fname configs/eval/vitl/icml/echojepa_l_k_lvef_d4_h100.yaml \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7
