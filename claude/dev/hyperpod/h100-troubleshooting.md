# H100 Troubleshooting Guide

Lessons learned migrating V-JEPA 2.1 pretraining from A100 (p4de) to H100 (p5.48xlarge) on SageMaker HyperPod.

## Issue 1: CUBLAS_STATUS_INVALID_VALUE during backward pass

### Symptoms
```
RuntimeError: CUDA error: CUBLAS_STATUS_INVALID_VALUE when calling `cublasGemmEx(
  handle, opa, opb, m, n, k, &falpha, a, CUDA_R_16BF, lda, b, CUDA_R_16BF, ...)`
```
- Crash occurs on the first training iteration during `loss.backward()`
- Happens with bf16 (bfloat16) dtype only
- Even a trivial 64x1024 bf16 matmul backward reproduces the error

### Root cause
**cuBLAS library version mismatch.** The SageMaker HyperPod AMI ships multiple CUDA toolkits (12.6, 12.8, 12.9, 13.0). The system `LD_LIBRARY_PATH` puts `/usr/local/cuda-12.9/lib` first, so PyTorch (compiled with CUDA 12.8) loads the system cuBLAS 12.9.1.4 instead of its bundled cuBLAS 12.8. This specific cuBLAS 12.9.1.4 has a bf16 GemmEx bug on H100 (sm_90).

### Diagnosis steps
```bash
# 1. Confirm the error is cuBLAS, not OOM
#    OOM shows "CUDA out of memory", not "CUBLAS_STATUS_INVALID_VALUE"

# 2. Check which cuBLAS PyTorch actually loads
srun -w <node> --gpus-per-node=1 bash -c '
  source /opt/vjepa2-312/bin/activate
  python -c "
import torch; torch.randn(2,2).cuda()
import os
maps = open(f\"/proc/{os.getpid()}/maps\").read()
for line in maps.split(chr(10)):
    if \"cublas\" in line and \".so\" in line:
        print(line)
        break
"'
# If it shows /usr/local/cuda-12.9/ instead of the conda env path, that's the bug.

# 3. Check LD_LIBRARY_PATH
echo $LD_LIBRARY_PATH
# System paths like /usr/local/cuda-12.9/lib should NOT come before PyTorch's libs.

# 4. Verify PyTorch's bundled cuBLAS location
find /opt/vjepa2-312 -name "libcublas*.so*" 2>/dev/null
```

### Fix
Prepend PyTorch's bundled NVIDIA libraries to `LD_LIBRARY_PATH` before launching training:
```bash
export LD_LIBRARY_PATH="/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cublas/lib:/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cudnn/lib:/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH"
```

### Quick verification
```bash
# This should PASS after the fix and FAIL without it
srun -w <node> --gpus-per-node=1 bash -c '
  source /opt/vjepa2-312/bin/activate
  export LD_LIBRARY_PATH="/opt/vjepa2-312/lib/python3.12/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH"
  python -c "
import torch
a = torch.randn(200704, 1024, device=\"cuda\", dtype=torch.bfloat16, requires_grad=True)
b = torch.randn(1024, 4096, device=\"cuda\", dtype=torch.bfloat16)
c = torch.matmul(a, b)
c.sum().backward()
print(\"PASSED\")
"'
```

---

## Issue 2: CUDA_VISIBLE_DEVICES override in train.py

### Symptoms
All 8 GPU processes appear to initialize successfully but crash during the first backward pass with CUBLAS or OOM errors. GPU memory monitoring shows only 1 of 8 GPUs is actually being used.

### Root cause
`app/vjepa_2_1/train.py` has a top-level line:
```python
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]
```
This was designed for multi-node Slurm launches with `--ntasks-per-node=8` (one Slurm task per GPU, each with its own `SLURM_LOCALID`).

However, with `--ntasks-per-node=1` (single Slurm task, app spawns 8 processes via `mp.Process`), `SLURM_LOCALID=0` for the parent, and all child processes inherit this value. Since `train.py` is imported in each child process, it overrides the correct per-rank `CUDA_VISIBLE_DEVICES` that `main.py` already set, forcing all 8 ranks onto GPU 0.

### Fix
Comment out the override in `app/vjepa_2_1/train.py`:
```python
# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
# NOTE: Disabled — main.py already sets CUDA_VISIBLE_DEVICES per rank.
# With ntasks-per-node=1, SLURM_LOCALID=0 for all spawned processes,
# which would force all ranks onto GPU 0.
# try:
#     os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]
# except Exception:
#     pass
```

---

## Issue 3: GradScaler with bfloat16

### Symptoms
Not a crash by itself, but contributes to instability. The A100 checkpoint may carry an extreme GradScaler scale factor (e.g., 2^33 = 8.59 billion).

### Root cause
The training code enables `GradScaler` for all mixed-precision modes including bf16. Unlike fp16 (5-bit exponent, narrow range), bf16 has the same 8-bit exponent as fp32, so loss scaling is unnecessary. Loading a checkpoint with a large scale factor can amplify gradients during backward, causing overflow or numerical issues.

### Fix
Disable GradScaler when using bf16 (after `init_opt` in `train.py`):
```python
if dtype == torch.bfloat16:
    scaler = None
    logger.info("BF16 detected — disabling GradScaler (not needed for bfloat16)")
```
And change scaler usage from `if mixed_precision:` to `if scaler is not None:`.

---

## Issue 4: /dev/shm leaks

### Symptoms
Training crash or OOM on relaunch even though GPUs show 0 MiB used. New processes fail to allocate shared memory.

### Cause
PyTorch DataLoader with `num_workers > 0` uses `/dev/shm` for IPC. Crashes leave stale shared memory segments.

### Fix
Always clear before relaunching:
```bash
srun -w <node> bash -c 'rm -rf /dev/shm/* 2>/dev/null'
```

---

## Environment details (March 2026)

| Component | Version |
|-----------|---------|
| NVIDIA Driver | 580.126.09 |
| System CUDA | 13.0, 12.9, 12.8, 12.6 |
| PyTorch | 2.10.0+cu128 |
| Python | 3.12 (conda env at /opt/vjepa2-312) |
| Instance | ml.p5.48xlarge (8x H100 80GB HBM3) |
| GPU arch | sm_90 |
