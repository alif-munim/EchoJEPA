---
name: H100 cuBLAS bf16 fix
description: Critical fix for bf16 CUBLAS_STATUS_INVALID_VALUE on HyperPod H100 nodes — must set LD_LIBRARY_PATH in sbatch scripts
type: feedback
---

On HyperPod H100 nodes, the system LD_LIBRARY_PATH loads cuBLAS 12.9.1.4 from /usr/local/cuda-12.9/ instead of PyTorch's bundled cuBLAS 12.8. This causes CUBLAS_STATUS_INVALID_VALUE on every bf16 backward pass. The sbatch script MUST prepend PyTorch's bundled NVIDIA libs to LD_LIBRARY_PATH.

**Why:** Spent significant debugging time on this — the error looks like a code bug but is actually a library mismatch. Even a trivial bf16 matmul backward fails without the fix.

**How to apply:** Any new sbatch script for H100 training must include the LD_LIBRARY_PATH override. See `scripts/vjepa2_pretrain_h100.sbatch` for the canonical version. Also, train.py has two code fixes: SLURM_LOCALID override disabled, GradScaler disabled for bf16.
