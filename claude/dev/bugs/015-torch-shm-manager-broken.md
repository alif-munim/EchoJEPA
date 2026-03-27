# Bug 015: `torch_shm_manager` broken on SageMaker A100 node

**Severity**: HIGH
**Status**: **FIXED** (workaround)
**Date**: 2026-03-27
**Affected**: Pretraining and any DataLoader with `num_workers > 0`
**Files modified**: `app/vjepa_2_1/train.py`, `configs/train/vitl16/pretrain-21-mimic-224px-16f.yaml`

## Symptom

Any training or eval job with `num_workers > 0` crashes immediately with:

```
RuntimeError: torch_shm_manager at "/opt/conda/lib/python3.12/site-packages/torch/bin/torch_shm_manager": Invalid argument
Exception raised from start_manager at /pytorch/torch/lib/libshm/core.cpp:62
```

The binary `/opt/conda/lib/python3.12/site-packages/torch/bin/torch_shm_manager` always returns `ERROR: Invalid argument` regardless of arguments. The binary itself is broken on this SageMaker node — all IPC mechanisms (Unix sockets, POSIX shared memory, System V shm) work fine, so it's not a kernel/security restriction. Likely a build/runtime incompatibility specific to this container image (PyTorch 2.10.0+cu128).

## Root Cause Chain

Three issues compound:

### 1. Meta upstream sets `file_system` sharing strategy

`app/vjepa_2_1/train.py` (and `app/vjepa/train.py`, `app/byol_video/train.py`) set at module level:

```python
mp.set_sharing_strategy("file_system")
```

This is Meta's upstream default. The `file_system` strategy uses `torch_shm_manager` to manage memory-mapped temporary files. When the binary is broken, this strategy fails.

### 2. `file_system` and `file_descriptor` are NOT equivalent

Counter-intuitively:
- **`file_system` strategy**: Uses `storage._share_filename_cpu_()` → calls `THManagedMapAllocator` → **requires `torch_shm_manager`**
- **`file_descriptor` strategy**: Uses `storage._share_fd_cpu_()` → passes file descriptors via Unix domain sockets → **does NOT require `torch_shm_manager`**

The `file_descriptor` strategy is the PyTorch default on Linux and works fine on this node.

### 3. `TMPDIR` path length exceeds Unix socket limit

The `file_descriptor` strategy uses Unix domain sockets for IPC. Socket paths have a 108-byte limit. The default `TMPDIR` on this node is:

```
/mnt/custom-file-systems/efs/fs-0049217cdf69186d7_fsap-0fa7145b64eaa046b/_tmp  (77 chars)
```

With multiprocessing's auto-generated socket suffix (e.g., `listener-XXXXX-XXXXX`), total path exceeds 108 bytes → `OSError: AF_UNIX path too long`.

## Fix (3 parts)

### Part 1: Change sharing strategy to `file_descriptor`

In `app/vjepa_2_1/train.py` (line ~32):

```python
# BEFORE (Meta upstream)
mp.set_sharing_strategy("file_system")

# AFTER (works without torch_shm_manager)
mp.set_sharing_strategy("file_descriptor")
```

Same fix needed in `app/vjepa/train.py` and `app/byol_video/train.py` if used.

### Part 2: Set `TMPDIR=/tmp` at launch

Required to keep Unix domain socket paths under the 108-byte limit:

```bash
TMPDIR=/tmp LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH python -m app.main --fname <config> --devices cuda:0 ...
```

### Part 3: Enable DataLoader workers in config

```yaml
data:
  num_workers: 4
  persistent_workers: true
```

## Performance Impact

Without workers (`num_workers: 0`), data loading is serial and blocks GPU:

| Setting | iter time | data time | gpu time | ETA (104 epochs) |
|---------|-----------|-----------|----------|-------------------|
| `num_workers: 0` | ~23s | ~15s | ~9s | ~10 days |
| `num_workers: 4` | ~10s | ~1.5s | ~8s | ~4.5 days |

The 2.4x speedup comes entirely from overlapping S3 data fetching with GPU compute.

## Launch Template

Full working command for V-JEPA 2.1 ViT-L pretraining on SageMaker A100:

```bash
cd /mnt/custom-file-systems/efs/fs-0049217cdf69186d7_fsap-0fa7145b64eaa046b/vjepa2 && \
TMPDIR=/tmp LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH \
nohup python -m app.main \
  --fname configs/train/vitl16/pretrain-21-mimic-224px-16f.yaml \
  --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
  > /home/sagemaker-user/user-default-efs/vjepa2/logs/pretrain_21_vitl_resume.log 2>&1 &
```

## Debugging History (what didn't work)

1. **Cleaning `/dev/shm`**: Removed 478 stale files → didn't help (binary itself is broken, not a resource issue)
2. **`torch.multiprocessing.set_sharing_strategy("file_system")` in `app/main.py`**: Set in parent process and `process_main` → still failed because `app/vjepa_2_1/train.py` overrides it at import time, AND `file_system` strategy also uses `torch_shm_manager`
3. **`set_sharing_strategy` in `worker_init_fn`**: Set inside DataLoader worker init → still failed because the strategy affects the SENDING process (worker), and `file_system` always needs `torch_shm_manager`
4. **`file_descriptor` without `TMPDIR=/tmp`**: Socket path too long → `OSError: AF_UNIX path too long`

## Also Required: `LD_LIBRARY_PATH`

The conda `libstdc++` must be on `LD_LIBRARY_PATH` because the system libstdc++ lacks `GLIBCXX_3.4.31` (needed by `optree._C`):

```bash
LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH
```

This is a separate issue but always needed on this node for any PyTorch command.
