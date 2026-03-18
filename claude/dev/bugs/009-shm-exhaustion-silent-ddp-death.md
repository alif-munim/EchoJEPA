# Bug 009: /dev/shm Exhaustion Causes Silent DDP Worker Death

**Severity:** HIGH
**Status:** FIXED (2026-03-18)
**Files:** `scripts/run_lvef_pred_avg.sh`, system-level

## Summary

DDP inference workers silently died during prediction averaging, causing the process to "complete" successfully (exit code 0) with only a fraction of test clips processed. Study-level prediction averaging produced garbage results because most clips were never scored.

## Root Cause

Two compounding issues:

### 1. Shared memory (/dev/shm) exhaustion

`/dev/shm` (144GB tmpfs) was 99.3% full (143GB used, 1GB free). The leaked memory came from **dozens of orphaned DataLoader worker processes** (`multiprocessing.spawn`) from previous crashed training/inference runs. These workers held shared memory mappings even after their parent processes died.

With only ~1GB free, new DataLoader workers failed immediately:

```
RuntimeError: unable to allocate shared memory(shm) for file </torch_2937163_2983410088_1>: Success (0)
```

This crashed individual DDP workers. The surviving workers processed their data partition and exited. The parent process collected partial results and exited 0.

### 2. Silent failure masking

When DDP workers die mid-inference:
- The remaining workers finish their partition of the data
- The parent `mp.Process` launcher exits with code 0
- The bash script sees exit code 0 and prints "DONE"
- Study-level metrics are computed on incomplete predictions (most studies have 0 predictions from the dead workers' partition, so they're simply missing from the average)

The runs appeared to complete in 60-90 seconds instead of the expected 80+ minutes — a clear signal, but easy to miss.

## Detection Signals

- **Impossible completion times**: G "finished" 266K-clip inference in 82 seconds
- **Only 2-3 of 4 DDP workers visible** in `nvidia-smi --query-compute-apps`
- **`/dev/shm` at 99%+**: `df -h /dev/shm` showed 143GB/144GB used
- **Orphaned processes**: `ps aux | grep multiprocessing.spawn` showed 40+ zombie workers
- **`free -h` shared column**: 142GB in shared memory

## Fixes Applied

### In `scripts/run_lvef_pred_avg.sh`:

1. **Orphan cleanup between model runs**: Added `pkill -f multiprocessing.spawn` and `pkill -f multiprocessing.resource_tracker` before each model inference starts.

2. **Reduced `num_workers` from 8 to 4**: Halves per-process shm pressure. Each worker prefetches batches into shared memory; fewer workers = less shm.

3. **Reduced G `val_batch_size` from 192 to 128**: Each batch allocates `batch_size * frames * H * W * channels * dtype_size` in shm for the collated tensor. Smaller batches = smaller allocations.

### Manual cleanup procedure:

```bash
# Kill orphaned workers
pkill -f "multiprocessing.spawn"
pkill -f "multiprocessing.resource_tracker"

# Clean shm files
rm -f /dev/shm/torch_* /dev/shm/__KMP_* /dev/shm/sem.*

# Verify
df -h /dev/shm   # should show <1% used
free -h           # shared column should be near 0
```

## Prevention

- Always check `df -h /dev/shm` before launching multi-GPU jobs
- Any script that runs multiple sequential DDP jobs should clean orphans between runs
- Monitor for impossible completion times (inference should take minutes-to-hours, not seconds)
- Consider adding a shm check at the start of eval.py that warns if `/dev/shm` usage exceeds 50%
- The `run_uhn_probe.sh` script already cleans `/dev/shm/torch_*` between models but does not kill orphaned spawn workers — should be updated to match the fix in `run_lvef_pred_avg.sh`
