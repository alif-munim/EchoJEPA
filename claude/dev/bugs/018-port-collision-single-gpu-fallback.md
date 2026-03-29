# Bug 018: Port collision causes silent single-GPU fallback

**Severity:** HIGH
**Status:** FIXED (2026-03-29, workaround)
**Affected file:** `src/utils/distributed.py`, launch procedures
**Related:** Bug 010 (orphan DDP workers), Bug 019 (orphan GPU processes)

## Summary

When `init_distributed()` fails to bind the default port (37129), the eval process silently falls back to `rank: 0/1` (single GPU), iterating through the entire dataset on one GPU instead of splitting across 8. The run appears to work but takes 8x longer, with no obvious error unless you check the init log.

## How Discovered

After killing a running RVSP probe (pid 1783559) and relaunching, the 5K subset run showed `iterations per epoch: 5000` instead of the expected 625 (5000/8 GPUs). Epoch timing was ~48 min — identical to the full 41K run on 8 GPUs — which made no sense for 1/8 the data.

The init log revealed:
```
Rank: 0. Distributed training not available The server socket has failed to listen
on any local network address. port: 37129, useIpv6: false, code: -98, name: EADDRINUSE
Running... (rank: 0/1)
```

Port 37129 was held by orphan child processes from the killed run (see Bug 019). The parent process fell back to world_size=1, creating a DataLoader with no DistributedSampler. Meanwhile, `mp.spawn` still launched 8 child workers that formed their own distributed group on a different port — but the parent's DataLoader was already configured for single-GPU.

## Root Cause

`src/utils/distributed.py` line 18:
```python
def init_distributed(port=37129, rank_and_world_size=(None, None)):
```

Port 37129 is hardcoded. When it's unavailable:
1. `init_distributed()` catches the socket error and returns `(1, 0)` (world_size=1, rank=0)
2. The DataLoader is created with `DistributedSampler(world_size=1)`, which gives every sample to the single process
3. Training runs correctly but 8x slower — no crash, no warning beyond the init log

The dangerous part is the **silent degradation**. The logged `iterations per epoch` shows the correct value for the 8-GPU case (because child workers also log it), burying the parent's single-GPU behavior.

## Detection Checklist

Signs that a run fell back to single GPU:
1. `Running... (rank: 0/1)` in the init log (should be `rank: 0/8`)
2. `EADDRINUSE` message
3. Training iteration counter exceeds `len(dataset) / world_size` (e.g., iter 1900+ with ipe=625)
4. Epoch time matches a run on the full dataset (dataset_size / subset_size ratio should predict speedup)

## Fix

**Immediate (workaround):** Set `MASTER_PORT` to an unused port at launch time:
```bash
MASTER_PORT=55585 python -m evals.main --fname config.yaml --devices cuda:0 ...
```

**Proper fix (not yet implemented):** `init_distributed()` should auto-find a free port when the default is busy, or raise an error instead of silently falling back:
```python
# Option A: Auto-find free port
import socket
def find_free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('', 0))
    port = s.getsockname()[1]
    s.close()
    return port

# Option B: Fail loudly instead of silent fallback
if not distributed_available:
    raise RuntimeError(f"Port {port} in use. Set MASTER_PORT env var or kill orphan processes.")
```

**Prevention:** Always kill orphan child processes before relaunching (see Bug 019).

## Impact

Any eval run launched while port 37129 is occupied (by orphans, another job, or TIME_WAIT) silently runs on 1 GPU. The results are correct but wall-clock time is 8x worse. Multiple RVSP rebuttal runs were affected before diagnosis.
