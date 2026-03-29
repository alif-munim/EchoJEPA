# Bug 019: Orphan GPU processes survive parent kill, block future runs

**Severity:** HIGH
**Status:** DOCUMENTED (2026-03-29, manual procedure)
**Affected file:** Process management (no code fix — operational procedure)
**Related:** Bug 010 (pkill cleanup), Bug 018 (port collision)

## Summary

Killing the parent `evals.main` process does NOT kill its `mp.spawn` child workers. The children become orphans (ppid=1), continue holding GPU memory and network ports (including the NCCL port), and block future runs from initializing distributed training.

## How Discovered

During RVSP probe training for the rebuttal, we killed and relaunched the eval process 4 times (switching checkpoints, fixing configs). Each kill left 8-9 orphan children alive. By the fourth launch:

- **19 GPU processes** running (8 from run 1 + 1 + 1 + 1 + 8 from current)
- **Port 37129** held by orphans from run 1, causing Bug 018 (silent single-GPU fallback)
- **GPU memory** consumed by orphan models (ViT-L encoder × 11 orphans)

Discovery was triggered by the 5K subset RVSP run taking the same wall-clock time as the 41K full run — impossible with 8x less data.

Diagnosis:
```bash
# nvidia-smi showed 19 GPU processes
nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u | wc -l  # 19

# All orphans had ppid=1
for pid in 1783624 1783625 ...; do
    ps -o ppid= -p $pid  # → 1
done

# Port held by orphans
python3 -c "import socket; s=socket.socket(); s.bind(('',37129))"  # EADDRINUSE
```

## Root Cause

`evals.main` uses `mp.spawn` (or `mp.Process`) to create 8 worker processes. When the parent is killed with `kill <pid>`:
1. Parent receives SIGTERM and dies
2. Children are reparented to init (pid 1) — they do NOT receive a signal
3. Children continue running: holding GPU memory, NCCL sockets, /dev/shm files
4. Port 37129 remains bound by the children's NCCL listener

This is the same underlying mechanism as Bug 010, but Bug 010 addressed it in shell scripts with ppid=1 filtering. Manual `kill` of interactive launches has no such safety net.

## Correct Kill Procedure

```bash
# Step 1: Kill parent
kill <parent_pid>

# Step 2: Wait for children to notice
sleep 2

# Step 3: Kill orphan children (ppid=1 only — safe for concurrent jobs)
pgrep -P 1 -f "evals.main|video_classification" | xargs -r kill 2>/dev/null

# Step 4: Verify GPUs are free
nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l  # should be 0

# Step 5: Verify port is free (may need to wait for TIME_WAIT)
python3 -c "import socket; s=socket.socket(); s.bind(('',37129)); s.close(); print('FREE')"
```

Alternatively, kill the entire process group:
```bash
# Kill parent + all children in one shot
kill -- -$(ps -o pgid= -p <parent_pid> | tr -d ' ')
```

## What NOT To Do

- `pkill -f "multiprocessing.spawn"` — kills ALL DDP workers on the machine, including concurrent jobs (Bug 010)
- Launching a new run without checking for orphans — causes Bug 018 (silent single-GPU fallback)
- Assuming `kill <parent>` is sufficient — it never is for mp.spawn processes

## Possible Code Fixes (not yet implemented)

1. **Signal forwarding in scaffold**: Trap SIGTERM in the parent and forward to all children before exiting
2. **Port auto-retry in `init_distributed()`**: If default port busy, find a free one (see Bug 018)
3. **Startup orphan check**: Before creating DataLoaders, scan for orphan GPU processes and warn/fail

## Timeline

| Time | Event |
|------|-------|
| Mar 28 ~18:00 | Run 1: RVSP full 41K (pid 1783559) started |
| Mar 29 01:54 | Run 1 killed. 8 children orphaned (pid 1783624-1783631) |
| 01:54 | Run 2: RVSP full 41K pt50 (pid 2746968) launched — port 37129 held by orphans |
| 02:03 | Run 2 killed. 1 child orphaned (pid 2747176) |
| 02:03 | Run 3: RVSP 5K pt50 (pid 2766931) — port still busy, MASTER_PORT not set |
| 02:27 | Run 3 killed. 1 child orphaned (pid 2767111) |
| 02:27 | Run 4: with MASTER_PORT=39751 — port 39751 also busy (from run 3 orphan) |
| 02:27 | Run 4 killed. 1 child orphaned (pid 2789985) |
| 02:42 | All 11 orphans killed. Port 37129 in TIME_WAIT. MASTER_PORT=55585 used. |
| 02:43 | Run 5: Clean launch. 8/8 GPUs, 625 iters/epoch, 6 min/epoch. Correct. |
