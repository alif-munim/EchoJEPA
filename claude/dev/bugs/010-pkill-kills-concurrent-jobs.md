# Bug 010: pkill Orphan Cleanup Kills Concurrent DDP Jobs

**Severity:** HIGH
**Status:** FIXED (2026-03-18)
**Files:** `scripts/run_lvef_pred_avg.sh`, `scripts/run_pred_avg.sh`, `scripts/run_uhn_probe.sh`

## Summary

The orphan cleanup added in the Bug 009 fix used `pkill -f "multiprocessing.spawn"`, which kills ALL `multiprocessing.spawn` processes on the machine — including workers belonging to other active DDP jobs running on different GPUs. This caused TAPSE retraining (GPUs 4-7) to crash when the LVEF pred avg script (GPUs 0-3) ran its between-model cleanup.

## Root Cause

Bug 009 introduced orphan cleanup between sequential model runs:

```bash
pkill -f "multiprocessing.spawn" 2>/dev/null || true
pkill -f "multiprocessing.resource_tracker" 2>/dev/null || true
```

`pkill -f` matches the process command line against ALL processes owned by the user. It cannot distinguish between:
- **Orphaned workers** (ppid=1, leaked from crashed jobs, holding /dev/shm)
- **Active workers** (ppid=<real_parent>, belonging to a running DDP job on other GPUs)

When LVEF pred avg finished EchoJEPA-G at 01:44 and ran cleanup before starting L, it killed all 4 of TAPSE's DDP workers on GPUs 4-7. TAPSE was mid-epoch 14, and the process died with "56 leaked semaphore objects" — the resource tracker tried to clean up after the abrupt kill.

## Detection

- TAPSE log shows training at batch 3150 at 01:45:15, then immediate resource_tracker cleanup warnings and death
- LVEF-G pred avg log_r0.csv last modified at 01:44 — exactly when the between-model cleanup ran
- `run_uhn_probe.sh` detected the incomplete training: "only 13/15 epochs in log — likely crashed"
- No NCCL timeout, no OOM, no /dev/shm exhaustion — a clean kill from outside

## Fix

Replace `pkill -f` with targeted orphan-only cleanup that checks ppid=1:

```bash
# OLD (kills everything):
pkill -f "multiprocessing.spawn" 2>/dev/null || true

# NEW (kills only orphans):
ps -eo pid,ppid,args | grep "multiprocessing.spawn" | grep -v grep | awk '$2 == 1 {print $1}' | xargs -r kill 2>/dev/null || true
```

Orphaned workers are adopted by init (PID 1) when their parent dies. Active workers have their real parent's PID. Filtering on `ppid == 1` ensures we only clean up leaked processes.

Applied to all three scripts:
- `scripts/run_lvef_pred_avg.sh`
- `scripts/run_pred_avg.sh`
- `scripts/run_uhn_probe.sh`

## Lesson

When running concurrent GPU jobs on the same machine, process cleanup must be scoped. `pkill -f` is a machine-wide sledgehammer. Always check ppid or use process group IDs to target only the processes you own.
