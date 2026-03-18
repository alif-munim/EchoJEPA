# Bug 011: /dev/shm File Cleanup Kills Concurrent Jobs

**Severity:** HIGH
**Status:** FIXED (2026-03-18)
**Files:** `scripts/run_lvef_pred_avg.sh`, `scripts/run_pred_avg.sh`, `scripts/run_uhn_probe.sh`

## Summary

The `rm -f /dev/shm/torch_*` cleanup added in the Bug 009 fix indiscriminately deletes ALL torch shared memory files, including those actively used by concurrent DDP jobs on other GPUs. This caused a TAPSE retrain on GPUs 4-7 to crash three times while LVEF prediction averaging ran on GPUs 0-3.

## Root Cause

Bug 009's fix had two cleanup steps between model runs:

1. **Kill orphaned processes** (ppid=1 filtered) — safe, targeted
2. **Delete `/dev/shm/torch_*` files** — **indiscriminate, kills concurrent jobs**

When `rm -f /dev/shm/torch_*` runs, it deletes shm backing files for ALL processes, not just orphans. If a concurrent job's DataLoader worker has just created a shm-backed tensor but the main process hasn't consumed it yet, the main process tries to open a file that no longer exists. This causes the concurrent job's DDP workers to crash.

## Evidence

TAPSE G retrain (GPUs 4-7) crashed exactly when LVEF pred avg (GPUs 0-3) transitioned between models (when cleanup runs):

```
TAPSE epoch checkpoints              LVEF pred avg transitions
─────────────────────────            ─────────────────────────
ep01-10: 19:01 → 21:46 ✓
  ❌ CRASH #1                        ~22:51 Earlier pred avg attempt ran shm cleanup
  (3-hour gap)
ep11: 00:53                          00:23 Current script starts → G inference
ep12: 01:12
ep13: 01:30
                                     01:44 G finishes → shm cleanup → start L
  ❌ CRASH #2                        ← cleanup between G→L
  (config regen at 02:03)
ep14: 02:23
                                     02:40 L finishes → shm cleanup → start EP
  ❌ CRASH #3                        ← cleanup between L→EP
  (never restarted)
```

Each crash correlates within minutes of a model transition in the LVEF pred avg script.

## Why File Cleanup Was Unnecessary

After killing orphaned processes (ppid=1), their memory mappings close automatically. The orphaned `/dev/shm/torch_*` files from dead processes are just empty directory entries consuming negligible tmpfs space — the actual memory was freed when the mappings closed. The `rm` was belt-and-suspenders that turned out to be actively harmful.

## Fix Applied

Removed `rm -f /dev/shm/torch_* ...` from all three run scripts:
- `scripts/run_lvef_pred_avg.sh` (line 117)
- `scripts/run_pred_avg.sh` (line 249)
- `scripts/run_uhn_probe.sh` (lines 303, 315, 327)

The ppid=1 filtered process kill remains — it's safe and sufficient.

Also fixed residual Bug 010 regression: `run_uhn_probe.sh` error paths (lines 315-317, 327-329) still used unfiltered `pkill -f "multiprocessing.spawn"`. Replaced with ppid=1-filtered cleanup.

## If /dev/shm Cleanup Is Truly Needed

Only clean shm files when NO concurrent jobs are running:

```bash
# Manual cleanup — ONLY when no other GPU jobs are active
ps -eo pid,ppid,args | grep "multiprocessing.spawn" | grep -v grep | awk '$2 == 1 {print $1}' | xargs -r kill
ps -eo pid,ppid,args | grep "multiprocessing.resource_tracker" | grep -v grep | awk '$2 == 1 {print $1}' | xargs -r kill
sleep 2
# Only THEN is it safe to clean files
rm -f /dev/shm/torch_* /dev/shm/__KMP_* /dev/shm/sem.*
```

## Impact

- TAPSE G lost 1 epoch (14/15 completed, best R²=0.554 at epoch 11)
- Multiple manual restarts wasted ~3 hours
- Masked the actual crash cause (appeared as random DDP instability)

## Related

- [Bug 009](009-shm-exhaustion-silent-ddp-death.md) — the original shm exhaustion fix that introduced this cleanup
- [Bug 010](010-pkill-kills-concurrent-jobs.md) — same class of bug (indiscriminate cleanup kills concurrent jobs), but for process kills instead of file deletion
