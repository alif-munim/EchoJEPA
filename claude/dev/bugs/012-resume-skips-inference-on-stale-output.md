# Bug 012: Resume Logic Skips Inference When Stale Output Dir Exists

**Severity:** HIGH
**Status:** FIXED (2026-03-18)
**Files:** `scripts/run_lvef_pred_avg.sh`, `scripts/run_pred_avg.sh`

## Summary

When `val_only: true` + `resume_checkpoint: true`, eval.py's resume logic checks the output directory for existing logs. If a stale `log_r0.csv` exists from a previous failed run (even header-only, no data rows), eval.py interprets this as "already completed" for `num_epochs: 1` and exits 0 without running inference. The run script sees exit code 0 and continues to the next model, silently producing no results.

## Root Cause

eval.py's resume logic:
1. Checks if the output directory exists
2. Reads `log_r0.csv` to determine last completed epoch
3. With `num_epochs: 1`, if the output dir exists with any log, it considers the work done
4. Exits 0 without running validation or printing any warning

A previous failed run (e.g., from before the Bug 008 fix) creates the output directory and writes the CSV header before failing. The header persists as a 43-byte file. On the next run, the resume logic finds this stale artifact and skips.

## Impact

- LVEF prediction averaging: **EchoJEPA-L-K and PanEcho silently skipped** — the script ran G, L, (L-K skipped), EchoPrime, (PanEcho skipped), and exited successfully
- No error message, no warning, exit code 0
- Only detectable by checking that the output log has actual data rows

## Detection Signals

- Output log contains only the CSV header (43 bytes, no data rows)
- Inference "completes" in under 1 second (should take 30-90 minutes for 266K clips)
- Missing models in results when all 5 were expected

## Fix Applied

Added stale output dir cleanup before each inference run in both prediction averaging scripts:

```bash
# Clear stale output dir to prevent resume logic from skipping inference (Bug 012)
local OUT_TAG_DIR="${OUT_DIR}/video_classification_frozen/${TASK}-predavg-${MODEL}"
if [ -d "$OUT_TAG_DIR" ]; then
    log ">>> Clearing stale output dir: ${OUT_TAG_DIR}"
    rm -rf "$OUT_TAG_DIR"
fi
```

This is safe because prediction averaging is inference-only — there's no training state to preserve. The probe checkpoint lives in a separate directory (`checkpoints/probes/`), not in the output dir.

**Note:** This fix is specific to prediction averaging scripts (`val_only: true`). Training scripts should NOT clear output dirs — the resume logic is correct and useful for resuming interrupted training runs.

## Related

- [Bug 008](008-inference-probe-not-loaded.md) — the original bug that caused the failed runs leaving stale output dirs
- [Bug 009](009-shm-exhaustion-silent-ddp-death.md) — another silent success masking bug (exit code 0 with partial results)
