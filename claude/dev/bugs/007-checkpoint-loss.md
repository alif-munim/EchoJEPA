# Bug 007: Probe Checkpoint Loss — No Backup, No S3 Push

**Severity:** CRITICAL
**Status:** FIXED (2026-03-16)
**File:** `scripts/run_uhn_probe.sh`

## Summary

Probe training checkpoints (best.pt, log_r0.csv, latest.pt) were stored only in the eval output directory (`evals/vitg-384/nature_medicine/uhn/video_classification_frozen/{task}-{model}/`) with no backup or S3 push. All checkpoint directories for LVEF, TAPSE, MR severity, and AS severity (20 probe runs) are gone.

**Root cause: unknown.** Investigation found no evidence of deletion:
- No `rm` commands in any version of `run_uhn_probe.sh` (all 3 git commits checked)
- No `rm` in bash history
- No `rm` in any previous Claude Code session logs (5 sessions checked)
- No `shutil.rmtree` targeting eval dirs in any repo script
- eval.py never deletes directories
- No EFS audit logs available

The directories definitely existed — training logs in `logs/` confirm successful runs with `folder: 'evals/vitg-384/nature_medicine/uhn'` and tags like `tapse-echojepa-g`. Surviving directories (tr_severity, aov_vmax, trajectory_lvef) in the same parent dir are intact, ruling out wholesale parent-directory deletion.

## What Was Lost

| Task | Models Lost | Runs |
|------|------------|------|
| LVEF | G, L, L-K, EchoPrime, PanEcho | 5 |
| TAPSE | G, L, L-K, EchoPrime, PanEcho | 5 |
| MR severity | G, L, L-K, EchoPrime, PanEcho | 5 |
| AS severity | G, L, L-K, EchoPrime, PanEcho | 5 |
| **Total** | | **20 runs** |

These were run Mar 11-15. Probe weights (best.pt) are gone and must be retrained. However, **training log files survive** in `logs/` with historical AUROC numbers:

| Task | Log files | Notes |
|------|-----------|-------|
| TAPSE | `tapse_5model_vf_20260311_224038.log`, `jobA_tapse_lvef_vbs64_20260313_065551.log` | 5 models completed |
| LVEF | `jobA_tapse_lvef_vbs64_20260313_065551.log` | EchoPrime completed here |
| MR severity | `jobA_mr_severity_ep20_v3.log`, `jobA_mr_severity_bal3_20260314_0836.log` | Multiple attempts, G 0.857, L 0.767 |
| AS severity | `jobB_as_severity_ep20_v3.log`, `jobB_as_severity_bal3_20260314_0836.log` | Multiple attempts |

Historical AUROC values are also preserved in `experiments/nature_medicine/TASK_TRACKER.md`.

Additionally, AV Vmax G/L/L-K log_r0.csv files were overwritten during model cycling. Only EchoPrime and PanEcho AV Vmax checkpoints survived.

## What Survived (19 checkpoints)

| Task | Models | Status |
|------|--------|--------|
| tr_severity | G, L, L-K, EchoPrime | PanEcho still training at time of fix |
| aov_vmax | EchoPrime, PanEcho | G/L/L-K logs lost |
| trajectory_lvef (3-class) | G, L, L-K | EchoPrime/PanEcho not trained |
| trajectory_lvef_onset | G, L, L-K, EchoPrime, PanEcho | All 5 complete |
| trajectory_lvef_v1 (backup) | G, L, L-K, EchoPrime, PanEcho | Old 30-365d version |

All 19 surviving checkpoints archived retroactively to:
- Local: `checkpoints/probes/{task}/{model}/`
- S3: `s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts/checkpoints/probes/{task}/{model}/`

## Fixes Applied

1. **`archive_model()` function** (lines 246-273): After each model completes training, copies best.pt + log_r0.csv + latest.pt to `checkpoints/probes/{task}/{model}/` (separate from eval output dir) and pushes best.pt + log_r0.csv to S3.

2. **Archive on skip** (line 285): When a model is skipped as already-complete, still runs `archive_model()` to catch previously un-archived runs. Idempotent.

3. **Archive after success** (line 330): Runs `archive_model()` immediately after each model finishes training.

4. **`best.pt` verification** (lines 323-327): Warns if training exits 0 and all epochs logged but best.pt is missing.

5. **`is_complete()` respects `FRESH=true`** (lines 203-205): Previously, `is_complete()` always skipped models that had reached target epochs, even when `FRESH=true` was set. Now returns false in fresh mode.

6. **`NO_S3` env var** (line 23): Set `NO_S3=true` to skip S3 upload (local archive still saved).

## Retraining Required

The following 20 runs must be re-done:

```bash
# LVEF (5 models)
bash scripts/run_uhn_probe.sh lvef

# TAPSE (5 models)
bash scripts/run_uhn_probe.sh tapse

# MR severity (5 models)
bash scripts/run_uhn_probe.sh mr_severity

# AS severity (5 models)
bash scripts/run_uhn_probe.sh as_severity

# AV Vmax G/L/L-K (3 models — EchoPrime/PanEcho survived)
bash scripts/run_uhn_probe.sh --models "echojepa-g echojepa-l echojepa-l-k" aov_vmax
```

Each task takes ~3-4 hours for 5 models on 8 GPUs. Total retraining: ~20 hours.

## Prevention

The `archive_model()` function now runs automatically after every model. Even if the eval output directory is deleted, checkpoints persist in `checkpoints/probes/` and on S3. To restore from S3:

```bash
aws s3 sync s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts/checkpoints/probes/ checkpoints/probes/
```
