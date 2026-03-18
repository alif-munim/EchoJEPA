# Bug 008: Inference Config Missing `resume_checkpoint`, Probe Weights Never Loaded

**Severity:** CRITICAL
**Status:** FIXED (2026-03-18)
**File:** `scripts/run_lvef_pred_avg.sh`, `evals/video_classification_frozen/eval.py`

## Summary

The prediction averaging inference script set `probe_checkpoint: /path/to/best.pt` and `val_only: true` in the generated YAML config, but omitted `resume_checkpoint: true`. In eval.py, checkpoint loading is gated on `resume_checkpoint` (line 493):

```python
if resume_checkpoint and os.path.exists(latest_path):
    classifiers, optimizer, scaler, start_epoch = load_checkpoint(...)
```

Since `resume_checkpoint` defaults to `False` (line 181), the probe checkpoint was **never loaded**. All 12 attentive probe heads ran with random Xavier-initialized weights, producing R²=0.006 (essentially chance) instead of the expected R²~0.71.

## Detection

The bug was not caught immediately because:
- The inference process ran to completion without errors
- The progress bar showed a running "Inf MAE" that appeared reasonable at first glance (8.73 in Z-score space)
- The final R² was only visible in a single log line after 521 batches completed

The key diagnostic signal was the Z-score MAE: 8.73 means predictions are 8.73 standard deviations from truth on average — clearly random. With loaded weights, the MAE drops to ~4.8.

## Fix

Added `resume_checkpoint: true` to the YAML template in `scripts/run_lvef_pred_avg.sh`:

```yaml
val_only: true
resume_checkpoint: true    # <-- was missing
probe_checkpoint: ${PROBE_DIR}/${MODEL_TAG}/best.pt
```

## Verification

After fix, the log confirms checkpoint loading:
```
[load_checkpoint] read-path: .../checkpoints/probes/lvef/echojepa-g/best.pt
[load_checkpoint] loaded pretrained classifier (val_only) with msg: [<All keys matched successfully> x12]
```

## Lesson

Any new inference script must include `resume_checkpoint: true` alongside `probe_checkpoint`. The `probe_checkpoint` field only sets the *path* — it does not trigger loading. Consider refactoring eval.py so that `probe_checkpoint` implies `resume_checkpoint: true` automatically, since there's no valid use case for specifying a checkpoint path but not loading it.
