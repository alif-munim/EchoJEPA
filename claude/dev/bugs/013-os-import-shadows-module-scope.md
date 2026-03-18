# Bug 013: Local `import os` Shadows Module-Level Import, Breaks study_predictions Save

**Severity:** MEDIUM
**Status:** FIXED (2026-03-18)
**Files:** `evals/video_classification_frozen/eval.py`

## Summary

A conditional `import os` inside `run_one_epoch()` caused Python to treat `os` as a local variable throughout the entire function. When a different code path (the study_predictions save block) tried to use `os.path.join()`, Python raised `UnboundLocalError` because the local `import os` hadn't executed in that branch. The error was caught by a broad `except Exception` and logged as a warning, silently preventing study_predictions.csv from being saved.

## Root Cause

Python's scoping rules: if any assignment to a name (including `import`) appears anywhere in a function body, Python treats that name as local to the entire function, even before the assignment executes.

```python
# Line 8 (module level) — works everywhere EXCEPT functions with a local 'os'
import os

def run_one_epoch(...):
    # ...

    # Line 980 — inside an `if val_only and predictions_save_path is not None` block
    # This import makes 'os' a LOCAL variable for the ENTIRE function
    if val_only and predictions_save_path is not None:
        import os  # <-- shadows module-level os for all of run_one_epoch()
        os.makedirs(...)

    # ...

    # Line 1188 — inside a different conditional block (study_predictions save)
    # When predictions_save_path is None (pred avg path), the import above never runs,
    # but Python still treats 'os' as local → UnboundLocalError
    save_path = os.path.join(output_dir, "study_predictions.csv")  # CRASH
```

The study_predictions save code was added in Session 20 but the `import os` at line 980 predated it. The conflict only manifests when `predictions_save_path is None` (prediction averaging path) but `output_dir is not None` (study predictions path) — a combination that didn't exist before Session 20.

## Symptom

```
[WARNING] R²/Pearson computation failed: cannot access local variable 'os' where it is not associated with a value
```

The warning is misleading — R²/Pearson computation actually succeeded (values are logged on the next line). The error occurs in the study_predictions save block which is inside the same try/except. The broad except catches the `os` error and attributes it to the R²/Pearson computation.

## Impact

- LVEF pred avg G completed with correct R²=0.778 / Pearson=0.889 in `log_r0.csv`
- But `study_predictions.csv` was NOT saved — the per-study predictions needed for downstream analysis were lost
- Required killing and restarting the entire 5-model pred avg pipeline (~5 hours of GPU time)

## Fix

Removed the redundant `import os` at line 980. The module-level `import os` at line 8 is sufficient.

```python
# Before (line 978-981):
if val_only and predictions_save_path is not None and len(all_predictions) > 0:
    import pandas as pd
    import os                    # <-- REMOVED: shadows module-level import
    os.makedirs(...)

# After:
if val_only and predictions_save_path is not None and len(all_predictions) > 0:
    import pandas as pd
    os.makedirs(...)             # uses module-level os
```

## Lessons

1. **Never `import` standard library modules inside functions when they're already imported at module level.** Conditional imports are fine for heavy optional deps (pandas, torch.distributed), but not for `os` which is always available.
2. **Broad `except Exception` blocks mask unrelated errors.** The try/except around R²/Pearson also covered the study_predictions save, making the error message misleading. Consider narrower try blocks or separate try/except for independent operations.
3. **Python 3.12 changed the error message** from `UnboundLocalError: local variable 'os' referenced before assignment` to the less obvious `cannot access local variable 'os' where it is not associated with a value`.

## Related

- Session 20 added the study_predictions save block that exposed this latent scoping issue
- The `import os` at line 980 was added by a previous contributor for the predictions_save_path feature
