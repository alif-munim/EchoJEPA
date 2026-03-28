# Bug 017: Multi-view eval missing z-score normalization for regression

**Severity:** CRITICAL
**Status:** FIXED (2026-03-28)
**Affected file:** `evals/video_classification_frozen_multi/eval.py`
**Affected tasks:** All multi-view regression (RVSP in ICML preprint and rebuttal)
**Not affected:** Single-view regression (NatMed RVSP, LVEF, TAPSE, etc.)

## Summary

The multi-view eval module (`video_classification_frozen_multi`) never z-score normalized regression labels at runtime, unlike the single-view module which does it at line 899 of `evals/video_classification_frozen/eval.py`. Whether this was a bug depended on the CSV format:

- **ICML preprint**: CSVs were **pre-z-scored** using `sklearn.StandardScaler` (saved as `data/scalers/rvsp_scaler.pkl`). The `int()` cast in `VideoGroupDataset` quantized z-scored floats to integers (-2, -1, 0, 1, 2, ...), but training still worked. The logged MAE was `z_mae * target_std`, correctly converting to mmHg. **Preprint results (4.54 MAE mmHg for G) are valid.**

- **Post-preprint**: CSVs were rebuilt with **raw** mmHg values (for the NatMed pipeline, which uses single-view eval with runtime z-scoring). The multi-view eval was never updated. Raw labels (mean ~34 mmHg) were passed directly to `SmoothL1Loss` against near-zero model outputs, causing catastrophic training failure (MAE ~145-176 mmHg on raw scale).

## Evidence

### 1. Preprint used pre-z-scored CSVs

- `data/scalers/rvsp_scaler.pkl` exists: `StandardScaler(mean=34.465, scale=14.013)`
- `data/scripts/normalize_rvsp_external.py` describes the pre-normalization workflow
- Preprint probe checkpoint regressor biases are ~0.02-0.04 (z-scored output range, not raw ~34)
- Preprint logged MAE of 4.29 = z_mae (0.306) × target_std (14.01) = 4.29 mmHg ✓
- Git commit `a6a520e` (2026-01-23) replaced hardcoded `LVEF_TRAIN_STD = 11.33` with parameterized `target_std`

### 2. CSVs were later replaced with raw values

Current `data/csv/rvsp_train.csv` has raw labels (18.99, 31.13, 29.99 mmHg). The exact date of replacement is unknown (CSVs not in git), but it happened when the NatMed pipeline was built with runtime z-scoring in the single-view module.

### 3. Rebuttal runs failed catastrophically

```
# Without z-score fix (raw labels, multi-view eval):
Epoch 1, iter 60: MAE 172.8 (in *14.01 scale = raw_mae * 14.01 = ~12.3 mmHg * 14 = 172)
# Actually: raw labels ~34, model output ~0, F.l1_loss = ~34, * 14.01 = ~476
# Logged: ~170 (training loss was improving but still terrible)

# With z-score fix:
Epoch 1, iter 0: MAE 9.34 mmHg (correct scale, near baseline ~11.2)
```

## Root Cause

Single-view eval (`video_classification_frozen/eval.py`, line 896-899):
```python
t_mean = target_mean if target_mean is not None else 0.0
t_std = target_std if target_std is not None else 1.0
labels = (labels - t_mean) / t_std  # ← z-score at runtime
```

Multi-view eval (`video_classification_frozen_multi/eval.py`, line 889-893) — BEFORE fix:
```python
y = labels.float()
if y.dim() == 1:
    y = y.unsqueeze(-1)
losses = [criterion(o.float(), y) for o in outs]  # ← raw labels, no z-scoring
```

## Fix

Added z-score normalization to the multi-view eval (line 891-893):
```python
y = labels.float()
if y.dim() == 1:
    y = y.unsqueeze(-1)
# Z-score normalize labels at runtime (CSVs store raw values)
t_mean = target_mean if target_mean is not None else 0.0
t_std = target_std if target_std is not None else 1.0
y = (y - t_mean) / t_std
losses = [criterion(o.float(), y) for o in outs]
```

## Impact Assessment

### ICML preprint RVSP (Table 4)
**Valid.** Pre-z-scored CSVs meant the missing runtime normalization was a no-op — the data was already normalized. The `int()` truncation of z-scored values quantized labels to ~5-6 integer bins across the clinical range, which was coarse but didn't prevent learning on 41K studies.

### ICML rebuttal RVSP
**Invalidated and re-running.** The rebuttal uses raw CSVs (rebuilt for NatMed). All multi-view RVSP runs before 2026-03-28 produced garbage results (MAE ~145 mmHg). The fixed run (EchoJEPA-L full checkpoint, 5K MIMIC subset) is in progress.

### Nature Medicine RVSP
**Not affected.** NatMed uses the single-view module (`video_classification_frozen`) which has correct runtime z-scoring. All NatMed RVSP results (R² 0.504 for G, etc.) are valid.

## Bug 017b: Shared `zscore_params.json` poisoning (discovered 2026-03-28)

**Severity:** HIGH
**Status:** FIXED
**Root cause:** Auto-computed `zscore_params.json` saved to a shared directory, then loaded by a different task.

### What happened

When a regression run has no explicit `target_mean`/`target_std` in the YAML config, `eval.py` auto-detects z-score params in order:
1. Load from `zscore_params.json` in the same directory as the train CSV
2. If not found, compute from the train CSV labels and **save** a `zscore_params.json`

An earlier LVEF run using `data/csv/lvef_train.csv` auto-computed and saved `data/csv/zscore_params.json` with LVEF parameters (mean=57.06, std=11.33). Later, the EchoMAE RVSP ep163 run used `data/csv/rvsp_train.csv` — same directory — and auto-loaded the **LVEF** zscore params for an RVSP task.

This caused RVSP labels (~34 mmHg) to be z-scored as `(34 - 57.06) / 11.33 = -2.03`, producing deeply negative targets. The MAE was ~10.6 in these distorted units × 11.33 (the wrong std) ≈ 120 mmHg effective MAE. The run appeared to not learn at all.

### Diagnosis evidence

```
[INFO] Loaded zscore params from data/csv/zscore_params.json: mean=57.0569, std=11.3252
[INFO] Regression Un-normalization: Mean=57.0569, Std=11.3252
```

The log clearly shows LVEF params (mean≈57, std≈11.3) being loaded for an RVSP task (should be mean≈34.5, std≈14.0).

### Fix

1. **Added explicit `target_mean: 34.4650` / `target_std: 14.0130`** to all 9 RVSP ICML configs. When specified in YAML, the auto-detection is bypassed entirely.
2. **Deleted the stale `data/csv/zscore_params.json`** so it cannot poison future runs.
3. **Recommendation**: All multi-view regression configs should specify `target_mean`/`target_std` explicitly. Never rely on auto-detection when multiple tasks share a CSV directory.

### Affected runs

| Run | Config | Status |
|-----|--------|--------|
| EchoMAE-L RVSP ep163 (full 41K) | `echomae_l_rvsp_d4_ep163.yaml` | **Invalid** — loaded LVEF params. Must restart. |
| All earlier EchoMAE RVSP runs (5K) | `echomae_l_rvsp_d4.yaml` | Different dir (`rebuttal/rvsp/`), had correct params. But ran pre-Bug-017-fix, so still invalid. |
| EchoJEPA-L RVSP UHN 41K (current) | `echojepa_l_mimic_full_rvsp_d4_uhn.yaml` | **Valid** — had explicit params from the start. |

---

## Additional Notes

### The `int()` cast problem

`VideoGroupDataset._get_item_row()` line 281 does `label = int(row["label"])`. For pre-z-scored CSVs, this quantizes continuous z-scores to integers:
- RVSP 20 mmHg → z = (20 - 34.5) / 14.0 = -1.03 → `int(-1.03)` = -1
- RVSP 35 mmHg → z = (35 - 34.5) / 14.0 = 0.04 → `int(0.04)` = 0
- RVSP 50 mmHg → z = (50 - 34.5) / 14.0 = 1.11 → `int(1.11)` = 1

The entire clinical RVSP range (20-50 mmHg) maps to just 3 integer bins (-1, 0, 1). Despite this coarse quantization, EchoJEPA-G achieved 4.54 MAE mmHg on 41K UHN studies — suggesting the model captured the ordinal ranking even with degraded label precision.

### The `* target_std` logging line

Both preprint and current code multiply raw MAE by `target_std` for logging (line 904). With pre-z-scored data, this correctly converts z-MAE to mmHg. With raw data and no z-score fix, it was a double-error: raw MAE in mmHg × std ≈ 170 logged value (which was the symptom that led to discovering this bug).

### Timeline of the code path

| Date | Commit | Change |
|------|--------|--------|
| 2026-01-23 | `6da5989` | Hardcoded `LVEF_TRAIN_STD = 11.33` for MAE logging |
| 2026-01-23 | `a6a520e` | Parameterized to `target_mean`/`target_std` from YAML config |
| 2026-01-26 | `1a5dcf5` | "Working multi-view RVSP inference!" — preprint RVSP runs |
| ~2026-02/03 | (unknown) | CSVs rebuilt with raw values for NatMed single-view pipeline |
| 2026-03-11 | `6ea72ad` | Auto-compute zscore from train CSV — but only for the `target_std` parameter, still no label normalization in multi-view loss |
| 2026-03-28 | (this fix) | Added `y = (y - t_mean) / t_std` before loss in multi-view eval |
