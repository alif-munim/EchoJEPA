# Prediction Averaging: NPZ Files and Study-Level Inference

**Scope: EchoJEPA-G, EchoPrime, PanEcho** (primary 3-model comparison)

See `inference-tracker-additional.md` for EchoJEPA-B, EchoJEPA-L-K, and EchoJEPA-L.

Last updated: 2026-04-12

## Overview

Each echocardiography study contains multiple video clips (different views, different time points within the same clinical encounter). The prediction averaging pipeline:

1. **Probe training:** Train a frozen probe (d=1 attentive) on single clips
2. **Clip-level inference:** Run the probe on ALL clips per study → save to `clip_outputs.npz`
3. **Study-level aggregation:** Average predictions across clips within each study → study-level metric

The NPZ files are the bridge between GPU inference (step 2) and CPU-only statistics (step 3).

---

## NPZ File Format

All NPZ files are at:
```
evals/vitg-384/nature_medicine/{uhn,mimic}/video_classification_frozen/{task}-predavg-{model}/clip_outputs.npz
```

### Regression Tasks

| Key | Shape | Description |
|-----|-------|-------------|
| `clip_predictions_all_heads` | `(N_clips, N_heads)` | Per-clip prediction from each probe head (z-scored) |
| `clip_labels` | `(N_clips,)` | Per-clip ground truth label (z-scored) |
| `clip_features_best_head` | `(N_clips, embed_dim)` | Mean-pooled encoder features for best head |
| `clip_study_ids` | `(N_clips,)` | DICOM study UID per clip |
| `best_head_idx` | `(1,)` | Index of best head (selected on val MAE) |
| `r2_per_head` | `(N_heads,)` | Clip-level R² per head |
| `pearson_per_head` | `(N_heads,)` | Clip-level Pearson per head |
| `zscore_mean` | `(1,)` | Z-score mean (for un-normalizing predictions/labels) |
| `zscore_std` | `(1,)` | Z-score std |

Example: RVSP regression, EchoJEPA-G → 100,184 clips × 12 heads, 10,015 unique studies.

### Classification Tasks

| Key | Shape | Description |
|-----|-------|-------------|
| `clip_probs_all_heads` | `(N_clips, N_heads, N_classes)` | Per-clip class probabilities from each head |
| `clip_labels` | `(N_clips,)` | Per-clip ground truth class (integer) |
| `clip_features_best_head` | `(N_clips, embed_dim)` | Mean-pooled encoder features for best head |
| `clip_study_ids` | `(N_clips,)` | DICOM study UID per clip |
| `best_head_idx` | `(1,)` | Index of best head (selected on val AUROC) |
| `auroc_per_head` | `(N_heads,)` | Clip-level AUROC per head |

Example: Diastolic function (4-class), EchoJEPA-G → 139,920 clips × 12 heads × 4 classes, 4,756 unique studies.

---

## Study-Level Aggregation (CPU-only)

```python
import numpy as np
from collections import defaultdict

data = np.load("clip_outputs.npz", allow_pickle=True)
best_head = int(data['best_head_idx'].item())
study_ids = data['clip_study_ids']
labels = data['clip_labels']

# For regression:
preds = data['clip_predictions_all_heads'][:, best_head]

# For classification:
# probs = data['clip_probs_all_heads'][:, best_head, :]  # (N_clips, N_classes)

# Average predictions per study
study_preds = defaultdict(list)
study_label = {}
for sid, pred, lab in zip(study_ids, preds, labels):
    study_preds[sid].append(pred)
    study_label[sid] = lab

studies = sorted(study_preds.keys())
agg_preds = np.array([np.mean(study_preds[s]) for s in studies])
agg_labels = np.array([study_label[s] for s in studies])

# Now compute R², Pearson, AUROC, etc. on agg_preds / agg_labels
```

For classification, average the probability vectors per study, then take argmax for predicted class or use the averaged probabilities directly for AUROC.

---

## Task Coverage (EchoJEPA-G / EchoPrime / PanEcho)

### Have predavg NPZ (study-level stats can be computed now, CPU-only):

**UHN (9 tasks, 26 NPZ files):**

| Task | G | EP | Pan |
|------|---|-----|-----|
| cardiac_output | ✅ | ✅ | ✅ |
| diastolic_function | ✅ | ✅ | ✅ |
| edv | ✅ | ✅ | ✅ |
| esv | ✅ | ✅ | ✅ |
| rv_function | ❌ | ✅ | ✅ |
| rvsp | ✅ | ✅ | ✅ |
| trajectory_lvef | ✅ | ✅ | ✅ |
| trajectory_lvef_onset | ✅ | ✅ | ✅ |
| trajectory_mr_severity_onset | ✅ | ✅ | ✅ |

**MIMIC (15 tasks, 43 NPZ files including 10 trainfeat-G variants):**

| Task | G | EP | Pan | Notes |
|------|---|-----|-----|-------|
| creatinine | ✅ | ✅ | ✅ | +trainfeat-G |
| discharge_destination | ✅ | ✅ | ✅ | +trainfeat-G |
| ef_note_extracted-xfer | ✅ | ✅ | ✅ | |
| in_hospital_mortality | ✅ | ✅ | ✅ | |
| lactate | ✅ | ✅ | ✅ | +trainfeat-G |
| los_remaining | ✅ | ✅ | ✅ | +trainfeat-G |
| lvef_structured | ✅ | ❌ | ❌ | |
| mitral_regurg-xfer | ✅ | ✅ | ✅ | |
| mortality_1yr | ✅ | ✅ | ✅ | +trainfeat-G |
| mortality_30d | ✅ | ✅ | ✅ | +trainfeat-G |
| mortality_90d | ✅ | ✅ | ✅ | +trainfeat-G |
| nt_probnp | ✅ | ✅ | ✅ | +trainfeat-G |
| readmission_30d | ✅ | ✅ | ✅ | +trainfeat-G |
| tricuspid_regurg-xfer | ✅ | ✅ | ✅ | |
| troponin_t | ✅ | ✅ | ✅ | +trainfeat-G |

**Total ready:** 69 NPZ files (26 UHN + 43 MIMIC) → study-level stats at `predictions/nature_medicine/study_level_statistics/`.

### Need GPU inference (probes trained, NPZ not generated):

**UHN (19 tasks, 55 inference runs):**

| Task | G | EP | Pan |
|------|---|-----|-----|
| aov_mean_grad | ❌ | ❌ | ❌ |
| aov_vmax | ❌ | ❌ | ❌ |
| ar_severity | ❌ | ❌ | ❌ |
| as_severity | ❌ | ❌ | ❌ |
| disease_amyloidosis | ❌ | ❌ | ❌ |
| disease_bicuspid_av | ❌ | ❌ | ❌ |
| disease_dcm | ❌ | ❌ | ❌ |
| disease_hcm | ❌ | ❌ | ❌ |
| disease_myxomatous_mv | ❌ | ❌ | ❌ |
| disease_rheumatic_mv | ❌ | ❌ | ❌ |
| disease_stemi | ❌ | ❌ | ❌ |
| lvef | ❌ | ❌ | ❌ |
| mr_severity | ❌ | ❌ | ❌ |
| mv_ee_medial | ❌ | ❌ | ❌ |
| rv_fac | ❌ | ❌ | ❌ |
| rv_function | ❌ | | |
| rv_sp | ❌ | ❌ | ❌ |
| tapse | ❌ | ❌ | ❌ |
| tr_severity | ❌ | ❌ | ❌ |

**MIMIC (1 task, 2 inference runs):**

| Task | G | EP | Pan |
|------|---|-----|-----|
| lvef_structured | ✅ | ❌ | ❌ |

**Total needed:** 57 GPU inference runs (55 UHN + 2 MIMIC).

Each run requires:
- Encoder checkpoint (on S3/EFS)
- Trained probe (`checkpoints/probes/{task}/{model}/best.pt`, all on S3)
- Test CSV with S3 video paths
- Config YAML with `val_only: true` and predavg settings

Estimated time: ~5-30 min per run depending on test set size. Total ~6-8 hours on 8 GPUs.

---

## Pipeline

```
[Probe training]     →  best.pt (trained probe weights)
       ↓                    ↓
[GPU inference]      →  clip_outputs.npz (clip-level predictions + study IDs)
       ↓                    ↓
[CPU aggregation]    →  study-level R²/AUROC/CIs (predictions/nature_medicine/study_level_statistics/)
```

Step 1 is complete for all tasks. Step 2 is the bottleneck (57 runs missing for G/EP/Pan). Step 3 is instant once step 2 is done.

---

## File Locations

| Asset | Path |
|-------|------|
| Probes (EFS) | `checkpoints/probes/{task}/{model}/best.pt` |
| Probes (S3) | `s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts/checkpoints/probes/{task}/{model}/best.pt` |
| NPZ outputs (EFS) | `evals/vitg-384/nature_medicine/{uhn,mimic}/video_classification_frozen/{task}-predavg-{model}/clip_outputs.npz` |
| Extracted CSVs | `predictions/nature_medicine/{uhn,mimic}/{task}-predavg-{model}.csv` |
| Study-level stats | `predictions/nature_medicine/study_level_statistics/{uhn,mimic}/{task}-predavg-{model}.json` |
| Probe inventory | `predictions/nature_medicine/probe_inventory.json` |
