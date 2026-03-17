# Nature Medicine — Task Tracker

Last updated: 2026-03-17 (GLS demoted to ED/Batch 8, AV mean grad + AV area promoted to Batch 2, prioritized run queue updated)

## Evaluation Protocol

**Strategy E**: d=1 attentive probes from video + prediction averaging across all clips per study.
- **5 models**: EchoJEPA-G, EchoJEPA-L, EchoJEPA-L-K, EchoPrime, PanEcho
- **Probe grid**: 12 HP combos (4 LR × 3 WD), 15 epochs. LVEF/TAPSE used 20-head grid (5 LR × 4 WD); all subsequent tasks use 12-head.
- **Convergence**: 15 epochs sufficient for all tasks tested. Extending to 20 yields <0.005 AUROC improvement. Use 15 epochs going forward.
- **Run script**: `scripts/run_uhn_probe.sh <task>`
- **Config template**: auto-generated YAML at `/tmp/nm_{task}_{model}.yaml`
- **Checkpoint output**: `evals/vitg-384/nature_medicine/uhn/video_classification_frozen/{task}-{model}/`
- **Checkpoint archive**: `checkpoints/probes/{task}/{model}/` (local) + `s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts/checkpoints/probes/{task}/{model}/` (S3)
- **CSV source**: `experiments/nature_medicine/uhn/probe_csvs/{task}/`

---

## Historic Baseline Results (Linear Probes on Mean-Pooled Embeddings)

> **Superseded by Strategy E** (d=1 attentive probes from video). These results used the old NPZ pipeline: frozen encoder → mean-pool 1568 tokens → mean-pool ~56-72 clips/study → sklearn linear probe. They serve as a reference floor — Strategy E results should improve on these across the board.

### UHN — EchoJEPA-G (in-domain, 2026-03-09)

Frozen linear probes (LogisticRegression / Ridge) on mean-pooled study-level embeddings (319,815 studies, 1408-dim). HP grid search on val, evaluation on test. Script: `scripts/run_uhn_probes.py`. Results: `results/probes/nature_medicine/uhn/`.

**Classification (26 tasks, mean AUC: 0.874):**

| Task | AUC |
|------|-----|
| AS severity | 0.947 |
| Pericardial effusion | 0.940 |
| LV systolic function | 0.936 |
| LV cavity size | 0.934 |
| Diastolic function | 0.918 |
| HCM | 0.877 |
| Amyloidosis | 0.835 |
| STEMI | 0.828 |
| Takotsubo | 0.815 |

**Regression (21 tasks, mean R²: 0.625):**

| Task | R² |
|------|-----|
| ESV | 0.853 |
| EDV | 0.798 |
| LVEF | 0.775 |
| LV mass | 0.738 |
| LA vol | 0.726 |

**Trajectory (paired-study delta prediction):**

| Task | R² |
|------|-----|
| LVEF delta | 0.456 |
| MR severity delta | 0.234 |
| TAPSE delta | 0.129 |

### UHN — EchoJEPA-L (out-of-domain — complete failure)

- All 26 classification tasks: AUC ~0.50 (chance)
- All 21 regression tasks: R² ≤ 0 (worse than mean predictor)
- All 5 trajectory tasks: R² < 0
- **Root cause**: L pretrained on ~7K MIMIC studies → extreme embedding concentration on UHN (mean pairwise cosine 0.998, var/dim ratio 0.0005 vs G's 0.293). Confirmed L works in-domain on MIMIC (HF AUC 0.761).

### MIMIC — All 7 Models (161 runs, 2026-03-08/09)

23 tasks × 7 models. Frozen linear probes, HP selection on val, evaluation on test. Script: `scripts/run_mimic_probes_v2.py`. Results: `results/probes/nature_medicine/mimic_v2/`.

**Mortality & Outcomes (AUC-ROC):**

| Model | 30-day | 90-day | 1-year | In-hospital | Readmit 30d | ICU Transfer |
|-------|--------|--------|--------|-------------|-------------|--------------|
| EchoJEPA-G | 0.822 | 0.801 | 0.769 | 0.787 | 0.586 | 0.486 |
| EchoJEPA-L | 0.797 | 0.783 | 0.766 | 0.709 | 0.567 | 0.526 |
| EchoJEPA-L-K | 0.797 | 0.771 | 0.781 | 0.760 | 0.552 | 0.555 |
| EchoMAE-L | 0.739 | 0.753 | 0.722 | 0.688 | 0.583 | 0.501 |
| **EchoPrime** | **0.902** | **0.880** | **0.824** | **0.831** | **0.609** | **0.593** |
| PanEcho | 0.817 | 0.808 | 0.779 | 0.787 | 0.550 | 0.443 |
| EchoFM | 0.745 | 0.668 | 0.644 | 0.628 | 0.559 | 0.503 |

**Disease Detection (AUC-ROC):**

| Model | AFib | Heart Failure | Amyloidosis | DCM | HCM | STEMI | Takotsubo |
|-------|------|---------------|-------------|-----|-----|-------|-----------|
| **EchoJEPA-G** | **0.847** | **0.825** | 0.741 | 0.734 | 0.639 | 0.631 | 0.391 |
| EchoJEPA-L | 0.781 | 0.761 | 0.669 | 0.788 | 0.613 | 0.611 | 0.423 |
| EchoJEPA-L-K | 0.830 | 0.808 | 0.608 | 0.806 | 0.651 | **0.655** | 0.362 |
| EchoMAE-L | 0.731 | 0.701 | 0.695 | 0.687 | 0.623 | 0.577 | 0.366 |
| EchoPrime | 0.834 | 0.816 | **0.859** | **0.852** | **0.678** | 0.577 | **0.614** |
| PanEcho | 0.806 | 0.790 | 0.728 | 0.651 | 0.665 | 0.489 | 0.540 |
| EchoFM | 0.626 | 0.618 | 0.545 | 0.432 | 0.493 | 0.496 | 0.618 |

**Regression:**

| Model | EF R² | EF r | NT-proBNP R² | NT-proBNP r | Creatinine R² | TnT R² |
|-------|-------|------|-------------|-------------|---------------|--------|
| EchoJEPA-G | 0.444 | 0.676 | 0.144 | 0.403 | 0.067 | 0.000 |
| EchoJEPA-L | 0.271 | 0.546 | 0.097 | 0.334 | 0.012 | -0.041 |
| EchoJEPA-L-K | 0.360 | 0.620 | -0.027 | 0.276 | 0.002 | -0.030 |
| EchoMAE-L | 0.215 | 0.471 | -0.012 | 0.221 | 0.012 | -0.102 |
| **EchoPrime** | **0.547** | **0.747** | **0.155** | 0.395 | **0.076** | **0.070** |
| PanEcho | 0.351 | 0.624 | 0.191 | **0.444** | 0.083 | 0.010 |
| EchoFM | 0.052 | 0.242 | -0.049 | 0.006 | 0.003 | -0.000 |

**Key findings:**
1. EchoPrime dominates mortality/outcomes and structural disease detection — text-supervised contrastive geometry is ideal for linear probes
2. EchoJEPA-G leads AFib (0.847) and HF (0.825) — broad multi-view phenotypes where 18M pretraining helps
3. JEPA > MAE confirmed: EchoJEPA-L beats EchoMAE-L on 19/23 tasks (same ViT-L, same data, same probe)
4. EchoFM weakest overall (near-collapsed representations)
5. ICU transfer uninformative (0.486-0.593) across all models
6. Linear probes structurally favor contrastive models (see `probe-results.md` for 5-factor analysis)

### MIMIC — CY Standalone Results (EchoJEPA-G only, sklearn, 2026-03-13)

Separate analysis by CY with prediction averaging on study-level embeddings (mean-pool across clips, then average predictions). Ensemble of multiple classifiers.

| Task | AUC |
|------|-----|
| Mortality 30d | 0.912 |
| Mortality 90d | 0.883 |
| Mortality 1yr | 0.846 |
| In-hospital mortality | 0.875 |
| Readmission 30d | 0.634 |
| ICU transfer | 0.570 |

**EHR-only baselines (XGBoost + TabPFN, 54 features):**
- Mortality 1yr: 0.856 (XGBoost), echo ≈ EHR
- Readmission 30d: 0.624 (XGBoost), echo > EHR
- Prediction averaging validated: +0.08-0.09 AUC over single mean-pool embedding

**Pending**: Combined echo+EHR model, acuity conditioning (H_confound).

---

## Completed Tasks — Per-Task Protocol

### LVEF (regression)

| Setting | Value |
|---------|-------|
| Task type | Regression (smooth L1 loss) |
| Views | A4C, A2C |
| B-mode only | No |
| Study sampling | Yes (`DistributedStudySampler`, 1 random clip/study/epoch) |
| Class balance | N/A (regression) |
| Train | 406,634 clips, 23,863 studies |
| Val | 215,714 clips, 12,431 studies |
| Test | 266,516 clips, 15,097 studies |
| Z-score | mean=57.914, std=10.979 |
| HP grid | 20 heads (5 LR × 4 WD): LR ∈ {1e-3, 5e-4, 1e-4, 5e-5, 1e-5}, WD ∈ {0.001, 0.01, 0.1, 0.4} |
| Epochs | 15 (G ran 20) |
| Batch size | 2 per GPU × 4 GPUs = 8 effective |
| Val batch size | 64 (EchoPrime: 16) |
| Warmup | 2 epochs cosine |
| Inference | Single random clip per study per val epoch. **No prediction averaging.** |
| CSV source | `experiments/nature_medicine/uhn/probe_csvs/lvef/train_vf.csv` |

### TAPSE (regression)

| Setting | Value |
|---------|-------|
| Task type | Regression (smooth L1 loss) |
| Views | A4C only |
| B-mode only | No |
| Study sampling | Yes |
| Class balance | N/A (regression) |
| Train | 280,637 clips, 25,263 studies |
| Val | 151,624 clips, 13,311 studies |
| Test | 191,567 clips, 16,336 studies |
| Z-score | mean=1.955, std=0.497 |
| HP grid | 20 heads (5 LR × 4 WD): same as LVEF |
| Epochs | 15 (G/L/EchoPrime ran 20) |
| Batch size | 2 per GPU × 4 GPUs = 8 effective |
| Inference | Single random clip per study per val epoch. **No prediction averaging.** |
| CSV source | `experiments/nature_medicine/uhn/probe_csvs/tapse/train_vf.csv` |

### MR Severity (classification, B-mode only)

| Setting | Value |
|---------|-------|
| Task type | Classification, 5-class ordinal (none/trace/mild/moderate/severe) |
| Views | A4C, A2C, A3C, PLAX |
| B-mode only | **Yes** (colour/spectral/tissue Doppler excluded via color classifier) |
| Study sampling | Yes |
| Class balance | `class_balance_ratio=3` (cap each class at 3× minority count). 89,597 → ~29K studies |
| Train | 1,648,090 clips, 89,597 studies (pre-balance) |
| Val | 225,770 clips, 11,703 studies |
| Test | 399,594 clips, 20,825 studies |
| HP grid | 12 heads (4 LR × 3 WD): LR ∈ {5e-4, 1e-4, 5e-5, 1e-5}, WD ∈ {0.001, 0.01, 0.1} |
| Epochs | 15 (G/L ran 20; L-K ran 19) |
| Batch size | 2 per GPU × 4 GPUs = 8 effective |
| Val batch size | 64 (EchoPrime: 16) |
| Warmup | 3 epochs cosine |
| num_workers | 4 (reduced from 8 to avoid shm exhaustion) |
| Inference | Single random clip per study per val epoch. **No prediction averaging.** |
| CSV source | `experiments/nature_medicine/uhn/probe_csvs/mr_severity/train_vf.csv` |

### AS Severity (classification, B-mode only)

| Setting | Value |
|---------|-------|
| Task type | Classification, 4-class ordinal (none+sclerosis/mild/moderate/severe) |
| Views | PLAX, PSAX-AV, A3C |
| B-mode only | **Yes** |
| Study sampling | Yes |
| Class balance | `class_balance_ratio=3`. 130,863 → ~22K studies |
| Train | 1,487,708 clips, 130,863 studies (pre-balance) |
| Val | 153,372 clips, 12,869 studies |
| Test | 270,360 clips, 23,176 studies |
| HP grid | 12 heads (4 LR × 3 WD): same as MR severity |
| Epochs | 15 (G/L/L-K ran 20; EchoPrime ran 19) |
| Batch size | 2 per GPU × 4 GPUs = 8 effective |
| Val batch size | 64 (EchoPrime: 16) |
| Warmup | 3 epochs cosine |
| num_workers | 4 |
| Inference | Single random clip per study per val epoch. **No prediction averaging.** |
| CSV source | `experiments/nature_medicine/uhn/probe_csvs/as_severity/train_vf.csv` |

## Running / Queued Tasks — Per-Task Protocol

### TR Severity (classification, B-mode only) — ALL 5 MODELS DONE

| Setting | Value |
|---------|-------|
| Task type | Classification, 5-class ordinal (none/trivial-trace/mild/moderate/severe) |
| Views | A4C, Subcostal, PLAX |
| B-mode only | **Yes** |
| Study sampling | Yes |
| Class balance | `class_balance_ratio=3` |
| Train | 1,365,676 clips, 95,162 studies (pre-balance) |
| Val | 189,236 clips, 12,495 studies |
| Test | 340,626 clips, 22,652 studies |
| HP grid | 12 heads (4 LR × 3 WD) |
| Epochs | 15 |
| Batch size | 2 per GPU × 4 GPUs = 8 effective |
| Val batch size | 64 (EchoPrime: 16) |
| Warmup | 2 epochs cosine |
| num_workers | 4 |
| Inference | Single random clip per study per val epoch. **No prediction averaging.** |
| CSV source | `experiments/nature_medicine/uhn/probe_csvs/tr_severity/train_vf.csv` |

### AR Severity (classification, B-mode only) — QUEUED

| Setting | Value |
|---------|-------|
| Task type | Classification, 5-class ordinal (none/trace/mild/moderate/severe) |
| Views | A4C, A2C, A3C, PLAX |
| B-mode only | **Yes** |
| Study sampling | Yes |
| Class balance | `class_balance_ratio=3` |
| Train | 969,896 clips, 51,212 studies (pre-balance) |
| Val | 172,027 clips, 8,781 studies |
| Test | 285,696 clips, 14,756 studies |
| HP grid | 12 heads (4 LR × 3 WD) |
| Epochs | 15 |
| Batch size | 2 per GPU × 4 GPUs = 8 effective |
| Val batch size | 64 (EchoPrime: 16) |
| Warmup | 2 epochs cosine |
| num_workers | 4 |
| Inference | Single random clip per study per val epoch. **No prediction averaging.** |
| CSV source | `experiments/nature_medicine/uhn/probe_csvs/ar_severity/train_vf.csv` |

### E/e' (mv_ee, regression, B-mode only) — QUEUED

| Setting | Value |
|---------|-------|
| Task type | Regression (smooth L1 loss) |
| Views | A4C |
| B-mode only | **Yes** (Doppler-derived measurement inferred from structure) |
| Study sampling | Yes |
| Class balance | N/A (regression) |
| Train | 71,562 clips, 10,775 studies (view-filtered + B-mode) |
| Val | 26,845 clips, 3,976 studies |
| Test | 33,701 clips, 4,886 studies |
| Z-score | mean=8.735, std=4.338 |
| HP grid | 12 heads (4 LR × 3 WD) |
| Epochs | 15 |
| Batch size | 2 per GPU × 4 GPUs = 8 effective |
| Val batch size | 64 (EchoPrime: 16) |
| Warmup | 2 epochs cosine |
| num_workers | 4 |
| Inference | Single random clip per study per val epoch. **No prediction averaging.** |
| CSV source | `experiments/nature_medicine/uhn/probe_csvs/mv_ee/train_vf.csv` |

### RVSP (regression, B-mode only) — QUEUED

| Setting | Value |
|---------|-------|
| Task type | Regression (smooth L1 loss) |
| Views | A4C, Subcostal |
| B-mode only | **Yes** (Doppler-derived measurement inferred from structure) |
| Study sampling | Yes |
| Class balance | N/A (regression) |
| Train | 139,861 clips, 14,772 studies (view-filtered + B-mode) |
| Val | 71,602 clips, 7,219 studies |
| Test | 100,183 clips, 10,015 studies |
| Z-score | mean=35.099, std=13.907 |
| HP grid | 12 heads (4 LR × 3 WD) |
| Epochs | 15 |
| Batch size | 2 per GPU × 4 GPUs = 8 effective |
| Val batch size | 64 (EchoPrime: 16) |
| Warmup | 2 epochs cosine |
| num_workers | 4 |
| Inference | Single random clip per study per val epoch. **No prediction averaging.** |
| CSV source | `experiments/nature_medicine/uhn/probe_csvs/rvsp/train_vf.csv` |

### RV S' (rv_sp, regression) — QUEUED

| Setting | Value |
|---------|-------|
| Task type | Regression (smooth L1 loss) |
| Views | A4C, Subcostal |
| B-mode only | No |
| Study sampling | Yes |
| Class balance | N/A (regression) |
| Train | 391,778 clips, 24,849 studies (view-filtered) |
| Val | 210,399 clips, 13,011 studies |
| Test | 264,046 clips, 16,001 studies |
| Z-score | mean=0.116, std=0.031 |
| HP grid | 12 heads (4 LR × 3 WD) |
| Epochs | 15 |
| Batch size | 2 per GPU × 4 GPUs = 8 effective |
| Val batch size | 64 (EchoPrime: 16) |
| Warmup | 2 epochs cosine |
| num_workers | 4 |
| Inference | Single random clip per study per val epoch. **No prediction averaging.** |
| CSV source | `experiments/nature_medicine/uhn/probe_csvs/rv_sp/train_vf.csv` |

### RV FAC (rv_fac, regression) — QUEUED

| Setting | Value |
|---------|-------|
| Task type | Regression (smooth L1 loss) |
| Views | A4C |
| B-mode only | No |
| Study sampling | Yes |
| Class balance | N/A (regression) |
| Train | 80,046 clips, 6,395 studies (view-filtered) |
| Val | 51,177 clips, 4,025 studies |
| Test | 72,232 clips, 5,666 studies |
| Z-score | mean=37.096, std=10.281 |
| HP grid | 12 heads (4 LR × 3 WD) |
| Epochs | 15 |
| Batch size | 2 per GPU × 4 GPUs = 8 effective |
| Val batch size | 64 (EchoPrime: 16) |
| Warmup | 2 epochs cosine |
| num_workers | 4 |
| Inference | Single random clip per study per val epoch. **No prediction averaging.** |
| CSV source | `experiments/nature_medicine/uhn/probe_csvs/rv_fac/train_vf.csv` |

### New-Onset Cardiomyopathy (trajectory_lvef_onset, binary) — DONE (all 5 models trained + inference)

| Setting | Value |
|---------|-------|
| Task type | Binary classification: 0=stable (future EF ≥ 50), 1=decline (future EF < 50) |
| Population | **Baseline EF ≥ 50 only** (preserved EF at index echo) |
| Views | A4C, A2C (view-filtered at CSV build time) |
| B-mode only | No |
| Study sampling | **Yes** (all clips per study_1 in CSV, sampler picks 1/epoch) |
| Class balance | `class_balance_ratio=3` |
| Train | 33,705 clips, 1,932 studies (1,763 stable / 169 decline = 8.7%) |
| Val | 49,952 clips, 2,752 studies (2,571 stable / 181 decline = 6.6%) |
| Test | 111,974 clips, 6,087 studies (5,634 stable / 453 decline = 7.4%) |
| Source pairs | 10,808 (from 14,235 total, filtered to baseline EF ≥ 50) |
| HP grid | 12 heads (4 LR × 3 WD) |
| Epochs | 15 |
| Batch size | 2 per GPU × 4 GPUs = 8 effective |
| Val batch size | 64 (EchoPrime: 16) |
| Warmup | 2 epochs cosine |
| num_workers | 4 |
| Inference | Single random clip per study per val epoch. Prediction averaging pending. |
| CSV source | `experiments/nature_medicine/uhn/probe_csvs/trajectory_lvef_onset/train.csv` |
| Build command | `python build_trajectory_csvs.py --task trajectory_lvef --onset --baseline_min 50 --future_below 50` |

**Clinical framing**: From an apparently normal echocardiogram (EF ≥ 50%), can the model identify patients who will develop reduced EF (< 50%) within 1 year? This tests whether frozen representations encode subclinical dysfunction beyond what the EF number captures.

**Why this reframing** (supersedes V1/V2 delta classification — see `TRAJECTORY_LVEF_EXPERIMENTS.md`):
- Delta prediction (V1 AUROC 0.649, V2 AUROC 0.610) failed because change is driven by external factors (treatment, regression to mean) not visible in a single echo
- Restricting to baseline EF ≥ 50 controls for baseline EF, forcing the model to find *other* visual features
- Event rate (7-9%) is stable across all time windows (30-89d, 90-179d, 180-365d) — no need to stratify
- Binary task is cleaner and more clinically actionable than 3-class delta

---

## Completed Runs — Summary

> **Checkpoint status**: LVEF, TAPSE, MR severity, AS severity, and AV Vmax G/L/L-K checkpoint directories were lost (cause unknown — see `claude/dev/bugs/007-checkpoint-loss.md`). Historical results below are from training logs. These tasks need retraining. Run script now archives to `checkpoints/probes/` + S3 after each model.

### Regression Tasks

| Task | Model | Best Val R² | Best Pearson | Best MAE | Best Head (lr, wd) | Best Epoch | Epochs | Checkpoint |
|------|-------|------------|-------------|---------|-------------------|-----------|--------|------------|
| LVEF | echojepa-g | **0.720** | **0.849** | 4.462 | 13 (5e-5, 0.01) | 17 | 20 | LOST |
| LVEF | echojepa-l | 0.415 | 0.646 | 6.152 | 5 (1e-4, 0.1) | 14 | 15 | LOST |
| LVEF | echojepa-l-k | 0.582 | 0.763 | 5.265 | 4 (1e-4, 0.01) | 13 | 15 | LOST |
| LVEF | echoprime | 0.559 | 0.751 | 5.399 | 11 (1e-5, 0.1) | 13 | 15 | LOST |
| LVEF | panecho | 0.555 | 0.746 | 5.442 | 6 (5e-5, 0.001) | 15 | 15 | LOST |
| TAPSE | echojepa-g | NaN | NaN | 0.264 | 13 (5e-5, 0.01) | 15 | 20 | LOST (also **BUG**) |
| TAPSE | echojepa-l | 0.323 | 0.572 | 0.325 | 8 (1e-4, 0.001) | 18 | 20 | LOST |
| TAPSE | echojepa-l-k | **0.450** | **0.671** | 0.292 | 6 (5e-5, 0.001) | 14 | 15 | LOST |
| TAPSE | echoprime | 0.343 | 0.588 | 0.321 | 17 (1e-5, 0.01) | 18 | 20 | LOST |
| TAPSE | panecho | 0.311 | 0.560 | 0.330 | 9 (1e-5, 0.001) | 15 | 15 | LOST |
| AV Vmax | echojepa-g | **0.582** | **0.763** | -- | -- | 13 | 15 | LOST |
| AV Vmax | echojepa-l | 0.232 | 0.487 | -- | -- | 14 | 15 | LOST |
| AV Vmax | echojepa-l-k | 0.388 | 0.623 | -- | -- | 14 | 15 | LOST |
| AV Vmax | echoprime | 0.476 | 0.692 | -- | -- | 15 | 15 | archived |
| AV Vmax | panecho | 0.390 | 0.627 | -- | -- | 15 | 15 | archived |

### Classification Tasks (B-mode only)

| Task | Model | Best AUROC | Best Acc | Best Bal Acc | Best Kappa | Best Head (lr, wd) | Best Epoch | Epochs | Checkpoint |
|------|-------|-----------|---------|-------------|-----------|-------------------|-----------|--------|------------|
| MR sev. | echojepa-g | **0.860** | 43.75 | 0.561 | 0.280 | 6 (5e-5, 0.001) | 17 | 20 | LOST |
| MR sev. | echojepa-l | 0.771 | 33.75 | 0.428 | 0.184 | 7 (5e-5, 0.01) | 17 | 20 | LOST |
| MR sev. | echojepa-l-k | 0.803 | 36.08 | 0.473 | 0.210 | 7 (5e-5, 0.01) | 19 | 19 | LOST |
| MR sev. | echoprime | 0.770 | 30.58 | 0.416 | 0.160 | 0 (5e-4, 0.001) | 4 | 15 | LOST |
| MR sev. | panecho | 0.724 | 23.87 | 0.373 | 0.106 | 0 (5e-4, 0.001) | 4 | 9 | LOST |
| AS sev. | echojepa-g | **0.908** | 70.51 | 0.594 | 0.432 | 5 (1e-4, 0.1) | 16 | 20 | LOST |
| AS sev. | echojepa-l | 0.786 | 54.25 | 0.427 | 0.239 | 3 (1e-4, 0.001) | 19 | 20 | LOST |
| AS sev. | echojepa-l-k | 0.821 | 53.26 | 0.467 | 0.235 | 6 (5e-5, 0.001) | 18 | 20 | LOST |
| AS sev. | echoprime | 0.827 | 57.00 | 0.496 | 0.253 | 10 (1e-5, 0.01) | 16 | 19 | LOST |
| AS sev. | panecho | 0.762 | 34.65 | 0.420 | 0.086 | 0 (5e-4, 0.001) | 2 | 15 | LOST |
| TR sev. | echojepa-g | **0.837** | 41.95 | 0.530 | 0.258 | -- | 15 | 15 | archived |
| TR sev. | echojepa-l | 0.753 | 32.10 | 0.401 | 0.167 | -- | 15 | 15 | archived |
| TR sev. | echojepa-l-k | 0.786 | 35.74 | 0.469 | 0.201 | -- | 15 | 15 | archived |
| TR sev. | echoprime | 0.758 | 33.38 | 0.399 | 0.165 | -- | 15 | 15 | archived |
| TR sev. | panecho | 0.715 | -- | -- | -- | -- | 15 | 15 | archived |

### Trajectory / Onset Tasks (Binary Classification)

#### Training Results (Val AUROC — single random clip per study per epoch)

| Task | Model | Best Val AUROC | Best Val Bal Acc | Best Val Kappa | Best Epoch | Epochs | Status |
|------|-------|---------------|-----------------|---------------|-----------|--------|--------|
| onset (V3) | echojepa-g | **0.733** | 0.666 | 0.154 | 15 | 15 | DONE |
| onset (V3) | echoprime | 0.700 | 0.622 | 0.132 | 13 | 15 | DONE |
| onset (V3) | panecho | 0.698 | 0.631 | 0.139 | 13 | 15 | DONE |
| onset (V3) | echojepa-l-k | 0.596 | 0.563 | 0.089 | 13 | 15 | DONE |
| onset (V3) | echojepa-l | 0.516 | 0.500 | 0.000 | 13 | 15 | DONE |

#### Inference Results (Test AUROC — prediction averaging across all clips per study)

| Task | Model | Test AUROC | Test Bal Acc | Test Kappa | Test Acc | Status |
|------|-------|-----------|-------------|-----------|---------|--------|
| onset (V3) | echojepa-g | **0.793** | 0.700 | 0.218 | 79.81 | DONE |
| onset (V3) | echoprime | 0.776 | 0.661 | 0.254 | 84.17 | DONE |
| onset (V3) | panecho | 0.759 | 0.549 | 0.147 | 89.25 | DONE |
| onset (V3) | echojepa-l-k | 0.677 | 0.500 | 0.000 | 91.82 | DONE |
| onset (V3) | echojepa-l | 0.514 | 0.500 | 0.000 | 92.05 | DONE (at chance, NCCL crash during inference but result valid) |

**Key findings:**
- **G test AUROC 0.793 passes the 0.75 decision gate -> Pillar 3 headline confirmed**
- **EchoPrime 0.776 and PanEcho 0.759** -- both strong, much closer to G than on hemodynamic tasks (+1.7pp and +3.4pp vs +8-10pp gap on valve severity)
- Prediction averaging boosts all models substantially: G +0.060, EchoPrime +0.076, PanEcho +0.061, L-K +0.081
- L-K predicts all-negative on test (bal_acc=0.500, kappa=0.000) despite 0.596 val AUROC -- overfitting to val set's class distribution
- L at chance on both training (0.516) and test (0.514) -- MIMIC-pretrained L lacks prognostic features for UHN trajectory
- **Model ranking differs from hemodynamic tasks**: EchoPrime nearly matches G on trajectory, suggesting text-supervised pretraining captures prognostic features. JEPA scale advantage is smaller for trajectory prediction than for hemodynamic inference.

#### L-K onset collapse analysis

L-K test AUROC 0.677 with bal_acc=0.500 and kappa=0.000 means it **never predicts the positive class**. The AUROC > 0.5 indicates the predicted probabilities have some ranking ability (patients who will decline get slightly higher scores), but the scores never cross the 0.5 decision threshold. The 91.82% accuracy equals the stable-class prevalence -- the model predicts all-stable. This is not a bug; it reflects genuinely weak signal at L-K's scale for this task.

#### JEPA objective: scale-efficiency tradeoff across task types

The G-vs-EchoPrime gap varies dramatically with task difficulty:

| Task type | G | L-K | EchoPrime | G-vs-EP gap |
|-----------|---|-----|-----------|-------------|
| Hemodynamics (cross-sectional structure) | 0.860-0.908 | 0.786-0.821 | 0.770-0.827 | +3-9pp |
| Standard regression (LVEF R2) | 0.720 | 0.582 | 0.559 | +16pp |
| Trajectory prognosis (onset) | 0.793 | 0.677 | 0.776 | **+1.7pp** |

**JEPA's masking objective** naturally learns spatial structure, motion patterns, and appearance -- exactly what hemodynamic inference needs. G dominates there (+8-10pp). But predicting future cardiomyopathy from an apparently-normal echo requires **subtle subclinical features** (early wall motion abnormalities, diastolic changes, myocardial texture) that predict decline before EF drops.

**Key insights:**

1. **JEPA at G scale (1,012M params, 18M echos) is the best approach on all tasks** -- sufficient data volume means the model sees enough "looked normal now, declined later" cases to learn prognostic patterns through masking alone.

2. **JEPA at L-K scale captures structure but not prognosis** -- Kinetics pretraining (220 epochs) + echo annealing (55 epochs) gives strong structural features (2nd on MR/TR severity) but misses the subtle prognostic signal. This is not a training bug; it's the inherent limitation of learning prognostic features from masking alone without sufficient echo-specific data.

3. **Text supervision (EchoPrime/CLIP) is more sample-efficient for prognostic features** -- when a cardiologist writes "subtle septal hypokinesis" or "mild diastolic dysfunction" in a report, CLIP contrastive training forces the vision encoder to represent exactly those features. This gives EchoPrime a shortcut to prognostic features that JEPA needs massive scale to discover.

4. **L's failure (0.514 = chance) is simply insufficient data** -- 7K MIMIC studies is far too little for SSL pretraining.

5. **What you pretrain on matters as much as scale** -- L-K has 300M params and saw Kinetics + echo data, but can't match EchoPrime's 0.776 on onset. Domain-relevant supervision (text or labels) is more efficient than domain-agnostic SSL for prognostic tasks.

**Paper framing**: G leads on all tasks (headline). But the margin depends on task difficulty -- largest on hemodynamic inference (structure), smallest on prognosis (trajectory). Text supervision provides a more efficient path to prognostic features, motivating multimodal JEPA + report pretraining as future work.

| delta ±10 (V1) | echojepa-g | 0.649 | 15 | superseded | 3-class, 30-365d |
| delta ±10 (V1) | panecho | 0.649 | 15 | superseded | |
| delta ±10 (V1) | echoprime | 0.642 | 15 | superseded | |
| delta ±10 (V1) | echojepa-l-k | 0.614 | 15 | superseded | |
| delta ±10 (V1) | echojepa-l | 0.608 | 15 | superseded | |
| delta ±8 (V2) | echojepa-g | 0.610 | 15 | superseded | 3-class, 90-365d. Worse than V1. |
| delta ±8 (V2) | echojepa-l | 0.545 | 15 | superseded | |

Full experiment log: `experiments/nature_medicine/uhn/TRAJECTORY_LVEF_EXPERIMENTS.md`

**MR/AS settings**: B-mode only view-filtered CSVs, `class_balance_ratio=3` (cap each class at 3× minority), `study_sampling=true`, `num_workers=4`.
MR views: A4C, A2C, A3C, PLAX. AS views: PLAX, PSAX-AV, A3C. Studies after balancing: MR ~29K, AS ~22K.

**AV Vmax settings**: B-mode only, views PLAX/A3C/PSAX-AV. Regression canary for hemodynamic pillar — confirms frozen representations predict continuous Doppler-derived measurements from structure alone. G leads by +10pp R² over next-best (EchoPrime). EchoPrime competitive (0.476) likely due to CLIP text supervision. G/L/L-K log_r0.csv lost (run script deletes checkpoint dirs when cycling models); results extracted from raw run log via grep. EchoPrime crashed mid-run (shm exhaustion) but checkpoint was saved; PanEcho completed normally after.

### Surviving Probe Checkpoints

All surviving checkpoints archived to `checkpoints/probes/{task}/{model}/` and S3. LVEF, TAPSE, MR severity, AS severity, and AV Vmax G/L/L-K directories are gone (see Bug 007).

| Task | Model | Archive path | Status |
|------|-------|-------------|--------|
| AV Vmax | echoprime | `checkpoints/probes/aov_vmax/echoprime/` | archived |
| AV Vmax | panecho | `checkpoints/probes/aov_vmax/panecho/` | archived |
| TR sev. | echojepa-g | `checkpoints/probes/tr_severity/echojepa-g/` | archived |
| TR sev. | echojepa-l | `checkpoints/probes/tr_severity/echojepa-l/` | archived |
| TR sev. | echojepa-l-k | `checkpoints/probes/tr_severity/echojepa-l-k/` | archived |
| TR sev. | echoprime | `checkpoints/probes/tr_severity/echoprime/` | archived |
| TR sev. | panecho | `checkpoints/probes/tr_severity/panecho/` | archived |
| Traj LVEF (3-class) | echojepa-g | `checkpoints/probes/trajectory_lvef/echojepa-g/` | archived |
| Traj LVEF (3-class) | echojepa-l | `checkpoints/probes/trajectory_lvef/echojepa-l/` | archived |
| Traj LVEF (3-class) | echojepa-l-k | `checkpoints/probes/trajectory_lvef/echojepa-l-k/` | archived |
| Onset | echojepa-g | `checkpoints/probes/trajectory_lvef_onset/echojepa-g/` | archived |
| Onset | echojepa-l | `checkpoints/probes/trajectory_lvef_onset/echojepa-l/` | archived |
| Onset | echojepa-l-k | `checkpoints/probes/trajectory_lvef_onset/echojepa-l-k/` | archived |
| Onset | echoprime | `checkpoints/probes/trajectory_lvef_onset/echoprime/` | archived |
| Onset | panecho | `checkpoints/probes/trajectory_lvef_onset/panecho/` | archived |

S3 restore: `aws s3 sync s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts/checkpoints/probes/ checkpoints/probes/`

Dropped (EchoMAE — excluded from Nature Medicine):
- `lvef-echomae/` — LVEF R²≈0
- `tapse-echomae/` — TAPSE R²≈0

Archived (old 20-head HP grid, superseded by 12-head):
- `lvef-echojepa-l.bak_20head/`
- `lvef-echojepa-l-k.bak_20head/`
- `tapse-echojepa-l-k.bak_20head/`
- `tapse-panecho.bak_20head/`

**Notes:**
- MAE for LVEF is in raw EF points (%). MAE for TAPSE is in Z-score units (multiply by std=0.497 for cm).
- TAPSE echojepa-g: R²/Pearson NaN on all 20 epochs. MAE is decreasing normally (model is learning). Likely metric computation bug — needs investigation.
- EchoJEPA-L still improving at epoch 15 for both tasks (not plateaued).
- MR/AS severity: ALL models extract hemodynamic signal from B-mode video (PanEcho 0.724-0.762). Story is "scale advantage" (+8-9pp for G), not unique capability.
- MR/AS EchoPrime/PanEcho: best heads are high-LR (5e-4), peak early (epoch 2-4). JEPA models prefer lower LR (5e-5), peak later (epoch 16-19).
- MR/AS used `class_balance_ratio=3` to cap severe class imbalance (AS: 131K→22K studies, MR: 89K→29K studies).

---

## Known Issues

1. **tapse-echojepa-g NaN metrics**: Val R² and Pearson are NaN for all 20 epochs. MAE looks correct (0.298→0.264). Possible causes: all_gather bug on multi-GPU, Z-score denormalization issue in metric computation, or constant predictions on some ranks. Needs re-run or debugging.

2. **EchoMAE dropped**: Checkpoint was pretrained with lr=3.5e-6 (~170× below standard). TAPSE R²≈0, LVEF R²≈0. Excluded from Nature Medicine; cite ICML for JEPA-vs-MAE comparison.

3. **Shared memory exhaustion (shmmni=4096)**: Running 2 concurrent DDP jobs with `num_workers=8` accumulates orphaned `/dev/shm/torch_*`, `__KMP_REGISTERED_LIB_*`, `sem.loky-*`, `sem.mp-*` files across crashes, hitting the kernel segment limit. **Fix**: `num_workers: 4` + cleanup between model runs (`rm -f /dev/shm/torch_* ...`). Applied in `scripts/run_uhn_probe.sh`.

4. **CSV header duplication on resume**: eval.py rewrites the CSV header each time it resumes, creating duplicate header rows. Epoch counting must use `grep -c "^[0-9]"` to skip headers, not `tail -n +2 | wc -l`.

5. **Prediction averaging implemented** (2026-03-16): Added to `evals/video_classification_frozen/eval.py`. Auto-enables when `val_only=True` and `study_sampling=True`. Disables `DistributedStudySampler` for val (scores all clips), gathers study IDs across ranks via `dist.all_gather_object`, averages predictions per study, then computes metrics. All current training run metrics are still single-clip; pred avg requires a separate val-only pass with the best checkpoint.

6. **Orphaned DDP worker processes**: When a parent run script is killed (`kill PID`), DDP child processes (one per GPU) may survive as orphans attached to init (ppid=1), holding GPU memory indefinitely. Symptom: `nvidia-smi` shows GPUs occupied with no matching Python process in `ps`. Fix: identify orphan PIDs via `nvidia-smi --query-compute-apps=pid`, then `kill -9`. Always check GPU memory before starting new runs.

7. **Checkpoint loss (Bug 007)**: All checkpoint directories for LVEF (5), TAPSE (5), MR severity (5), AS severity (5), and AV Vmax G/L/L-K (3) = 23 runs are gone. Cause unknown (no `rm` in any script, bash history, or Claude session). Historical results preserved in training logs. **Fix applied**: `run_uhn_probe.sh` now archives to `checkpoints/probes/{task}/{model}/` + S3 after each model. See `claude/dev/bugs/007-checkpoint-loss.md`.

---

## Failed Experiments

### Trajectory LVEF — Delta Regression (ABANDONED 2026-03-16)

**Setup**: Predict continuous Z-scored delta EF from baseline echo clip. 11,501 pairs (30-365d between echos), all clips per study_1, study_sampling enabled. EchoJEPA-G, 30 epochs.

**Result**: Best R²=0.043, Pearson=0.214 (epoch 24). Model barely beats predicting the mean.

**Root cause analysis**:
1. **Variable time horizon (30-365d)**: Model doesn't know prediction window. Same baseline should predict different deltas at 60d vs 300d. Regression requires precise continuous output → time ambiguity destroys signal.
2. **Near-zero-mean target**: Delta mean=0.24, std=7.85. Most patients don't change much (54% within ±5 EF). Probe fits a very noisy near-zero signal.
3. **Regression to the mean dominates**: r(baseline_EF, delta) = -0.371. Low EF → improve, high EF → decline. Model achieves R²=0.043 despite baseline EF alone explaining ~14% of variance → model is doing WORSE than a trivial "predict delta from baseline EF" linear regression.

**Key data**:
- Baseline EF 10-35 (n=754): mean delta +5.42 (improve)
- Baseline EF 35-50 (n=1161): mean delta +2.66
- Baseline EF 50-60 (n=2886): mean delta +1.87
- Baseline EF 60-70 (n=3495): mean delta -1.92
- Baseline EF 70-85 (n=472): mean delta -8.21 (decline)

**Epoch-level performance (EchoJEPA-G)**:
```
Epoch  R²       Pearson    Notes
  7   -0.003    0.106
  8   -0.004    0.101
  9    0.003    0.158
 10   -0.103    0.101
 11    0.008    0.113
 12    0.004    0.119
 13   -0.013    0.107
 14    0.002    0.127
 15   -0.018    0.129
 16    0.007    0.147
 17    0.021    0.146      Best R² before ep 24
 18   -0.007    0.120
 19    0.017    0.158
 20   -0.037    0.128
 21    0.004    0.137
 22    0.027    0.178
 23   -0.005    0.163
 24    0.043    0.214      Best overall
 25    0.015    0.152
 26    0.013    0.167
```

**Decision**: Switched to 3-class classification (declined/stable/improved at ±10 EF threshold). See "Trajectory LVEF Decision" below.

---

## Trajectory LVEF — Experiment Evolution (2026-03-16)

Three iterations of the trajectory prediction task, culminating in the onset framing. Full experiment log with per-epoch data: `experiments/nature_medicine/uhn/TRAJECTORY_LVEF_EXPERIMENTS.md`.

### Summary

| Version | Task | Time Window | Classes | Event Rate | G AUROC | Model Separation |
|---------|------|-------------|---------|------------|---------|-----------------|
| V0 | Delta regression | 30-365d | continuous | — | R²=0.043 | — |
| V1 | Delta ±10 | 30-365d | 3 (9/82/9%) | 18% | 0.649 | minimal (0.04 range) |
| V2 | Delta ±8 | 90-365d | 3 (15/72/13%) | 28% | 0.610 | — |
| **V3** | **Onset (EF≥50→<50)** | **30-365d** | **2 (93/7%)** | **7-9%** | **0.733 val / 0.793 test** | **large (+0.18 G vs L)** |

### Why delta prediction fails

- **r = -0.511** between baseline EF and delta → massive regression to the mean
- All models learn baseline EF as a proxy (which all models do well), leaving no room for model differentiation
- External factors (treatment, disease progression) dominate delta but are invisible in a single echo
- V2 (narrower window, lower threshold) was worse because it removed informative pairs and ±8 is at measurement noise floor

### Why onset framing works

1. **Controls for baseline EF by design** — restricting to EF ≥ 50 forces the model to find visual features *beyond* the EF number
2. **Differentiates model quality** — G at 0.793 test (pred avg) vs L at 0.514 (chance) vs V1 where all models clustered at 0.60-0.65
3. **Clinically compelling** — "from an apparently normal echo, identify patients at risk of developing cardiomyopathy" is a Nature Medicine headline
4. **Event rate stable across time windows** — 7-8% at 30-89d, 90-179d, and 180-365d → no need to stratify

### Next steps

1. ~~Complete all 5 models on onset task~~ -- **DONE** (all 5 trained, 15 epochs each)
2. ~~Run prediction averaging~~ -- **DONE** (G 0.793, EchoPrime 0.776, PanEcho 0.759, L-K 0.677, L 0.514)
3. ~~**Decision gate**: AUROC >= 0.75 with pred avg -> Pillar 3 headline~~ -- **PASSED** (G 0.793)
4. Compare to baseline-EF-only predictor (logistic regression on measured EF) to quantify value added
5. Time-stratified AUROC analysis for supplement
6. Consider recovery prediction subgroup (EF < 40 → improves, 40-50% event rate, smaller N)

**Confounders / interventions**: Not filtered. Most trajectory changes are treatment-driven. Paper limitation sentence sufficient.

---

## Task Inventory (52 UHN + 5 Trajectory)

### Revised Paper Structure — "Comprehensive Cardiac Representation" (2026-03-17)

Literature review (March 2026) revealed the competitive landscape is broader than initially assessed. EchoPrime covers 23 tasks (AUC 0.88-0.98), PanEcho covers 39 tasks including RV (TAPSE, S'). Revised framing emphasizes six qualitatively different evidence categories from ONE frozen model. See `context_files/literature_review/foundation_model_landscape.md`.

**Six Evidence Categories (the "comprehensive representation" argument):**

| # | Category | What it proves | Key tasks | Status |
|---|----------|---------------|-----------|--------|
| 1 | **Structure** | Anatomy, morphology, dimensions | LVEF, EDV, ESV, LV mass, chamber sizes | LVEF done |
| 2 | **Hemodynamics** ("Virtual Doppler") | Physics: structure → flow (B-mode only) | MR/AS/TR/AR sev, AV Vmax, E/e', RVSP, diastolic grade, PA pressure, AV grad, AV area | 4 done, rest queued |
| 3 | **Deformation** | Motion tracking / mechanics | TAPSE, RV S', RV FAC | TAPSE done |
| 4 | **Pathology** | Disease recognition | Amyloidosis, HCM, DCM, endocarditis, bicuspid AV | CSVs ready |
| 5 | **Biochemistry** | Cross-modality (image → blood) | NT-proBNP, troponin T, creatinine, lactate (MIMIC) | CY prelim |
| 6 | **Prognosis** | Temporal / future state | Onset cardiomyopathy, other trajectories | Onset DONE |

No other model can fill all six categories. EchoPrime covers 1 + partial 2 + partial 4. PanEcho covers 1 + partial 2 + partial 3. None cover 5 or 6.

**Main Text Pillars (revised):**

- **Pillar 1 "Virtual Doppler" (B-mode hemodynamics, EXPANDED)**: MR, AS, TR, AR, AV Vmax, E/e', RVSP. Consider promoting: diastolic function grade (B-mode), PA pressure grade (B-mode), AV mean grad (B-mode), AV area (B-mode), PR severity (B-mode).
  - *Novelty*: No model predicts RVSP, E/e', AV mean grad, AV area from B-mode. MR/TR/AR from B-mode also first for frozen SSL. AS from B-mode done by others (Holste 2023, Ahmadi 2024) but single-task supervised, not frozen SSL.
  - *Key distinction*: EchoPrime/PanEcho valve severity (AUC 0.88-0.95) likely includes color Doppler views as input. Our B-mode-only restriction is fundamentally harder. G beats EchoPrime when both restricted to B-mode (MR: 0.860 vs 0.770, AS: 0.908 vs 0.827).

- **Pillar 2 RV Mechanics (REFRAMED)**: TAPSE, RV S', RV FAC.
  - *NOT novel tasks*: PanEcho does TAPSE (supervised, MAE 3.4mm) and RV S' (MAE 1.9cm/s). EchoNet-RV does RVFAC (task-specific, MAE 5.8-6.4%).
  - *Novel approach*: Frozen SSL representations encode RV function without task-specific training. PanEcho is our direct comparison baseline. Three-way taxonomy: self-supervised (EchoJEPA) vs language-supervised (EchoPrime) vs supervised (PanEcho).
  - *RVSP moved to Pillar 1*: RVSP is a hemodynamic/Doppler measurement (requires TR jet velocity), not an RV mechanics measurement.

- **Pillar 3 Trajectory**: Onset cardiomyopathy (G 0.793). No foundation model does temporal risk stratification. Keep as is.

- **GLS (strain)**: Extended Data only. Not cross-modal (speckle tracking already works on B-mode input), PanEcho already reports it (MAE 1.89%), and commercial software automates it. Low priority relative to genuinely cross-modal tasks.

- **Disease panel**: Extended Data but essential for "comprehensive representation" argument. Run at least amyloidosis, HCM, DCM, endocarditis, bicuspid AV.

- **Biomarkers (MIMIC)**: Extended Data but qualitatively unique (cross-modality). Push NT-proBNP + troponin as priority.

**Extended Data (revised):**
- ED1 Structural (7): ivsd, la_size, la_vol, lv_cavity_size, rv_size, ra_size, ao_root
- ED2 Hemodynamics — B-mode (7+): aov_area, aov_mean_grad, cardiac_output, lvot_vti, mv_ea, mv_dt, mv_ee_medial. Consider promoting diastolic_function + pa_pressure here or to main text.
- ED3 Findings (7): lv_hypertrophy, lv_systolic_function, diastolic_function, rv_function, pericardial_effusion, rwma, pa_pressure
- ED4 Disease detection (8): disease_hcm, disease_amyloidosis, disease_dcm, disease_endocarditis, disease_stemi, disease_takotsubo, disease_bicuspid_av, disease_myxomatous_mv
- ED5 Additional (4): disease_rheumatic_mv, cardiac_rhythm, gls, pr_severity
- ED6 Outcomes & biomarkers — MIMIC (11): mortality (30d/90d/1yr), in-hospital mortality, readmission, ICU transfer, discharge destination, creatinine, troponin T, NT-proBNP, lactate
- ED7 Fairness: MIMIC demographics
- EDF1 View classification (1)

---

## Prioritized Run Queue (2026-03-17)

Based on literature review and "six evidence categories" framing. 8 GPUs available (~2 concurrent 4-GPU jobs). Each task = 5 models × 15 epochs ≈ 4-6 hours per model.

### Batch 1 — Complete Hemodynamic Pillar (highest manuscript impact)
| Priority | Task | Type | B-mode | GPUs | Est. time | Evidence category |
|----------|------|------|--------|------|-----------|-------------------|
| **1a** | ar_severity | Classification, 5-class | Yes | 0-3 | ~20h (5 models) | Hemodynamics |
| **1b** | rvsp | Regression | Yes | 4-7 | ~15h (5 models) | Hemodynamics |

### Batch 2 — E/e' + AV hemodynamics (expand Virtual Doppler)
| Priority | Task | Type | B-mode | GPUs | Est. time | Evidence category |
|----------|------|------|--------|------|-----------|-------------------|
| **2a** | mv_ee (E/e') | Regression | Yes | 0-3 | ~10h (5 models, small dataset) | Hemodynamics |
| **2b** | aov_mean_grad | Regression | Yes (exists) | 4-7 | ~12h (5 models) | Hemodynamics |
| **2c** | aov_area | Regression | Yes (exists) | 0-3 | ~10h (5 models) | Hemodynamics |

### Batch 3 — RV Mechanics (fills Pillar 2)
| Priority | Task | Type | B-mode | GPUs | Est. time | Evidence category |
|----------|------|------|--------|------|-----------|-------------------|
| **3a** | rv_sp (RV S') | Regression | No | 0-3 | ~20h (5 models) | Deformation |
| **3b** | rv_fac | Regression | No | 4-7 | ~10h (5 models, small dataset) | Deformation |

### Batch 4 — Disease Panel (fills Pathology category)
| Priority | Task | Type | B-mode | GPUs | Est. time | Evidence category |
|----------|------|------|--------|------|-----------|-------------------|
| **4a** | disease_amyloidosis | Binary | No | 0-3 | ~8h | Pathology |
| **4b** | disease_hcm | Binary | No | 4-7 | ~15h | Pathology |
| **4c** | disease_dcm | Binary | No | 0-3 | ~8h | Pathology |
| **4d** | disease_endocarditis | Binary | No | 4-7 | ~8h | Pathology |
| **4e** | disease_bicuspid_av | Binary | No | 0-3 | ~20h | Pathology |

### Batch 5 — Expanded Hemodynamics (B-mode, need CSV build first)
| Priority | Task | Type | B-mode | GPUs | Est. time | Evidence category |
|----------|------|------|--------|------|-----------|-------------------|
| **5a** | diastolic_function | Classification, 4-class | **Need B-mode filter** | 0-3 | ~12h | Hemodynamics |
| **5b** | pa_pressure | Classification, 4-class | **Need B-mode filter** | 4-7 | ~10h | Hemodynamics |

### Batch 6 — Prediction Averaging Inference (test numbers for all completed tasks)
| Priority | Task | Models | Status |
|----------|------|--------|--------|
| **6a** | tr_severity | 5 | Checkpoints archived, ready |
| **6b** | Other completed tasks | Varies | Need retrain for lost checkpoints (LVEF, TAPSE, MR, AS) |

### Batch 7 — MIMIC Biomarkers (CY or us, fills Biochemistry category)
| Priority | Task | Type | Studies | Evidence category |
|----------|------|------|---------|-------------------|
| **7a** | NT-proBNP | Regression | 867 | Biochemistry |
| **7b** | troponin_t | Regression | 1,686 | Biochemistry |
| **7c** | creatinine | Regression | 3,883 | Biochemistry |
| **7d** | lactate | Regression | 1,226 | Biochemistry |

### Batch 8 — Additional Extended Data (structural, findings, GLS, remaining)
Lower priority. Run after Batches 1-7 complete. ~25 tasks remaining. Includes GLS (not cross-modal, PanEcho already reports it, commercial speckle tracking software automates it — Extended Data only).

### Prerequisites
- **Batch 2b, 2c**: B-mode filtered CSVs already built for `aov_mean_grad` and `aov_area`. Ready to run.
- **Batch 5a, 5b**: Need B-mode filtered CSVs for `diastolic_function` and `pa_pressure`. Run `build_viewfiltered_csvs.py` with `--bmode_only` flag.
- **Batch 6**: Depends on checkpoint availability. TR severity has all 5 archived. Others need retrain.
- **Batch 7**: Coordinate with CY. May run attentive probes ourselves or CY runs Strategy E.

---

## All Tasks — Metadata

### Regression Tasks (26)

| Task | Train Clips | Val Clips | Test Clips | VF Train | Views | B-mode | Mean | Std |
|------|------------|----------|-----------|----------|-------|--------|------|-----|
| ao_root | 1,046,757 | 545,130 | 653,762 | 155,779 | PLAX | N | 3.164 | 0.469 |
| aov_area | 442,423 | 230,217 | 298,367 | 80,046 | PLAX,A3C,PSAX-AV | **Y** | 1.881 | 0.866 |
| aov_mean_grad | 618,337 | 310,793 | 403,527 | 109,542 | PLAX,A3C,PSAX-AV | **Y** | 11.307 | 12.553 |
| aov_vmax | 1,410,302 | 749,691 | 912,460 | 269,567 | PLAX,A3C,PSAX-AV | **Y** | 1.599 | 0.766 |
| cardiac_output | 449,322 | 238,281 | 306,267 | 49,331 | A5C,A3C | N | 4.832 | 1.628 |
| edv | 1,451,832 | 766,832 | 943,931 | 407,368 | A4C,A2C | N | 110.576 | 45.687 |
| esv | 1,447,265 | 763,837 | 941,202 | 405,920 | A4C,A2C | N | 48.978 | 33.863 |
| gls | 45,719 | 34,435 | 46,164 | -- | (none) | -- | -19.353 | 3.000 |
| ivsd | 1,769,112 | 912,801 | 1,139,929 | 266,476 | PLAX | N | 1.040 | 0.264 |
| la_vol | 801,055 | 416,825 | 516,323 | 224,934 | A4C,A2C | N | 65.340 | 32.430 |
| lv_mass | 1,736,399 | 907,572 | 1,136,654 | 1,138,288 | PLAX,A4C,A2C,PSAX-* | N | 193.540 | 73.758 |
| lvef | 1,448,916 | 764,269 | 942,267 | 406,635 | A4C,A2C | N | 57.914 | 10.979 |
| lvot_vti | 727,314 | 356,708 | 464,298 | 78,553 | A5C,A3C | N | 0.197 | 0.053 |
| mv_dt | 1,074,322 | 534,125 | 669,677 | 364,652 | A4C,A2C,A3C | N | 211.131 | 63.475 |
| mv_ea | 1,461,676 | 730,981 | 871,872 | 495,368 | A4C,A2C,A3C | N | 1.228 | 0.679 |
| mv_ee | 637,208 | 237,542 | 297,640 | 71,562 | A4C | **Y** | 8.735 | 4.338 |
| mv_ee_medial | 1,057,885 | 581,920 | 684,852 | 190,270 | A4C | N | 11.429 | 5.251 |
| rv_fac | 413,564 | 260,622 | 367,025 | 80,046 | A4C | N | 37.096 | 10.281 |
| rv_sp | 1,493,083 | 789,682 | 990,767 | 391,778 | A4C,Subcostal | N | 0.116 | 0.032 |
| rvsp | 910,977 | 451,081 | 631,846 | 139,861 | A4C,Subcostal | **Y** | 35.099 | 13.907 |
| tapse | 1,524,418 | 812,242 | 1,017,414 | 280,638 | A4C | N | 1.955 | 0.497 |
| trajectory_lv_mass | 3,922 | 4,879 | 11,293 | -- | PLAX,A4C,A2C,PSAX-* | N | -5.119 | 52.059 |
| trajectory_lvef | 2,543 | 3,539 | 8,056 | -- | A4C,A2C | N | -0.805 | 9.626 |
| trajectory_mr_severity | 24,735 | 4,602 | 11,817 | -- | A4C,A2C,A3C,PLAX | N | -0.037 | 0.708 |
| trajectory_rv_sp | 2,757 | 3,596 | 8,198 | -- | A4C,Subcostal | N | -0.004 | 0.034 |
| trajectory_tapse | 2,872 | 3,780 | 8,646 | -- | A4C | N | -0.060 | 0.531 |

### Classification Tasks (26)

| Task | Train Clips | Val Clips | Test Clips | VF Train | Views | B-mode | Classes |
|------|------------|----------|-----------|----------|-------|--------|---------|
| ar_severity | 3,257,600 | 565,527 | 949,552 | 969,896 | A4C,A2C,A3C,PLAX | **Y** | 5 |
| as_severity | 7,776,258 | 795,910 | 1,419,201 | 1,487,709 | PLAX,PSAX-AV,A3C | **Y** | 4 |
| cardiac_rhythm | 1,586,839 | 432,894 | 682,209 | -- | (none) | -- | 6 |
| diastolic_function | 1,947,597 | 158,040 | 282,672 | 951,453 | A4C,A2C,A3C,PLAX | N | 4 |
| disease_amyloidosis | 368,111 | 128,621 | 282,658 | 201,734 | PLAX,A4C,PSAX-* | N | 2 |
| disease_bicuspid_av | 3,479,345 | 1,278,760 | 2,493,189 | 1,082,588 | PLAX,PSAX-AV,A3C | N | 2 |
| disease_dcm | 626,699 | 194,684 | 442,354 | -- | (none) | -- | 2 |
| disease_endocarditis | 1,091,024 | 200,344 | 433,827 | -- | (none) | -- | 2 |
| disease_hcm | 1,972,743 | 738,330 | 1,510,955 | 794,493 | PLAX,PSAX-PM,PSAX-MV,A4C | N | 2 |
| disease_myxomatous_mv | 269,665 | 105,235 | 217,935 | 103,606 | A4C,A2C,PLAX | N | 2 |
| disease_rheumatic_mv | 139,391 | 40,690 | 72,830 | 53,595 | A4C,A2C,PLAX | N | 2 |
| disease_stemi | 307,305 | 119,685 | 229,584 | -- | (none) | -- | 2 |
| disease_takotsubo | 232,503 | 104,520 | 203,879 | -- | (none) | -- | 2 |
| la_size | 6,001,934 | 782,412 | 1,315,708 | 2,523,887 | A4C,A2C,PLAX | N | 4 |
| lv_cavity_size | 10,254,343 | 1,172,095 | 2,073,388 | 6,584,841 | PLAX,A4C,A2C,PSAX-* | N | 5 |
| lv_hypertrophy | 1,876,056 | 155,352 | 282,885 | 734,431 | PLAX,A4C,PSAX-PM,PSAX-MV | N | 5 |
| lv_systolic_function | 5,227,379 | 729,422 | 1,234,474 | 2,127,485 | A4C,A2C,PLAX | N | 6 |
| mr_severity | 5,742,099 | 754,838 | 1,338,908 | 1,648,091 | A4C,A2C,A3C,PLAX | **Y** | 5 |
| pa_pressure | 1,627,875 | 274,341 | 470,933 | 680,896 | A4C,Subcostal,PLAX | N | 4 |
| pericardial_effusion | 7,842,019 | 673,742 | 1,285,981 | 3,263,297 | A4C,PLAX,Subcostal | N | 5 |
| pr_severity | 3,185,141 | 545,029 | 901,031 | 582,923 | PSAX-AV,Subcostal,PLAX | **Y** | 5 |
| ra_size | 2,563,332 | 531,172 | 878,492 | 645,981 | A4C,Subcostal | N | 4 |
| rv_function | 5,221,854 | 773,325 | 1,361,546 | 2,072,321 | A4C,Subcostal,PLAX | N | 5 |
| rv_size | 3,551,977 | 631,235 | 1,085,634 | 2,156,950 | A4C,Subcostal,PLAX,PSAX-* | N | 4 |
| rwma | 2,683,316 | 520,495 | 867,849 | 1,176,666 | A4C,A2C,A3C,PSAX-PM,PSAX-AP | N | 2 |
| tr_severity | 5,925,783 | 791,052 | 1,427,082 | 1,365,676 | A4C,Subcostal,PLAX | **Y** | 5 |

---

## Classification Class Maps

See `experiments/nature_medicine/uhn/CLASS_MAPS.md` for detailed valve severity class-to-label mappings.

Key severity scales:
- **MR/AR/PR/TR**: 5 classes — none(0), trace(1), mild(2), moderate(3), severe(4)
- **AS**: 4 classes — none/sclerosis(0), mild(1), moderate(2), severe(3)
- **TR**: class 1 = "trivial/trace" (merged)

Intermediate grades (mild-to-moderate, moderate-to-severe) collapsed into the adjacent higher class.

---

## Run Priority (Phase 1)

### URGENT — Retrain Lost Checkpoints (RUNNING)

23 runs retraining via `scripts/retrain_lost_checkpoints.sh` on GPUs 4-7 (port 29501).
Started: 2026-03-16 23:26. Log: `logs/retrain_all_20260316_232654.log`.
Chain: LVEF (5) → TAPSE (5) → MR severity (5) → AS severity (5) → AV Vmax G/L/L-K (3).

| Task | Models to retrain | Historical best (G) | Status |
|------|-------------------|-------|--------|
| lvef | all 5 | R² 0.720 | **RUNNING** (G epoch 1, GPUs 4-7) |
| tapse | all 5 | R² 0.450 (L-K; G had NaN bug) | Queued |
| mr_severity | all 5 | AUROC 0.860 | Queued |
| as_severity | all 5 | AUROC 0.908 | Queued |
| aov_vmax | G, L, L-K only | R² 0.582 | Queued |

### Tier 1 — Main text Pillar 2: Hemodynamics (B-mode only)

| Task | Type | Classes | VF Train Clips | Status |
|------|------|---------|---------------|--------|
| mr_severity | classification | 5 | 1,648,091 | **RETRAIN** (checkpoints lost, was G 0.860) |
| as_severity | classification | 4 | 1,487,709 | **RETRAIN** (checkpoints lost, was G 0.908) |
| aov_vmax | regression | -- | 269,567 | **RETRAIN G/L/L-K** (EchoPrime+PanEcho archived) |
| tr_severity | classification | 5 | 1,365,676 | **ALL 5 DONE** + archived. G 0.838, L-K 0.787, EchoPrime 0.758, L 0.755, PanEcho 0.715. |
| ar_severity | classification | 5 | 969,896 | READY |
| mv_ee | regression | -- | 71,562 | READY (B-mode filter rebuilt) |
| rvsp | regression | -- | 139,861 | READY (B-mode filter rebuilt) |

### Tier 2 — Main text Pillar 1: RV Mechanics

| Task | Type | VF Train Clips | Status |
|------|------|---------------|--------|
| tapse | regression | 280,638 | **RETRAIN** (checkpoints lost, was L-K 0.450 best) |
| rv_sp | regression | 391,778 | READY |
| rv_fac | regression | 80,046 | READY |
| rv_basal_dim | regression | -- | **BLOCKED** (labels not built) |

### Tier 3 — Main text Pillar 3: Trajectory Forecasting

| Task | Type | Train Clips | Status |
|------|------|------------|--------|
| trajectory_lvef_onset | binary classification | 33,705 (1,932 studies) | **DONE** all 5 models trained + inference. G 0.793 test AUROC. |
| trajectory_lvef (3-class) | classification | 155,053 (8,471 studies) | G/L/L-K trained (archived). No inference yet. |
| trajectory_tapse | regression | 2,872 | READY (may switch to classification) |
| trajectory_lv_mass | regression | 3,922 | READY |
| trajectory_rv_sp | regression | 2,757 | READY |
| trajectory_mr_severity | regression | 24,735 | READY |

### Tier 4 — Extended Data (batch after main text tasks)

All remaining 35 tasks. Run in order: ED1 structural -> ED3 findings -> ED2 hemodynamics -> ED4+ED5 disease detection.

---

## Checkpoint Paths

All Nature Medicine probes are saved under:
```
evals/vitg-384/nature_medicine/uhn/video_classification_frozen/{task}-{model}/
```

Each directory contains:
- `best.pt` — best validation checkpoint (selected by lowest val MAE for regression, highest AUROC for classification)
- `latest.pt` — final epoch checkpoint
- `epoch_NNN.pt` — per-epoch checkpoints
- `log_r0.csv` — per-epoch training log (columns vary by task type)

### Completed checkpoint sizes
| Model | Checkpoint Size |
|-------|----------------|
| echojepa-g | ~4.9 GB |
| echojepa-l | ~1.6–2.6 GB |
| echojepa-l-k | ~1.6 GB |
| echoprime | ~400–660 MB |
| panecho | ~893 MB |

---

## CSV Paths

All UHN probe CSVs:
```
experiments/nature_medicine/uhn/probe_csvs/{task}/
    train.csv          # all clips, all views
    val.csv
    test.csv
    train_vf.csv       # view-filtered (task-relevant views only)
    val_vf.csv
    test_vf.csv
    viewfilter_meta.json   # allowed_views, bmode_only flag
    zscore_params.json     # regression tasks only: target_mean, target_std
```

Trajectory CSVs:
```
experiments/nature_medicine/uhn/probe_csvs/trajectory_{task}/
    train.csv, val.csv, test.csv
    viewfilter_meta.json
    zscore_params.json          # regression tasks only (removed for classification)
    pairs_metadata.json         # full pair info: delta_raw, days_between, class, etc.
```
Trajectory LVEF: switched to 3-class classification (2026-03-16). All clips per study_1 in CSV, study_sampling enabled. Format: `clip_path class_int`. Other trajectory tasks still use regression (1 clip per pair, `clip_path delta_value`).

Build script: `build_trajectory_csvs.py --task trajectory_lvef --classification --threshold 10`

MIMIC CSVs (23 tasks):
```
experiments/nature_medicine/mimic/probe_csvs/{task}/
    train.csv, val.csv, test.csv
    zscore_params.json (regression only)
```

---

## TEE/Stress Filtering

Applied 2026-03-14 to ALL 52 UHN task CSV directories.
- 7,341 deidentified study UIDs excluded (TEE, stress echo, IVUS, ICE)
- ~3.2M clips removed across all tasks
- Contamination was 0–3.2% per task
- Mapping chain: `syngo_study_details.STUDY_DESCRIPTION` → `STUDY_REF` → `uhn_uid_to_studyref.csv` → deid UID
- Key file: `experiments/nature_medicine/uhn/mapping/uhn_uid_to_studyref.csv`

---

## Detailed Training Curves

### LVEF — Per-epoch Val R² (12-head grid, best head)

```
Epoch   G       L       L-K     Prime   PanEcho
  1   0.658   0.038   0.370   0.510   0.509
  2   0.667   0.151   0.448   0.543   0.525
  3   0.647   0.266   0.466   0.527   0.517
  4   0.632   0.255   0.433   0.520   0.518
  5   0.690   0.301   0.522   0.534   0.502
  6   0.693   0.326   0.536   0.531   0.528
  7   0.693   0.360   0.536   0.545   0.509
  8   0.694   0.348   0.541   0.552   0.551
  9   0.700   0.349   0.557   0.542   0.538
 10   0.701   0.390   0.568   0.554   0.549
 11   0.690   0.379   0.573   0.553   0.538
 12   0.710   0.398   0.579   0.552   0.546
 13   0.694   0.411   0.582*  0.559*  0.550
 14   0.711   0.415*  0.578   0.553   0.551
 15   0.705   0.409   0.581   0.555   0.555*
 16   0.709   --      --      --      --
 17   0.720*  --      --      --      --
 18   0.712   --      --      --      --
 19   0.708   --      --      --      --
 20   0.714   --      --      --      --
```
`*` = best epoch for that model. `--` = run ended.

### TAPSE — Per-epoch Val R² (12-head grid, best head)

```
Epoch   G       L       L-K     Prime   PanEcho
  1    NaN    0.083   0.268   0.288   0.240
  2    NaN    0.148   0.318   0.268   0.220
  3    NaN    0.154   0.336   0.283   0.230
  4    NaN    0.228   0.387   0.309   0.254
  5    NaN    0.211   0.394   0.298   0.258
  6    NaN    0.254   0.394   0.309   0.273
  7    NaN    0.249   0.403   0.305   0.260
  8    NaN    0.113   0.402   0.307   0.295
  9    NaN    0.266   0.413   0.312   0.293
 10    NaN    0.260   0.438   0.311   0.301
 11    NaN    0.254   0.431   0.318   0.294
 12    NaN    0.286   0.439   0.329   0.306
 13    NaN    0.285   0.432   0.323   0.303
 14    NaN    0.295   0.450*  0.328   0.306
 15    NaN    0.297   0.443   0.335   0.311*
 16    NaN    0.306   --      0.330   --
 17    NaN    0.313   --      0.335   --
 18    NaN    0.323*  --      0.343*  --
 19    NaN    0.310   --      0.342   --
 20    NaN    0.299   --      0.339   --
```

**TAPSE ranking** (excluding G due to NaN): L-K (0.450) > EchoPrime (0.343) > L (0.323) > PanEcho (0.311).
G has best MAE (0.264 Z-score ≈ 0.131 cm raw) so likely would rank first if metrics were fixed.

---

### MR Severity — Per-epoch Val AUROC (12-head grid, best head)

```
Epoch   G       L       L-K     Prime   PanEcho
  1   0.810   0.667   0.740   0.700   0.666
  2   0.786   0.671   0.733   0.725   0.657
  3   0.833   0.685   0.751   0.731   0.677
  4   0.843   0.710   0.768   0.770*  0.693
  5   0.828   0.713   0.772   0.768   0.705
  6   0.840   0.713   0.774   0.766   0.704
  7   0.850   0.732   0.784   0.762   0.710
  8   0.846   0.722   0.787   0.764   0.718
  9   0.851   0.732   0.791   0.769   0.724*
 10   0.852   0.727   0.792   0.759   --
 11   0.857   0.742   0.795   0.764   --
 12   0.854   0.737   0.792   0.762   --
 13   0.856   0.747   0.800   0.765   --
 14   0.856   0.752   0.798   0.768   --
 15   0.852   0.754   0.797   0.764   --
 16   0.858   0.764   0.800   --      --
 17   0.860*  0.771*  0.801   --      --
 18   0.854   --      0.802   --      --
 19   0.857   --      0.803*  --      --
 20   --      --      --      --      --
```
`*` = best epoch for that model. `--` = run ended or not reached.
EchoPrime/PanEcho best early (epoch 4/9) — consistent with high-LR heads peaking fast.

### AS Severity — Per-epoch Val AUROC (12-head grid, best head)

```
Epoch   G       L       L-K     Prime   PanEcho
  1   0.831   0.669   0.704   0.761   0.708
  2   0.877   0.710   0.737   0.770   0.743
  3   0.880   0.731   0.751   0.781   0.746
  4   0.877   0.727   0.757   0.790   0.749
  5   0.884   0.737   0.773   0.793   0.755
  6   0.899   0.744   0.783   0.802   0.760
  7   0.891   0.742   0.782   0.798   0.755
  8   0.893   0.739   0.786   0.805   0.762*
  9   0.903   0.754   0.796   0.803   0.758
 10   0.890   0.761   0.799   0.806   0.756
 11   0.898   0.763   0.800   0.812   0.756
 12   0.904   0.766   0.803   0.817   0.753
 13   0.903   0.773   0.806   0.820   0.756
 14   0.904   0.771   0.805   0.822   0.755
 15   0.907   0.778   0.808   0.827*  0.757
 16   0.908*  0.774   0.814   0.826   --
 17   0.906   0.777   0.813   0.825   --
 18   0.906   0.780   0.821*  0.822   --
 19   0.907   0.786*  0.818   0.818   --
 20   0.907   0.784   --      --      --
```

### AV Vmax — Per-epoch Val R² (12-head grid, best head, B-mode only)

```
Epoch  EchoPrime  PanEcho
  1    0.401      0.312
  2    0.422      0.330
  3    0.433      0.344
  4    0.409      0.337
  5    0.450      0.379
  6    0.447      0.337
  7    0.454      0.353
  8    0.462      0.362
  9    0.464      0.377
 10    0.458      0.375
 11    0.454      0.358
 12    0.462      0.376
 13    0.470      0.379
 14    0.468      0.385
 15    0.476*     0.390*
```
G/L/L-K log_r0.csv lost (run script overwrites checkpoint dirs). From raw log: G best R²=0.582 (ep13), L best R²=0.232 (ep14), L-K best R²=0.388 (ep14).

**AV Vmax ranking**: G (0.582) >> EchoPrime (0.476) > PanEcho (0.390) ≈ L-K (0.388) >> L (0.232).
G leads by +10pp over next-best. Confirms hemodynamic regression works from B-mode, not just classification.

### TR Severity — Per-epoch Val AUROC (12-head grid, best head, B-mode only)

```
Epoch   G       L       L-K
  1   0.790   0.676   0.724
  2   0.783   0.693   0.718
  3   0.817   0.712   0.741
  4   0.825   0.727   0.767
  5   0.824   0.736   0.766
  6   0.822   0.732   0.763
  7   0.827   0.728   0.773
  8   0.831   0.742   0.773
  9   0.837   0.751   0.780
 10   0.836   0.745   0.781
 11   0.838   0.755*  0.790
 12   0.839*  0.754   0.791*
 13   0.838   0.755   --
 14   0.837   0.747   --
 15   0.837   0.755   --
```
`*` = best epoch for that model. `--` = not yet reached (L-K at epoch 12, running).
EchoPrime and PanEcho queued after L-K completes.

### Convergence Analysis

15→20 epoch gains across all models:

| Model | AS Δ AUROC | MR Δ AUROC |
|-------|-----------|-----------|
| echojepa-g | +0.001 | +0.003 |
| echojepa-l | +0.008 | +0.017 |
| echojepa-l-k | +0.013 | +0.003 |
| echoprime | +0.000 | — (stopped at 15) |
| panecho | — (stopped at 15) | — (stopped at 9) |

**Conclusion**: <0.005 gain for G, modest gains for smaller models. 15 epochs is the default going forward.

---

## Model Checkpoints (Pretrained Encoders)

| Model | Checkpoint | Tokens/clip | Notes |
|-------|-----------|-------------|-------|
| EchoJEPA-G | `checkpoints/vitg-384/echojepa-g.pth` | 1,568 | ViT-G/14, 384px, 16 frames |
| EchoJEPA-L | `checkpoints/vitl16/echojepa-l.pth` | 1,568 | ViT-L/16, 224px, 16 frames |
| EchoJEPA-L-K | `checkpoints/vitl16/echojepa-l-k.pth` | 1,568 | ViT-L/16, Kinetics init |
| EchoPrime | (external) | 1 | Global avg pool → single token |
| PanEcho | (external) | 1 | Global avg pool → single token |

---

## Experiment Prioritization & Decision Gates (2026-03-15)

Ordered by narrative damage if the experiment fails. Run the experiments you're most scared of first.

### Tier 1: Paper collapses without these — PASSED

**Hemodynamic inference from B-mode (MR severity, AS severity)**

The headline claim. Abstract leads with it, discussion centers on it, "world model" framing depends on it.

**Status: DONE. Both canaries passed.** All 5 models extract hemodynamic signal from B-mode video.

| Task | EchoJEPA-G | EchoJEPA-L | EchoJEPA-L-K | EchoPrime | PanEcho |
|------|-----------|-----------|-------------|----------|---------|
| MR severity (B-mode) | **0.860** | 0.771 | 0.803 | 0.770 | 0.724 |
| AS severity (B-mode) | **0.908** | 0.786 | 0.821 | 0.827 | 0.762 |

Key finding: ALL models show signal, so the story is "scale advantage" (+8-9pp for G over baselines), not unique JEPA capability. EchoPrime is competitive on AS (0.827) due to its text-supervised pretraining on reports that describe aortic stenosis.

### Tier 2: A major pillar drops, paper needs restructuring

**2a. Trajectory prediction (future LVEF)** — PASSED

The "model predicts future cardiac states" claim is one of three pillars. Delta regression and 3-class classification both failed (see Failed Experiments). **Onset framing (V3) succeeded**: from apparently normal EF (≥50%), predict new-onset cardiomyopathy (future EF<50%). EchoJEPA-G test AUROC **0.793** with prediction averaging (pred avg boosted +0.060 over val).

- **Onset framing works** because it controls for baseline EF by design, forcing the model to find subclinical features beyond the EF number
- **Strong model separation**: G 0.793 >> L-K 0.677 >> L ~chance. EchoPrime/PanEcho test results pending (~0.70 expected from training val).
- **Decision gate PASSED**: 0.793 > 0.75 → trajectory confirmed as Pillar 3 headline

**2b. Biomarkers (attentive probes, NT-proBNP + creatinine)** — NOT STARTED

Preliminary linear probe results: NT-proBNP r=0.40, creatinine r=0.29 (small test sets: 125 and 564). If attentive probes don't improve these, the cross-modal prediction section is thin and the cardiac output bridge experiment is pointless.

- Gates the CO experiment: if biomarkers work (NT-proBNP r > 0.5 or creatinine r > 0.4) -> run CO from B-mode. If not -> skip CO, redirect compute.
- Keep +/-24h temporal window as primary (tight linkage is a strength). +/-48h as sensitivity analysis in supplement.

### Tier 3: Section shrinks but paper stands

**MIMIC outcomes (attentive probes)** — NOT STARTED

De-risked by preliminary sklearn results (mortality AUC 0.846). Question is whether attentive probes push echo above the EHR baseline (0.856). Sklearn results are a safety net.

**RV mechanics (RV S', RV FAC)** — NOT STARTED

TAPSE already works (MAE 0.264 Z-score). Extensions failing doesn't break anything.

**Disease detection, fairness, SAE** — NOT STARTED

Supporting evidence / Extended Data. Failures shrink scope but don't touch core claims.

### Updated Run Order

| Priority | Experiment | Dataset | Status | Probe runs |
|----------|-----------|---------|--------|------------|
| ~~1~~ | ~~MR severity from B-mode~~ | ~~UHN~~ | **DONE** | ~~5~~ |
| ~~2~~ | ~~AS severity from B-mode~~ | ~~UHN~~ | **DONE** | ~~5~~ |
| ~~3~~ | ~~AV Vmax from B-mode~~ | ~~UHN~~ | **DONE** | ~~5~~ |
| ~~4~~ | ~~Trajectory LVEF onset (V3)~~ | ~~UHN (6K studies)~~ | **DONE** — G test 0.793 (pred avg). Passes 0.75 gate. | ~~5~~ |
| ~~5~~ | ~~TR severity from B-mode~~ | ~~UHN (1.4M clips)~~ | **DONE** — G 0.838, L-K 0.787, EchoPrime 0.758, L 0.755, PanEcho 0.715 | ~~5~~ |
| **6** | **AR severity from B-mode** | UHN (970K clips) | READY | 5 |
| **7** | **NT-proBNP + creatinine (attentive)** | MIMIC (852 / 3,883) | READY | 10 |
| **8** | **E/e' + RVSP from B-mode** | UHN | READY (B-mode filters rebuilt) | 10 |
| **9** | **RV mechanics (rv_sp, rv_fac)** | UHN | READY | 10 |
| **10** | **MIMIC outcomes (attentive)** | MIMIC (7,243) | READY | 35 |
| **11** | **Remaining trajectory tasks** | UHN | READY (switch to classification if LVEF works) | 20 |
| **12** | **Extended Data tasks** | UHN + MIMIC | READY | ~200 |

### Decision Points

**After priority 3 (AV Vmax): PASSED**
- ✓ Hemodynamic regression works. G achieves R²=0.582 (Pearson 0.763) predicting AV Vmax from B-mode. All 5 models show signal. Proceed with full hemodynamic table.

**After priority 4 (trajectory LVEF): PASSED**
- Delta regression FAILED (R²=0.043). 3-class classification (V1/V2) showed modest signal but poor model separation.
- Onset framing (V3): EF≥50 → predict future EF<50. **G test AUROC 0.793 with prediction averaging — passes 0.75 gate.**
- ✓ Trajectory is confirmed as Pillar 3 headline. Proceed with remaining 4 trajectory tasks (reframe as onset/classification where appropriate).

**After priority 6 (biomarkers):**
- NT-proBNP r > 0.5 or creatinine r > 0.4 -> Run cardiac output from B-mode (5 runs on UHN). The mechanistic bridge is worth pursuing.
- No improvement over linear probes -> Skip CO. Report biomarker results as-is in compressed section. Focus compute on remaining hemodynamic and outcome tasks.

### Cardiac Output — Conditional Experiment

**Run only if biomarker attentive probes show signal (decision gate after priority 6).**

Derives CO from UHN structured measurements: LVOT_VTI x pi x (LVOT_diam/2)^2 x HR. 16,137 UHN studies with all components. Also available on MIMIC via structured-measurement.csv.gz: lvot_vti (93%), lvot_diam (96%), resting_hr (100%).

Predicting CO from B-mode is Tier S (requires Doppler to measure). If it works, the mechanistic chain is airtight: frozen representations -> CO -> renal perfusion (creatinine), tissue oxygenation (lactate), cardiac strain (NT-proBNP), mortality. Transforms the biomarker section from correlational to mechanistically grounded.

Labels already built on UHN (`cardiac_output.npz`, 16K studies). MIMIC labels need to be derived from structured measurements.
