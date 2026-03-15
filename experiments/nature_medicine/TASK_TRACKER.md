# Nature Medicine — Task Tracker

Last updated: 2026-03-15

## Evaluation Protocol

**Strategy E**: d=1 attentive probes from video + prediction averaging across all clips per study.
- **5 models**: EchoJEPA-G, EchoJEPA-L, EchoJEPA-L-K, EchoPrime, PanEcho
- **Probe grid**: 12 HP combos (4 LR × 3 WD), 15 epochs. LVEF/TAPSE used 20-head grid (5 LR × 4 WD); all subsequent tasks use 12-head.
- **Convergence**: 15 epochs sufficient for all tasks tested. Extending to 20 yields <0.005 AUROC improvement. Use 15 epochs going forward.
- **Run script**: `scripts/run_uhn_probe.sh <task>`
- **Config template**: auto-generated YAML at `/tmp/nm_{task}_{model}.yaml`
- **Checkpoint output**: `evals/vitg-384/nature_medicine/uhn/video_classification_frozen/{task}-{model}/`
- **CSV source**: `experiments/nature_medicine/uhn/probe_csvs/{task}/`

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

### AV Vmax (regression, B-mode only) — RUNNING

| Setting | Value |
|---------|-------|
| Task type | Regression (smooth L1 loss) |
| Views | PLAX, A3C, PSAX-AV |
| B-mode only | **Yes** |
| Study sampling | Yes (`DistributedStudySampler`, 1 random clip/study/epoch) |
| Class balance | N/A (regression) |
| Train | 269,567 clips, 22,417 studies (view-filtered) |
| Val | 143,772 clips, 11,896 studies |
| Test | 173,060 clips, 14,235 studies |
| Z-score | mean=1.599, std=0.766 |
| HP grid | 12 heads (4 LR × 3 WD) |
| Epochs | 15 |
| Batch size | 2 per GPU × 4 GPUs = 8 effective |
| Val batch size | 64 (EchoPrime: 16) |
| Warmup | 2 epochs cosine |
| num_workers | 4 |
| Inference | Single random clip per study per val epoch. **No prediction averaging.** |
| CSV source | `experiments/nature_medicine/uhn/probe_csvs/aov_vmax/train_vf.csv` |

### TR Severity (classification, B-mode only) — RUNNING

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

### Trajectory LVEF (regression, delta prediction) — RUNNING

| Setting | Value |
|---------|-------|
| Task type | Regression (smooth L1 loss). Predicts delta EF (future − baseline). |
| Views | A4C, A2C (from base task; no additional view filter) |
| B-mode only | No |
| Study sampling | **No** (1 clip per pair, no grouping) |
| Class balance | N/A (regression) |
| Train | 8,475 clips (pairs) — cleaned v2 |
| Val | 1,147 clips |
| Test | 1,879 clips |
| Z-score | mean=−0.024, std=8.037 |
| Cleaning | delta_max=±30, EF range 5–85, max 10 pairs/patient, temporal split by study2 date |
| HP grid | 12 heads (4 LR × 3 WD) |
| Epochs | 30 (extended for small dataset) |
| Batch size | 2 per GPU × 4 GPUs = 8 effective |
| Val batch size | 64 (EchoPrime: 16) |
| Warmup | 2 epochs cosine |
| num_workers | 4 |
| Inference | Single clip per pair. **No prediction averaging.** |
| CSV source | `experiments/nature_medicine/uhn/probe_csvs/trajectory_lvef/train.csv` |
| Critical baseline | "Predict no change" (delta=0). If model can't beat this, pillar 3 drops. |

---

## Completed Runs — Summary

### Regression Tasks

| Task | Model | Best Val R² | Best Pearson | Best MAE | Best Head (lr, wd) | Best Epoch | Epochs | Status |
|------|-------|------------|-------------|---------|-------------------|-----------|--------|--------|
| LVEF | echojepa-g | **0.720** | **0.849** | 4.462 | 13 (5e-5, 0.01) | 17 | 20 | DONE |
| LVEF | echojepa-l | 0.415 | 0.646 | 6.152 | 5 (1e-4, 0.1) | 14 | 15 | DONE |
| LVEF | echojepa-l-k | 0.582 | 0.763 | 5.265 | 4 (1e-4, 0.01) | 13 | 15 | DONE |
| LVEF | echoprime | 0.559 | 0.751 | 5.399 | 11 (1e-5, 0.1) | 13 | 15 | DONE |
| LVEF | panecho | 0.555 | 0.746 | 5.442 | 6 (5e-5, 0.001) | 15 | 15 | DONE |
| TAPSE | echojepa-g | NaN | NaN | 0.264 | 13 (5e-5, 0.01) | 15 | 20 | **BUG** |
| TAPSE | echojepa-l | 0.323 | 0.572 | 0.325 | 8 (1e-4, 0.001) | 18 | 20 | DONE |
| TAPSE | echojepa-l-k | **0.450** | **0.671** | 0.292 | 6 (5e-5, 0.001) | 14 | 15 | DONE |
| TAPSE | echoprime | 0.343 | 0.588 | 0.321 | 17 (1e-5, 0.01) | 18 | 20 | DONE |
| TAPSE | panecho | 0.311 | 0.560 | 0.330 | 9 (1e-5, 0.001) | 15 | 15 | DONE |

### Classification Tasks (B-mode only)

| Task | Model | Best AUROC | Best Acc | Best Bal Acc | Best Kappa | Best Head (lr, wd) | Best Epoch | Epochs | Status |
|------|-------|-----------|---------|-------------|-----------|-------------------|-----------|--------|--------|
| MR sev. | echojepa-g | **0.860** | 43.75 | 0.561 | 0.280 | 6 (5e-5, 0.001) | 17 | 20 | DONE |
| MR sev. | echojepa-l | 0.771 | 33.75 | 0.428 | 0.184 | 7 (5e-5, 0.01) | 17 | 20 | DONE |
| MR sev. | echojepa-l-k | 0.803 | 36.08 | 0.473 | 0.210 | 7 (5e-5, 0.01) | 19 | 19 | DONE |
| MR sev. | echoprime | 0.770 | 30.58 | 0.416 | 0.160 | 0 (5e-4, 0.001) | 4 | 15 | DONE |
| MR sev. | panecho | 0.724 | 23.87 | 0.373 | 0.106 | 0 (5e-4, 0.001) | 4 | 9 | DONE |
| AS sev. | echojepa-g | **0.908** | 70.51 | 0.594 | 0.432 | 5 (1e-4, 0.1) | 16 | 20 | DONE |
| AS sev. | echojepa-l | 0.786 | 54.25 | 0.427 | 0.239 | 3 (1e-4, 0.001) | 19 | 20 | DONE |
| AS sev. | echojepa-l-k | 0.821 | 53.26 | 0.467 | 0.235 | 6 (5e-5, 0.001) | 18 | 20 | DONE |
| AS sev. | echoprime | 0.827 | 57.00 | 0.496 | 0.253 | 10 (1e-5, 0.01) | 16 | 19 | DONE |
| AS sev. | panecho | 0.762 | 34.65 | 0.420 | 0.086 | 0 (5e-4, 0.001) | 2 | 15 | DONE |

**MR/AS settings**: B-mode only view-filtered CSVs, `class_balance_ratio=3` (cap each class at 3× minority), `study_sampling=true`, `num_workers=4`.
MR views: A4C, A2C, A3C, PLAX. AS views: PLAX, PSAX-AV, A3C. Studies after balancing: MR ~29K, AS ~22K.

### Trained Probe Paths

Base: `evals/vitg-384/nature_medicine/uhn/video_classification_frozen/`

| Task | Model | best.pt | log |
|------|-------|---------|-----|
| LVEF | echojepa-g | `lvef-echojepa-g/best.pt` | `lvef-echojepa-g/log_r0.csv` |
| LVEF | echojepa-l | `lvef-echojepa-l/best.pt` | `lvef-echojepa-l/log_r0.csv` |
| LVEF | echojepa-l-k | `lvef-echojepa-l-k/best.pt` | `lvef-echojepa-l-k/log_r0.csv` |
| LVEF | echoprime | `lvef-echoprime/best.pt` | `lvef-echoprime/log_r0.csv` |
| LVEF | panecho | `lvef-panecho/best.pt` | `lvef-panecho/log_r0.csv` |
| TAPSE | echojepa-g | `tapse-echojepa-g/best.pt` | `tapse-echojepa-g/log_r0.csv` |
| TAPSE | echojepa-l | `tapse-echojepa-l/best.pt` | `tapse-echojepa-l/log_r0.csv` |
| TAPSE | echojepa-l-k | `tapse-echojepa-l-k/best.pt` | `tapse-echojepa-l-k/log_r0.csv` |
| TAPSE | echoprime | `tapse-echoprime/best.pt` | `tapse-echoprime/log_r0.csv` |
| TAPSE | panecho | `tapse-panecho/best.pt` | `tapse-panecho/log_r0.csv` |
| MR sev. | echojepa-g | `mr_severity-echojepa-g/best.pt` | `mr_severity-echojepa-g/log_r0.csv` |
| MR sev. | echojepa-l | `mr_severity-echojepa-l/best.pt` | `mr_severity-echojepa-l/log_r0.csv` |
| MR sev. | echojepa-l-k | `mr_severity-echojepa-l-k/best.pt` | `mr_severity-echojepa-l-k/log_r0.csv` |
| MR sev. | echoprime | `mr_severity-echoprime/best.pt` | `mr_severity-echoprime/log_r0.csv` |
| MR sev. | panecho | `mr_severity-panecho/best.pt` | `mr_severity-panecho/log_r0.csv` |
| AS sev. | echojepa-g | `as_severity-echojepa-g/best.pt` | `as_severity-echojepa-g/log_r0.csv` |
| AS sev. | echojepa-l | `as_severity-echojepa-l/best.pt` | `as_severity-echojepa-l/log_r0.csv` |
| AS sev. | echojepa-l-k | `as_severity-echojepa-l-k/best.pt` | `as_severity-echojepa-l-k/log_r0.csv` |
| AS sev. | echoprime | `as_severity-echoprime/best.pt` | `as_severity-echoprime/log_r0.csv` |
| AS sev. | panecho | `as_severity-panecho/best.pt` | `as_severity-panecho/log_r0.csv` |

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

5. **Prediction averaging NOT implemented**: All reported val metrics are single random clip per study per epoch (via `DistributedStudySampler`). Study-level prediction aggregation at inference time is needed for final numbers.

---

## Task Inventory (52 UHN + 5 Trajectory)

### Paper Section Assignments

**Main Text — 17 tasks:**
- **Standard benchmark (1)**: lvef
- **Hemodynamics / B-mode (7)**: mr_severity, as_severity, aov_vmax, tr_severity, ar_severity, mv_ee (E/e'), rvsp
- **RV mechanics (4)**: tapse, rv_sp (RV S'), rv_fac, *rv_basal_dim (labels not built yet)*
- **Trajectory forecasting (5)**: trajectory_lvef, trajectory_tapse, trajectory_lv_mass, trajectory_rv_sp, trajectory_mr_severity

**Extended Data — ~35 tasks:**
- ED1 Structural (7): ivsd, la_size, la_vol, lv_cavity_size, rv_size, ra_size, ao_root
- ED2 Hemodynamics (7): aov_area, aov_mean_grad, cardiac_output, lvot_vti, mv_ea, mv_dt, mv_ee_medial
- ED3 Findings (7): lv_hypertrophy, lv_systolic_function, diastolic_function, rv_function, pericardial_effusion, rwma, pa_pressure
- ED4 Disease detection (8): disease_hcm, disease_amyloidosis, disease_dcm, disease_endocarditis, disease_stemi, disease_takotsubo, disease_bicuspid_av, disease_myxomatous_mv
- ED5 Additional diseases (2): disease_rheumatic_mv, cardiac_rhythm
- EDF1 View classification (1): *separate pipeline*
- Remaining: edv, esv, gls, lv_mass, pr_severity

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

### Tier 1 — Main text Pillar 2: Hemodynamics (B-mode only)

These are the headline results. All use `bmode_only: true` view-filtered CSVs.

| Task | Type | Classes | VF Train Clips | Status |
|------|------|---------|---------------|--------|
| mr_severity | classification | 5 | 1,648,091 | **DONE** (5 models, B-mode only) |
| as_severity | classification | 4 | 1,487,709 | **DONE** (5 models, B-mode only) |
| aov_vmax | regression | -- | 269,567 | **RUNNING** (GPUs 0-3, priority 3) |
| tr_severity | classification | 5 | 1,365,676 | READY (priority 5, after trajectory LVEF) |
| ar_severity | classification | 5 | 969,896 | READY |
| mv_ee | regression | -- | 71,562 | READY (B-mode filter rebuilt) |
| rvsp | regression | -- | 139,861 | READY (B-mode filter rebuilt) |

### Tier 2 — Main text Pillar 1: RV Mechanics

| Task | Type | VF Train Clips | Status |
|------|------|---------------|--------|
| tapse | regression | 280,638 | **DONE** (5 models) |
| rv_sp | regression | 391,778 | READY |
| rv_fac | regression | 80,046 | READY |
| rv_basal_dim | regression | -- | **BLOCKED** (labels not built) |

### Tier 3 — Main text Pillar 3: Trajectory Forecasting

| Task | Type | Train Clips | Status |
|------|------|------------|--------|
| trajectory_lvef | regression | 2,543 | **RUNNING** (GPUs 4-7, priority 4, 30 epochs) |
| trajectory_tapse | regression | 2,872 | READY |
| trajectory_lv_mass | regression | 3,922 | READY |
| trajectory_rv_sp | regression | 2,757 | READY |
| trajectory_mr_severity | regression | 24,735 | READY |

### Tier 4 — Extended Data (batch after main text tasks)

All remaining 35 tasks. Run in order: ED1 structural → ED3 findings → ED2 hemodynamics → ED4+ED5 disease detection.

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
    zscore_params.json
```
Trajectory tasks use 1 clip per pair (no study_sampling). Format: `clip_path delta_value`.

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

**2a. Trajectory prediction (future LVEF)** — NOT STARTED

The "model predicts future cardiac states" claim is one of three pillars. If the model can't beat a "predict no change" baseline, this pillar drops. Paper survives on hemodynamics + outcomes but loses ~2 pages of main text.

- Likelihood of failure: High. 72% of patients are stable (EF change <5pp).
- Train one model on all 30-365d pairs (14,235 UHN). Stratify by time window (30-90d / 90-180d / 180-365d) at evaluation only.
- Critical baselines: "predict no change" (future EF = baseline EF); EF change classification (improved >5pp / stable / declined >5pp).
- Predict absolute future EF for training; compute delta at eval by subtracting known baseline EF.

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
| **3** | **AV Vmax from B-mode** | UHN (50K) | READY | 5 |
| **4** | **Future LVEF (30-365d)** | UHN (14K pairs) | READY | 5 |
| **5** | **TR severity + AR severity from B-mode** | UHN (139K / 75K) | READY | 10 |
| **6** | **NT-proBNP + creatinine (attentive)** | MIMIC (852 / 3,883) | READY | 10 |
| **7** | **E/e' + RVSP from B-mode** | UHN | BLOCKED (B-mode filter) | 10 |
| **8** | **RV mechanics (rv_sp, rv_fac)** | UHN | READY | 10 |
| **9** | **MIMIC outcomes (attentive)** | MIMIC (7,243) | READY | 35 |
| **10** | **Remaining trajectory tasks** | UHN | READY | 20 |
| **11** | **Extended Data tasks** | UHN + MIMIC | READY | ~200 |

### Decision Points

**After priority 3 (AV Vmax):**
- Pass -> Hemodynamic regression works too, not just classification. Proceed with full table.
- Fail -> Classification signal (MR/AS) is real but continuous prediction may be harder. Reframe hemodynamic section as severity grading rather than quantitative prediction.

**After priority 4 (trajectory LVEF):**
- Beats "predict no change" baseline -> Trajectory is a pillar. Run remaining trajectory tasks (TAPSE, MR, LV mass, RV S').
- Does not beat baseline -> Drop trajectory from main text. Compress to 1 paragraph or move to Extended Data as exploratory.

**After priority 6 (biomarkers):**
- NT-proBNP r > 0.5 or creatinine r > 0.4 -> Run cardiac output from B-mode (5 runs on UHN). The mechanistic bridge is worth pursuing.
- No improvement over linear probes -> Skip CO. Report biomarker results as-is in compressed section. Focus compute on remaining hemodynamic and outcome tasks.

### Cardiac Output — Conditional Experiment

**Run only if biomarker attentive probes show signal (decision gate after priority 6).**

Derives CO from UHN structured measurements: LVOT_VTI x pi x (LVOT_diam/2)^2 x HR. 16,137 UHN studies with all components. Also available on MIMIC via structured-measurement.csv.gz: lvot_vti (93%), lvot_diam (96%), resting_hr (100%).

Predicting CO from B-mode is Tier S (requires Doppler to measure). If it works, the mechanistic chain is airtight: frozen representations -> CO -> renal perfusion (creatinine), tissue oxygenation (lactate), cardiac strain (NT-proBNP), mortality. Transforms the biomarker section from correlational to mechanistically grounded.

Labels already built on UHN (`cardiac_output.npz`, 16K studies). MIMIC labels need to be derived from structured measurements.
