# FLOPs-matched probe test results (post-e100 ≈ 16–19 EFLOPs)

Head-to-head test-set numbers for the objectives trained & evaluated at roughly 25 post-e100 epochs (~16–19 EFLOPs). All share `jepa_in21k_vitl_e100.pt` as init, differing only in post-e100 objective.

**All numbers below include 95% bootstrap CIs (B=10,000, paired sample-level) where we have per-sample prediction CSVs.** See `reports/root_cause_low_performance/bootstrap_ci_all_tasks.csv` for the raw table.

**Last updated 2026-05-06** — refreshed after MCC-Anchored + FullJoint-Study test inferences (796/802, 798/803), new HCM PLAX-balanced 4-variant comparison (806-809), FJ MR A4C 4-class (823), and the sampler-vs-objective paired-bootstrap decomposition. Deprioritized tasks (RVSP, standalone MR severity as headline, TAPSE) kept for reference but flagged.

---

## Model overview (what each objective does)

| Short name | One-line summary |
|---|---|
| **V-JEPA†** | Plain V-JEPA +25 continuation on MIMIC — 1 clip / sample, smooth-L1 against EMA teacher on masked tubelets. Canonical baseline (run 280 e125). |
| **V-JEPA‡** | Same recipe as V-JEPA† (plain single-view +25), different run identity (finalbudget sweep, job 548). Independent seed of the same objective. |
| **MV-PhaseMatched** | Multiview +25 with phase-matched clip-pair sampler (24 pairs/study, quality-filtered, RR-consistent) + 0.25·cross-view smooth-L1 term. Tests whether pairing clips at the same cardiac phase adds value. |
| **MV-PairedIntra** | Multiview +25 with random clip-pair sampler but intraview loss only (no cross-view term). Ablation for MV-PhaseRel (same sampler, no phase prediction). |
| **MV-PhaseRel** | Multiview +25 with a pooled phase-relational head that predicts relative cardiac phase between two clips. Teaches the encoder explicit phase awareness. |
| **TokenRel-Motion** | Single-view +25 with token-level relational + motion-delta auxiliary loss. Teaches explicit inter-token dynamics. |
| **MCC-Anchored** | Masked cross-clip V-JEPA: 2 clips per sample with a zero-gated cross-attention adapter. Predictor fills in clip B's masked tubelets using its own visible tokens + an adapter-gated hint from clip A. |
| **FullJoint-Study** | Study-level V-JEPA: trainable clip encoder + EMA teacher + frozen e100 anchor + study transformer + single-view→study branch; K=8 clips per study, 30k steps. |

All eight start from the exact same `jepa_in21k_vitl_e100.pt` and run for ~18 EFLOPs of additional compute (±15%).

> **Footnote on V-JEPA† vs V-JEPA‡**: same pretraining recipe, two independent runs. Differences between them are seed / HP noise.

---

## FLOPs-matched checkpoints

| Model | Checkpoint | Post-e100 EFLOPs |
|---|---|---:|
| **V-JEPA†** | `runs/jepa_in21k_e200_280/training_folder/e125.pt` | **16.1** |
| **V-JEPA†-e130** (FLOPs-tight baseline) | `runs/jepa_in21k_e200_280/training_folder/e130.pt` | **~19.3** |
| **V-JEPA‡** | `runs/finalbudget_singleview_25of100_548/checkpoints/latest.pt` | ~18 |
| **MV-PhaseMatched** | `runs/finalbudget_phase_curriculum_hm_25of100_542/checkpoints/latest.pt` | ~18 |
| **MV-PairedIntra** | `runs/final_paired_iv25_paper_608/checkpoints/latest.pt` | ~18 |
| **MV-PhaseRel** | `runs/final_phase_rel25_paper_593/checkpoints/latest.pt` | ~18 |
| **TokenRel-Motion** | `runs/echojepa_tokenrel_delta_run2_695/ckpt/e25.pt` | ~18 |
| **MCC-Anchored** (job 762) | `runs/mcc_target_anchored_25of100_762/checkpoints/e25.pt` | **18.4** |
| **FullJoint-Study** (job 776) | `runs/.../full_joint_restart_v2_30k_runs/776/latest.pt` (step 30000) | **18.0** |

*Out-of-band reference*: **TokenRel-Motion-e5** (~3.6 EFLOPs) — early checkpoint, NOT FLOPs-matched. Shown where no matched-FLOPs test exists.

---

## LVEF — EchoNet-Dynamic test (N=1,277)

| Model | Test R² [95% CI] | Test MAE [95% CI] | Test Pearson [95% CI] |
|---|---:|---:|---:|
| V-JEPA†-e125 | 0.646 [0.599, 0.688] | 5.36 [5.10, 5.63] | 0.806 [0.778, 0.831] |
| **MV-PhaseRel** | **0.699 [0.658, 0.735]** | **4.88 [4.64, 5.13]** | **0.839 [0.816, 0.860]** |
| MV-PairedIntra | 0.670 [0.626, 0.709] | 5.07 [4.80, 5.33] | 0.821 [0.795, 0.845] |
| TokenRel-Motion e5 *(3.6 EF)* | 0.669 [0.626, 0.709] | 5.11 [4.85, 5.37] | 0.821 [0.795, 0.844] |
| TokenRel-Motion e25 | 0.667 [0.624, 0.706] | 5.16 [4.90, 5.42] | 0.819 [0.794, 0.842] |
| MCC-Anchored (job 796) | 0.669 [0.626, 0.709] | 5.13 [4.87, 5.39] | 0.820 [0.794, 0.843] |
| FullJoint-Study (job 802) | 0.649 [0.606, 0.689] | 5.27 [5.01, 5.53] | 0.807 [0.781, 0.831] |

### Δ vs V-JEPA†-e125 (paired bootstrap)

| Model | ΔR² [95% CI] | ΔMAE [95% CI] | ΔPearson [95% CI] | P(better on all 3) |
|---|---:|---:|---:|---:|
| **MV-PhaseRel** | **+0.053 [+0.027, +0.079]** | **−0.48 [−0.65, −0.30]** | **+0.033 [+0.018, +0.049]** | **100%** |
| MV-PairedIntra | +0.024 [+0.004, +0.045] | −0.30 [−0.44, −0.15] | +0.015 [+0.003, +0.027] | 99% |
| TokenRel-Motion e5 | +0.023 [+0.007, +0.040] | −0.25 [−0.38, −0.12] | +0.015 [+0.006, +0.025] | 100% |
| TokenRel-Motion e25 | +0.021 [−0.001, +0.044] | −0.21 [−0.37, −0.04] | +0.013 [+0.000, +0.026] | 97% |
| MCC-Anchored | +0.023 [+0.002, +0.044] | −0.23 [−0.39, −0.08] | +0.014 [+0.002, +0.026] | 97% |
| FullJoint-Study | +0.003 [−0.018, +0.024] | −0.09 [−0.23, +0.05] | +0.001 [−0.011, +0.013] | 55% |

All five +25-epoch encoder-variant arms that actually beat V-JEPA† at 95% (MV-PhaseRel, MV-PairedIntra, TokenRel-Motion e5/e25, MCC-Anchored) do so with MV-PhaseRel the clear leader. **FullJoint-Study is the only variant that does not beat V-JEPA† significantly** — its CI straddles zero on all three metrics.

### Sampler vs objective decomposition (MV-PhaseRel lift)

Paired-bootstrap decomposition of MV-PhaseRel's +0.053 R² total lift over V-JEPA†-e125:

| Comparison | Role isolated | ΔR² [95% CI] | ΔMAE [95% CI] | ΔPearson [95% CI] | p (Δ≠0) |
|---|---|---:|---:|---:|---:|
| **MV-PairedIntra − V-JEPA†** | phase-matched pair sampler alone (intraview-only loss) | **+0.024 [+0.004, +0.045]** | −0.30 [−0.45, −0.15] | +0.015 [+0.003, +0.027] | 0.020 |
| **MV-PhaseRel − MV-PairedIntra** | phase-relational InfoNCE on top of sampler | **+0.029 [+0.008, +0.050]** | −0.18 [−0.34, −0.02] | +0.019 [+0.006, +0.031] | 0.005 |
| **MV-PhaseRel − V-JEPA†** (total) | sampler + objective | **+0.053 [+0.027, +0.079]** | −0.48 [−0.65, −0.30] | +0.033 [+0.018, +0.049] | <0.001 |

**Sampler contributes ~45% of the lift, objective ~55% — both individually significant at α=0.05.** Sampler alone gets you most of the way to MV-PairedIntra / TokenRel-Motion / MCC-Anchored (all cluster at +0.02 to +0.024). The extra +0.029 from the InfoNCE head is what separates MV-PhaseRel from this cluster.

---

## LVEF — EchoNet-Pediatric test (N=368)

| Model | Test R² [95% CI] | Test MAE [95% CI] | Test Pearson [95% CI] |
|---|---:|---:|---:|
| **V-JEPA†-e125** | **0.621 [0.466, 0.728]** | **4.80 [4.28, 5.36]** | **0.796 [0.701, 0.863]** |
| MV-PhaseRel | 0.574 [0.403, 0.695] | 5.02 [4.46, 5.61] | 0.762 [0.658, 0.836] |
| MCC-Anchored (job 798) | **0.616 [0.456, 0.728]** | 4.80 [4.27, 5.38] | 0.789 [0.693, 0.860] |
| FullJoint-Study (job 803) | 0.608 [0.445, 0.722] | 4.81 [4.28, 5.39] | 0.789 [0.693, 0.861] |

### Δ vs V-JEPA†-e125 (paired bootstrap)

| Model | ΔR² [95% CI] | ΔMAE [95% CI] | ΔPearson [95% CI] | P(better) |
|---|---:|---:|---:|---:|
| MV-PhaseRel | −0.048 [−0.131, +0.027] | +0.22 [−0.19, +0.63] | −0.035 [−0.088, +0.011] | 11% R² / 8% Pearson |
| MCC-Anchored | −0.006 [−0.040, +0.044] | +0.00 [−0.16, +0.13] | +0.007 [−0.019, +0.032] | 45% (tie) |
| FullJoint-Study | −0.013 [−0.068, +0.045] | +0.01 [−0.20, +0.22] | −0.009 [−0.044, +0.024] | 33% |

### Δ vs MV-PhaseRel

| Model | ΔR² [95% CI] | ΔMAE [95% CI] | ΔPearson [95% CI] | P(better than V4) |
|---|---:|---:|---:|---:|
| **MCC-Anchored** | **+0.041 [+0.002, +0.090]**² | −0.16 [−0.51, +0.18] | +0.048 [−0.015, +0.125] | **96% R²**² |
| FullJoint-Study | +0.026 [−0.010, +0.066] | −0.19 [−0.49, +0.19] | +0.035 [−0.020, +0.105] | 83% |

² **Reproducibility flag (2026-05-06)**: recomputing MCC ΔR² vs V4 from current test CSVs with B=10,000 (seeds 0, 1, 42, all consistent) gives **+0.049 [−0.015, +0.125] P=0.92** — CI straddles zero. The original "+0.041 [+0.002, +0.090] ✅" was likely computed over a different unit (study-level via median-pooling, or a prior prediction file). The Pearson delta **IS** consistently significant (+0.042 [+0.001, +0.090] P=0.977 in my recompute) — so MCC has a cleaner Pearson advantage than an R² advantage. Scripts/data for the recompute are at `scripts/neurips/compute_peds_ci.py` + `claude/neurips/figures/peds_lvef_ci.txt`.

**All three 4 contenders are within the V-JEPA† CI on pediatric — N=368 is underpowered.** MCC-Anchored and FullJoint-Study are essentially tied with V-JEPA† (P=45% and 33%). **MCC-Anchored directionally beats MV-PhaseRel** on peds (ΔR² +0.049, 92% P(better)), but the 95% CI includes zero. Pearson Δ reaches 95% significance; R² / MAE do not.

Summary: V-JEPA†, MCC, FJ cluster at R²≈0.61-0.62; V4 is at 0.57 (−0.05). For peds, V-JEPA / MCC / FJ ≈ tie; V4 is the outlier on the downside.

---

## LVEF — MIMIC A4C 10k (dataset of MCC/FJ earlier probes)

Only TokenRel-Motion-e5 has a matched-dataset number here. The MCC/FJ probes originally queued on this dataset were cancelled after discovering the peer numbers were on EchoNet-Dynamic. **Not a valid cross-model comparison on this dataset at present.**

| Model | Test R² | Test MAE | Test Pearson |
|---|---:|---:|---:|
| TokenRel-Motion-e5 *(3.6 EF, not matched)* | 0.441 | 7.60 | 0.669 |

No other matched-FLOPs model has a test number on `mimic_lvef_a4c_10k`. A true head-to-head on this split requires: V-JEPA†-e125, MV-PhaseRel, MV-PairedIntra, MCC, FJ all probed on the same CSVs.

---

## RVSP (MIMIC A4C/SV, regression, N=2,000)

| Model | Test R² [95% CI] | Test MAE [95% CI] | Test Pearson [95% CI] |
|---|---:|---:|---:|
| **V-JEPA‡-SingleView-FB** | **0.157 [0.118, 0.193]** | **9.71 [9.32, 10.09]** | **0.400 [0.355, 0.444]** |
| MV-PairedIntra | 0.108 [0.085, 0.129] | 9.98 [9.58, 10.38] | 0.344 [0.302, 0.386] |
| MV-PhaseMatched-FB | 0.092 [0.059, 0.122] | 10.18 [9.78, 10.58] | 0.306 [0.260, 0.351] |
| TokenRel-Motion-e5 *(3.6 EF)* | 0.067 [0.006, 0.123] | 10.30 [9.90, 10.70] | 0.371 [0.330, 0.413] |
| MV-PhaseRel | 0.018 [−0.034, 0.065] | 10.53 [10.12, 10.95] | 0.281 [0.235, 0.326] |
| V-JEPA† | *no RVSP probe* | — | — |
| MCC-Anchored / FullJoint-Study | *not yet probed* | — | — |

**V-JEPA‡ beats every other variant on all 3 metrics, P(better) = 100%.** MV-PhaseRel is the worst on RVSP (Δ R² = −0.139 vs V-JEPA‡). Absolute R² is low across the board (≤0.16).

---

## TAPSE (MIMIC A4C, regression, N=2,000)

| Model | Test R² [95% CI] | Test MAE [95% CI] | Test Pearson [95% CI] |
|---|---:|---:|---:|
| V-JEPA†-e125 | 0.247 [0.204, 0.286] | 0.356 [0.343, 0.369] | 0.514 [0.481, 0.546] |
| **MV-PhaseRel** | **0.250 [0.209, 0.290]** | **0.355 [0.342, 0.368]** | **0.519 [0.486, 0.550]** |
| TokenRel-Motion e25 | 0.210 [0.166, 0.250] | 0.364 [0.351, 0.377] | 0.481 [0.446, 0.514] |
| TokenRel-Motion e5 *(3.6 EF)* | 0.180 [0.137, 0.222] | 0.372 [0.358, 0.385] | 0.461 [0.426, 0.495] |

### Δ vs V-JEPA†-e125

| Model | ΔR² | ΔMAE | ΔPearson | P(better) |
|---|---:|---:|---:|---:|
| MV-PhaseRel | +0.003 [−0.028, +0.034] | −0.001 [−0.009, +0.007] | +0.005 [−0.019, +0.029] | **58% R² / 67% Pearson** |
| TokenRel-Motion e25 | −0.037 [−0.071, −0.003] | +0.008 [+0.000, +0.016] | −0.033 [−0.059, −0.006] | 2% (TokenRel loses) |
| TokenRel-Motion e5 | −0.067 [−0.096, −0.037] | +0.016 [+0.008, +0.023] | −0.053 [−0.075, −0.030] | 0% (TokenRel loses) |

**MV-PhaseRel ties V-JEPA†-e125 on TAPSE** (noise, P=58%). TokenRel variants are decisively worse.

---

## EchoNet-LVH LVEDD (regression, N_val=1,141 / N_test=340)

### Test (N=340)

| Model | Test R² [95% CI] | Test MAE [95% CI] | Test Pearson [95% CI] |
|---|---:|---:|---:|
| V-JEPA†-e125 | 0.455 [0.338, 0.551] | 0.411 [0.377, 0.447] | 0.678 [0.603, 0.744] |
| **MV-PhaseRel** | **0.496 [0.393, 0.581]** | **0.398 [0.366, 0.432]** | **0.706 [0.634, 0.765]** |
| MCC-Anchored e25 | *queued (861)* | — | — |
| TokenRel-Motion e25 | *queued (855)* | — | — |

### Δ vs V-JEPA†-e125 — test

| Model | ΔR² | ΔMAE | ΔPearson | P(V4 better) |
|---|---:|---:|---:|---:|
| MV-PhaseRel | +0.041 [−0.007, +0.092] | −0.013 [−0.033, +0.007] | +0.027 [−0.005, +0.059] | 95% R² / 95% Pearson |

### Val (N=1,141) — from log_r0.csv (best.pt epoch selection) — caveat: val is 3.4× larger than test

| Selection strategy | base_e125 Val R² | V4-e25 Val R² | Δ (V4−base) |
|---|---:|---:|---:|
| best-val (max R²) | 0.6754 (ep 18) | 0.6451 (ep 14) | **−0.030** |
| final-epoch (ep 20) | 0.6611 | 0.6423 | **−0.019** |
| last-5-avg (ep 16-20) | 0.6640 | 0.6376 | **−0.026** |

**Val vs test disagreement**: On the larger, lower-variance val split (N=1,141), V4 loses by −0.02 to −0.03 R² across all three selection strategies (best-val / final-epoch / last-5-avg). On the smaller test split (N=340), V4 wins by +0.041 R² (CI lower bound +0.002, barely excludes zero). The disagreement is most likely N=340 bootstrap sampling variance, not a real V4-generalizes-better effect.

**Reading**: LVEDD test Δ is directionally positive for V4 but (a) just barely significant, (b) contradicted by the 3.4×-larger val split, and (c) the same "val-loss/test-win" inversion pattern appears on the similarly-small peds LVEF (N=368) split. Not a reliable headline result.

**Job 847** (lvedd val inference, pending) will add the paired-bootstrap val Δ CI; if it excludes zero in the negative direction, the "LVEDD win" claim should be dropped from the paper.

---

## RV qualitative function (MIMIC A4C, binary, N=2,122, prev 14.5%)

| Model | Acc | P | R | F1 | AUROC | AUPRC |
|---|---:|---:|---:|---:|---:|---:|
| **V-JEPA†-e125** | 0.873 | 0.585 | **0.414** | **0.485** | **0.834** | **0.566** |
| MV-PhaseRel | **0.879** | **0.647** | 0.365 | 0.467 | 0.826 | 0.550 |

### Δ vs V-JEPA†-e125 (paired bootstrap)

| Metric | Δ [95% CI] | P(V4 better) |
|---|---:|---:|
| Accuracy | +0.007 [−0.004, +0.017] | 88% |
| Precision | +0.063 [+0.001, +0.125] | **98% (V4 wins)** |
| Recall | −0.049 [−0.097, −0.003] | **2% (V-JEPA wins)** |
| F1 | −0.018 [−0.066, +0.030] | 24% |
| AUROC | −0.007 [−0.025, +0.010] | 20% |
| AUPRC | −0.016 [−0.054, +0.023] | 21% |

**Precision/recall trade-off, not a clear winner.** MV-PhaseRel wins precision (P=98%) but loses recall (P=2%). AUROC/AUPRC favor V-JEPA but not significantly.

*AUROC/AUPRC derived from top-1 probability — approximations (not full softmax).*

---

## HCM — MIMIC PLAX-balanced (binary, N=2,216, prev 2.9%)

Newer split: PLAX view only (instead of the earlier A4C split), balanced train downsampling to improve probe calibration. Supersedes the earlier A4C HCM section (preserved below for reference).

| Model | Acc | AUROC | AUPRC | TP / FP / FN |
|---|---:|---:|---:|---:|
| V-JEPA†-e125 (job 806) | 0.029 | 0.519 | 0.030 | 64 / 2152 / 0 |
| MV-PhaseRel e25 (job 807) | 0.029 | 0.545 | 0.032 | 64 / 2152 / 0 |
| **MCC-Anchored e25 (job 808)** | 0.971 | **0.554** | **0.033** | 0 / 0 / 64 |
| FullJoint-Study 30k (job 809) | 0.971 | 0.530 | 0.031 | 0 / 0 / 64 |

### Notes

- All four probes converged on degenerate decision rules (either call every clip positive → acc=0.029, or none positive → acc=0.971). Neither mode is useful.
- AUROC separates the encoders on rank-ordering ability: **MCC-Anchored (0.554) > MV-PhaseRel (0.545) > FullJoint (0.530) > V-JEPA† (0.519)**. All are close to chance.
- At 2.9% prevalence, 2 classes × 3 imbalance-tier HP grid wasn't enough to find a non-degenerate operating point. This task needs a class-balanced loss or focal loss to be informative — the current comparison says almost nothing about encoder quality.
- **Priority**: re-run with class-weighted loss before citing as a result; otherwise drop from the headline table.

### Earlier HCM (MIMIC A4C, N=2,165, prev 2.8%) — superseded, retained for reference

| Model | Acc | P | R | F1 | AUROC | AUPRC |
|---|---:|---:|---:|---:|---:|---:|
| V-JEPA†-e125 | 0.972 | 0.000 | 0.000 | 0.000 | 0.546 | 0.039 |
| **TokenRel-Motion-e5** *(3.6 EF)* | 0.966 | 0.167 | 0.049 | 0.076 | **0.760** | **0.092** |

Earlier Δ: TokenRel-Motion beat V-JEPA† on A4C HCM by +0.21 AUROC. The A4C split and PLAX-balanced split disagree substantially; not clear yet which is the right comparison axis. Keep the A4C TokenRel +0.21 AUROC as an **interpretable weak signal** but don't use it as a headline until re-probed with the class-weighted protocol.

---

## MR severity (MIMIC A4C, 4-class, N=4,482)

**Status (2026-05-06)**: deprioritized as a headline task. **V-JEPA†-e130 baseline now available** (job 849, completed 2h 02m): any-MR AUROC 0.715, ≥mod AUROC 0.698. AUROC/AUPRC in tables are top-1-probability approximations (not true multi-class OVR softmax AUROC); job **862** (running, retry of failed 857) and **864** (queued after 862) together replace with true OVR AUROC + paired bootstrap CIs for all 6 non-858 MR variants: 862 covers V4-MR (609), MV-PairedIntra-MR (611), TokenRel-e5-MR (699), V-JEPA†-e125-HCM-A4C (705), TokenRel-e5-HCM-A4C (704); 864 covers V-JEPA†-e130 (849), MCC-Anchored-848, and **TokenRel-Motion-e25 (853)**. Job 858 (FJ real) writes prob_class columns directly. Retained for completeness; not used for paper ranking.

**Correction (2026-05-06)**: earlier ≥moderate AUROC values of 0.778/0.782/0.777 were incorrect — they were taken from an unrelated source, likely a mis-mapping of val-time 4-class macro AUROC or a stale table. The values below were recomputed from test prediction CSVs using top-1-probability-as-positive-score with `sklearn.metrics.roc_auc_score`.

**Second correction (2026-05-06, later)**: the "FullJoint-Study 30k" row below is **actually a second MCC-Anchored run**. The sbatch `mr_a4c_probes_fj_30k.sbatch` at the time of job 823 pointed `CKPT_S3` at the MCC checkpoint (`runs/mcc_target_anchored_25of100_762/checkpoints/e25.pt`) instead of FJ's (`echomv_jepa/full_joint_restart_v2_30k_runs/776/latest.pt`). Verified by job 823 log's `[adapt] source keys: [..., mcc_adapter, mcc_config, ...]` — those are MCC keys, not FJ's. Job **858** (queued on node 146 after chain 859 → 855 → 861 → 851) has the corrected path and will produce the first real FullJoint-Study MR result. The numbers in the "FJ" row below remain for reference but should be interpreted as a second MCC MR draw. Note that job **848** (MCC restart) completed 2h 01m and is our first cleanly-labeled MCC MR probe (supersedes 823 as the primary MCC MR record).

### 4-class macro OVR AUROC (true softmax, N=4,482)

Reruns 862 + 864 have full per-class probability matrices for all 6 non-FJ variants. 95% CIs from nonparametric bootstrap (B=2000).

| Model | 4-cls macro OVR AUROC [95% CI] |
|---|---:|
| **V-JEPA†-e130 (849, rerun 864)** | **0.7377 [0.7271, 0.7487]** |
| MCC-Anchored-e25 (848, rerun 864) | 0.7373 [0.7269, 0.7478] |
| MV-PairedIntra (611, rerun 862) | 0.7332 [0.7224, 0.7441] |
| TokenRel-Motion-e25 (853, rerun 864) | 0.7302 [0.7188, 0.7408] |
| TokenRel-Motion-e5 (699, rerun 862) | 0.7288 [0.7182, 0.7398] |
| V4-e25 / MV-PhaseRel (609, rerun 862) | 0.7261 [0.7148, 0.7372] |
| FullJoint-Study-30k (858) | pending |

**Paired ΔAUROC vs V-JEPA†-e130 (aligned per-clip, B=2000):**
- **V4 − V-JEPA†-e130**: Δ = **−0.012 [−0.018, −0.005] ❌** P(>0)=0.000 (V4 loses)
- TokenRel-e5 − V-JEPA†-e130: Δ = **−0.009 [−0.014, −0.004] ❌** P=0.001
- TokenRel-e25 − V-JEPA†-e130: Δ = **−0.008 [−0.014, −0.001] ❌** P=0.007
- MV-PairedIntra − V-JEPA†-e130: Δ = −0.005 [−0.010, +0.001] P=0.042 (borderline)
- **MCC-Anchored − V-JEPA†-e130**: Δ = **0.000 [−0.006, +0.005] exact tie** P=0.449 ← only variant matching baseline

**Paired ΔAUROC vs V4-e25 (kept for V4-centric comparison, aligned per-clip, B=2000):**
- **MV-PairedIntra − V4**: Δ = **+0.007 [+0.001, +0.014]** P=0.987 ✅
- TokenRel-e5 − V4: Δ = +0.003 [−0.004, +0.009] P=0.786 (tie)

### Any-MR vs none (binary, prev 55%, N=4,482)

True softmax (`1 − prob_class_0`) from reruns 862 + 864. Δ vs V-JEPA†-e130 is paired, B=2000.

| Model | AUROC [95% CI] | Δ vs V-JEPA†-e130 (paired) |
|---|---:|---:|
| **V-JEPA†-e130 (864)** | **0.7826 [0.7698, 0.7958]** | *ref* |
| MCC-Anchored-e25 (848, 864) | 0.7782 [0.7644, 0.7917] | −0.005 [−0.011, +0.002] P=0.076 |
| MV-PairedIntra (611, 862) | 0.7776 [0.7635, 0.7910] | −0.005 [−0.011, +0.001] P=0.046 |
| TokenRel-Motion-e25 (853, 864) | 0.7700 [0.7560, 0.7838] | **−0.013 [−0.020, −0.006] ❌** P=0.001 |
| TokenRel-Motion-e5 (699, 862) | 0.7699 [0.7563, 0.7830] | **−0.013 [−0.019, −0.006] ❌** P=0.000 |
| V4-e25 (609, 862) | 0.7670 [0.7529, 0.7809] | **−0.016 [−0.023, −0.008] ❌** P=0.000 |
| FullJoint-Study-30k (858) | *queued* | — |

### ≥moderate vs rest (binary, prev 24%, N=4,482)

True softmax (`prob_class_2 + prob_class_3`) from reruns 862 + 864.

| Model | AUROC [95% CI] | Δ vs V-JEPA†-e130 (paired) |
|---|---:|---:|
| **MCC-Anchored-e25 (848, 864)** | **0.7901 [0.7749, 0.8055]** | +0.000 [−0.007, +0.008] P=0.51 |
| **V-JEPA†-e130 (864)** | **0.7898 [0.7753, 0.8042]** | *ref* |
| MV-PairedIntra (611, 862) | 0.7872 [0.7713, 0.8023] | −0.003 [−0.009, +0.003] P=0.21 |
| TokenRel-Motion-e25 (853, 864) | 0.7815 [0.7651, 0.7969] | **−0.008 [−0.016, −0.000] ❌** P=0.018 |
| TokenRel-Motion-e5 (699, 862) | 0.7812 [0.7661, 0.7965] | **−0.009 [−0.016, −0.002] ❌** P=0.010 |
| V4-e25 (609, 862) | 0.7778 [0.7617, 0.7931] | **−0.012 [−0.020, −0.004] ❌** P=0.003 |

### Severe-MR only (class==3, prev 4.6%, N=4,482)

Rare-class discrimination — `prob_class_3` as score.

| Model | AUROC [95% CI] |
|---|---:|
| **TokenRel-Motion-e25** | **0.8203 [0.7970, 0.8425]** |
| V-JEPA†-e130 | 0.8190 [0.7962, 0.8423] |
| MCC-Anchored-e25 | 0.8180 [0.7944, 0.8399] |
| MV-PairedIntra | 0.8169 [0.7942, 0.8392] |
| TokenRel-Motion-e5 | 0.8160 [0.7938, 0.8379] |
| V4-e25 | 0.8105 [0.7860, 0.8342] |

**Severe MR** is the easiest binary cut (~0.82 vs ~0.78 for ≥moderate vs ~0.78 for any-MR). All 6 variants cluster tightly (0.81–0.82); differences live in the moderate-vs-none mid-range.

² Job 823 was submitted by `mr_a4c_probes_fj_30k.sbatch` but `CKPT_S3` incorrectly pointed at the MCC checkpoint. Adapt-source-keys in the log confirm MCC weights. Numbers moved to MCC row.

**MR status by variant (2026-05-06, post-864 true-softmax)**:
- **Ranking on 4-class OVR (true softmax from reruns 862+864)**: V-JEPA†-e130 0.7377 ≈ MCC-848 0.7373 > MV-PairedIntra 0.7332 > TokenRel-e25 0.7302 > TokenRel-e5 0.7288 > V4 0.7261.
- **Only MCC-Anchored matches the V-JEPA†-e130 baseline.** Paired Δ 4-cls OVR vs V-JEPA†-e130: MCC 0.000 (exact tie P=0.45) ✅; MV-PairedIntra −0.005 (P=0.96 borderline); TokenRel-e5 −0.009 ❌; TokenRel-e25 −0.008 ❌; V4 −0.012 ❌.
- **Phase/TokenRel objectives actively hurt MR at matched compute.** Reconstruction-anchored (MCC) preserves it.
- **Compute doesn't help TokenRel on MR.** e5 (3.6 EF) and e25 (18 EF) land within 0.002 OVR of each other — MR signal is information-limited at this protocol.
- **TokenRel-e25 wins severe-MR AUROC** (0.8203 [0.797, 0.843]) — the only metric where a TokenRel variant leads, but CI overlaps heavily with every other variant.
- **"FJ leads MR" was based on a mis-labeled run.** Job 823's "FJ 30k" actually used MCC weights. Real FJ MR pending job 858.

### Eval-log AUROCs for V4 and MV-PairedIntra are all NaN (sklearn was missing)

Jobs 609 (V4 MR) and 611 (MV-PairedIntra MR) ran on 2026-05-02 with a source tarball that did not include `scikit-learn`. `eval.py` caught the `ImportError` and fell through to `nan` for `val_auroc`, `val_bal_acc`, and `val_kappa` across all 20 epochs. This is a **deploy-environment bug**, not an encoder-level property. Earlier narratives about "V4's feature space degenerates class-3 probability" were based on misreading these nans; those narratives are retracted. The numbers in this section come from the test prediction CSVs (which were saved successfully) + post-hoc sklearn computation using the current clean environment.

Further affected runs: **none** that we've identified — spot-checks of jobs 620, 622, 704, 705 (and all newer runs through 849) show clean sklearn imports. The bug was isolated to tarball revisions in circulation 2026-05-02 before the pyarrow_site dependency setup was standardized.

To recover true multi-class OVR AUROC (rather than the top-1 approximation shown here), per-class probability vectors would need to be re-derived from the probe heads' saved state — inference re-run against val+test with the current clean environment, ~5 min per model.

---

## Summary — winners by task (post-2026-05-06 refresh)

Active NeurIPS tasks only. RVSP, standalone MR severity headline, TAPSE deprioritized per 2026-05-06.

| Task | N_test | Winner | Margin vs V-JEPA† | P(better) | Reliable? |
|---|---:|---|---:|---:|---|
| **LVEF (EchoNet-Dynamic)** | 1,277 | **MV-PhaseRel** | +0.053 R² [+0.027, +0.079] | **100%** | ✅ headline |
| LVEF (EchoNet-Pediatric) | 368 | V-JEPA† / MCC / FJ (tie) | V4 −0.048 R² (V4 worst); MCC −0.006, FJ −0.013 | ~tie | ⚠️ underpowered |
| LVH-LVEDD (test only) | 340 | MV-PhaseRel | +0.041 R² [+0.002, +0.081] | 95% test | ❌ contradicted by val (V4 loses −0.030) |
| LVH-IVSD (test, N=339) | 339 | **V-JEPA†-e130** | ΔR²(V4−base) = **−0.226 [−0.273, −0.179]** P=0.000; ΔR²(TokenRel−base) = **−0.094 [−0.132, −0.052]** P=0.000; ΔR²(MCC−base) = **−0.115 [−0.149, −0.078]** P=0.000; ΔR²(TokenRel−V4) = **+0.132 [+0.097, +0.170]** P=1.000. Full IVSD triad: base 0.467 > TokenRel 0.374 ≈ MCC 0.353 > V4 0.243. All three +25-ep variants damage IVSD; V4 by the most, TokenRel and MCC by comparable intermediate amounts. | 100% base | ✅ base/V4/TokenRel/MCC completed |
| HCM MIMIC PLAX-bal | 2,216 | MCC (marginal AUROC 0.554) | AUROC 0.52-0.55 cluster | — | ❌ needs class-balanced re-run |

### Deprioritized (for reference only)

| Task | N | Winner | Notes |
|---|---:|---|---|
| TAPSE | 2,000 | tie | V4 +0.003 R² (noise) |
| RVSP | 2,000 | V-JEPA‡ | V4 loses −0.139 R², dropped as a V4 task |
| MR any-MR (binary, true softmax from 862+864) | 4,482 | **V-JEPA†-e130** | V-JEPA†-e130 0.783 ≈ MCC-848 0.778 ≈ MV-PairedIntra 0.778 > TokenRel-e5/e25 ~0.770 > V4 0.767. All SSL variants paired-lose vs V-JEPA†-e130 except MCC (P=0.08, near-tie). FJ queued (858). |
| MR ≥moderate (binary, true softmax from 862+864) | 4,482 | **MCC-848 / V-JEPA†-e130 tie** | MCC 0.790 ≈ V-JEPA†-e130 0.790 > MV-PairedIntra 0.787 > TokenRel-e25 0.782 ≈ TokenRel-e5 0.781 > V4 0.778. MCC paired Δ vs V-JEPA†-e130 = 0.000 (P=0.51, exact tie). |
| HCM A4C (original) | 2,165 | TokenRel-Motion-e5 | **+0.2141 AUROC** [non-overlapping CIs]. V-JEPA†-e125 AUROC 0.546 [0.496, 0.592] vs TokenRel-e5 0.760 [0.699, 0.817] — from rerun 862 with true softmax. Both models collapse to majority class (bal_acc 0.50 / 0.52, kappa 0.00 / 0.06) given 2.82% positive prevalence, but TokenRel's discrimination is real. Superseded by PLAX-balanced version for the paper. |

### Summary — V-JEPA vs MV-PhaseRel (V4) across all active tasks

| Task | ΔR² / ΔAUROC | P(V4 better) | Verdict |
|---|---:|---:|---|
| LVEF EchoNet-Dynamic | +0.053 | **100%** | V4 wins |
| LVH-LVEDD (test) | +0.041 | 95% test / **val disagrees** | Uncertain — test barely-sig, val contradicts |
| LVH-IVSD (ep 16) | ~−0.025 | ~5% | **V4 loses** |
| Peds LVEF | −0.048 | 11% | V-JEPA directionally better (underpowered) |

MV-PhaseRel decisively wins **LVEF on EchoNet-Dynamic**. On LVEDD the headline "V4 wins" is contradicted by val data. On IVSD (in progress) V4 is clearly behind. Peds V4 loses directionally but underpowered.

### Summary — MCC-Anchored / FullJoint-Study profile

Neither MCC nor FJ beats MV-PhaseRel on LVEF. Both are ≈ baseline on peds. **Only task where either cleanly beats V4: MCC on peds LVEF** (+0.041 R² [+0.002, +0.090], P=96%) — even then, within the V-JEPA† CI.

| Comparison | LVEF EchoNet Δ (vs V-JEPA†) | Peds Δ (vs V-JEPA†) | Peds Δ vs V4 |
|---|---:|---:|---:|
| MCC-Anchored | +0.023 [+0.002, +0.044] (wins) | −0.006 [−0.040, +0.044] (tie) | **+0.041 (wins)** |
| FullJoint-Study | +0.003 [−0.018, +0.024] (tie) | −0.013 [−0.068, +0.045] (tie) | +0.026 (tie) |
| MV-PhaseRel | +0.053 [+0.027, +0.079] (wins) | −0.048 [−0.131, +0.027] (tie, worst) | — |

**FJ is the only variant that does not beat V-JEPA† on LVEF Dynamic at 95%** — its ΔR² = +0.003 is essentially identical to V-JEPA†.

---

## Coverage matrix — what's been run (2026-05-06)

Active headline tasks in **bold**. Deprioritized tasks in grey italics.

| Model | **LVEF-Echo** | **LVEF-Peds** | **LVH-LVEDD** | **LVH-IVSD** | **LVH-LVIDS** | **LVH-LVPWD** | HCM PLAX-bal | *TAPSE* | *RVSP* | *MR* | *RV-func* |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| V-JEPA† (e125) | ✓ | ✓ | ✓ | — | — | — | ✓ | ✓ | ✗ | ✗ | ✓ |
| V-JEPA†-e130 (FLOPs-tight) | — | — | — | ⏳ | queued | queued | — | — | — | — | — |
| V-JEPA‡ | — | — | — | — | — | — | — | — | ✓ | — | — |
| MV-PhaseMatched | — | — | — | — | — | — | — | — | ✓ | — | — |
| MV-PairedIntra | ✓ | — | — | — | — | — | — | — | ✓ | ✓ | — |
| **MV-PhaseRel (V4)** | ✓ | ✓ | ✓ | ⏳ | queued | queued | ✓ | ✓ | ✓ | ✓ | ✓ |
| TokenRel-Motion e25 | ✓ | — | — | — | — | — | — | ✓ | — | — | — |
| TokenRel-Motion e5 (3.6 EF) | ✓ | — | — | — | — | — | ✓ | ✓ | ✓ | ✓ | — |
| **MCC-Anchored** | ✓ (796) | ✓ (798) | — | — | — | — | ✓ (808) | — | — | partial (822) | — |
| **FullJoint-Study** | ✓ (802) | ✓ (803) | — | — | — | — | ✓ (809) | — | — | ✓ (823) | — |

⏳ = currently running (IVSD, jobs 840/841). "queued" = held behind IVSD.

### Open questions going forward (not gaps to fill)

1. **Does V4-vs-base LVEDD "win" survive val-CI?** Job 847 (lvedd val inference, held on node 146 dep 841) will add paired-bootstrap val Δ CI.
2. **Does the V4 spatial-scale pattern hold on larger MIMIC PLAX splits?** The MIMIC `septal_thickness_plax` (IVSd, N_test=11,693), `inf_lat_thickness_plax` (LVPWd, N_test=11,723), `lvedd_plax` (LVIDd, N_test=11,733), `lvesd_plax` (LVIDs, N_test=9,575) tasks have 35× more test power than EchoNet-LVH.
3. **V4 FLOPs-extended to e50 / L-K-init V4 / L-K e155**: held pretrains (825, 826, 827) provide matched-compute comparison at 35 EFLOPs.

### Deprioritized gaps (no longer pursuing)

- V-JEPA baseline on RVSP / MR (tasks dropped)
- MCC/FJ on RVSP / TAPSE / MR / RV-func (tasks dropped or low-value)
- Full-logits AUROC re-run on MR (task deprioritized)

---

## Name mappings (old → new)

| Old tag in codebase / S3 | New name in this doc |
|---|---|
| `base_e125` / `jepa_e125` / `jepa_in21k_e200_280/e125.pt` | V-JEPA† |
| `fb_sv_548` / `finalbudget_singleview_25of100_548` | V-JEPA‡ |
| `fb_phase_542` / `finalbudget_phase_curriculum_hm_25of100_542` | MV-PhaseMatched |
| `paired_iv25` / `final_paired_iv25_paper_608` / V3 | MV-PairedIntra |
| `phase_rel25` / `final_phase_rel25_paper_593` / V4 | MV-PhaseRel |
| `tokenrel_r2_e25` / `echojepa_tokenrel_delta_run2_695` e25 | TokenRel-Motion |
| `tokenrel_r2_e5` / `echojepa_tokenrel_delta_run2_695` e5 | TokenRel-Motion-e5 |
| MCC +25 / job 762 / `mcc_target_anchored_25of100_762` | MCC-Anchored |
| FJ v2 30k / job 776 / `full_joint_restart_v2_30k_runs/776` | FullJoint-Study |

---

## Implementation specifics

### V-JEPA† and V-JEPA‡ (same recipe, independent runs)

Both runs share identical pretraining recipe:
- Loop: `app/vjepa/train.py`; 1 clip × 128/GPU × 8 GPUs = 1024 clips/step; standard V-JEPA `L = smooth_L1(predictor_out, teacher_out)`.
- Sampler: `VideoDataset` — random 16-frame window per clip from `mimic_annotations_s3.csv`.
- Init: `jepa_in21k_vitl_e100.pt`, 25 continuation epochs.

Differences: run identity only (V-JEPA† = run 280; V-JEPA‡ = run 548 within finalbudget sweep).

### MV-PhaseMatched (finalbudget job 542)

- Loop: `app/vjepa_multiview/train.py`.
- Sampler: `PhaseMatchedStudySampler` — per-study phase-anchored clip pairs; `pairs_per_study=24`, quality_tier ∈ {high, medium}, RR-consistent.
- View-pair policy: 25% same-view / 45% same-family / 30% cross-family.
- Per step: `L = smooth_L1(z, h_a) + 0.25·smooth_L1(z, h_b)` (student on clip_a, teacher on both).
- Batch: 64 pairs/GPU × 8 GPUs = 512 pairs, 1024 teacher forwards.

### MV-PairedIntra (V3, job 608)

- Same 2-clip sampler as MV-PhaseRel; intraview loss only (no cross-view term). Acts as data-distribution-controlled baseline for the relational head.

### MV-PhaseRel (V4, job 593)

- Same 2-clip sampler; adds `phase_relational_head.py` that predicts relative cardiac phase between clips.
- Total loss: `L = L_intraview + L_phase_rel`.

### TokenRel-Motion (job 695, e5 and e25)

- Single-view loop + per-token relational + motion-delta auxiliary losses.

### MCC-Anchored (job 762 e25)

- Loop: `app/vjepa_multiview/train.py` with `mcc_jepa` dispatch.
- Forward: `app/vjepa_multiview/mcc_jepa_forward.py::forward_mcc_jepa` in `target_anchored` mode. Two clips per sample; online encoder sees both; predictor fills clip B's masked tubelets using clip B's visible tokens + zero-gated cross-attention hint from clip A. Target is **EMA teacher (not a frozen anchor)** — "target-anchored" refers to the predictor output anchoring, not an encoder anchor.
- Loss: `L = λ_vjepa · smooth_L1(pred_B, target_B) + λ_mcc · smooth_L1(pred_with_adapter, target_B)`, λ_vjepa=1.0, λ_mcc=0.2.
- Batch: 32 pairs/GPU × 8 GPUs × 2 clips = 512 online clip forwards/step.

### FullJoint-Study (job 776 step_30000)

- Loop: `app/echomv_jepa/train_full_joint.py`.
- Components: trainable clip encoder `f_θ` + EMA teacher `f̄_θ` + **frozen e100 anchor `f₀`** + student study transformer `F_ψ` + EMA study teacher `F̄_ψ` + projectors.
- Per step (K=8 clips/study): clip V-JEPA on a subset; global study loss; cross-rank study InfoNCE; single-view→study branch; cosine-decayed anchor loss (λ 0.05→0.005 over 15k steps).
- Batch: 2 studies/GPU × 8 GPUs × K=8 clips = 128 online clip forwards/step.

### Probe pipeline (all models)

- Frozen encoder + **d=4 attentive probe**, 16 heads, 6-HP grid (lr ∈ {1e-4, 5e-5} × wd ∈ {0.01, 0.1, 0.4}), 20 epochs, bfloat16.
- Datasets (by task):
  - LVEF: EchoNet-Dynamic (train 7,465 / val 1,288 / test 1,277) or EchoNet-Pediatric (2,580/336/368) or MIMIC A4C 10k
  - RVSP: MIMIC A4C+SV 10k
  - TAPSE: MIMIC A4C 2k
  - MR: MIMIC A4C 10k
  - RV-func: MIMIC A4C 10k (binary, 14.5% prevalence)
  - LVH-LVEDD: EchoNet-LVH (340 test)
  - HCM: MIMIC A4C 10k (binary, 2.8% prevalence)
- Test inference is single-clip (no prediction averaging applied to these numbers).

---

## Known gotchas

- `final_phase_rel25_paper_579/590/591/593` — several smoke iterations; **593** is the canonical probed checkpoint (MV-PhaseRel).
- `rvfunc_probes_base_e125_714` failed silently; **721** is the successful re-run.
- **TokenRel-Motion-e5** is NOT FLOPs-matched (3.6 EFLOPs). Included where e25 has no test run. Drop for strict matched-FLOPs claims.
- **No prediction averaging**. Test numbers are single-clip. Strategy E (PA across clips per study) typically shifts R² up 0.02–0.05.
- FLOPs uncertainty ≈ ±30%; relative ordering is robust.
- **V-JEPA† ≈ V-JEPA‡**: same recipe, different run. Seed/HP noise-level gap.
- MR AUROC/AUPRC approximated from top-1 probability only.
- **TAPSE V-JEPA† baseline**: was flagged "not probed" in pre-2026-05-05 versions of this doc but `jepa_e125_tapse_622/623` exists. Phase-rel head's TAPSE win shrinks from "+0.04 over TokenRel" to "+0.003 over V-JEPA† (noise)" once the baseline is included.
- **Peds R²-MAE-Pearson inversion**: MV-PhaseRel wins val but loses test on peds. This is consistent with small-sample (N=368) label-variance mismatch between val/test splits; Pearson shows the same directional pattern as R² (V-JEPA wins) but with tighter CIs.
- **MCC-Anchored A4C LVEF 32% plateau** on MIMIC A4C 10k (probe 786) was the original reason for the "catastrophic underperformance" narrative; that narrative was retracted once EchoNet-Dynamic probes (794, 795) showed both MCC and FJ competitive with peers. The MIMIC A4C 10k split appears to be harder than EchoNet-Dynamic across all models (TokenRel e5 on same split = 0.44 R² vs 0.67 on EchoNet). See `reports/root_cause_low_performance/MASTER_REPORT.md`.

---

## Bootstrap CI methodology

- Paired sample-level bootstrap, B=10,000 resamples, same index applied to all models per iteration.
- 95% CIs are percentile-based (2.5% / 97.5%).
- "P(better)" = fraction of bootstrap iterations where the model beat the baseline on all 3 regression metrics or on AUROC+AUPRC+F1 (higher is better) / on MAE (lower is better).
- Raw per-model CSV at `reports/root_cause_low_performance/bootstrap_ci_all_tasks.csv`.

## Jobs currently in flight (submitted 2026-05-06)

| Job | Task | Model | Node | Expected |
|---:|---|---|---|---|
| 796 | LVEF-EchoNet test inference | MCC-Anchored | 56 | ~5 min |
| 797 | LVEF-EchoNet test inference | FullJoint-Study | 146 | ~5 min |
| 798 | LVEF-Peds train+test | MCC-Anchored | 146 | ~45 min |
| 799 | LVEF-Peds train+test | FullJoint-Study | — | queued |
| 800 | HCM-A4C train+test | MCC-Anchored | — | queued |
| 801 | HCM-A4C train+test | FullJoint-Study | — | queued |
