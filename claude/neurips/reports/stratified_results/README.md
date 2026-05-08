# Stratified regression test results — adult LVEF, LVEDD, pediatric LVEF

Per-stratum breakdown of test predictions for clinical regression probes. Stratification splits the test set into clinical categories (reduced/mildly reduced/normal/hyperdynamic) and reports per-stratum MAE, Pearson, and R². Also reports binary clinical-decision metrics (sensitivity, specificity, AUROC) at clinically-actionable thresholds.

**Overall patterns**:
- **MV-PhaseRel** dominates adult cohorts (EchoNet-Dynamic LVEF, EchoNet-LVH LVEDD) across every stratum.
- **V-JEPA† and FullJoint-Study** are competitive / best on pediatric cohort (EchoNet-Pediatric LVEF).
- **Different best model per cohort** — strong signal that MV-PhaseRel's phase-relational objective is optimized for adult cardiac patterns; generalization to pediatric is limited.

---

## 1. Adult LVEF — EchoNet-Dynamic test (N=1,277)

Label distribution: mean=55.5, std=12.2, range=[10.2, 84.5]. Clinical strata per AHA 2022 / ASE.

### Sample counts per stratum

| Stratum | N | % |
|---|---:|---:|
| Reduced (≤40, HFrEF) | 160 | 12.5% |
| Mildly reduced (41–49, HFmrEF) | 104 | 8.1% |
| Normal (50–70) | 954 | 74.7% |
| Hyperdynamic (>70) | 38 | 3.0% |

### MAE by stratum (lower = better)

| Model | Reduced ≤40 | Mild. red. 41–49 | Normal 50–70 | Hyper. >70 |
|---|---:|---:|---:|---:|
| V-JEPA†-e125 | 9.23 | 8.40 | 4.18 | 9.46 |
| **MV-PhaseRel** | 8.71 | 7.86 | **3.70** | 9.00 |
| MV-PairedIntra | 9.48 | 7.89 | 3.80 | 9.32 |
| TokenRel-Motion-e25 | 9.18 | 7.87 | 3.98 | 8.92 |
| **MCC-Anchored** | 9.59 | **7.73** | 3.89 | **8.58** |
| **FullJoint-Study** | 9.04 | **7.51** | 4.18 | 9.48 |

### Pearson by stratum (higher = better)

| Model | Reduced | Mild. red. | Normal | Hyper. |
|---|---:|---:|---:|---:|
| V-JEPA†-e125 | 0.609 | 0.275 | 0.430 | 0.283 |
| **MV-PhaseRel** | **0.640** | **0.359** | **0.493** | 0.205 |
| MV-PairedIntra | 0.623 | 0.336 | 0.465 | 0.237 |
| TokenRel-Motion-e25 | 0.617 | 0.320 | 0.440 | 0.224 |
| MCC-Anchored | 0.608 | 0.316 | 0.472 | 0.208 |
| FullJoint-Study | 0.582 | 0.271 | 0.435 | 0.202 |

### Clinical binary: LVEF < 50 detection (prev = 22.3%, 285 positives)

| Model | Sensitivity | Specificity | PPV | NPV | Bal-acc | **AUROC** |
|---|---:|---:|---:|---:|---:|---:|
| V-JEPA†-e125 | 0.660 | 0.945 | 0.774 | 0.906 | 0.802 | 0.906 |
| **MV-PhaseRel** | 0.642 | **0.953** | **0.796** | 0.903 | 0.797 | **0.926** |
| MV-PairedIntra | 0.649 | **0.959** | **0.819** | 0.905 | 0.804 | 0.916 |
| TokenRel-Motion-e25 | 0.663 | 0.954 | 0.804 | 0.908 | 0.808 | 0.917 |
| MCC-Anchored | 0.646 | 0.952 | 0.793 | 0.903 | 0.799 | 0.915 |
| **FullJoint-Study** | **0.688** | 0.939 | 0.763 | **0.913** | **0.813** | 0.905 |

**Clinical trade-offs**:
- **MV-PhaseRel is the best ranker** (AUROC 0.926).
- **FullJoint-Study has highest sensitivity** (0.688) — best at catching reduced EF.
- **MV-PairedIntra has highest specificity / PPV** — most conservative flagger.
- All 6 models cluster at bal-acc ~0.80 for this threshold.

### Reading

- Overall R² is dominated by the Normal stratum (75% of test).
- MV-PhaseRel wins every LVEF stratum on Pearson except Hyperdynamic (N=38, noise).
- On the clinically-actionable binary, MV-PhaseRel is best ranker; FullJoint wins sensitivity.
- Reduced-LVEF stratum (≤40): all models have Pearson ~0.6 but R² < 0 (predictions regress to mean) — known regression-on-skewed-distribution artifact.

### TokenRel-Motion e25 — only +25 EF variant with uniform MAE improvement over V-JEPA† (noted 2026-05-06)

TokenRel-Motion e25 is the only +25-epoch variant that beats V-JEPA† on **MAE in every stratum** of the adult LVEF test:

| Stratum | V-JEPA† | TokenRel e25 | ΔMAE | Pearson Δ |
|---|---:|---:|---:|---:|
| Reduced (≤40) | 9.23 | 9.18 | **−0.05** ✓ | +0.008 |
| Mild red (41–49) | 8.40 | 7.87 | **−0.53** ✓ | +0.045 |
| Normal (50–70) | 4.18 | 3.98 | **−0.20** ✓ | +0.010 |
| Hyper (>70) | 9.46 | 8.92 | **−0.54** ✓ | **−0.059** ✗ |

**Caveats**:

- **Reduced-stratum gain is marginal** (−0.05 EF %). Within per-stratum noise.
- **Hyperdynamic MAE win (−0.54) is partly regression-to-mean**. Pearson drops by −0.059 on the same 38 cases — lower absolute error, worse rank-ordering. TokenRel is predicting closer to the population mean on hyperdynamic extremes, trading directional fidelity for calibration.
- **MV-PairedIntra and MCC both LOSE to V-JEPA† on Reduced MAE** (+0.25 and +0.36 EF % respectively). TokenRel is distinct in not regressing on Reduced.
- **Test-wide ΔMAE for TokenRel-Motion e25** = −0.21 EF % [−0.37, −0.04], 97% P(better) — defensible.
- **MV-PhaseRel's Reduced MAE gain is larger** (−0.52) but loses on Mild-red Pearson tradeoff axes.

Reading: TokenRel-Motion e25's strength here is **uniform non-regression** across strata, not per-stratum magnitudes. On the clinically-critical Reduced stratum where MAE improvement translates to better HFrEF detection, V4/MV-PhaseRel still has the larger absolute gain (−0.52 vs −0.05). But TokenRel avoids the "Reduced gets worse while Normal improves" pattern that MV-PairedIntra and MCC show. This is consistent with TokenRel's token-level (vs V4's pooled) phase supervision preserving per-sample feature diversity that helps extreme-value calibration.

---

## 2. Adult LV geometry — EchoNet-LVH LVEDD (N=340)

Label distribution: mean=4.56 cm, std=0.71, range=[2.8, 7.7]. ASE-guideline LV dilation strata.

### Sample counts

| Stratum | N | % |
|---|---:|---:|
| Small/low-normal (<4.2) | 102 | 30.0% |
| Normal (4.2–5.1) | 181 | 53.2% |
| Mildly dilated (5.2–5.7) | 42 | 12.4% |
| Moderate+ dilated (≥5.8) | 15 | 4.4% |

### MAE by stratum

| Stratum | V-JEPA†-e125 | **MV-PhaseRel V4** |
|---|---:|---:|
| Small (<4.2) | 0.454 | **0.421** |
| Normal (4.2–5.1) | 0.328 | **0.318** |
| Mild dilated (5.2–5.7) | **0.478** | 0.498 |
| Moderate+ (≥5.8) | 0.935 | **0.931** |

### Pearson by stratum

| Stratum | V-JEPA | **MV-PhaseRel** |
|---|---:|---:|
| Small | 0.422 | **0.479** |
| Normal | 0.338 | **0.344** |
| Mild dilated | 0.189 | **0.292** |
| Moderate+ | 0.709 | **0.745** |

### Binary: LVEDD ≥ 5.2 cm (mild+ dilation, prev=16.8%)

| Model | Sens | Spec | AUROC |
|---|---:|---:|---:|
| V-JEPA†-e125 | 0.351 | 0.954 | 0.847 |
| **MV-PhaseRel** | 0.351 | **0.965** | **0.856** |

### Reading

- **MV-PhaseRel wins every LVEDD stratum on Pearson**, 3 of 4 on MAE.
- Small but consistent advantage (+0.01 AUROC, +0.01 spec at matched sens).
- Matches the overall finding (ΔR² = +0.041, P(V4 better on all 3) = 89%).
- **Only 2 models probed so far** — remaining 4 (MV-PairedIntra, TokenRel, MCC, FullJoint) are in the queue (12 LVH probe jobs submitted for 3 remaining endpoints × 4 models).

---

## 3. Pediatric LVEF — EchoNet-Pediatric test (N=368)

Label distribution: mean=60.2, std=11.6, range=[11.9, 73.0]. Peds strata mirror adult LVEF thresholds.

### Sample counts

| Stratum | N | % |
|---|---:|---:|
| Reduced (≤50) | 39 | 10.6% |
| Low-normal (51–55) | 22 | 6.0% |
| Normal (56–70) | 282 | 76.6% |
| Hyperdynamic (>70) | 25 | 6.8% |

### MAE by stratum

| Model | Reduced ≤50 | Low-norm 51–55 | Normal 56–70 | Hyper. >70 |
|---|---:|---:|---:|---:|
| **V-JEPA†-e125** | **12.32** | 9.45 | 3.29 | **6.02** |
| MV-PhaseRel | 12.94 | 9.37 | 3.44 | 6.67 |
| MCC-Anchored | 13.69 | 9.05 | **3.21** | 6.13 |
| **FullJoint-Study** | 13.60 | **8.49** | **3.21** | 6.94 |

### Pearson by stratum

| Model | Reduced | Low-norm | Normal | Hyper. |
|---|---:|---:|---:|---:|
| V-JEPA†-e125 | 0.728 | 0.081 | **0.175** | -0.037 |
| MV-PhaseRel | 0.678 | **0.270** | 0.106 | -0.001 |
| **MCC-Anchored** | **0.746** | 0.264 | 0.124 | -0.140 |
| FullJoint-Study | 0.745 | 0.226 | 0.164 | **0.059** |

### Binary: peds LVEF ≤ 55 (reduced/low-normal detection, prev=16.6%)

| Model | Sens | Spec | PPV | NPV | **AUROC** |
|---|---:|---:|---:|---:|---:|
| **V-JEPA†-e125** | **0.508** | 0.974 | 0.795 | 0.909 | 0.863 |
| MV-PhaseRel | 0.426 | 0.980 | 0.812 | 0.896 | 0.867 |
| **MCC-Anchored** | 0.459 | **0.984** | **0.848** | 0.901 | 0.873 |
| **FullJoint-Study** | 0.443 | 0.980 | 0.818 | 0.899 | **0.885** |

### Reading

**Pediatric is noisier across the board** — smaller per-stratum sample sizes, within-stratum Pearson often <0.3 (except in Reduced stratum which has clear signal).

- **Reduced peds (≤50)**: all models have MAE 12–14 on true values 20–50, meaning predictions regress to ~60. Pearson within Reduced is 0.68–0.75, so direction-rank is correct; severity is underestimated. Classic regression-to-mean on the skewed distribution.
- **Low-normal (51–55)** is the most unreliable within-stratum bin: FullJoint wins MAE (8.49); V-JEPA has Pearson near 0 (essentially no signal within 5-pp range).
- **Normal (56–70)** is the bulk of peds test (77%); MCC and FullJoint tied on MAE (3.21).
- **Hyperdynamic (>70)** is uninformative: all models have Pearson near 0 or negative, N=25.

### Clinical binary on peds:

- **V-JEPA† wins sensitivity** (0.508 > FullJoint 0.443 > MCC 0.459 > V4 0.426) — catches more reduced peds LVEF.
- **MCC wins specificity / PPV** (0.984 / 0.848) — most conservative.
- **FullJoint wins AUROC** (0.885) — best overall ranker.
- **Different best model per metric**, all within ~0.02 AUROC of each other.

### Pediatric ordering contradicts adult

| Task | Winner | Margin |
|---|---|---:|
| Adult LVEF (EchoNet-Dynamic) | **MV-PhaseRel** | +0.053 R² over V-JEPA† |
| Adult LV geometry (EchoNet-LVH LVEDD) | **MV-PhaseRel** | +0.041 R² over V-JEPA† |
| Pediatric LVEF (EchoNet-Pediatric) | **V-JEPA† / MCC / FJ (tied)** | MV-PhaseRel worst (Δ = −0.048 R²) |

**Strong signal: MV-PhaseRel's phase-relational head does not generalize to pediatric.** The adult MIMIC training distribution (quality-filtered, RR-consistent, 24 phase-matched pairs per study) creates features specific to adult cardiac phase patterns. These features don't transfer to pediatric hearts, which have:
- Higher baseline heart rates (70–200+ bpm vs adult 50–100)
- Different chamber proportions
- Different view availability (younger children have better PLAX/A4C windows)

The phase-matched sampler may encode cycle structure that's adult-specific.

---

## Caveats

1. **R² within narrow strata is inflated-negative** because predictions need to vary within the stratum to explain its variance. Focus on MAE and Pearson within strata.
2. **Hyperdynamic strata (N<50) across all datasets are uninformative** — small samples, wide clinical meaning, predictions regress to mean.
3. **Reduced-class performance is the clinically-actionable metric.** High overall R² can mask catastrophic failure on reduced LVEF (the actual indication for clinical decision-making). MV-PhaseRel's 0.64 Pearson on Reduced is the relevant number for "can this probe detect HFrEF?"
4. **No bootstrap CIs on per-stratum metrics** in this doc; with N=15–40 per stratum, CIs would be wide. Report point estimates with sample-size caveats.
5. **LVEDD has only 2 models** probed currently (V-JEPA, MV-PhaseRel). MCC, FJ, MV-PairedIntra, TokenRel queued.

---

## Source data

Per-sample test predictions at:

- LVEF EchoNet-Dynamic:
  - `runs/base_e125_lvef_test_698/predictions/base_e125_lvef_test.csv`
  - `runs/final_phase_rel25_lvef_test_596/predictions/final_phase_rel25_lvef_test.csv`
  - `runs/final_paired_iv25_lvef_test_630/predictions/final_paired_iv25_lvef_test.csv`
  - `runs/tokenrel_r2_e25_lvef_719/lvef_encoder_pool/predictions/tokenrel_r2_e25_e5_lvef_test.csv`
  - `runs/mcc_e25_echonet_lvef_test_796/predictions/mcc_e25_echonet_lvef_test.csv`
  - `runs/fj_30k_echonet_lvef_test_802/predictions/fj_30k_echonet_lvef_test.csv`

- LVEDD EchoNet-LVH:
  - `runs/lvh_lvedd_base_e125_724/echonet_lvh_lvedd_encoder_pool/predictions/base_e125_echonet_lvh_lvedd_test.csv`
  - `runs/lvh_lvedd_v4_e25_725/echonet_lvh_lvedd_encoder_pool/predictions/v4_e25_echonet_lvh_lvedd_test.csv`

- Peds EchoNet-Pediatric:
  - `runs/pediatric_probes_base_e125_722/echonet_pediatric_lvef_encoder_pool/predictions/base_e125_echonet_pediatric_lvef_test.csv`
  - `runs/pediatric_probes_v4_e25_723/echonet_pediatric_lvef_encoder_pool/predictions/v4_e25_echonet_pediatric_lvef_test.csv`
  - `runs/pediatric_probes_mcc_e25_798/echonet_pediatric_lvef_encoder_pool/predictions/mcc_e25_echonet_pediatric_lvef_test.csv`
  - `runs/pediatric_probes_fj_30k_803/echonet_pediatric_lvef_encoder_pool/predictions/fj_30k_echonet_pediatric_lvef_test.csv`

All under `s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts/`.

## Stratum thresholds

| Cohort | Reduced | Mild | Normal | Hyper |
|---|---|---|---|---|
| Adult LVEF | ≤40 (HFrEF) | 41–49 (HFmrEF) | 50–70 | >70 |
| Peds LVEF | ≤50 | 51–55 | 56–70 | >70 |
| Adult LVEDD (cm) | — | 4.2–5.1 (normal) | 5.2–5.7 (mild dilate) | ≥5.8 (mod+ dilate) |
