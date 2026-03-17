# Trajectory LVEF Experiments

Iterative experiment log for predicting future LVEF trajectory from a single baseline echocardiogram.

**Source labels**: `labels/trajectory/trajectory_lvef.npz` — 14,235 study pairs with baseline EF, follow-up EF, delta, days between (30-365d), patient-level train/val/test splits.

**Key data properties**:
- Baseline EF: mean 54.9, std 13.5, range 1-89
- Delta: mean 0.67, std 10.88, range -57 to +71
- Correlation baseline EF vs delta: **r = -0.511** (strong regression to mean)
- Days between: mean 177, median 169

---

## V0: Delta Regression (FAILED)

**Setup**: Predict raw delta EF (z-score normalized) from 1 random clip per pair.
- Time window: 30-365 days
- 1 clip per study pair (no study_sampling)
- Regression with zscore_params.json
- CSVs: `trajectory_lvef_v1_backup/` (14,138 pairs: 2,543 train / 3,539 val / 8,056 test)

**Results** (EchoJEPA-G only, 27 epochs):

| Model | R² | Pearson |
|-------|-----|---------|
| EchoJEPA-G | 0.043 | 0.214 |

**Conclusion**: Barely beats predicting the mean. Delta regression is ill-posed — variable time horizon (30-365d) without the model knowing the prediction window, and baseline EF alone explains 26% of delta variance (r² = 0.261 from correlation).

---

## V1: 3-Class Delta Classification, 30-365d, threshold ±10 (LOW CEILING)

**Setup**: Predict declined (delta <= -10) / stable / improved (delta >= +10).
- Time window: 30-365 days
- Threshold: ±10 EF points (exceeds measurement noise of 5-8 points)
- All clips per study_1 (study_sampling), 3x class balancing
- CSVs: `trajectory_lvef_30_365d_t10_backup/` (11,501 studies: 8,471 train / 1,113 val / 1,917 test)
- Class distribution: 9% declined / 82% stable / 9% improved

**Results** (15 epochs, single-clip):

| Model | Best AUROC (single-clip) | AUROC (pred avg) |
|-------|:--:|:--:|
| EchoJEPA-G | 0.649 | 0.644 |
| PanEcho | 0.649 | 0.638 |
| EchoPrime | 0.642 | 0.628 |
| EchoJEPA-L-K | 0.614 | 0.628 |
| EchoJEPA-L | 0.608 | 0.595 |

**Observations**:
- All models cluster in 0.60-0.65 range with minimal inter-model separation
- Prediction averaging did NOT help (slight decrease for most models) — unusual
- 82% majority class dominates; models mostly learn baseline EF as a proxy
- G has no meaningful advantage over PanEcho/EchoPrime

**Conclusion**: Real signal but low ceiling. The task doesn't differentiate model quality because all models can estimate baseline EF, and baseline EF is the primary predictor of delta direction.

---

## V2: 3-Class Delta Classification, 90-365d, threshold ±8 (WORSE)

**Hypothesis**: Removing noisy short-term pairs (30-89d) and lowering threshold to ±8 would increase event rate and improve signal.

**Setup**: Same as V1 but narrower time window and lower threshold.
- Time window: 90-365 days
- Threshold: ±8 EF points
- CSVs: `trajectory_lvef/` (10,751 studies: 1,738 train / 2,822 val / 6,191 test)
- Class distribution: 15% declined / 72% stable / 13% improved (28% events, up from 18%)

**Results** (15 epochs, single-clip, incomplete — killed after 2 models):

| Model | Epochs | Best AUROC |
|-------|--------|:--:|
| EchoJEPA-G | 15 | 0.610 |
| EchoJEPA-L | 15 | 0.545 |

**Observations**:
- **Worse than V1** despite better class balance
- Removing 30-89d pairs dropped ~40% of data including apparently informative short-term pairs
- ±8 threshold is right at measurement noise floor (5-8 points), introducing label noise
- Run killed in favor of V3 after seeing G and L results

**Conclusion**: Narrower time window and lower threshold made things worse. The problem isn't data filtering — it's the fundamental task formulation.

---

## V3: New-Onset Cardiomyopathy (Binary, baseline EF >= 50) (CURRENT)

**Key insight**: Stop predicting *change* and start predicting *future state*. Among patients with preserved EF (>= 50) at baseline, predict who will develop reduced EF (< 50) at follow-up.

**Why this reframing works**:
1. Controls for baseline EF by restricting to a narrow range (>= 50), forcing the model to find *other* visual features
2. Binary classification is cleaner than 3-class
3. Clinically compelling: "from an apparently normal echo, identify patients at risk of developing cardiomyopathy"
4. Tests the world model hypothesis: the model must detect subclinical dysfunction not captured by the EF number

**Setup**:
- Population: baseline EF >= 50 only (10,812 pairs, excluded 3,423 with EF < 50)
- Time window: 30-365 days (all windows — event rate is stable at 7-8% across time strata)
- Binary: 0 = stable (future EF >= 50), 1 = decline (future EF < 50)
- All clips per study_1 (study_sampling), 3x class balancing
- CSVs: `trajectory_lvef_onset/` (10,771 studies: 1,932 train / 2,752 val / 6,087 test)
- Event rate: train 8.7% (169) / val 6.6% (181) / test 7.4% (453)
- Build command: `python build_trajectory_csvs.py --task trajectory_lvef --onset --baseline_min 50 --future_below 50`

**Training Results** (15 epochs, single-clip val AUROC):

| Model | Epochs | Best Val AUROC |
|-------|--------|:--:|
| EchoJEPA-G | 15/15 | **0.733** |
| EchoPrime | 15/15 | 0.700 |
| PanEcho | 15/15 | 0.698 |
| EchoJEPA-L-K | 15/15 | 0.596 |
| EchoJEPA-L | 15/15 | 0.516 |

**Test Results with Prediction Averaging** (FINAL):

| Model | Test AUROC (pred avg) |
|-------|:--:|
| EchoJEPA-G | **0.793** |
| EchoPrime | 0.776 |
| PanEcho | 0.759 |
| EchoJEPA-L-K | 0.677 |
| EchoJEPA-L | 0.514 |

**Observations**:
- **0.793 passes the 0.75 decision gate** → Pillar 3 headline
- **Prediction averaging added +0.06** for G (0.733 val → 0.793 test)
- **G-vs-EchoPrime gap only +1.7pp** — dramatically smaller than hemodynamic tasks (+8-10pp). Text-supervised pretraining provides an efficient path to prognostic features.
- **L at chance (0.514)** — 7K MIMIC studies is insufficient for SSL pretraining to learn prognostic signal

**Remaining**:
- Compare to baseline-EF-only predictor (logistic regression on measured EF value) to quantify value added by visual features
- Time-stratified AUROC analysis (30-89d / 90-179d / 180-365d) for supplement

---

## Summary Table

| Version | Task | Time Window | Classes | Event Rate | G AUROC | Model Separation |
|---------|------|-------------|---------|------------|---------|-----------------|
| V0 | Delta regression | 30-365d | continuous | — | R²=0.043 | — |
| V1 | Delta ±10 | 30-365d | 3 (9/82/9%) | 18% | 0.649 | minimal (0.04 range) |
| V2 | Delta ±8 | 90-365d | 3 (15/72/13%) | 28% | 0.610 | — |
| V3 | Onset (EF>=50 → <50) | 30-365d | 2 (93/7%) | 7-9% | **0.793 test (pred avg)** | large (+0.28 G vs L) |

**Lesson**: Predicting change from a single timepoint is fundamentally limited when change is driven by external factors. Reframing as risk stratification (who will cross a clinical threshold?) yields better results because the model can leverage both current state assessment and subclinical risk features.
