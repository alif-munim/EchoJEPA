# Completed Experiments — NeurIPS Inventory

All experiments below are complete with results in hand. Organized by NeurIPS paper section, not chronological execution order. Source references point to `claude/rebuttals/` where full details live.

---

## 1. Three-Way Controlled Comparison (NeurIPS §3)

**Setup:** ViT-L (304M params), MIMIC-IV-Echo 525K clips, 50 pretraining epochs, d=4 attentive probes. Only variable: prediction target.

### 1a. LVEF Regression (UHN, 10K train / 53K test)

| Model | Prediction Target | Test R² | Test Pearson | Test MAE |
|-------|------------------|---------|-------------|----------|
| **EchoJEPA-L** | Local masked tokens | **0.409** | **0.650** | **6.508** |
| EchoBYOL-L | Global mean pool | 0.384 | 0.625 | 6.656 |
| EchoMAE-L | Pixels | 0.283 | 0.572 | 7.031 |

Bootstrap CIs (n=53K, 10K resamples): JEPA-BYOL ΔR²=+0.025 [+0.018, +0.033]; JEPA-MAE ΔR²=+0.126 [+0.112, +0.140]. All pairwise significant.

**Source:** `rebuttals/10-rebuttal-experiment-results.md` §1a, §6a
**Predictions:** `predictions/icml/echo{jepa,byol,mae}_l_pt50_lvef_test.csv`

### 1b. RVSP Regression (UHN, 41K train / 5K test, multi-view)

| Model | Test R² | Test Pearson | Test MAE |
|-------|---------|-------------|----------|
| **EchoJEPA-L** | **0.220** | **0.484** | **9.101** |
| EchoBYOL-L | 0.193 | 0.446 | 9.183 |
| EchoMAE-L | 0.179 | 0.438 | 9.275 |

**Source:** `rebuttals/10-*` §1b
**Predictions:** `predictions/icml-echo{jepa,byol,mae}-l-pt50-rvsp-test.csv`

### 1c. CAMUS Segmentation (400 train / 50 test patients)

| Model | Test Dice | LV | MYO | LA |
|-------|----------|-----|-----|-----|
| EchoMAE-L | **0.822** | 0.887 | 0.760 | 0.818 |
| EchoBYOL-L | 0.821 | 0.880 | 0.769 | 0.813 |
| EchoJEPA-L | 0.815 | 0.878 | 0.760 | 0.807 |

**Key:** MAE wins segmentation despite worst LVEF. The anatomy-function dissociation.
**Source:** `rebuttals/10-*` §1c

---

## 2. Cross-Dataset Transfer (NeurIPS §3)

### 2a. EchoNet-Dynamic LVEF (7.5K train / 1,277 test, public)

#### pt50 (rebuttal) — JEPA had unfair init advantage

| Model | Test R² | Test Pearson | Test MAE |
|-------|---------|-------------|----------|
| **EchoJEPA-L** (pt50, **235ep init confound**) | **0.552** | **0.753** | **5.938** |
| EchoBYOL-L (pt50, IN21K init) | 0.440 | 0.669 | 6.666 |
| EchoMAE-L (pt50, IN21K init) | 0.351 | 0.609 | 7.283 |

Bootstrap CIs (rebuttal): JEPA-MAE ΔR²=+0.201 [+0.168, +0.235]; JEPA-BYOL Δr=+0.083 [+0.055, +0.114].

**⚠️ Init confound discovered post-rebuttal**: the "JEPA pt50" checkpoint was actually a 235-epoch fully-trained model, while BYOL/MAE pt50 were 50-epoch IN21K-init checkpoints. The pt50 numbers above overstate JEPA's advantage. See e100 init-matched results below.

#### e100 init-matched (canonical, all IN21K-init at ~100 epochs)

| Model | Test R² [95% CI] | Test Pearson [95% CI] | Test MAE |
|-------|-----------------|----------------------|----------|
| **EchoJEPA-IN21K e100** | **0.652** [0.608, 0.691] | **0.808** [0.781, 0.832] | **5.32** |
| EchoBYOL e100 | 0.511 [0.452, 0.564] | 0.720 [0.680, 0.756] | 6.18 |
| EchoMAE e99 | 0.447 [0.389, 0.500] | 0.688 [0.645, 0.728] | 6.59 |
| **SALT v1 e79** (frozen teacher) | 0.416 [0.347, 0.478] | 0.659 [0.613, 0.702] | 6.66 |

Bootstrap 95% CIs (n=1,277, 10K resamples, all 4 models paired on same samples — SALT aggregated from 1,280 clips to 1,277 videos).

**Pairwise differences (all paired bootstrap):**

| Comparison | ΔR² [95% CI] | Δr [95% CI] | R² sig? | r sig? |
|-----------|-------------|------------|---------|--------|
| JEPA vs BYOL | +0.141 [+0.109, +0.175] | +0.088 [+0.066, +0.111] | **SIG** | **SIG** |
| JEPA vs MAE | +0.205 [+0.164, +0.247] | +0.120 [+0.090, +0.152] | **SIG** | **SIG** |
| JEPA vs SALT | +0.237 [+0.188, +0.289] | +0.149 [+0.116, +0.184] | **SIG** | **SIG** |
| BYOL vs MAE | +0.064 [+0.016, +0.110] | +0.032 [-0.001, +0.066] | **SIG** | n.s. |
| BYOL vs SALT | +0.096 [+0.042, +0.151] | +0.061 [+0.025, +0.098] | **SIG** | **SIG** |
| MAE vs SALT | +0.032 [-0.022, +0.086] | +0.029 [-0.010, +0.066] | n.s. | n.s. |

**Ranking preserved** (JEPA >> BYOL > MAE ≈ SALT). All JEPA pairwise comparisons highly significant. MAE and SALT are statistically equivalent.

**Source:** Bootstrap computed 2026-04-08 from `clip_outputs.npz` files in `evals/vitl/icml/`. SALT had 1,280 clips (3 videos with duplicate clips) aggregated to 1,277 by averaging. See `experiments/representation-analysis.md` (canonical), SALT from `experiments/salt-comparison.md`.

### 2b. Pathology-Stratified LVEF (EchoNet-Dynamic, 1,277 test)

| EF Category | N | JEPA MAE | BYOL MAE | MAE MAE |
|-------------|---|----------|----------|---------|
| Normal (≥55%) | 876 | 4.3 | 5.0 | 5.1 |
| Mildly reduced (40-54%) | 241 | 7.6 | 7.8 | 7.1 |
| **Reduced (<40%)** | 160 | **12.4** | 14.4 | **19.3** |

MAE predicts 48% for patients with true EF 29% — misses severe heart failure. JEPA advantage 8× larger on reduced EF.

**Source:** `rebuttals/10-*` §6d

---

## 3. Cross-Population Transfer (NeurIPS §3)

### 3a. Pediatric Zero-Shot (UHN-trained probes → 368 pediatric test, NO retraining)

| Model | Test Pearson | Test MAE | Test R² |
|-------|-------------|----------|---------|
| **EchoJEPA-L** | **0.705** | **6.957** | **0.405** |
| EchoMAE-L | 0.626 | 7.857 | 0.187 |
| EchoBYOL-L | 0.602 | 8.004 | 0.206 |

### 3b. Pediatric Zero-Shot (END-trained probes → 368 pediatric test)

| Model | Test Pearson | Test MAE | Test R² |
|-------|-------------|----------|---------|
| **EchoJEPA-L** | **0.615** | **7.358** | **0.293** |
| EchoMAE-L | 0.531 | 9.203 | 0.041 |
| EchoBYOL-L | 0.498 | 12.132 | -0.847 |

BYOL collapses on cross-population transfer from END (R²=-0.847).

**Source:** `rebuttals/10-*` §4b, §4c

---

## 4. Temporal Ablation — Frame Shuffling (NeurIPS §4)

### 4a. ICML Rebuttal 6-condition (pt50, confounded JEPA init)

6 disruption conditions on EchoNet-Dynamic test (1,277 videos), LVEF R²:

| Condition | JEPA | BYOL | MAE |
|-----------|------|------|-----|
| clean | **0.549** | 0.460 | 0.396 |
| tubelet | **0.554** | 0.442 | 0.410 |
| reverse | **0.535** | 0.444 | 0.388 |
| matched (RoPE remap) | **0.549** | 0.460 | 0.396 |
| shuffle (mean, 3 seeds) | **0.365** | 0.174 | 0.318 |
| matched_frame (RoPE remap) | **0.324** | 0.099 | 0.286 |

**Source:** `rebuttals/experiments/frame-shuffling.md` (full writeup with 24 log files)

### 4b. Init-matched Severity Gradient (NeurIPS canonical, 4 models × 4 epochs)

Partial frame shuffling at 0/25/50/75/100%, 3 seeds per fraction. R² (mean):

| Fraction | JEPA e100 | BYOL e100 | MAE e99 | SALT e79 |
|----------|-----------|-----------|---------|----------|
| 0.00 | **0.591** | 0.468 | 0.445 | 0.296 |
| 0.25 | **0.542** | 0.410 | 0.421 | 0.048 |
| 0.50 | **0.507** | 0.336 | 0.436 | -0.161 |
| 0.75 | **0.485** | 0.300 | 0.414 | -0.256 |
| 1.00 | **0.488** | 0.291 | 0.428 | -0.270 |

Four distinct profiles: JEPA gentle slope (−17%), BYOL steep (−38%), MAE flat (−4%), SALT cliff (−191%).

### 4c. Init-matched 6-Condition (NeurIPS canonical, 4 models at convergence)

| Condition | JEPA e100 | BYOL e100 | MAE e99 | SALT e79 |
|-----------|-----------|-----------|---------|----------|
| clean | **.591** | .468 | .445 | .296 |
| tubelet | **.582** | .402 | .424 | .294 |
| reverse | **.539** | .373 | .431 | .120 |
| matched | **.580** | .415 | .419 | .296 |
| shuffle | **.484** | .291 | .422 | −.283 |
| matched_frame | **.477** | .280 | .449 | −.310 |

### 4d. Training Dynamics — Full 4-model × 4-epoch matrix

Training dynamics frame shuffling complete for all 4 models across 4 comparable epochs (~24/50/74/99). Full results in `experiments/severity-gradient.md` and `experiments/6-condition-shuffling.md`.

**SALT training dynamics (S2 e4/e29/e54/e79):** The cliff profile persists at every training stage. No consolidation — unlike JEPA (−42%→−17%), SALT stays at −187% to −256% from e29 onward. The frozen teacher cannot drive temporal consolidation.

**Source:** `claude/neurips/experiments/severity-gradient.md`, `claude/neurips/experiments/6-condition-shuffling.md`

### 4e. CAMUS Segmentation Frame Shuffling (spatial task control)

Same shuffling protocol applied to **segmentation** (per-frame spatial task) on CAMUS test set (100 samples = 50 patients × 2 views). Frozen encoder + frozen LinearSegDecoder. All CIs: paired bootstrap, n=100, 10K resamples.

**Severity gradient (% Dice degradation from clean):**

| Shuffle % | JEPA e100 | BYOL e100 | MAE e99 | SALT e79 |
|-----------|-----------|-----------|---------|----------|
| 25% | 1.6% [1.1, 2.0] | 1.3% [0.9, 1.7] | 2.1% [1.7, 2.5] | 0.8% [0.5, 1.1] |
| 50% | 3.6% [2.9, 4.3] | 2.7% [2.2, 3.3] | 4.9% [4.1, 5.6] | 2.1% [1.5, 2.6] |
| 100% | 7.0% [6.0, 8.0] | 6.0% [5.2, 6.7] | **8.6% [7.6, 9.6]** | 4.9% [4.0, 5.8] |

**6-condition (% Dice degradation from clean):**

| Condition | JEPA e100 | BYOL e100 | MAE e99 | SALT e79 |
|-----------|-----------|-----------|---------|----------|
| reverse | **14.5% [13.4, 15.7]** | 12.2% [11.0, 13.4] | 12.7% [11.6, 13.8] | 11.9% [10.6, 13.3] |
| tubelet | 5.8% [5.0, 6.7] | 4.6% [3.9, 5.3] | 5.5% [4.6, 6.3] | 5.0% [4.1, 5.9] |
| shuffle | 7.1% [6.0, 8.1] | 6.0% [5.2, 6.7] | 8.5% [7.5, 9.5] | 4.8% [4.0, 5.7] |
| matched_frame | 7.6% [6.6, 8.5] | 7.1% [6.2, 7.9] | 8.5% [7.6, 9.3] | 5.7% [4.8, 6.6] |

**Tracked extraction (Version A vs B) — isolating content vs temporal encoding:**

After shuffling, extract features where ED/ES content actually landed (inverse permutation) instead of the original fixed position. Separates content misalignment from temporal encoding disruption.

| Model | A drop (orig pos) | B drop (tracked) | Interpretation |
|-------|-------------------|------------------|----------------|
| JEPA | 5.8% | 5.5% | B ~ A — position-invariant features |
| BYOL | 4.9% | **14.3%** | B >> A — decoder position-locked via RoPE |
| MAE | 7.0% | **10.3%** | B > A — moderately position-dependent |
| SALT | 3.7% | **2.8%** | B < A — content was the main issue |

**Key findings:** (1) Reverse is catastrophic (12–15%), ~2× worse than full shuffle — temporal direction matters even for segmentation. (2) MYO is most sensitive structure (MAE: 12.7% degradation). (3) SALT degrades least (4.9–5.7%), MAE most (8.5–8.6%). (4) Segmentation is NOT temporally invariant — predicted <2% but actual 5–9%. (5) JEPA/SALT learn position-invariant spatial features; BYOL/MAE are position-locked (tracked extraction worse than original).

**Source:** `claude/neurips/experiments/camus-frame-shuffling.md`

---

## 5. Representation-Level Analysis (NeurIPS §4 — Mechanistic Evidence)

### ⚠️ The ICML rebuttal speckle claim is RETRACTED (init confound)

The ICML rebuttal numbers (JEPA 0.674, BYOL 0.775, MAE 0.875, "23% less speckle") used a confounded JEPA "pt50" that was actually a 235-epoch checkpoint. With init-matched e100 models, the gap shrinks dramatically and the ranking changes:

| Comparison | JEPA | BYOL | MAE | JEPA−MAE gap |
|---|---|---|---|---|
| ICML rebuttal pt50 (confounded) | 0.674 | 0.775 | 0.875 | −0.201 (23%) |
| **e100 init-matched (canonical)** | **0.848** | **0.716** | **0.885** | **−0.037 (4%)** |

Under init-matching, BYOL is the *best* speckle filter, not JEPA. The "JEPA filters speckle via EMA target averaging" narrative is **not supported**.

### Effective dimensionality (REVISED 2026-04-07)

⚠️ **Prior numbers retracted.** Consistent 4-model comparison with `scripts/neurips/rankme.py` (500 EchoNet-Dynamic test videos, same code/GPU, HyperPod jobs 510/525):

| Model | Effective Dimensionality | % of embed_dim (1024) |
|-------|--------------------------|----------------------|
| JEPA IN21K e95 | **245.3** | 24.0% |
| BYOL e100 | **220.7** | 21.6% |
| MAE e99 | **206.4** | 20.2% |
| SALT v1 e79 | **202.7** | 19.8% |

All four models are in the **200-245 range**. The prior MAE=63 (Goodfire report) is not reproducible with the consistent pipeline and should not be cited. Effective dimensionality does **not** explain MAE's weakness — the gap is modest (~20%), not 3×.

### Other supporting analyses (e100 init-matched)

- **Layer-wise speckle**: BYOL filters most aggressively across depth (−31% from layer 1 to 24); JEPA modest (−9%); MAE retains throughout (−4%)
- **Token-level speckle**: MAE 0.941 > JEPA 0.926 > BYOL 0.891 (same ranking as mean-pooled)
- **Temporal consistency**: BYOL 0.976 > JEPA 0.954 > MAE 0.950 (JEPA and MAE essentially identical, contradicting EMA filtering hypothesis)
- **Noise autocorrelation sweep**: static spatial noise is the *worst* perturbation for all models, opposite of EMA-filtering prediction

### Cross-temporal attention analysis (2026-04-09)

Fraction of attention flowing between tokens at different temporal positions. Random baseline = 0.875. Lower = more within-frame (spatial) attention.

**Epoch ~100:**

| Model | Layers 0-1 | Layers 2-10 | Layers 11-23 | Overall |
|-------|-----------|-------------|-------------|---------|
| **SALT S2 e79** | **0.44-0.49** | **0.39-0.56** | 0.83-0.88 | **0.672** |
| **JEPA e100** | **0.57-0.60** | 0.82-0.87 | 0.87-0.88 | **0.839** |
| BYOL e100 | 0.77-0.86 | 0.81-0.86 | 0.87-0.88 | 0.855 |
| MAE e99 | 0.86 | 0.82-0.87 | 0.87 | 0.861 |

SALT develops the strongest spatial→temporal hierarchy: layers 0-10 are heavily within-frame, sharp transition at layer 11. JEPA shows a milder version (layers 0-1 only). BYOL and MAE show no spatial-first specialization. The hierarchy deepens with training: at SALT e29, only layer 0 is spatial-biased (0.27); by e79, the entire first half of the network specializes for spatial processing.

**Four hypotheses for "why JEPA outperforms MAE on functional tasks":**

| Hypothesis | Status |
|---|---|
| EMA filters frame-varying noise | ❌ Not supported (multiple tests) |
| JEPA encodes temporal dynamics MAE doesn't | ✅ Supported (frame shuffling, severity gradient) |
| JEPA uses representational capacity more efficiently | ❌ Not supported (revised). All models 200-245 range. |
| Predictive objectives induce spatial→temporal layer specialization | ✅ Supported (cross-temporal attention analysis) |

**Surviving mechanisms:** Temporal structure encoding and spatial→temporal layer specialization. Predictive objectives (JEPA, SALT) force early layers to attend within-frame (spatial features) before integrating across time. This hierarchical processing may be the mechanism by which temporal dynamics are encoded more robustly — a model that first builds spatial features, then reasons temporally over them, captures cardiac dynamics more effectively than one that mixes both uniformly (MAE, BYOL).

**Source:** `claude/neurips/experiments/representation-analysis.md` §7 (canonical), `claude/neurips/experiments/speckle-probing.md` (with retraction)
**Data:** `scripts/neurips/temporal_attention/{jepa_e100,byol_e100,mae_e99,salt_s2v1_e79,jepa_pt50,byol_pt50,mae_pt50,salt_s2v1_e29}_temporal_attention.csv`

---

## 6. Noise Robustness — EchoBench (NeurIPS §5)

**Full details:** `experiments/echobench-e100.md` (4-model bootstrap CIs, checkpoints, scripts, issues)

### 6a. LVEF Robustness (EchoNet-Dynamic, 1,277 test, R² with 95% bootstrap CIs)

**Init-matched e100 models (authoritative, 4-model):**

| Condition | JEPA | BYOL | MAE | SALT |
|---|---|---|---|---|
| clean | **0.591** [0.538, 0.638] | 0.465 [0.401, 0.523] | 0.445 [0.377, 0.505] | 0.293 [0.215, 0.362] |
| depth_atten/severe | **0.396** [0.321, 0.463] | 0.342 [0.288, 0.392] | 0.090 [0.051, 0.129] | 0.137 [0.094, 0.179] |
| shadow/severe | **0.457** [0.385, 0.518] | 0.320 [0.240, 0.390] | 0.400 [0.340, 0.455] | 0.208 [0.128, 0.278] |
| haze/severe | **0.553** [0.498, 0.603] | 0.431 [0.368, 0.488] | 0.159 [0.099, 0.217] | 0.217 [0.145, 0.283] |

Robustness ranking (LVEF): JEPA > BYOL > SALT > MAE. MAE collapses under depth attenuation (0.090) and haze (0.159).

### 6b. CAMUS Segmentation Robustness (100 samples, Dice with 95% bootstrap CIs)

| Condition | JEPA | BYOL | MAE | SALT |
|---|---|---|---|---|
| clean | 0.815 [0.801, 0.829] | 0.823 [0.811, 0.835] | **0.827** [0.814, 0.838] | 0.777 [0.759, 0.794] |
| depth_atten/severe | **0.683** [0.663, 0.703] | 0.368 [0.345, 0.391] | 0.654 [0.625, 0.681] | 0.508 [0.486, 0.529] |
| shadow/severe | 0.717 [0.697, 0.736] | 0.587 [0.556, 0.616] | **0.737** [0.717, 0.755] | 0.645 [0.621, 0.668] |
| haze/severe | 0.794 [0.778, 0.808] | **0.815** [0.804, 0.826] | 0.778 [0.763, 0.792] | 0.767 [0.749, 0.785] |

| | JEPA | BYOL | MAE | SALT |
|---|---|---|---|---|
| **Avg severe drop** | **10.3%** [9.4, 11.3] | 28.4% [26.8, 29.9] | 12.6% [11.4, 13.8] | 17.6% [16.4, 18.9] |

Robustness ranking (CAMUS): JEPA (10.3%) > MAE (12.6%) > SALT (17.6%) >> BYOL (28.4%).

### 6c. Pediatric Zero-Shot Robustness (Pearson, UHN probes, pt50)

| Perturbation | JEPA | BYOL | MAE |
|---|---|---|---|
| Clean | **0.695** | 0.589 | 0.613 |
| Depth severe | **0.596** | 0.347 | 0.544 |
| Shadow severe | **0.654** | 0.574 | 0.598 |
| Haze severe | **0.592** | 0.532 | 0.481 |

JEPA highest at every severity level.

### 6d. RVSP Multi-View Noise Robustness (Pearson, 5,103 test, pt50)

| | Multi-view severe | A4C severe | PSAX severe |
|---|---|---|---|
| Avg drop | **-5.4%** | -9.8% | -7.5% |

Multi-view at severe ≈ single-view clean. Cross-view integration nearly halves degradation.

### 6e. Key finding (4-model)

**JEPA is most robust on BOTH tasks.** Clean performance fails to predict robustness: MAE leads clean CAMUS (0.827) but JEPA has lowest avg severe drop (10.3%). BYOL catastrophically collapses under depth attenuation on CAMUS (0.823→0.368, −55%). MAE collapses on functional tasks (LVEF R²: 0.445→0.090 under depth attenuation).

**Source:** `experiments/echobench-e100.md` (e100 4-model), `rebuttals/10-*` §5m, §5n, §5o, §6c (pt50 3-model)

---

## 7. Multi-View Ablation (NeurIPS Appendix)

RVSP single-view vs multi-view (JEPA-L pt50):

| View | Test Pearson | Test R² |
|------|-------------|---------|
| Multi-view (A4C + PSAX) | **0.484** | **0.220** |
| A4C only | 0.447 | 0.181 |
| PSAX only | 0.449 | 0.188 |

Multi-view +3.9pp R² over best single view.

**Source:** `rebuttals/10-*` §6b

---

## 8. Scaling (NeurIPS §6, brief)

| Model | Params | Data | LVEF R² |
|-------|--------|------|---------|
| EchoJEPA-B (V-JEPA 2.1) | 86M | MIMIC 525K | 0.650 |
| EchoJEPA-L (V-JEPA 2.0, 50ep) | 304M | MIMIC 525K | 0.436 |
| EchoJEPA-G (V-JEPA 2.0) | 1,012M | UHN 18M | 0.778 |

**Caveat:** B uses V-JEPA 2.1 (different architecture version + more epochs); L→G confounds data scale.
**Source:** `rebuttals/10-*` §2

---

## 9. SALT — Frozen Teacher vs EMA Self-Distillation (NeurIPS §3 row)

SALT (Li et al., Apple 2025) replaces JEPA's co-evolving EMA teacher with a frozen pixel-reconstruction teacher. Tested as a fourth row in the controlled comparison.

### EchoNet-Dynamic LVEF — full e100 init-matched comparison (test set)

Uses bootstrap-verified numbers (see §2a above for full CIs table):

| Method | Test MAE | Test R² | Test Pearson |
|---|---|---|---|
| **JEPA-IN21K e100** | **5.32** | **0.652** | **0.808** |
| BYOL e100 | 6.18 | 0.511 | 0.720 |
| MAE e99 | 6.59 | 0.447 | 0.688 |
| **SALT v1 e79** (best variant) | **6.66** | **0.416** | **0.659** |

**Headline:** Same ranking as the rebuttal three-way comparison at pt50, now confirmed at e100 with init-matching. JEPA significantly outperforms all alternatives (all pairwise CIs exclude zero). MAE and SALT are statistically equivalent.

**Conservative interpretation:** Replacing JEPA's co-evolving EMA teacher with a frozen pixel-reconstruction teacher (SALT) reduces LVEF R² from 0.652 to 0.416 (−36%), placing it statistically equivalent to MAE. This suggests co-evolution of the target encoder contributes to representation quality independent of the prediction target.

### SALT variant robustness (appendix)

Three SALT variants tested. All land in the same neighborhood:

| Variant | Predictor arch | Hyperparameters | S2 epochs | Test MAE | Test R² |
|---|---|---|---|---|---|
| v1 e79 (best) | hierarchical 4-layer | LR 1.75e-4 const, weak aug | 80 | **6.66** | **0.414** |
| v1 e199 | hierarchical 4-layer | (same as v1, extended) | 200 | 7.02 | 0.360 |
| v3 e79 | single-level (paper-spec) | LR 2.55e-4 cosine, paper aug | 80 | 7.03 | 0.348 |

The SALT failure is robust to predictor architecture and hyperparameter regime — not an artifact of any particular implementation choice.

**Caveats:**
- v1 e79 → v1 e199 regression (more training hurts) is most parsimoniously explained by overfitting from constant LR (no decay) on a small homogeneous dataset, NOT by a SALT-specific pathology.
- We do NOT have evidence that SALT inherits speckle from the frozen pixel-reconstruction teacher. The original speckle-pollution argument depends on the retracted ICML rebuttal claim (see §5 above).

**Source:** `claude/neurips/experiments/salt-comparison.md` (full writeup)
**Configs:** `configs/eval/vitl/icml/salt_s2_*_predavg.yaml`

---

## 10. MR Severity Cross-Dataset Transfer (NeurIPS §3 — Hemodynamic Generalization)

Two MR severity probes (same frozen EchoJEPA-G encoder, same d=1 attentive architecture), one trained on MIMIC, one on UHN. Tested on MIMIC MR test set (1,003 studies, pred-avg).

| Probe | Accuracy | Balanced Acc | Quad Kappa | Macro AUROC |
|-------|----------|-------------|------------|-------------|
| **MIMIC (in-distribution)** | **0.591** | **0.391** | **0.538** | **0.806** |
| UHN (cross-dataset) | 0.531 | 0.341 | 0.410 | 0.799 |

**Headline:** AUROC nearly preserved cross-institution (−0.9%). The UHN probe's ranking ability transfers; only classification thresholds degrade. Both probes fail on Severe (n=56, class imbalance).

**Source:** `claude/neurips/experiments/mr-cross-dataset-transfer.md`
**Artifacts:** `s3://.../runs/echojepa_g_mr_compare_549/logs/mr_comparison.csv`

---

## Checkpoint Reference

### S3 (canonical, persistent)

All pt50 encoders and EchoNet-Dynamic LVEF probes are mirrored to a clean S3 location:

```
s3://echodata25/neurips/
├── encoders/
│   ├── echojepa_l_pt50.pt        (4.8 GB — ViT-L, JEPA, 50ep MIMIC)
│   ├── echobyol_l_pt50.pt        (2.3 GB — ViT-L, BYOL, 50ep MIMIC)
│   └── echomae_l_pt50.pth        (3.6 GB — ViT-L, MAE, 50ep MIMIC)
└── probes/
    └── end_lvef_pt50/
        ├── echojepa_l_pt50/best.pt   (3.3 GB — d=4 attentive, 20ep, head 3)
        ├── echobyol_l_pt50/best.pt   (3.3 GB — d=4 attentive, 20ep, head 1)
        └── echomae_l_pt50/best.pt    (3.3 GB — d=4 attentive, 20ep, head 5)
```

These are the checkpoints used for frame shuffling and noise robustness experiments. The probes were trained on EchoNet-Dynamic (7,460 train) and evaluated on EchoNet-Dynamic test (1,277 videos).

**Provenance:**
- JEPA encoder: EFS `checkpoints/echojepa-l-pt50.pt`; probe from HyperPod job 294
- BYOL encoder: EFS `checkpoints/byol_vitl_imagenet_v2_e50.pt`; probe trained on local A100 (EFS `evals/vitl/icml/echobyol_pt50_end_lvef_224/.../best.pt`)
- MAE encoder: EFS `checkpoints/videomae_l_mimic_ep50.pth`; probe from HyperPod job 296

**HyperPod S3 (original training outputs, less organized):**
- `s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts/runs/echojepa_pt50_end_lvef_294/`
- `s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts/runs/echobyol_pt50_end_lvef_284/` (earlier run, different HP selection)
- `s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts/runs/echomae_pt50_end_lvef_296/`

### EFS (local, fast access)

| Model | Encoder (EFS) | Probe (EFS) |
|-------|--------------|-------------|
| EchoJEPA-L pt50 | `checkpoints/echojepa-l-pt50.pt` | `evals/vitl/icml/echojepa_pt50_end_lvef_224/.../best.pt` |
| EchoBYOL-L pt50 | `checkpoints/byol_vitl_imagenet_v2_e50.pt` | `evals/vitl/icml/echobyol_pt50_end_lvef_224/.../best.pt` |
| EchoMAE-L pt50 | `checkpoints/videomae_l_mimic_ep50.pth` | `evals/vitl/icml/echomae_pt50_end_lvef_224/.../best.pt` |
| SALT (to be trained) | `checkpoints/pretrain/mimic/salt_s{1,2}_vitl_224px_16f/latest.pt` | — |

Full checkpoint reference (all tasks, all models): `rebuttals/12-checkpoint-reference.md`
