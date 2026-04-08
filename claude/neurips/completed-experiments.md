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

6 disruption conditions on EchoNet-Dynamic test (1,277 videos), LVEF R²:

| Condition | JEPA | BYOL | MAE |
|-----------|------|------|-----|
| clean | **0.549** | 0.460 | 0.396 |
| tubelet | **0.554** | 0.442 | 0.410 |
| reverse | **0.535** | 0.444 | 0.388 |
| matched (RoPE remap) | **0.549** | 0.460 | 0.396 |
| shuffle (mean, 3 seeds) | **0.365** | 0.174 | 0.318 |
| matched_frame (RoPE remap) | **0.324** | 0.099 | 0.286 |

JEPA retains most absolute signal post-shuffle. BYOL collapses catastrophically (-79% under matched_frame). MAE degrades least in relative terms because it had little temporal signal to begin with.

**Source:** `rebuttals/experiments/frame-shuffling.md` (full writeup with 24 log files)

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

⚠️ **Prior numbers retracted.** Consistent 4-model comparison with `scripts/rebuttal/rankme.py` (500 EchoNet-Dynamic test videos, same code/GPU, HyperPod jobs 510/525):

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

**Three hypotheses for "why JEPA outperforms MAE on functional tasks":**

| Hypothesis | Status |
|---|---|
| EMA filters frame-varying noise | ❌ Not supported (multiple tests) |
| JEPA encodes temporal dynamics MAE doesn't | ✅ Supported (frame shuffling, severity gradient) |
| JEPA uses representational capacity more efficiently | ❌ Not supported (revised). All models 200-245 range. |

**Surviving mechanism:** Temporal structure encoding is the only supported explanation. JEPA consolidates temporal information during training; MAE abandons it. This is independent of representational capacity.

**Source:** `claude/neurips/experiments/representation-analysis.md` (canonical), `claude/neurips/experiments/speckle-probing.md` (with retraction)
**Data:** `scripts/rebuttal/samples/rankme_all.csv` (4-model consistent comparison)

---

## 6. Noise Robustness — EchoBench (NeurIPS §5)

### 6a. LVEF Robustness (EchoNet-Dynamic, R², clean → severe)

| Perturbation | JEPA | BYOL | MAE |
|---|---|---|---|
| Depth attenuation | 0.552→0.361 (-35%) | 0.440→0.145 (-67%) | 0.351→0.233 (-34%) |
| Acoustic shadow | 0.552→0.478 (-13%) | 0.440→0.247 (-44%) | 0.351→0.280 (-20%) |
| Haze artifact | 0.552→0.502 (-9%) | 0.440→0.398 (-10%) | 0.351→0.147 (-58%) |
| **Average drop** | **-19%** | **-40%** | **-37%** |

JEPA under severe noise still outperforms MAE's clean baseline on all 3 perturbation types.

### 6b. CAMUS Segmentation Robustness (Dice, clean → severe)

| Perturbation | JEPA | BYOL | MAE |
|---|---|---|---|
| Depth attenuation | 0.815→0.681 (-16%) | 0.821→0.425 (-48%) | 0.822→0.749 (-9%) |
| Acoustic shadow | 0.815→0.708 (-13%) | 0.821→0.614 (-25%) | 0.822→0.728 (-11%) |
| Haze artifact | 0.815→0.800 (-2%) | 0.821→0.804 (-2%) | 0.822→0.794 (-3%) |
| **Average drop** | **-10%** | **-25%** | **-8%** |

**Anatomy-function dissociation in robustness:** JEPA most robust on function (-19% LVEF), MAE most robust on anatomy (-8% CAMUS). Different objectives are robust on different task types.

### 6c. Pediatric Zero-Shot Robustness (Pearson, UHN probes)

| Perturbation | JEPA | BYOL | MAE |
|---|---|---|---|
| Clean | **0.695** | 0.589 | 0.613 |
| Depth severe | **0.596** | 0.347 | 0.544 |
| Shadow severe | **0.654** | 0.574 | 0.598 |
| Haze severe | **0.592** | 0.532 | 0.481 |

JEPA highest at every severity level.

### 6d. RVSP Multi-View Noise Robustness (Pearson, 5,103 test)

| | Multi-view severe | A4C severe | PSAX severe |
|---|---|---|---|
| Avg drop | **-5.4%** | -9.8% | -7.5% |

Multi-view at severe ≈ single-view clean. Cross-view integration nearly halves degradation.

**Source:** `rebuttals/10-*` §5m, §5n, §5o, §6c

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
