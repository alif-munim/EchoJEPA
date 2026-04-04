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

| Model | Test R² | Test Pearson | Test MAE |
|-------|---------|-------------|----------|
| **EchoJEPA-L** | **0.552** | **0.753** | **5.938** |
| EchoBYOL-L | 0.440 | 0.669 | 6.666 |
| EchoMAE-L | 0.351 | 0.609 | 7.283 |

Bootstrap CIs: JEPA-MAE ΔR²=+0.201 [+0.168, +0.235]; JEPA-BYOL Δr=+0.083 [+0.055, +0.114]. JEPA advantage **amplifies** on external data (+45% in-dist → +57% cross-dataset).

**Source:** `rebuttals/10-*` §3a

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

## 5. Speckle Probing — Information Probing (NeurIPS §4)

Linear probes predicting speckle energy from frozen embeddings (EchoNet-Dynamic, 2,554 clips, 5-fold CV):

| Metric | JEPA | BYOL | MAE |
|--------|------|------|-----|
| Speckle energy (raw R²) | 0.764 | 0.835 | 0.910 |
| Mean intensity (R²) | 0.998 | 0.984 | 0.995 |
| **Speckle (partial R², controlling for intensity)** | **0.674** | 0.775 | **0.875** |

JEPA encodes 23% less speckle than MAE. Monotonic ordering matches prediction. Mechanism: EMA target averages over frame-to-frame speckle variation.

**Source:** `rebuttals/10-*` §6e
**Data:** `scripts/rebuttal/samples/information_probing_{model}.npz`

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

## Checkpoint Reference

All encoder and probe checkpoint paths are in `rebuttals/12-checkpoint-reference.md`. Key encoders:

| Model | Checkpoint |
|-------|-----------|
| EchoJEPA-L pt50 | `checkpoints/echojepa-l-pt50.pt` |
| EchoBYOL-L pt50 | `checkpoints/byol_vitl_imagenet_v2_e50.pt` |
| EchoMAE-L pt50 | `checkpoints/videomae_l_mimic_ep50.pth` |
| SALT (to be trained) | `checkpoints/pretrain/mimic/salt_s{1,2}_vitl_224px_16f/latest.pt` |
