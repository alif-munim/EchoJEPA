# Three-Way Controlled Comparison

**Date:** 2026-03-28 to 2026-03-30
**Status:** Complete.
**NeurIPS section:** §3 (Core Finding)

---

## Overview

Epoch-matched comparison of three SSL objectives — JEPA (local masked latent prediction), BYOL (global self-distillation), MAE (pixel reconstruction) — on three clinical tasks. Same ViT-L encoder, same MIMIC-IV-Echo 525K clips, same 50-epoch budget. Only variable: prediction target. This is the backbone of the NeurIPS paper.

## Setup

**Architecture:** ViT-L (304M params), patch 16, tubelet 2, 16 frames, 224px
**Data:** MIMIC-IV-Echo 525K clips (4,600 patients)
**Pretraining:** 50 epochs each, identical LR schedule, EMA, weight decay
**Probes:** d=4 attentive, 20 epochs, 6-head HP grid (best head selected on val)

| Model | Encoder Checkpoint | Pretraining Objective |
|-------|-------------------|----------------------|
| EchoJEPA-L pt50 | `checkpoints/echojepa-l-pt50.pt` | L1 on local masked tokens, EMA target |
| EchoBYOL-L pt50 | `checkpoints/byol_vitl_imagenet_v2_e50.pt` | Cosine on global mean pool, EMA target + projector |
| EchoMAE-L pt50 | `checkpoints/videomae_l_mimic_ep50.pth` | MSE on pixels, no target encoder |

## Results

### LVEF Regression (UHN, 10K train / 1K val / 53K test)

| Model | Val R² | Val Pearson | Val MAE | Test R² | Test Pearson | Test MAE |
|-------|--------|-----------|---------|---------|-------------|----------|
| **JEPA** | 0.436 | 0.667 | 6.329 | **0.409** | **0.650** | **6.508** |
| BYOL | 0.421 | 0.652 | 6.297 | 0.384 | 0.625 | 6.656 |
| MAE | 0.325 | 0.584 | 6.866 | 0.283 | 0.572 | 7.031 |

Z-score: mean=57.07, std=11.28. Predict-mean baseline MAE: 9.0.

Bootstrap CIs (n=53K, 10K resamples):
- JEPA-BYOL: ΔR²=+0.025 [+0.018, +0.033], ΔPearson=+0.025 [+0.019, +0.030]
- JEPA-MAE: ΔR²=+0.126 [+0.112, +0.140], ΔPearson=+0.092 [+0.081, +0.103]
- BYOL-MAE: ΔR²=+0.101 [+0.087, +0.115], ΔPearson=+0.067 [+0.056, +0.079]

All 6 pairwise CIs exclude zero.

**Predictions:** `predictions/icml/echo{jepa,byol,mae}_l_pt50_lvef_test.csv`

### RVSP Regression (UHN, 41K train / 5K val / 5K test, multi-view A4C + PSAX-AV)

| Model | Val R² | Val Pearson | Val MAE | Test R² | Test Pearson | Test MAE |
|-------|--------|-----------|---------|---------|-------------|----------|
| **JEPA** | 0.241 | 0.504 | 9.044 | **0.220** | **0.484** | **9.101** |
| BYOL | 0.206 | 0.465 | 9.252 | 0.193 | 0.446 | 9.183 |
| MAE | 0.198 | 0.453 | 9.287 | 0.179 | 0.438 | 9.275 |

Z-score: mean=34.465, std=14.013. Multi-view data audit: 96.7% genuine A4C+PSAX-AV pairs.

**Predictions:** `predictions/icml-echo{jepa,byol,mae}-l-pt50-rvsp-test.csv`

### CAMUS Segmentation (400 train / 50 val / 50 test patients)

| Model | Test Dice | LV | MYO | LA |
|-------|----------|-----|-----|-----|
| **MAE** | **0.822** | 0.887 | 0.760 | 0.818 |
| BYOL | 0.821 | 0.880 | 0.769 | 0.813 |
| JEPA | 0.815 | 0.878 | 0.760 | 0.807 |

Frozen linear decoder (~4.1K params), 50 probe epochs, 7 HP configs per model.

## Key Finding

**Rankings invert by task type.** JEPA leads all hemodynamic/functional tasks (LVEF, RVSP). MAE leads spatial anatomy (CAMUS segmentation) despite having the worst LVEF signal. The pretraining objective determines what clinical information is encoded: pixel reconstruction → anatomy; latent prediction → function.

The 0.7pp segmentation spread (<1pp) contrasts with the 12.6pp LVEF R² spread — the dissociation is large and asymmetric.

## References

**Source:** `claude/rebuttals/10-rebuttal-experiment-results.md` §1a, §1b, §1c, §6a
**Probe configs:** `configs/eval/vitl/icml/echo{jepa,byol,mae}_l_pt50_{lvef,rvsp}_d4*.yaml`
**CAMUS scripts:** `scripts/neurips/camus_frozen_*.py`
**BYOL architecture audit:** `claude/rebuttals/09-three-way-comparison-results.md`
