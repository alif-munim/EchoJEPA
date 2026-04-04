# Cross-Dataset and Cross-Population Transfer

**Date:** 2026-03-30
**Status:** Complete.
**NeurIPS section:** §3 (Core Finding — Transfer)

---

## Overview

Tests whether the prediction target advantage generalizes beyond the training distribution. Evaluates on two external datasets: EchoNet-Dynamic (cross-dataset, same modality, different institution) and EchoNet-Pediatric (cross-population, adult-trained probes applied to children zero-shot).

## Setup

**Probes trained on:** MIMIC-IV-Echo (UHN) or EchoNet-Dynamic, d=4 attentive, 20 epochs
**Evaluation:** Frozen encoder, frozen probe (zero-shot = no retraining on target population)

| Model | Encoder Checkpoint |
|-------|-------------------|
| JEPA-L pt50 | `checkpoints/echojepa-l-pt50.pt` |
| BYOL-L pt50 | `checkpoints/byol_vitl_imagenet_v2_e50.pt` |
| MAE-L pt50 | `checkpoints/videomae_l_mimic_ep50.pth` |

## Results

### EchoNet-Dynamic LVEF (7,460 train / 1,277 test, public)

| Model | Test R² | Test Pearson | Test MAE |
|-------|---------|-------------|----------|
| **JEPA** | **0.552** | **0.753** | **5.938** |
| BYOL | 0.440 | 0.669 | 6.666 |
| MAE | 0.351 | 0.609 | 7.283 |

Bootstrap CIs (n=1,277, 10K resamples):
- JEPA-BYOL: Δr=+0.083 [+0.055, +0.114]
- JEPA-MAE: ΔR²=+0.201 [+0.168, +0.235]
- BYOL-MAE: Δr=+0.061 [+0.026, +0.098]

All pairwise differences significant.

**Amplification effect:** JEPA advantage grows from +2.5pp R² in-distribution (UHN) → +11.2pp cross-dataset (END). The prediction target matters most when you generalize.

### Pediatric Zero-Shot — UHN-Trained Probes (368 test, NO retraining)

| Model | Test Pearson | Test MAE | Test R² |
|-------|-------------|----------|---------|
| **JEPA** | **0.705** | **6.957** | **0.405** |
| MAE | 0.626 | 7.857 | 0.187 |
| BYOL | 0.602 | 8.004 | 0.206 |

### Pediatric Zero-Shot — END-Trained Probes (368 test, NO retraining)

| Model | Test Pearson | Test MAE | Test R² |
|-------|-------------|----------|---------|
| **JEPA** | **0.615** | **7.358** | **0.293** |
| MAE | 0.531 | 9.203 | 0.041 |
| BYOL | 0.498 | 12.132 | -0.847 |

BYOL collapses completely on cross-population transfer from END (R²=-0.847, MAE=12.1).

## Key Finding

**JEPA's advantage amplifies out-of-distribution:**
- In-distribution (UHN test): +2.5pp R² over BYOL
- Cross-dataset (END test): +11.2pp R² over BYOL
- Cross-population (Pediatric ZS, UHN probes): +10.3pp Pearson over BYOL

Local latent prediction learns more transferable representations than either global self-distillation or pixel reconstruction. The gap widens with distribution shift — the prediction target's advantage is not just "better features" but "more generalizable features."

## References

**Source:** `claude/rebuttals/10-rebuttal-experiment-results.md` §3a, §4b, §4c
**Probe configs:** `configs/eval/vitl/icml/echo{jepa,byol,mae}_l_pt50_end_lvef_d4.yaml`
**Predictions:** `predictions/icml-echo{jepa,byol,mae}-l-pt50-{end,enp}-lvef-*.csv`
**Test CSV:** `data/csv/echonet_dynamic_test_local.csv` (1,277 videos), `data/csv/echonet_pediatric_test_*.csv` (368 videos)
