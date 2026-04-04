# Noise Robustness (EchoBench)

**Date:** 2026-03-31
**Status:** Complete.
**NeurIPS section:** §5 (Robustness Under Physics-Based Perturbations)

---

## Overview

Frozen probes (trained on clean data, not retrained) evaluated under physics-based ultrasound perturbations at 3 severity levels. Tests representation robustness, not adaptation. Reveals task-specific failure modes invisible from clean benchmarks.

## Perturbation Types

| Perturbation | Physical Basis | Effect |
|---|---|---|
| **Depth attenuation** | Signal falloff with depth | Progressive darkening of far-field |
| **Acoustic shadow** | Signal blocked by structure | Localized sector dropout |
| **Haze artifact** | Reverberation / contrast reduction | Diffuse quality degradation |

Implementation: `scripts/rebuttal/echo_perturbations.py` via `scripts/rebuttal/noised_inference.py`

## Results

### LVEF Robustness (EchoNet-Dynamic test, 1,277 videos, R²)

| Perturbation | Severity | JEPA | BYOL | MAE |
|---|---|---|---|---|
| **Depth attenuation** | Clean | **0.552** | 0.440 | 0.351 |
| | Mild | **0.513** | 0.372 | 0.306 |
| | Moderate | **0.438** | 0.260 | 0.267 |
| | Severe | **0.361** | 0.145 | 0.233 |
| | Drop | **-34.6%** | -67.0% | -33.6% |
| **Acoustic shadow** | Clean | **0.552** | 0.440 | 0.351 |
| | Mild | **0.543** | 0.417 | 0.333 |
| | Moderate | **0.512** | 0.347 | 0.298 |
| | Severe | **0.478** | 0.247 | 0.280 |
| | Drop | **-13.4%** | -43.9% | -20.2% |
| **Haze artifact** | Clean | **0.552** | 0.440 | 0.351 |
| | Mild | **0.547** | 0.435 | 0.336 |
| | Moderate | **0.530** | 0.422 | 0.279 |
| | Severe | **0.502** | 0.398 | 0.147 |
| | Drop | **-9.1%** | -9.5% | -58.1% |

**Average clean→severe R² drop:** JEPA **-19.0%**, BYOL -40.1%, MAE -37.3%

JEPA under severe noise outperforms MAE's *clean* baseline on all 3 perturbation types (R² 0.361-0.502 vs 0.351).

### CAMUS Segmentation Robustness (50 test patients, mean Dice)

| Perturbation | Severity | JEPA | BYOL | MAE |
|---|---|---|---|---|
| **Depth attenuation** | Clean | 0.815 | 0.821 | **0.822** |
| | Severe | 0.681 | 0.425 | **0.749** |
| | Drop | -16.4% | -48.2% | **-8.9%** |
| **Acoustic shadow** | Clean | 0.815 | 0.821 | **0.822** |
| | Severe | 0.708 | 0.614 | **0.728** |
| | Drop | -13.1% | -25.2% | **-11.4%** |
| **Haze artifact** | Clean | 0.815 | 0.821 | **0.822** |
| | Severe | **0.800** | 0.804 | 0.794 |
| | Drop | **-1.8%** | -2.1% | -3.4% |

**Average Dice drop:** MAE **-7.9%**, JEPA -10.4%, BYOL -25.2%

### Pediatric Zero-Shot Robustness (UHN probes → 368 pediatric test, Pearson)

| Perturbation | Severity | JEPA | BYOL | MAE |
|---|---|---|---|---|
| Clean | — | **0.695** | 0.589 | 0.613 |
| Depth attenuation | Severe | **0.596** | 0.347 | 0.544 |
| Acoustic shadow | Severe | **0.654** | 0.574 | 0.598 |
| Haze artifact | Severe | **0.592** | 0.532 | 0.481 |

JEPA maintains highest absolute Pearson at every severity level across all perturbation types.

## Key Finding

**Anatomy-function dissociation extends to robustness:**
- On LVEF (function): JEPA most robust (-19%) > MAE (-37%) > BYOL (-40%)
- On CAMUS (anatomy): MAE most robust (-8%) > JEPA (-10%) > BYOL (-25%)

Each objective is most robust on the task it encodes best. BYOL collapses on both. Clean performance fails to predict robustness: all three converge on clean CAMUS (<1pp); under severe depth attenuation, 32pp gap emerges.

## References

**Source:** `claude/rebuttals/10-rebuttal-experiment-results.md` §5m, §5n, §5o
**Scripts:** `scripts/rebuttal/noised_inference.py` (LVEF), `scripts/rebuttal/noised_segmentation.py` (CAMUS)
**CSVs:** `scripts/rebuttal/samples/{jepa,byol,mae}_end_lvef_noised_inference.csv`, `*_noised_segmentation.csv`
**Perturbation code:** `scripts/rebuttal/echo_perturbations.py`
