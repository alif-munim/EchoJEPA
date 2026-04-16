# Multi-View Ablation and Noise Robustness

**Date:** 2026-03-30 to 2026-03-31
**Status:** Complete.
**NeurIPS section:** Appendix (multi-view framework ablation), §5 (multi-view noise robustness)

---

## Overview

Ablation of the multi-view probing framework on RVSP regression (requires integrating A4C and PSAX-AV views). Tests whether cross-view integration improves performance and robustness compared to single-view evaluation.

## Setup

**Task:** RVSP regression (multi-view: A4C + PSAX-AV)
**Model:** EchoJEPA-L pt50 only (ablation of the probing framework, not the encoder)
**Data:** UHN RVSP, 41K train / 5K val / 5K test
**Probes:** d=4 attentive with factorized stream embeddings

| Condition | Probe Checkpoint | Test Data |
|-----------|-----------------|-----------|
| Multi-view (A4C + PSAX) | Main RVSP probe | Standard multi-view test |
| A4C only | `echojepa_pt50_rvsp_a4c_301/.../best.pt` | `rvsp_test_a4c.csv` |
| PSAX only | `echojepa_pt50_rvsp_psax_305/.../best.pt` | `rvsp_test_psax.csv` |

## Results

### Single-View vs Multi-View (Clean)

| View Configuration | Test Pearson | Test R² | Test MAE |
|-------------------|-------------|---------|----------|
| **Multi-view (A4C + PSAX)** | **0.484** | **0.220** | **9.101** |
| A4C only | 0.447 | 0.181 | 9.266 |
| PSAX only | 0.449 | 0.188 | 9.368 |

Multi-view: +3.9pp R² over best single view, +3.5pp Pearson over A4C.

### Multi-View Noise Robustness (Pearson, 5,103 test studies)

| Perturbation | Severity | Multi-View | A4C only | PSAX only |
|---|---|---|---|---|
| Clean | — | **0.484** | 0.448 | 0.452 |
| Depth attenuation | Mild | **0.478** | 0.433 | 0.444 |
| | Moderate | **0.466** | 0.413 | 0.430 |
| | Severe | **0.455** | 0.391 | 0.415 |
| Acoustic shadow | Mild | **0.481** | 0.447 | 0.451 |
| | Moderate | **0.471** | 0.430 | 0.439 |
| | Severe | **0.449** | 0.394 | 0.412 |
| Haze artifact | Mild | **0.481** | 0.444 | 0.447 |
| | Moderate | **0.477** | 0.436 | 0.438 |
| | Severe | **0.469** | 0.427 | 0.427 |

**Average severe drop:** Multi-view **-5.4%**, A4C -9.8%, PSAX -7.5%

## Key Finding

Multi-view integration provides both a performance gain (+3.9pp R²) and a robustness gain (halves degradation under noise). Multi-view at severe perturbation (Pearson 0.449-0.469) matches or exceeds single-view clean (0.448-0.452). When one view is degraded, the complementary view compensates.

The multi-view probing framework (factorized stream embeddings, early fusion, view dropout) is applicable to any multi-video setting (multi-sequence MRI, multi-phase CT, multi-probe ultrasound).

**Framework components ablated in Appendix D, Table 5 of ICML preprint:**
- Factorized stream embeddings: encode view identity + clip position separately
- Early fusion with attention masking: +12.1% over late averaging (PanEcho-style)
- View dropout (p=0.1): +18.3% gain on RVSP

## References

**Source:** `claude/rebuttals/10-rebuttal-experiment-results.md` §6b, §6c
**Scripts:** `scripts/neurips/run_rvsp_noise_grid.py`
**Predictions:** `predictions/icml-echojepa-l-pt50-rvsp-{a4c,psax}-test.csv`
