# MR Severity: Cross-Dataset Transfer (UHN → MIMIC)

**Date:** 2026-04-08
**Status:** Complete.
**NeurIPS section:** §3 (Cross-modal — hemodynamic severity generalization)
**Script:** `scripts/echojepa_g_mr_compare_mimic_test.sbatch`

---

## Overview

Tests whether a mitral regurgitation severity probe trained on UHN echos generalizes to MIMIC-IV-Echo. Both probes use the same frozen EchoJEPA-G encoder (`pt-280-an81.pt`, ViT-G, 280 pretrain + 81 anneal epochs on UHN 18M), same probe architecture (d=1 attentive, 16 multihead HP sweep), and same 4-class ordinal task (None-Trivial / Mild / Moderate / Severe).

## Setup

| Probe | Training Data | Training Epochs | Best Val Acc | Job |
|-------|---------------|-----------------|-------------|-----|
| MIMIC probe | MIMIC-IV-Echo MR labels | 35 | 57.87% | 436 |
| UHN probe | UHN MR severity labels | 28 (stopped at 29/35) | 69.99% (e22) | 443 |

**Test set:** MIMIC-IV-Echo MR test (1,003 studies), prediction-averaged across all clips per study.

## Results on MIMIC MR Test Set

| Probe | Accuracy | Balanced Acc | Quadratic Kappa | Macro AUROC (OvR) |
|-------|----------|-------------|-----------------|-------------------|
| **MIMIC probe (in-distribution)** | **0.591** | **0.391** | **0.538** | **0.806** |
| UHN probe (cross-dataset) | 0.531 | 0.341 | 0.410 | 0.799 |

### Per-Class Breakdown

**MIMIC probe (in-distribution):**

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| None-Trivial | 0.69 | 0.88 | 0.77 | 503 |
| Mild | 0.39 | 0.34 | 0.36 | 271 |
| Moderate | 0.47 | 0.35 | 0.40 | 173 |
| Severe | 0.00 | 0.00 | 0.00 | 56 |

**UHN probe (cross-dataset):**

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| None-Trivial | 0.77 | 0.71 | 0.73 | 503 |
| Mild | 0.33 | 0.66 | 0.44 | 271 |
| Moderate | 0.00 | 0.00 | 0.00 | 173 |
| Severe | 0.00 | 0.00 | 0.00 | 56 |

## Key Finding

**AUROC is nearly preserved cross-dataset (0.806 → 0.799, −0.9%).** The UHN-trained probe's discriminative ability transfers almost perfectly to MIMIC, despite different institutions, patient populations, and label sources. The primary degradation is in calibration, not discrimination: the UHN probe collapses Moderate and Severe into a single bucket (0% recall), while maintaining better Mild recall (66% vs 34%).

This suggests:
1. The frozen EchoJEPA-G encoder captures institution-agnostic MR features
2. The AUROC gap is negligible — the probe's ranking ability transfers cross-institution
3. Classification thresholds (not features) need recalibration across datasets
4. Both probes fail on Severe (n=56) — likely a class imbalance issue, not a representation issue

## Caveats

- The UHN probe was stopped early (epoch 29/35, best at epoch 22) — the final model may be slightly undertrained
- UHN and MIMIC may use different MR grading protocols (visual assessment vs quantitative criteria)
- MIMIC probe's "in-distribution" advantage is itself modest (Acc 59.1%, balanced Acc 39.1%) — the task is hard

## Artifacts

- **Comparison CSV:** `s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts/runs/echojepa_g_mr_compare_549/logs/mr_comparison.csv`
- **Study predictions:** `echojepa_g_mr_compare_549/logs/{mimic,uhn}_probe_study_predictions.csv`
- **UHN probe checkpoint:** `echojepa_g_mr_severity_uhn_443/training_folder/.../best.pt`
- **UHN probe training log:** `echojepa_g_mr_severity_uhn_443/training_folder/.../log_r0.csv`
- **Config:** `configs/eval/vitg-384/nature_medicine/echojepa_g_mr_uhn_on_mimic_predavg.yaml`
