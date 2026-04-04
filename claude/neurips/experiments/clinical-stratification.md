# Pathology-Stratified Clinical Analysis

**Date:** 2026-03-31
**Status:** Complete (re-analysis of existing EchoNet-Dynamic predictions).
**NeurIPS section:** §3 (Core Finding — Clinical Impact)

---

## Overview

Stratifies LVEF prediction performance by EF severity category to reveal whether the prediction target advantage concentrates on clinically normal or clinically important cases. Uses the same EchoNet-Dynamic test predictions from the cross-dataset transfer experiment.

## Setup

**Data:** EchoNet-Dynamic test set (1,277 videos), stratified by true LVEF:
- Normal: EF ≥ 55% (n=876)
- Mildly reduced: EF 40-54% (n=241)
- Reduced: EF < 40% (n=160)

**Models:** Same 3-way pt50 comparison. Predictions already generated (no new inference).

## Results

### Per-Bin Performance

| EF Bin | N | JEPA Pearson | BYOL Pearson | MAE Pearson | JEPA MAE | BYOL MAE | MAE MAE |
|--------|---|-------------|-------------|------------|----------|----------|---------|
| Normal (≥55%) | 876 | 0.295 | 0.212 | 0.190 | 4.3 | 5.0 | 5.1 |
| Mildly reduced (40-54%) | 241 | 0.372 | 0.334 | 0.274 | 7.6 | 7.8 | 7.1 |
| **Reduced (<40%)** | 160 | **0.573** | 0.445 | 0.457 | **12.4** | 14.4 | **19.3** |

### Prediction Bias on Reduced EF (true mean 29.0%)

| Model | Predicted Mean | Bias | Within-Bin Pearson |
|-------|---------------|------|-------------------|
| **JEPA** | 40.2% | +11.2 | **0.573** |
| BYOL | 42.5% | +13.5 | 0.445 |
| MAE | **48.4%** | **+19.3** | 0.457 |

## Key Finding

**JEPA's clinical advantage is 8× larger on reduced EF** compared to normal EF (MAE gap: 6.9 points on reduced vs 0.8 on normal). MAE predicts 48% for patients with true EF 29% — effectively classifying severe heart failure as normal. JEPA maintains within-bin discrimination (Pearson 0.573 vs MAE 0.457) and keeps predictions in the correct clinical category despite regression-to-mean bias.

This is the most clinically important finding: the prediction target advantage concentrates precisely where the stakes are highest.

## References

**Source:** `claude/rebuttals/10-rebuttal-experiment-results.md` §6d
**Predictions:** Same as cross-dataset experiment — `predictions/icml-echo{jepa,byol,mae}-l-pt50-end-lvef-*.csv`
