# EchoJEPA-L-K MIMIC LVEF Structured Probe

**Date:** 2026-04-07
**Job:** HyperPod 429 (node ip-10-0-50-83)
**Status:** Completed 22/35 epochs (manually cancelled at e22 for fair G vs L-K comparison)

## Task
- **Measurement:** LVEF from structured echo reports
- **Type:** Regression (%), B-mode A4C + A2C
- **Dataset:** MIMIC-IV-Echo, study-level sampling
- **Train/Val:** 27,771 / 6,256 clips (same split as EchoJEPA-G job 401)

## Model
- **Backbone:** EchoJEPA-L-K (ViT-Large, vit_large, Kinetics-400 init)
- **Checkpoint:** vitl-kinetics-pt220-an55.pt (Kinetics init + pretrain 220ep + anneal 55ep on MIMIC)
- **Probe:** d=1 attentive, 16 heads, frozen backbone
- **HP grid:** 20 multihead configs (5 LR x 4 WD)

## Results

| Metric | Best Value | Epoch | Head |
|--------|-----------|-------|------|
| **Val MAE** | **7.190** (13.1% of mean) | 21 | — |
| **Val R²** | **0.442** | 20 | 8 |
| **Val Pearson** | **0.679** | 20 | 8 |
| Predict-mean baseline MAE | 10.319 | — | — |

### Epoch-by-Epoch (all epochs)
| Epoch | Train MAE | Val MAE | Best MAE | Val R² | Val Pearson |
|-------|-----------|---------|----------|--------|-------------|
| 1 | 9.517 | 8.857 | 8.857 | 0.118 | 0.368 |
| 2 | 9.366 | 8.515 | 8.515 | 0.212 | 0.474 |
| 3 | 9.057 | 8.121 | 8.121 | 0.266 | 0.533 |
| 4 | 8.776 | 8.362 | 8.121 | 0.212 | 0.509 |
| 5 | 8.665 | 7.846 | 7.846 | 0.305 | 0.568 |
| 6 | 8.637 | 7.656 | 7.656 | 0.361 | 0.617 |
| 7 | 8.641 | 7.719 | 7.656 | 0.359 | 0.620 |
| 8 | 8.408 | 7.541 | 7.541 | 0.363 | 0.623 |
| 9 | 8.383 | 7.423 | 7.423 | 0.376 | 0.615 |
| 10 | 8.286 | 7.461 | 7.423 | 0.387 | 0.632 |
| 11 | 8.271 | 7.550 | 7.423 | 0.367 | 0.614 |
| 12 | 8.402 | 7.651 | 7.423 | 0.331 | 0.639 |
| 13 | 8.258 | 7.296 | 7.296 | 0.409 | 0.640 |
| 14 | 7.927 | 7.441 | 7.296 | 0.396 | 0.639 |
| 15 | 7.975 | 7.353 | 7.296 | 0.390 | 0.628 |
| 16 | 8.064 | 7.410 | 7.296 | 0.396 | 0.639 |
| 17 | 7.887 | 7.432 | 7.296 | 0.384 | 0.640 |
| 18 | 8.015 | **7.192** | 7.192 | **0.434** | **0.659** |
| 19 | 7.851 | 7.275 | 7.192 | 0.403 | 0.671 |
| 20 | 7.851 | 7.310 | 7.192 | **0.442** | **0.679** |
| 21 | 7.870 | **7.190** | **7.190** | 0.407 | 0.639 |
| 22 | 7.696 | 7.208 | 7.190 | 0.414 | 0.654 |

## EchoJEPA-G vs EchoJEPA-L-K Comparison (22 epochs, same task/data/probe)

| Metric | EchoJEPA-G (job 401) | EchoJEPA-L-K (job 429) | Delta |
|--------|---------------------|----------------------|-------|
| **Val MAE** | **6.649** (12.1%) | 7.190 (13.1%) | G better by 0.541 (7.5%) |
| **Val R²** | **0.504** | 0.442 | G better by 0.062 |
| **Val Pearson** | **0.715** | 0.679 | G better by 0.036 |
| Best MAE epoch | 7 | 21 | G converges faster |
| Parameters | ~1.1B (ViT-Giant) | ~307M (ViT-Large) | 3.6x |

**Summary:** EchoJEPA-G outperforms EchoJEPA-L-K on LVEF structured regression across all metrics, with 7.5% lower MAE, +0.06 R², and +0.04 Pearson. G also converges much faster (best at e7 vs e21). The gap is meaningful but L-K still achieves 30% improvement over predict-mean baseline (7.190 vs 10.319).

## Artifacts
- **Best probe checkpoint:** `s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts/runs/echojepa_l_k_lvef_structured_429/training_folder/video_classification_frozen/echojepa-l-k-lvef-structured/best.pt`
- **Latest checkpoint:** `...429/.../latest.pt` (e22)
- **CSV log:** `...429/.../log_r0.csv`
- **Config:** `configs/eval/vitl/nature_medicine/echojepa_l_k_lvef_structured_hp.yaml`

## Notes
- Stopped at e22 to match EchoJEPA-G (which stalled at e22 due to credential expiry)
- L-K best MAE was still slowly improving (7.296 at e13 -> 7.190 at e21), suggesting more epochs might yield marginal gains
- Mean LVEF = 54.74%, std = 12.93%
- 30% better than predict-mean baseline (7.190 vs 10.319)
