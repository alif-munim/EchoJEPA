# EchoJEPA-G MIMIC LVEF Structured Probe

**Date:** 2026-04-06
**Job:** HyperPod 401 (node ip-10-0-50-83)
**Status:** Completed 22/35 epochs (stalled on S3 credential expiry at e23; best converged at e7)

## Task
- **Measurement:** LVEF from structured echo reports
- **Type:** Regression (%), B-mode A4C + A2C
- **Dataset:** MIMIC-IV-Echo, study-level sampling
- **Train/Val:** 27,771 / 6,256 clips

## Model
- **Backbone:** EchoJEPA-G (ViT-Giant, vit_giant_xformers)
- **Checkpoint:** pt-280-an81.pt (pretrain 280ep + anneal 81ep)
- **Probe:** d=1 attentive, 16 heads, frozen backbone
- **HP grid:** 20 multihead configs (5 LR × 4 WD)

## Results

| Metric | Best Value | Epoch | Head |
|--------|-----------|-------|------|
| **Val MAE** | **6.649** (12.1% of mean) | 7 | — |
| **Val R²** | **0.504** | 19 | 10 |
| **Val Pearson** | **0.715** | 19 | 6 |
| Predict-mean baseline MAE | 10.319 | — | — |

### Epoch-by-Epoch Val MAE (best head)
| Epoch | Train MAE | Val MAE | Best MAE | Val R² | Val Pearson |
|-------|-----------|---------|----------|--------|-------------|
| 1 | 9.586 | 7.551 | 7.551 | 0.370 | 0.616 |
| 2 | 8.643 | 6.914 | 6.914 | 0.456 | 0.687 |
| 7 | 7.646 | **6.649** | **6.649** | **0.501** | **0.713** |
| 15 | 7.407 | 6.793 | 6.649 | 0.476 | 0.690 |
| 19 | 7.204 | 6.656 | 6.649 | **0.504** | **0.715** |
| 22 | 7.149 | 6.791 | 6.649 | 0.480 | 0.699 |

## Artifacts
- **Best probe checkpoint:** `s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts/runs/echojepa_g_lvef_structured_401/training_folder/video_classification_frozen/echojepa-g-lvef-structured/best.pt`
- **Latest checkpoint:** `...401/.../latest.pt` (e22)
- **Config:** `configs/eval/vitg-384/nature_medicine/echojepa_g_lvef_structured_hp.yaml`

## Notes
- Best val MAE converged early (e7) and was stable through e22 — remaining epochs unlikely to improve significantly
- Job stalled at e23 due to AWS credential expiry (IMDSv2 token timeout after ~1h)
- Mean LVEF = 54.74%, std = 12.93%
- 36% better than predict-mean baseline (6.649 vs 10.319)
