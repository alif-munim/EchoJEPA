# Forward Prediction & Zero-Shot Anomaly Detection

Experiments exploiting the JEPA **predictor network** — a component absent from competing foundation models (EchoPrime, PanEcho, MAE). Full results and methodology in `evals/forward_prediction/RESULTS.md`.

## Code

```
evals/forward_prediction/
├── eval.py              # Prediction error anomaly detection (random block masks)
├── forward_predict.py   # Temporal forward prediction (first T/2 → last T/2)
├── anomaly_repr.py      # Representation distance (mean-pooled + token-level)
├── models.py            # Load frozen encoder + predictor + target_encoder
├── masking.py           # RandomBlockMask + FutureFrameMask generators
└── RESULTS.md           # Full experiment log with all results
```

Results stored in:
- `results/anomaly_detection/` — UHN prediction error
- `results/anomaly_repr/` — UHN mean-pooled repr distance
- `results/anomaly_repr_token/` — UHN token-level repr distance
- `results/mimic_anomaly_repr/` — MIMIC repr distance
- `results/mimic_anomaly_pred/` — MIMIC prediction error
- `results/mimic_forward_pred/` — MIMIC forward prediction

## Key Findings

### What works: Representation distance on MIMIC (population negatives)

Tested across 11 tasks. Performance scales with visual distinctiveness:

| Task | Studies | Best AUROC | Method |
|------|---------|:----------:|--------|
| **Takotsubo** | 61 | **0.711** | Mahalanobis (inv) |
| **Amyloidosis** | 33 | **0.698** | Cosine |
| **Tamponade** | 90 | **0.605** | Mahalanobis / Cosine |
| STEMI | 222 | 0.592 | Cosine |
| HCM | 749 | 0.586 | Mahalanobis (inv) |
| In-hosp mortality | 462 | 0.557 | Mahalanobis |
| Mortality 1yr | 1049 | 0.543 | Mahalanobis |
| DCM / HF / Mort 30d / AFib | — | 0.51-0.53 | — |

Zero-shot (no training, no labels) — the encoder's representation space naturally separates dramatic pathologies from the general population. Clear gradient: visual phenotypes (takotsubo, amyloidosis, tamponade) > non-imaging phenotypes (mortality, AFib).

### What doesn't work

1. **Prediction error** (eval.py): AUROC ~0.50 across all tasks (UHN and MIMIC). The predictor reconstructs all pathologies equally well because it was trained on all cardiac dynamics.

2. **Forward prediction** (forward_predict.py): AUROC ~0.52 across all MIMIC tasks. Per-frame errors are flat (0.574-0.584), showing no temporal discrimination.

3. **Any non-takotsubo UHN task**: All 0.51-0.55. Tested 12 tasks including LVEF extremes (0.542), AS severity (0.546), RWMA (0.516), pericardial effusion (0.514), RV function (0.516), TAPSE extremes (0.515), diastolic function (0.525). UHN data was used for pretraining — ALL pathologies are in-distribution, even extreme ones.

4. **Non-imaging phenotypes on MIMIC**: MR (0.522), TR (0.528), AFib (0.509), mortality (0.52-0.54), LV wall thickness (0.523) — Doppler findings and non-imaging outcomes don't separate in B-mode representation space.

5. **Token-level repr distance**: AUROC 0.536 vs 0.523 mean-pooled on UHN HCM. Mean pooling is NOT the bottleneck.

### Two factors for zero-shot detection (both contribute)

1. **Visually dramatic B-mode phenotype** — stronger effect. UHN takotsubo still gets 0.640 even on in-distribution data
2. **Out-of-distribution** test data — amplifies signal. MIMIC consistently > UHN for matched tasks

Head-to-head: Takotsubo MIMIC 0.711 > UHN 0.640 | Amyloidosis MIMIC 0.698 >> UHN 0.543 | PE/Tamponade MIMIC 0.605 >> UHN 0.514

### Interpretation

- The JEPA predictor is a **universal reconstruction model**, not a normality model. It trained on everything and reconstructs everything.
- Representation distance works when there's a **large distributional gap** (takotsubo apical ballooning vs general population). Fails on matched controls or non-imaging phenotypes.
- Supervised d=1 attentive probes get >0.90 AUROC on the same tasks where zero-shot gets ~0.50. The information IS in the representations — extracting it requires learned discrimination.

## Multi-Model Comparison

Zero-shot anomaly detection is NOT JEPA-specific. Tested 4 models on top 3 MIMIC tasks:

| Task | EchoJEPA-G (1012M) | VideoMAE-L (305M) | PanEcho (42M) | EchoPrime (35M) |
|------|:------------------:|:-----------------:|:-------------:|:---------------:|
| Takotsubo | 0.711 | **0.871** | 0.617 | 0.663 |
| Amyloidosis | 0.698 | **0.726** | 0.670 | 0.667 |
| Tamponade | 0.605 | 0.575 | **0.660** | 0.630 |

All models show signal. VideoMAE-L leads on takotsubo/amyloidosis. General property of self-supervised cardiac video encoders, not JEPA-specific. Multi-model support via `--model` flag in `anomaly_repr.py`.

## Environment

- **Conda**: Must use `vjepa2-312` (base has torch/torchvision conflict)
- **Checkpoint**: `checkpoints/vitg-384.pt` (encoder 1012M, predictor 22M, target_encoder 1012M)
- **Datasets**: UHN CSVs in `experiments/nature_medicine/uhn/probe_csvs/disease_*/test.csv`, MIMIC CSVs in `experiments/nature_medicine/mimic/probe_csvs/*/test.csv`
