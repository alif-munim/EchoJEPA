# Representation-Level Analysis Results

**Date:** 2026-04-07
**Models:** JEPA IN21K e100, BYOL e100, MAE e99 (init-matched, ~100 epochs on MIMIC)

---

## 1. Effective Dimensionality (spectral entropy)

Computed as exp(-Σ σ̄ᵢ log σ̄ᵢ) where σ̄ᵢ are normalized singular values of the [N_videos, D] embedding matrix. Higher = more diverse/informative features.

| Model | Effective Dim | Embed Dim | Usage |
|-------|-------------|-----------|-------|
| BYOL e100 | **209** | 1024 | 20% |
| JEPA IN21K e100 | **197** | 1024 | 19% |
| MAE e99 | **63** | 1024 | 6% |

**Key finding:** MAE representations occupy a 3× lower-dimensional subspace than JEPA/BYOL. Pixel reconstruction produces highly redundant features — many dimensions encode similar pixel-level information. Latent prediction objectives (JEPA, BYOL) encourage representational diversity.

**Mechanistic interpretation:** MAE's low effective dimensionality explains its weakness on functional tasks. With only 63 effective dimensions, there's less capacity for encoding complex temporal/functional information. JEPA and BYOL are close (~197 vs 209), consistent with both using objectives that encourage feature diversity.

---

## 2. Speckle Probing (mean-pooled, final layer)

Partial R² for speckle energy, controlling for mean intensity. Measures how much high-frequency content (anatomy + noise) is retained in the representation.

| Model | Speckle Partial R² |
|-------|--------------------|
| MAE e99 | 0.885 |
| JEPA IN21K e100 | 0.848 |
| BYOL e100 | 0.716 |

**⚠️ Revised from ICML rebuttal:** At pt50 the ordering was JEPA (0.674) < BYOL (0.775) < MAE (0.875). The pt50 result was confounded by JEPA's unfair 235ep init. With init-matching, the ordering changes to BYOL < JEPA < MAE.

**Interpretation:** MAE retains the most high-frequency information (both noise and fine anatomy). BYOL discards the most (global contrastive objective → invariance to local spatial detail). JEPA is intermediate. The "JEPA filters noise" narrative from the ICML rebuttal does not hold with init-matched models.

**Caveat (from web session analysis):** High-frequency power includes both speckle and fine anatomical detail (valve leaflets, trabeculations). This metric conflates stochastic noise with deterministic anatomy. See temporal consistency for a cleaner test.

---

## 3. Layer-wise Speckle Probing

Speckle partial R² at layers 1, 6, 12, 18, 24 (ViT-L has 24 blocks).

| Layer | JEPA IN21K e100 | BYOL e100 | MAE e99 |
|-------|-----------------|-----------|---------|
| 1 | 0.838 | 0.889 | 0.840 |
| 6 | 0.790 | 0.885 | 0.884 |
| 12 | 0.784 | 0.754 | 0.849 |
| 18 | 0.766 | 0.613 | 0.813 |
| 24 | 0.759 | 0.617 | 0.808 |

**Depth filtering profile:**
- BYOL: 0.889 → 0.617 (−31%) — steepest filtering, actively discards high-frequency info
- JEPA: 0.838 → 0.759 (−9%) — modest filtering
- MAE: 0.840 → 0.808 (−4%) — retains high-frequency info throughout

BYOL's global contrastive objective drives progressive abstraction across depth. MAE retains pixel-level detail through all layers (needed for reconstruction). JEPA is intermediate.

---

## 4. Token-level Speckle Probing

Per-patch prediction: train Ridge probe from individual spatial token embeddings to predict speckle energy of the corresponding 16×16 image patch.

| Model | Token Speckle R² |
|-------|-----------------|
| MAE e99 | **0.941** |
| JEPA IN21K e100 | 0.926 |
| BYOL e100 | 0.891 |

Same ordering as mean-pooled: MAE > JEPA > BYOL. Individual tokens also reflect the model's pixel-level information retention.

---

## 5. Temporal Consistency (frame-to-frame cosine similarity)

Mean cosine similarity between embeddings of consecutive temporal positions (mean-pooled over spatial tokens per temporal position). Higher = more consistent across frames (discards frame-specific variation).

| Model | Mean Cosine Sim | Std |
|-------|----------------|-----|
| BYOL e100 | **0.976** | 0.023 |
| JEPA IN21K e100 | 0.954 | 0.020 |
| MAE e99 | 0.950 | 0.012 |

**Key finding:** BYOL is most temporally consistent, not JEPA. The "JEPA filters frame-specific noise via EMA temporal averaging" hypothesis is **not supported**. BYOL's high consistency reflects its global contrastive objective (discards local spatial variation), not noise filtering. JEPA and MAE are very close (0.954 vs 0.950).

---

## 6. Noise Autocorrelation Sweep

Multiplicative Rayleigh speckle with controllable temporal correlation τ. See `experiments/noise-autocorrelation-sweep.md` for full details.

| τ | JEPA e100 | BYOL e100 | MAE e99 |
|---|-----------|-----------|---------|
| clean | 0.591 | 0.468 | 0.445 |
| ∞ (static) | 0.422 (−29%) | 0.262 (−44%) | −0.122 (−127%) |
| 4.0 (optimal) | 0.574 (−3%) | 0.345 (−26%) | 0.171 (−62%) |
| 0.0 (iid) | 0.508 (−14%) | 0.270 (−42%) | 0.253 (−43%) |

Static noise is worst for all models (opposite of predicted). JEPA most robust at every τ.

---

## Synthesis: What IS JEPA's mechanism?

Three hypotheses tested and their status:

| Hypothesis | Evidence | Verdict |
|-----------|---------|---------|
| "JEPA filters frame-varying noise via EMA" | Speckle probing, temporal consistency, autocorrelation sweep | **Not supported.** BYOL filters more noise; temporal consistency is similar for JEPA/MAE; static noise is worst. |
| "JEPA encodes temporal dynamics that MAE doesn't" | Frame shuffling (3 regimes), severity gradient | **Supported.** JEPA consolidates temporal encoding (−17% at convergence); MAE abandons it (−4%). |
| "JEPA uses representational capacity more efficiently" | Effective dimensionality | **Supported.** JEPA d_eff=197, MAE d_eff=63. MAE wastes capacity on redundant pixel-level features. |

**Revised mechanistic story for §4:**

The prediction target determines two things: (1) **what temporal structure is encoded** (frame shuffling: JEPA consolidates, MAE abandons, BYOL stabilizes) and (2) **how efficiently the representational space is used** (effective dimensionality: MAE 3× lower than JEPA/BYOL). JEPA's advantage is NOT primarily about noise filtering — it's about learning diverse, temporally-structured features that transfer well to functional tasks. MAE's pixel reconstruction objective produces redundant, low-dimensional, temporally-invariant representations that excel at spatial tasks but lack the diversity needed for functional tasks.

---

## Output Files

| Experiment | Files |
|-----------|-------|
| Effective dimensionality | `scripts/rebuttal/samples/rankme_e100.csv` |
| Speckle probing (mean-pooled) | `scripts/rebuttal/samples/speckle_probing_{JEPA_IN21K_e100,BYOL_e100,MAE_e99}.png` |
| Layer-wise speckle | `scripts/rebuttal/samples/layerwise_speckle_{JEPA-IN21K-e100,BYOL-L-e100,MAE-L-e99}.csv` |
| Token-level speckle | `scripts/rebuttal/samples/token_speckle_{JEPA-IN21K-e100,BYOL-L-e100,MAE-L-e99}.csv` |
| Temporal consistency | `scripts/rebuttal/samples/temporal_consistency_{JEPA,BYOL,MAE}.csv` |
| Autocorrelation sweep | `scripts/rebuttal/samples/autocorr_{JEPA_IN21K_e100,BYOL_e100,MAE_e99}.csv` |
