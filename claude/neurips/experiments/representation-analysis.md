# Representation-Level Analysis Results

**Date:** 2026-04-07
**Models:** JEPA IN21K e100, BYOL e100, MAE e99 (init-matched, ~100 epochs on MIMIC)

---

## 1. Effective Dimensionality (spectral entropy)

Computed as exp(-Σ σ̄ᵢ log σ̄ᵢ) where σ̄ᵢ are normalized singular values of the [N_videos, D] mean-pooled embedding matrix. Higher = more diverse/informative features. Script: `scripts/rebuttal/rankme.py`.

### Consistent 4-model comparison (2026-04-07, job 510/525)

All models run with the same script, same 500 EchoNet-Dynamic test videos (S3-streamed, seed=42), same GPU (node 184 H100). This supersedes the prior numbers.

| Model | Effective Dim | Embed Dim | Usage | Top-10 SV Energy | Top-50 SV Energy |
|-------|-------------|-----------|-------|-----------------|-----------------|
| JEPA IN21K e95 | **245.3** | 1024 | 24.0% | 66.7% | 89.5% |
| BYOL e100 | **220.7** | 1024 | 21.6% | 65.2% | 90.7% |
| MAE e99 | **206.4** | 1024 | 20.2% | 73.0% | 93.2% |
| SALT v1 e79 | **202.7** | 1024 | 19.8% | 71.3% | 92.0% |

**Key finding:** All four models are in the **200-245 range**. There is no dramatic dimensionality collapse for MAE or SALT. JEPA has the highest diversity (245), followed by BYOL (221), MAE (206), and SALT (203). The gap is modest (~20%), not the 3× difference previously reported.

**SALT dimensionality-collapse hypothesis: NOT SUPPORTED.** SALT's effective dimensionality (203) is essentially identical to MAE's (206), and both are in the same ballpark as JEPA/BYOL. The frozen pixel-reconstruction teacher does not constrain the student to a low-dimensional subspace. SALT's gap to JEPA on downstream tasks is about teacher dynamics (lack of co-evolving EMA target), not representational capacity.

**Retraction of prior numbers:** The earlier values (BYOL 209, JEPA 197, MAE 63) were computed with a different methodology/dataset (Goodfire report) and are not reproducible with the consistent pipeline above. The MAE=63 result in particular does not hold — MAE's effective dimensionality on EchoNet-Dynamic test videos is 206, comparable to the other models. All downstream analysis that relied on the "MAE 3× lower" claim should be revised.

**Revised interpretation:** The spectral structure differs subtly — MAE and SALT concentrate more energy in their top singular values (73% and 71% in top-10, vs JEPA's 67%) — but the overall effective dimensionality is similar. The difference between JEPA and MAE/SALT is in *what* gets encoded (temporal structure, functional features), not in *how much capacity* is used. See frame shuffling (§3) for the temporal encoding evidence.

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
| "JEPA uses representational capacity more efficiently" | Effective dimensionality | **Not supported (revised).** All models are in the 200-245 range (JEPA 245, BYOL 221, MAE 206, SALT 203). The prior MAE=63 number is not reproducible. Differences are in spectral concentration, not total dimensionality. |

**Revised mechanistic story for §4:**

JEPA's advantage is about **temporal structure encoding**, not noise filtering or representational capacity. Frame shuffling shows JEPA consolidates temporal information (−17% at convergence) while MAE abandons it (−4%). Effective dimensionality is similar across all models (200-245 range), ruling out capacity-based explanations. SALT's performance gap (frozen teacher, no EMA co-evolution) further isolates teacher dynamics as the key variable. The prediction target determines *what* temporal structure is encoded, not *how much* capacity is available.

---

## Output Files

| Experiment | Files |
|-----------|-------|
| Effective dimensionality (4-model) | `scripts/rebuttal/samples/rankme_all.csv` (job 510/525, consistent pipeline) |
| Speckle probing (mean-pooled) | `scripts/rebuttal/samples/speckle_probing_{JEPA_IN21K_e100,BYOL_e100,MAE_e99}.png` |
| Layer-wise speckle | `scripts/rebuttal/samples/layerwise_speckle_{JEPA-IN21K-e100,BYOL-L-e100,MAE-L-e99}.csv` |
| Token-level speckle | `scripts/rebuttal/samples/token_speckle_{JEPA-IN21K-e100,BYOL-L-e100,MAE-L-e99}.csv` |
| Temporal consistency | `scripts/rebuttal/samples/temporal_consistency_{JEPA,BYOL,MAE}.csv` |
| Autocorrelation sweep | `scripts/rebuttal/samples/autocorr_{JEPA_IN21K_e100,BYOL_e100,MAE_e99}.csv` |
