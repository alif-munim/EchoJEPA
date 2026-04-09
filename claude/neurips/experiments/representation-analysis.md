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

## 7. Cross-Temporal Attention Analysis

**Date:** 2026-04-09
**Script:** `scripts/rebuttal/temporal_attention_trial.py` (HyperPod jobs 919, 931)

For each layer, compute the fraction of attention flowing between tokens at different temporal positions (cross-temporal ratio). Approach: set `use_sdpa=False` to force explicit attention computation, hook into `attn_drop` (nn.Dropout) whose input is the post-softmax attention weight matrix `[B, H, N, N]`. Tokens are ordered `[T_patches × H_patches × W_patches]` = `[8 × 14 × 14]` = 1568. A token's temporal index = `token_id // 196`. Random baseline = 7/8 = 0.875 (7 of 8 temporal positions are "other").

All models: ViT-L (24 layers, 16 heads), 10 EchoNet-Dynamic test videos, seed=42.

### Epoch ~100 results

| Layer | JEPA e100 | BYOL e100 | MAE e99 | SALT S2 e79 |
|-------|-----------|-----------|---------|-------------|
| 0 | **0.601** | 0.857 | 0.861 | **0.439** |
| 1 | **0.567** | 0.775 | 0.860 | **0.491** |
| 2 | 0.837 | 0.744 | 0.825 | **0.419** |
| 3 | 0.822 | 0.815 | 0.834 | **0.463** |
| 4 | 0.859 | 0.828 | 0.822 | **0.387** |
| 5 | 0.818 | 0.813 | 0.868 | **0.422** |
| 6 | 0.854 | 0.848 | 0.867 | **0.420** |
| 7 | 0.846 | 0.848 | 0.860 | **0.433** |
| 8 | 0.844 | 0.856 | 0.842 | **0.490** |
| 9 | 0.864 | 0.863 | 0.869 | **0.447** |
| 10 | 0.873 | 0.864 | 0.841 | **0.557** |
| 11 | 0.874 | 0.882 | 0.841 | **0.839** |
| 12-23 | 0.87-0.88 | 0.87-0.88 | 0.87-0.87 | 0.83-0.88 |
| **Overall** | **0.839** | **0.855** | **0.861** | **0.672** |

### Epoch ~50 results (SALT e29 for comparison)

| Layer | JEPA pt50 | BYOL pt50 | MAE pt50 | SALT e29 |
|-------|-----------|-----------|----------|----------|
| 0 | **0.409** | 0.869 | 0.865 | **0.274** |
| 1 | **0.660** | 0.770 | 0.861 | 0.743 |
| 2 | **0.582** | 0.793 | 0.808 | 0.813 |
| 3-10 | 0.57-0.88 | 0.82-0.88 | 0.81-0.87 | 0.85-0.88 |
| 11-23 | 0.86-0.88 | 0.86-0.89 | 0.86-0.87 | 0.87-0.88 |
| **Overall** | **0.795** | **0.862** | **0.861** | **0.838** |

### Key findings

1. **SALT develops the strongest spatial→temporal hierarchy.** At e79, layers 0-10 are heavily within-frame (0.39-0.56), with a sharp transition at layer 11 to cross-temporal (~0.84). This hierarchy **deepens with training**: at e29 only layer 0 is spatial-biased (0.27), but by e79 the entire first half of the network specializes for spatial processing.

2. **JEPA shows a milder version of the same pattern.** Layers 0-1 are spatial-biased (0.57-0.60 at e100), then rapidly reach near-baseline by layer 2. The hierarchy is shallower than SALT's.

3. **BYOL and MAE are near-uniform across layers.** Both hover around the random baseline (~0.86-0.87) from layer 0 onwards. No clear spatial→temporal specialization.

4. **Predictive objectives (JEPA, SALT) induce spatial-first processing.** The prediction target forces early layers to build spatial features before integrating temporal context. Contrastive (BYOL) and reconstructive (MAE) objectives do not produce this hierarchy.

5. **SALT's distillation amplifies the teacher's hierarchy.** The S2 student has a much more pronounced spatial→temporal transition than its JEPA teacher, suggesting the frozen-teacher + predictor architecture drives stronger layer-wise specialization.

### Interpretation

The spatial-first hierarchy is consistent with the frame-shuffling results (§3 of synthesis): models that explicitly separate spatial and temporal processing (JEPA, SALT) are the ones that encode temporal dynamics most robustly. The attention analysis provides the mechanistic explanation — early layers attend within-frame to build spatial features, late layers integrate across frames for temporal dynamics. MAE and BYOL mix spatial and temporal attention uniformly, which may explain their weaker temporal encoding.

---

## Synthesis: What IS JEPA's mechanism?

Three hypotheses tested and their status:

| Hypothesis | Evidence | Verdict |
|-----------|---------|---------|
| "JEPA filters frame-varying noise via EMA" | Speckle probing, temporal consistency, autocorrelation sweep | **Not supported.** BYOL filters more noise; temporal consistency is similar for JEPA/MAE; static noise is worst. |
| "JEPA encodes temporal dynamics that MAE doesn't" | Frame shuffling (3 regimes), severity gradient, cross-temporal attention | **Supported.** JEPA consolidates temporal encoding (−17% at convergence); MAE abandons it (−4%). Attention analysis shows JEPA/SALT develop spatial-first processing hierarchy; MAE/BYOL do not. |
| "JEPA uses representational capacity more efficiently" | Effective dimensionality | **Not supported (revised).** All models are in the 200-245 range (JEPA 245, BYOL 221, MAE 206, SALT 203). The prior MAE=63 number is not reproducible. Differences are in spectral concentration, not total dimensionality. |
| "Predictive objectives induce spatial→temporal layer specialization" | Cross-temporal attention analysis (§7) | **Supported.** JEPA and SALT develop spatial-first attention (early layers attend within-frame), while BYOL and MAE distribute spatial/temporal attention uniformly. SALT's distillation amplifies this hierarchy (layers 0-10 spatial-biased at 0.39-0.56 vs JEPA's 0.57-0.60 in layers 0-1 only). |

**Revised mechanistic story for §4:**

JEPA's advantage is about **temporal structure encoding**, not noise filtering or representational capacity. Frame shuffling shows JEPA consolidates temporal information (−17% at convergence) while MAE abandons it (−4%). Cross-temporal attention analysis reveals the underlying mechanism: predictive objectives (JEPA, SALT) induce a spatial-first processing hierarchy where early layers attend within-frame and later layers integrate across time. MAE and BYOL show no such specialization. SALT amplifies this hierarchy through distillation — its entire first half (layers 0-10) specializes for spatial processing, compared to just layers 0-1 in JEPA. Effective dimensionality is similar across all models (200-245 range), ruling out capacity-based explanations. The prediction target determines *what* temporal structure is encoded and *how* the network organizes its processing, not *how much* capacity is available.

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
| Cross-temporal attention (~e100) | `scripts/rebuttal/temporal_attention/{jepa_e100,byol_e100,mae_e99,salt_s2v1_e79}_temporal_attention.csv` |
| Cross-temporal attention (~e50) | `scripts/rebuttal/temporal_attention/{jepa_pt50,byol_pt50,mae_pt50,salt_s2v1_e29}_temporal_attention.csv` |
| Cross-temporal attention (per-head) | `scripts/rebuttal/temporal_attention/*_per_head.csv` |
