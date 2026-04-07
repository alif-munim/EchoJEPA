# Noise Autocorrelation Sweep Results

**Date:** 2026-04-07
**Script:** `scripts/rebuttal/noise_autocorrelation_sweep.py`
**Dataset:** EchoNet-Dynamic test (1,277 videos)
**Noise model:** Multiplicative Rayleigh speckle (mean-normalized), moderate severity (σ=0.5)
**Temporal correlation:** AR(1) process with correlation time τ. τ=∞ (static), τ=0 (iid per-frame).
**Seeds:** 3 per condition (200, 201, 202)

---

## Results (R², mean ± std of 3 seeds)

| τ (frames) | JEPA IN21K e100 | BYOL e100 | MAE e99 |
|-----------|-----------------|-----------|---------|
| clean | **0.591** | 0.468 | 0.445 |
| ∞ (static) | 0.422 ± 0.006 | 0.262 ± 0.003 | −0.122 ± 0.002 |
| 8.0 | 0.542 ± 0.002 | 0.309 ± 0.001 | 0.065 ± 0.003 |
| 4.0 | **0.574 ± 0.001** | **0.345 ± 0.002** | 0.171 ± 0.003 |
| 2.0 | 0.564 ± 0.002 | 0.337 ± 0.002 | 0.205 ± 0.002 |
| 1.0 | 0.542 ± 0.001 | 0.313 ± 0.002 | 0.220 ± 0.002 |
| 0.5 | 0.522 ± 0.001 | 0.289 ± 0.003 | 0.239 ± 0.002 |
| 0.0 (iid) | 0.508 ± 0.002 | 0.270 ± 0.004 | 0.253 ± 0.002 |

## Relative degradation from clean

| τ | JEPA | BYOL | MAE |
|---|------|------|-----|
| ∞ (static) | −29% | −44% | −127% |
| 8.0 | −8% | −34% | −85% |
| 4.0 | **−3%** | **−26%** | −62% |
| 2.0 | −5% | −28% | −54% |
| 1.0 | −8% | −33% | −51% |
| 0.5 | −12% | −38% | −46% |
| 0.0 (iid) | −14% | −42% | −43% |

---

## Key Findings

### 1. Static noise is WORST, not iid — opposite of initial hypothesis

**Predicted:** MAE degrades as noise becomes more frame-varying (τ→0). JEPA stays robust.
**Found:** Static noise (τ=∞) is most damaging for ALL models. As noise becomes more frame-varying, all models IMPROVE.

- MAE: R² = −0.122 (static) → 0.253 (iid). Static noise destroys MAE; iid noise is tolerable.
- JEPA: R² = 0.422 (static) → 0.508 (iid). Static is 2× more damaging than iid.
- BYOL: R² = 0.262 (static) → 0.270 (iid). Roughly similar at extremes, best at τ=4.

**Why static is worst:** Static noise is a fixed spatial bias corrupting every frame identically — the encoder/probe cannot average it out across frames. Frame-varying noise is unbiased and self-averages over 16 frames. This is basic statistics (variance reduction by √N), not a property of the SSL objective.

### 2. Optimal correlation at τ≈4 for JEPA and BYOL

JEPA peaks at τ=4 (R²=0.574, only −3% from clean). BYOL also peaks at τ=4 (R²=0.345, −26%). This may reflect the temporal scale of cardiac dynamics — at τ=4 frames (≈0.27 seconds in a 1-second clip), the noise has similar temporal structure to real cardiac motion, and the encoders are somewhat calibrated to this timescale.

MAE shows monotonic improvement as τ decreases (no peak at τ=4), consistent with MAE having no temporal encoding to exploit.

### 3. JEPA is most robust at EVERY τ

| τ | Ranking |
|---|---------|
| ∞ (static) | JEPA (0.422) >> BYOL (0.262) >> MAE (−0.122) |
| 4.0 | JEPA (0.574) >> BYOL (0.345) >> MAE (0.171) |
| 0.0 (iid) | JEPA (0.508) >> BYOL (0.270) >> MAE (0.253) |

The ranking JEPA >> BYOL > MAE is maintained regardless of noise temporal structure. JEPA's advantage is about general representation quality, not specific to frame-varying noise.

### 4. MAE is most sensitive to noise type

MAE's degradation ranges from −127% (static) to −43% (iid) — a 3× difference. JEPA ranges from −29% to −14% (2× difference). BYOL ranges from −44% to −42% (~same). MAE's representations are most brittle under spatial corruption; BYOL is equally sensitive to all noise types; JEPA degrades gracefully.

---

## What This Means for the Paper

### The EMA temporal filtering hypothesis is NOT supported by this experiment

The original plan was to use this as "causal proof" that frame-varying noise determines the MAE/JEPA ranking. The results show the opposite pattern. Static noise is worst, not frame-varying. This means:

- The autocorrelation sweep does NOT prove "EMA filters frame-varying noise"
- It DOES show JEPA's robustness advantage is consistent across all noise temporal structures
- The advantage stems from representation quality, not a temporal-noise-specific mechanism

### What IS supported

- **Speckle probing (§4.3):** JEPA encodes 23% less REAL speckle. This still holds — real speckle (from acquisition) is different from synthetic noise (added post-acquisition).
- **Frame shuffling (§4.1-4.2):** Three temporal encoding regimes. Untouched by this result.
- **EchoBench (§5):** Uses static perturbations. Now better explained — static noise is the worst case, and JEPA handles it best.

### Recommended framing for the paper

Include as a supplementary/appendix result with honest framing:

> "To test whether JEPA's advantage depends on noise temporal structure, we sweep the autocorrelation time of synthetic multiplicative speckle from static (τ=∞) to frame-independent (τ=0). Contrary to our initial hypothesis, static noise is most damaging for all models, while frame-varying noise self-averages across the clip. JEPA maintains the highest R² at every correlation time (0.42–0.57 vs BYOL 0.26–0.35 vs MAE −0.12–0.25), suggesting its robustness stems from representation quality rather than a mechanism specific to frame-varying noise."

**Do NOT present this as the §4 centerpiece.** Move to appendix or brief mention in §5 (complements EchoBench). The frame shuffling three-regime finding remains the §4 star.

### Limitation

The noise is applied POST-acquisition to videos that already contain real speckle. This tests "robustness to additional synthetic perturbation" not "robustness to native noise." The spatial structure is also unrealistic — per-pixel iid Rayleigh, not spatially-correlated speckle grains. These limitations weaken the mechanistic interpretation.

---

## Output Files

| Model | CSV |
|-------|-----|
| JEPA IN21K e100 | `scripts/rebuttal/samples/autocorr_JEPA_IN21K_e100.csv` |
| BYOL e100 | `scripts/rebuttal/samples/autocorr_BYOL_e100.csv` |
| MAE e99 | `scripts/rebuttal/samples/autocorr_MAE_e99.csv` |
