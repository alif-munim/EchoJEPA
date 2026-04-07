# Speckle Probing (Information Probing)

**Date:** 2026-03-31 (initial), revised 2026-04-07
**Status:** Complete. **The original ICML rebuttal interpretation has been retracted** — see "Retraction" section.
**NeurIPS section:** §4 (Mechanistic Evidence) — but as a *non-load-bearing* result, not a primary mechanism.

---

## ⚠️ Retraction Notice (2026-04-07)

The ICML rebuttal claim that **"JEPA encodes 23% less speckle than MAE"** was based on a confounded comparison:

- **JEPA-L pt50** in the ICML rebuttal = a 235-epoch fully-trained checkpoint mislabeled as "pt50". It had 4.7× more pretraining than the BYOL/MAE pt50 baselines.
- **BYOL-L pt50** = 50 epochs of training, IN21K init.
- **MAE-L pt50** = 50 epochs of training, IN21K init.

When init-matched at e100 (all three models with the same IN21K initialization and ~100 epochs of training):

| Comparison | JEPA | BYOL | MAE | JEPA−MAE gap |
|---|---|---|---|---|
| ICML rebuttal pt50 (confounded) | 0.674 | 0.775 | 0.875 | **−0.201 (−23%)** |
| **e100 init-matched (canonical)** | **0.848** | **0.716** | **0.885** | **−0.037 (−4%)** |

**Two things changed:**
1. The gap shrank from −23% to −4% (essentially nothing).
2. **The ranking changed**: under init-matching, BYOL is the *best* speckle filter, not JEPA.

The "JEPA filters speckle via EMA target averaging" narrative is **not supported by init-matched data**. Do not cite the 0.674/0.875 numbers in any current document.

---

## Canonical Results (e100, init-matched)

**Models:** All three are ViT-L, IN21K-initialized, trained on MIMIC-IV-Echo for ~100 epochs. JEPA = JEPA-IN21K e100, BYOL = BYOL e100, MAE = MAE e99.

### Mean-pooled features, partial R² (speckle | intensity)

| Model | Speckle Partial R² |
|-------|--------------------|
| MAE e99 | **0.885** |
| JEPA IN21K e100 | 0.848 |
| BYOL e100 | **0.716** |

### Layer-wise speckle probing (depth profile)

| Layer | JEPA | BYOL | MAE |
|-------|------|------|-----|
| 1 | 0.838 | 0.889 | 0.840 |
| 6 | 0.790 | 0.885 | 0.884 |
| 12 | 0.784 | 0.754 | 0.849 |
| 18 | 0.766 | 0.613 | 0.813 |
| 24 | 0.759 | 0.617 | 0.808 |
| **Δ depth** | **−9%** | **−31%** | **−4%** |

BYOL filters most aggressively across depth (the global contrastive objective drives progressive abstraction). JEPA filters modestly. MAE retains pixel-level detail throughout.

### Token-level speckle probing

Per-patch prediction from individual spatial token embeddings to local 16×16 speckle energy:

| Model | Token Speckle R² |
|-------|-----------------|
| MAE e99 | **0.941** |
| JEPA IN21K e100 | 0.926 |
| BYOL e100 | 0.891 |

Same ordering as mean-pooled.

---

## Caveats with the Speckle Metric Itself

(Identified during 2026-04-07 web session analysis)

1. **High-frequency power conflates speckle with anatomy.** The metric is "mean FFT magnitude above Nyquist/2", which captures both stochastic noise (speckle) AND deterministic fine-scale structure (valve leaflets, trabeculations, papillary muscles). A model that retains anatomical detail will look like it "retains speckle".

2. **Cleaner test: temporal consistency.** Frame-to-frame cosine similarity of embeddings should detect whether models filter frame-specific noise (which is what EMA averaging is supposed to do). See `representation-analysis.md` §5.

| Model | Temporal Cosine Sim |
|-------|---------------------|
| BYOL e100 | **0.976** |
| JEPA IN21K e100 | 0.954 |
| MAE e99 | 0.950 |

JEPA and MAE are **essentially identical** on temporal consistency. The "JEPA filters frame-varying noise via EMA" hypothesis is **not supported** by this metric either.

3. **Cleanest test: noise autocorrelation sweep** (`experiments/noise-autocorrelation-sweep.md`). Result: static spatial noise is the *worst* perturbation for all models, opposite of the EMA-filtering prediction.

---

## What the Mechanism Actually Is (Revised)

Three hypotheses were tested for "why does JEPA outperform MAE on functional tasks". Status:

| Hypothesis | Evidence | Verdict |
|-----------|---------|---------|
| "JEPA filters frame-varying noise via EMA target averaging" | speckle probing, temporal consistency, autocorrelation sweep | **❌ Not supported.** Multiple independent tests fail to support this. |
| "JEPA encodes temporal dynamics that MAE doesn't" | frame shuffling (3 regimes), severity gradient | **✅ Supported.** JEPA consolidates temporal encoding (−17% post-shuffle at e100); MAE abandons it (−4% — invariant because there's nothing to disrupt). |
| "JEPA uses representational capacity more efficiently" | effective dimensionality (spectral entropy) | **❌ Not supported (revised 2026-04-07).** Consistent 4-model RankMe (`scripts/rebuttal/rankme.py`): JEPA 245, BYOL 221, MAE 206, SALT 203. All in the 200-245 range — no 3× collapse. Prior MAE=63 not reproducible. |

**Revised mechanistic story for §4 of the NeurIPS paper:**

The prediction target determines **what temporal structure is encoded** (frame shuffling: JEPA consolidates, MAE abandons, BYOL stabilizes). Effective dimensionality is similar across all models (200-245), so the difference is in feature *content*, not capacity. JEPA's advantage is about learning temporally-structured features that transfer well to functional tasks, not about noise filtering or representational diversity.

**Speckle probing should be reported as a secondary result** (§4 appendix), not as a primary mechanism. The main mechanistic evidence is frame shuffling. Effective dimensionality is no longer a supporting mechanism.

---

## ICML Rebuttal Reference (Historical, Confounded)

For historical reference only. **Do not cite these numbers in current documents.**

**Setup:** EchoNet-Dynamic train, 2,554 clips, ridge probes 5-fold CV, partial R² controlling for mean intensity.

| Variable | JEPA pt50* | BYOL pt50 | MAE pt50 |
|----------|------------|-----------|----------|
| Speckle energy (raw) | 0.764 | 0.835 | 0.910 |
| Mean intensity | 0.998 | 0.984 | 0.995 |
| Speckle (partial, controlling for intensity) | **0.674** | 0.775 | **0.875** |

\*JEPA "pt50" was actually a 235-epoch checkpoint.

**Claim made at the time:** JEPA encodes 23% less speckle than MAE; ordering JEPA < BYOL < MAE.

**Why the claim broke:** With init-matched models at e100, the JEPA−MAE gap collapsed from 0.201 to 0.037, and BYOL became the lowest speckle encoder. The "ordering" was an artifact of JEPA's extra training, not a property of the JEPA objective.

---

## References

- **Canonical (current):** `claude/neurips/experiments/representation-analysis.md` — full e100 init-matched analysis
- **NeurIPS framing:** `claude/neurips/paper-outline.md` §4 (Mechanism)
- **Historical (do not cite):** `claude/rebuttals/10-rebuttal-experiment-results.md` §6e
- **Scripts:** `scripts/rebuttal/information_probing.py`, `scripts/rebuttal/representation_analysis.py`
- **Data (e100):** `scripts/rebuttal/samples/representation_analysis_*.npz`
