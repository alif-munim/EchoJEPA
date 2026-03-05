# ICML Preprint: Claim Validity Assessment

Assessment of which preprint conclusions survive the probe architecture and fairness analysis documented in `encoder-fairness.md` and `probe-architecture-analysis.md`.

## Bulletproof (no confounds)

### JEPA > MAE objective (controlled comparison)
EchoJEPA-L vs EchoMAE-L: same ViT-L backbone, same probe, same data, same compute. The only variable is the training objective.

| Task | EchoJEPA-L | EchoMAE-L | Gap |
|------|-----------|-----------|-----|
| View classification | 85.5% | 40.4% | +45.1 |
| LVEF MAE | 5.97 | 8.15 | -2.18 |

No probe mismatch (identical architectures, identical token counts). This is the paper's anchor claim and it is solid.

### Sample efficiency
EchoJEPA-G at 1% labels (78.63%) exceeds the best baseline's architecture-agnostic linear probe ceiling at 100% labels (57.7% EchoPrime). Even granting the most generous probe correction for baselines, EchoJEPA still wins by 20+ points.

| Labels | EchoJEPA-L | EchoMAE-L | Gap |
|--------|-----------|-----------|-----|
| 1% | 57.55% | 21.86% | +35.7 |
| 10% | 80.06% | 34.47% | +45.6 |
| 100% | 85.5% | 40.4% | +45.1 |

The probe disadvantage is constant across label fractions, so relative sample efficiency within each model is unaffected.

### Pediatric zero-shot transfer
Tests encoder representations directly. Probe architecture is the same for all models and the claim is about generalization to an unseen patient population, not probe design.

## Valid but overstated

### EchoJEPA-G vs PanEcho/EchoPrime on view classification
The 87.4% vs ~42% gap is real but inflated. Attentive probes hurt CNN baselines due to token starvation. Linear probes show:

| Model | Attentive (test) | Linear (test) |
|-------|-----------------|---------------|
| EchoJEPA-G | 87.4% | 80.9% |
| EchoPrime | 42.1% | 57.7% |
| PanEcho | 41.9% | 52.8% |

The gap narrows from ~45 points to ~23-28 points. Still a large advantage for EchoJEPA-G, but the magnitude is exaggerated by roughly 2x in the attentive probe numbers.

### LVEF and RVSP vs baselines
Valid. The attentive probe works correctly for all models on complex temporal tasks — no inversion observed. EchoPrime achieves clinically useful 4.87 LVEF MAE, confirming the probe functions for CNN architectures on regression.

## Confounded (interpret with caution)

### EchoJEPA-G vs all baselines as evidence for JEPA superiority
This comparison conflates multiple variables:
- **Objective**: JEPA vs supervised/MAE (the thing we want to measure)
- **Scale**: 1B params vs 30-35M params (~30x difference)
- **Data**: 18M UHN echos vs smaller published datasets
- **Embedding dim**: 1408 vs 512-768 (more linear probe capacity)

This is a system-level comparison ("our best vs their best"), standard in clinical papers but should not be used to claim JEPA > supervised learning as a training objective.

### The "clustering pattern" (all non-JEPA baselines at ~40-42%)
The preprint observed that all non-JEPA baselines cluster at similar attentive probe accuracy regardless of architecture, training data, or model size. This is partially an artifact of token starvation — all non-ViT models suffer from the same probe mismatch similarly.

Linear probes spread them out:

| Model | Attentive (test) | Linear (test) |
|-------|-----------------|---------------|
| EchoMAE-L | 40.4% | 59.2% |
| EchoPrime | 42.1% | 57.7% |
| PanEcho | 41.9% | 52.8% |

The baselines aren't as uniformly bad as the attentive probe numbers suggest. EchoMAE-L in particular jumps from 40.4% to 59.2% — it was being dragged down not by token starvation (it's also ViT-L) but likely by the reconstruction objective failing to encode semantic categories. However, the apparent uniformity at ~40-42% is a coincidence amplified by the probe mismatch.

## Implications for Nature Medicine

The switch to linear probes on mean-pooled embeddings for Nature Medicine is the correct approach:

1. **Eliminates token starvation confound** — all models reduced to a single vector before the probe
2. **Produces more conservative but defensible gaps** — ~23 points instead of ~45
3. **Fairer across model scales** — though dimensionality confound (1408 vs 512) persists
4. **Shifts the narrative** from "attentive probe dominance" to "linearly accessible clinical information" — a stronger claim about representation quality

Remaining confound for Nature Medicine: embedding dimensionality. Can be addressed by reporting dims in tables and optionally PCA-projecting to common dimensionality.

## Related Documents
- `encoder-fairness.md` — encoder output comparison, dimensionality/scale/data confounds
- `probe-architecture-analysis.md` — attentive vs linear inversion, root cause, rebuttal strategy
- `probe_dim_analysis.md` — full conversation log (raw source)
