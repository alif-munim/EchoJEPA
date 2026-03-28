# Claim Validity Assessment

Which preprint conclusions survive the probe architecture and fairness analysis? Organized by confidence level.

---

## Bulletproof (no confounds)

### 1. JEPA > MAE objective (controlled comparison)

EchoJEPA-L vs EchoMAE-L: same ViT-L backbone, same probe, same data, same compute.

| Task | EchoJEPA-L | EchoMAE-L | Gap |
|------|-----------|-----------|-----|
| View classification | 85.5% | 40.4% | +45.1 |
| LVEF MAE | 5.97 | 8.15 | -2.18 |

No probe mismatch (identical architectures, identical token counts). This is the paper's anchor claim and it is solid.

### 2. Sample efficiency

EchoJEPA-G at 1% labels (78.63%) exceeds best baseline's linear probe ceiling at 100% labels (57.7% EchoPrime).

| Labels | EchoJEPA-L | EchoMAE-L | Gap |
|--------|-----------|-----------|-----|
| 1% | 57.55% | 21.86% | +35.7 |
| 10% | 80.06% | 34.47% | +45.6 |
| 100% | 85.5% | 40.4% | +45.1 |

Probe disadvantage is constant across label fractions, so relative sample efficiency is unaffected.

### 3. Pediatric zero-shot transfer

Tests encoder representations directly. Probe architecture is the same for all models. Claim is about generalization to an unseen patient population.

---

## Valid but Overstated

### 4. EchoJEPA-G vs PanEcho/EchoPrime on view classification

The attentive probe gap (~45 points) is real but inflated ~2x by token starvation.

| Model | Attentive (test) | Linear (test) |
|-------|-----------------|---------------|
| EchoJEPA-G | 87.4% | 80.9% |
| EchoPrime | 42.1% | 57.7% |
| PanEcho | 41.9% | 52.8% |

Gap narrows from ~45pts to ~23-28pts with linear probes. Still a large advantage but the magnitude is exaggerated in the attentive probe numbers.

### 5. LVEF and RVSP vs baselines

Valid. The attentive probe works correctly for all models on regression — no inversion observed. EchoPrime achieves clinically useful 4.87 LVEF MAE, confirming the probe functions for CNN architectures on these tasks.

---

## Confounded (interpret with caution)

### 6. EchoJEPA-G vs all baselines as evidence for JEPA superiority

This comparison conflates:
- **Objective:** JEPA vs supervised/MAE (the thing we want to measure)
- **Scale:** 1B params vs 30-35M params (~30x)
- **Data:** 18M UHN echos vs smaller published datasets
- **Embedding dim:** 1408 vs 512-768 (more linear probe capacity)

This is a system-level comparison ("our best vs their best"). Standard in clinical papers but should not be used to claim JEPA > supervised learning as a training objective.

### 7. The "clustering pattern" (all non-JEPA baselines at ~40-42%)

The preprint observed that all non-JEPA baselines cluster at similar attentive probe accuracy. This is partially a token starvation artifact.

Linear probes spread them:

| Model | Attentive (test) | Linear (test) |
|-------|-----------------|---------------|
| EchoMAE-L | 40.4% | 59.2% |
| EchoPrime | 42.1% | 57.7% |
| PanEcho | 41.9% | 52.8% |

The apparent uniformity at ~40-42% is coincidence amplified by probe mismatch.

---

## Implications

**Hills to die on:** Claims 1-3 (controlled comparison, sample efficiency, pediatric transfer). These are airtight.

**Claims to qualify:** Claims 4-5 (system-level view classification, regression tasks). Report both probe types, acknowledge the gap magnitude difference.

**Claims to reframe:** Claims 6-7 (system-level JEPA superiority, clustering pattern). Frame as "our best vs their best" and use linear probe numbers.

---

## Related Documents

- `01-paper-audit.md` — issue inventory with response templates
- `05-probe-fairness.md` — probe mismatch root cause and defense
- `../preprint/claim-validity.md` — extended version with Nature Medicine implications
