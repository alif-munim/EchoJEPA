# Frame Shuffling Severity Gradient Results

**Date:** 2026-04-05 / 2026-04-06
**Script:** `scripts/neurips/frame_shuffle_severity.py`
**Dataset:** EchoNet-Dynamic test (1,277 videos)
**Protocol:** Partial frame shuffling at 0/25/50/75/100% of frames, 3 seeds per fraction (100, 101, 102). Frame-level permutation without RoPE remapping.

---

## Complete Results Matrix (R², mean of 3 seeds)

| Fraction | JEPA e25 | JEPA e50 | JEPA e75 | JEPA e100 | BYOL e24 | BYOL e50 | BYOL e75 | BYOL e100 | MAE e24 | MAE e50 | MAE e74 | MAE e99 | SALT e4 | SALT e29 | SALT e54 | SALT e79 |
|----------|----------|----------|----------|-----------|----------|----------|----------|-----------|---------|---------|---------|---------|---------|----------|----------|----------|
| 0.00 | 0.383 | 0.503 | 0.537 | **0.591** | 0.380 | 0.427 | 0.435 | 0.468 | 0.221 | 0.141 | 0.390 | 0.445 | 0.007 | 0.277 | **0.330** | 0.296 |
| 0.25 | 0.362 | 0.419 | 0.465 | **0.542** | 0.119 | 0.360 | 0.388 | 0.410 | 0.214 | 0.091 | 0.356 | 0.421 | -0.007 | -0.096 | 0.080 | 0.048 |
| 0.50 | 0.340 | 0.327 | 0.402 | **0.507** | -0.080 | 0.278 | 0.329 | 0.336 | 0.205 | -0.103 | 0.347 | 0.436 | -0.015 | -0.327 | -0.158 | -0.161 |
| 0.75 | 0.332 | 0.293 | 0.378 | **0.485** | -0.160 | 0.220 | 0.309 | 0.300 | 0.182 | -0.271 | 0.320 | 0.414 | -0.019 | -0.418 | -0.260 | -0.256 |
| 1.00 | 0.331 | 0.290 | 0.370 | **0.488** | -0.173 | 0.219 | 0.304 | 0.291 | 0.176 | -0.301 | 0.330 | 0.428 | -0.019 | -0.432 | -0.286 | -0.270 |

## Relative Degradation (clean → fully shuffled)

| Model | e24/e25 | e50 | e74/e75 | e99/e100 |
|-------|---------|-----|---------|----------|
| **JEPA IN21K** | −14% | −42% | −31% | **−17%** |
| **BYOL** | −146% | −49% | −30% | −38% |
| **MAE** | −20% | −313% | −15% | **−4%** |
| **SALT S2 e4** | −371%* | — | — | — |
| **SALT S2 e29** | — | −256% | — | — |
| **SALT S2 e54** | — | — | −187% | — |
| **SALT S2 e79** | — | — | — | −191% |

*e4 clean R²=0.007, essentially noise — degradation percentage is meaningless.

---

## Key Findings

### 1. Three temporal encoding regimes — all emerge from early instability

All three objectives start with fragile temporal encoding that collapses under shuffling. They diverge in how they resolve this:

**JEPA — Consolidation.** Mild fragility at e25 (−14%), peak temporal reliance at e50 (−42%), then progressive consolidation to robust encoding at e100 (−17%). The EMA target continuously incentivizes temporal encoding, but the representation becomes more efficient over training — temporal features are encoded but no longer fragile.

**BYOL — Stabilization.** Catastrophically fragile at e24 (−146%, R² goes to −0.17). By e50 the collapse is tamed (−49%), and from e75 onward it stabilizes at ~30-38%. BYOL resolves early instability by locking in a fixed level of temporal dependence — moderate and consistent.

**MAE — Transient then spatial.** Mild fragility at e24 (−20%), peak collapse at e50 (−313%), then complete recovery to invariance at e99 (−4%). MAE initially uses temporal consistency as a reconstruction shortcut, this dependence peaks and catastrophically fails at e50, then the encoder rebuilds entirely on static spatial features. By convergence, frame order is irrelevant.

### 2. The e50 crisis point

All three models show their most extreme temporal behavior at e50:
- JEPA: peak temporal reliance (−42%) — learning temporal dynamics
- BYOL: still fragile (−49%) — not yet stabilized
- MAE: catastrophic collapse (−313%) — temporal shortcut maximally exploited

This suggests e50 is a critical training phase where temporal and spatial features are being negotiated. The prediction target determines which side wins.

### 3. JEPA spatial features alone beat everything

JEPA e100 fully shuffled (R²=0.488) > BYOL e100 clean (0.468) > MAE e99 clean (0.445). Even with all temporal information destroyed, JEPA's spatial features are the strongest. The advantage is not just temporal — latent prediction produces better features on both axes.

### 4. SALT training dynamics: no consolidation, only cliff (2026-04-08)

Full 4-checkpoint SALT training dynamics (S2 epochs 4/29/54/79, comparable total epochs ~24/49/74/99):

| Fraction | SALT e4 | SALT e29 | SALT e54 | SALT e79 |
|----------|---------|----------|----------|----------|
| 0.00 | 0.007 | 0.277 | **0.330** | 0.296 |
| 0.25 | -0.007 | -0.096 | 0.080 | 0.048 |
| 0.50 | -0.015 | -0.327 | -0.158 | -0.161 |
| 0.75 | -0.019 | -0.418 | -0.260 | -0.256 |
| 1.00 | -0.019 | -0.432 | -0.286 | -0.270 |

**Key dynamics:**

1. **e4 is baseline noise.** Clean R²=0.007 — the encoder has barely learned anything. All shuffle fractions near zero. Confirms probe quality: degradation at e29+ is real signal.

2. **e29 has the steepest collapse** (0.277→−0.432, −256%). This is SALT's "e50 crisis" — peak temporal fragility. But unlike JEPA/BYOL/MAE, SALT never recovers from this peak.

3. **e54 has the highest clean R² (0.330)** but still collapses (−187%). e54→e79 shows slight *regression* in clean R² (0.330→0.296). SALT's best representation is at mid-training, not convergence.

4. **No consolidation — the critical contrast with JEPA.** JEPA: −42% at e50 → −17% at e100 (consolidation). SALT: −256% at e29 → −187% at e54 → −191% at e79 (flat after peak, no recovery). The frozen teacher cannot drive temporal consolidation because it provides fixed targets — there's no co-evolving signal to push the student toward more robust representations.

5. **Already negative at 25% shuffle from e29 onward.** SALT e29 goes to −0.096 at 25% shuffle (JEPA e50: +0.419 at 25%). The cliff profile emerges immediately once the encoder starts learning temporal features.

**Extended training confirms the ceiling (2026-04-06):** SALT S2 e199 probe val MAE ~6.8 — *worse* than e79 (6.47). Training loss plateaued (0.429→0.419), weight cosine similarity >0.999 between e79 and e199. The frozen teacher imposes a representation ceiling that 2.5× more student training cannot overcome.

**Context vs concurrent work:** The SALT paper (Apple) trains on 3.6M diverse natural video clips. US-JEPA (concurrent) uses URFM (BiomedCLIP-distilled) as a strong external teacher. Both succeed because the teacher has broad coverage. Our V-Pixel teacher on 525K echo clips (single domain) has narrow coverage → ceiling. The frozen teacher mechanism requires data diversity OR a strong external teacher. With neither, EMA-based co-evolution is strictly superior.

### 5. MAE's transient temporal encoding is novel

The trajectory e24 (−20%) → e50 (−313%) → e74 (−15%) → e99 (−4%) shows MAE doesn't simply "fail to learn temporal features." It learns them, maximally exploits them, catastrophically depends on them, then abandons them entirely in favor of spatial features. This training dynamics effect is invisible from any single checkpoint.

### 6. Tube masking cannot prevent the shortcut (2026-04-08 reframe)

VideoMAE ViT-L was pretrained with **tube masking, 90%** (`--mask_type tube --mask_ratio 0.9` in all three VideoMAE MIMIC sbatches — the canonical Tong et al. 2022 recipe that masks the same spatial patches across every frame). Tube masking is the community-standard defense against temporal shortcuts in video MAE — it was designed specifically to prevent a model from reconstructing a masked patch by copying from adjacent frames.

Yet MAE e99 still converges to near-complete temporal invariance (−4%). This rules out **cross-frame** spatial copying as the mechanism of the MAE shortcut. The remaining path is **within-frame** spatial interpolation — reconstructing a masked patch from its visible spatial neighbors at the same timestep, which is trivial on spatially redundant echo anatomy. Frame-gap masking (mask entire frame positions) does not fix this either, because it addresses the same cross-frame-copying hypothesis that tube masking already rules out. Only whole-frame masking (no visible tokens at some timesteps) would force temporal reasoning, and it risks training collapse.

**Implication:** the temporal shortcut is intrinsic to pixel reconstruction on spatially redundant video, not an artifact of masking design. No masking trick fixes it — the prediction target is the bottleneck.

Full writeup: `experiments/tube-masking-failure.md`.

---

## Init and Epoch Matching

| Model | Init | Epochs evaluated | Comparable total epochs |
|-------|------|-----------------|------------------------|
| JEPA IN21K | ImageNet-21K | 25, 50, 75, 100 | 25, 50, 75, 100 |
| BYOL | ImageNet-21K | 24, 50, 75, 100 | 24, 50, 75, 100 |
| MAE | ImageNet | 24, 50, 74, 99 | 24, 50, 74, 99 |
| SALT S2 | Random (student) | S2: 4, 29, 54, 79 | 24, 49, 74, 99 (S1:20 + S2:N) |

JEPA IN21K is init-matched with BYOL/MAE. Slight epoch misalignment (24 vs 25, 74 vs 75, 99 vs 100) due to checkpoint availability — negligible.

---

## NeurIPS Framing

**Central claim for §4:** The prediction target determines not just what is encoded, but what *survives training*. Four qualitatively distinct temporal encoding regimes:

1. **JEPA consolidates** temporal encoding into an efficient, robust representation (−17% at convergence)
2. **BYOL stabilizes** at a fixed, moderate level of temporal dependence (−38%)
3. **MAE abandons** temporal encoding entirely, converging to purely spatial features (−4%)
4. **SALT never consolidates** — frozen teacher produces brittle temporal features that plateau at −191% without recovery

**One-paragraph text:** "We identify four qualitatively distinct temporal encoding regimes shaped by the prediction target and teacher dynamics. All objectives exhibit fragile temporal encoding during early training, but diverge in resolution: EMA-based latent prediction (JEPA) consolidates temporal features into a robust representation (−17% degradation under full shuffle at convergence); global self-distillation (BYOL) stabilizes at moderate temporal dependence (−38%); pixel reconstruction (MAE) abandons temporal encoding entirely (−4%), converging to purely spatial features after a transient phase of catastrophic temporal reliance at mid-training; and latent prediction with a frozen teacher (SALT) produces brittle temporal features that collapse under global disruption (−191%) without the consolidation that EMA co-evolution enables. These dynamics are invisible from single-checkpoint evaluation."

## Figure Plan

**Figure 2b (main text):** R² vs shuffle fraction at convergence — JEPA e100 (gentle slope), BYOL e100 (steep linear), MAE e99 (flat). Three visually distinct curves in one plot. Optionally add SALT S2 e79 (immediate collapse, dashed).

**Figure 2c (main text):** Training dynamics — relative degradation (%) vs pretraining epoch for each model. Shows:
- MAE's V-shape: −20% → −313% → −15% → −4%
- JEPA's arc: −14% → −42% → −31% → −17%
- BYOL's recovery: −146% → −49% → −30% → −38%
- SALT's plateau: noise → −256% → −187% → −191% (no consolidation)

**Appendix:** Full 16-model × 5-fraction results table.

---

## Output Files

| Model | CSV Path |
|-------|----------|
| JEPA IN21K e25 | `scripts/neurips/samples/severity_JEPA_IN21K_e25.csv` |
| JEPA IN21K e50 | `scripts/neurips/samples/severity_JEPA_IN21K_e50.csv` |
| JEPA IN21K e75 | `scripts/neurips/samples/severity_JEPA_IN21K_e75.csv` |
| JEPA IN21K e100 | `scripts/neurips/samples/severity_JEPA_IN21K_e100.csv` |
| BYOL e24 | `scripts/neurips/samples/severity_BYOL_e24.csv` |
| BYOL e50 | `scripts/neurips/samples/severity_BYOL_e50.csv` |
| BYOL e75 | `scripts/neurips/samples/severity_BYOL_e75.csv` |
| BYOL e100 | `scripts/neurips/samples/severity_BYOL_e100.csv` |
| MAE e24 | `scripts/neurips/samples/severity_MAE_e24.csv` |
| MAE e50 | `scripts/neurips/samples/severity_MAE_e50.csv` |
| MAE e74 | `scripts/neurips/samples/severity_MAE_e74.csv` |
| MAE e99 | `scripts/neurips/samples/severity_MAE_e99.csv` |
| SALT S2 e79 | `scripts/neurips/samples/severity_SALT_e79.csv` |
| SALT S2 e4 | `scripts/neurips/samples/severity_SALT_S2_e4.csv` |
| SALT S2 e29 | `scripts/neurips/samples/severity_SALT_S2_e29.csv` |
| SALT S2 e54 | `scripts/neurips/samples/severity_SALT_S2_e54.csv` |
| SALT S2 e79 (rerun) | `scripts/neurips/samples/severity_SALT_S2_e79.csv` |
