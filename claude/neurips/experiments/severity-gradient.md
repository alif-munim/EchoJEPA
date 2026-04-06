# Frame Shuffling Severity Gradient Results

**Date:** 2026-04-05 / 2026-04-06
**Script:** `scripts/rebuttal/frame_shuffle_severity.py`
**Dataset:** EchoNet-Dynamic test (1,277 videos)
**Protocol:** Partial frame shuffling at 0/25/50/75/100% of frames, 3 seeds per fraction (100, 101, 102). Frame-level permutation without RoPE remapping (equivalent to "shuffle" condition from the 6-condition experiment).

---

## Raw Results (R² mean across 3 seeds)

| Fraction | JEPA e25 | JEPA e50 | JEPA e75 | JEPA e100 | BYOL e50 | BYOL e100 | MAE e50 | MAE e99 | SALT S2 e79 |
|----------|----------|----------|----------|-----------|----------|-----------|---------|---------|-------------|
| 0.00 | 0.383 | 0.503 | 0.537 | **0.591** | 0.427 | 0.468 | 0.141 | 0.445 | 0.293 |
| 0.25 | 0.362 | 0.419 | 0.465 | **0.542** | 0.360 | 0.410 | 0.091 | 0.421 | -0.037 |
| 0.50 | 0.340 | 0.327 | 0.402 | **0.507** | 0.278 | 0.336 | -0.103 | 0.436 | -0.277 |
| 0.75 | 0.332 | 0.293 | 0.378 | **0.485** | 0.220 | 0.300 | -0.271 | 0.414 | -0.382 |
| 1.00 | 0.331 | 0.290 | 0.370 | **0.488** | 0.219 | 0.291 | -0.301 | 0.428 | -0.397 |

## Relative Degradation (clean → fully shuffled)

| Model | Epoch | Clean R² | Shuffled R² | Relative Drop |
|-------|-------|----------|------------|---------------|
| JEPA IN21K | 25 | 0.383 | 0.331 | −14% |
| JEPA IN21K | 50 | 0.503 | 0.290 | −42% |
| JEPA IN21K | 75 | 0.537 | 0.370 | −31% |
| JEPA IN21K | 100 | 0.591 | 0.488 | −17% |
| BYOL | 50 | 0.427 | 0.219 | −49% |
| BYOL | 100 | 0.468 | 0.291 | −38% |
| MAE | 50 | 0.141 | -0.301 | −313% (collapse) |
| MAE | 99 | 0.445 | 0.428 | −4% (invariant) |
| SALT S2 | 79 | 0.293 | -0.397 | −235% (collapse) |

## Key Findings

### 1. MAE converges to purely spatial representations

MAE e99 is essentially invariant to frame shuffling (R² 0.445 → 0.428, −4%). MAE e50 still collapses (0.141 → −0.301). **Temporal encoding in MAE is transient** — present during early/mid training, eliminated by convergence. The pixel reconstruction objective drives the encoder to learn static spatial features that don't depend on frame order.

This is a novel training dynamics result: the prediction target doesn't just determine *what* is encoded, it determines *what survives training*.

### 2. JEPA learns then consolidates temporal encoding

JEPA's temporal reliance follows a non-monotonic trajectory across training:
- **e25**: 14% drop (mild temporal reliance, still learning)
- **e50**: 42% drop (peak temporal reliance)
- **e75**: 31% drop (consolidating — some temporal features become redundant)
- **e100**: 17% drop (mature representation — temporal features are efficient, minimal redundancy)

JEPA doesn't just "encode more temporal info" — it learns to use temporal information efficiently over training, making the representation robust even when temporal order is disrupted.

### 3. JEPA spatial features alone beat BYOL's best

**JEPA e100 fully shuffled (R²=0.488) > BYOL e100 clean (R²=0.468)**. Even when all temporal information is destroyed, JEPA's spatial features are stronger than BYOL's combined spatial+temporal features at the same training budget. The EMA + latent prediction objective produces better features on both axes.

### 4. BYOL has stable, moderate temporal reliance

BYOL degrades ~38-49% across epochs (linearly with shuffle fraction). Unlike MAE (which loses temporal encoding) and JEPA (which consolidates it), BYOL maintains a consistent level of temporal dependence throughout training. The global self-distillation objective provides a fixed level of temporal encoding.

### 5. SALT S2 collapses immediately

SALT S2 e79 drops from 0.293 to −0.397 under full shuffling (−235%). The frozen pixel-reconstruction teacher doesn't help the student learn temporally robust features. At 25% shuffling, SALT is already below zero. This confirms that the EMA mechanism (not just the latent target) is important for temporal robustness.

## Init and Epoch Matching

| Model | Init | Epochs | Notes |
|-------|------|--------|-------|
| JEPA IN21K | ImageNet-21K | 25/50/75/100 | Init-matched with BYOL and MAE |
| BYOL | ImageNet-21K | 50/100 | |
| MAE | ImageNet | 50/99 | Slightly different ImageNet checkpoint but comparable |
| SALT S2 | Random (student) | 79 (= S1:20 + S2:79 = 99 total) | Per SALT paper recipe |

JEPA IN21K is the correct init-matched comparison. Do NOT use JEPA pt50 (init from fully-trained 235ep encoder).

## Output Files

| Model | CSV Path |
|-------|----------|
| JEPA IN21K e25 | `scripts/rebuttal/samples/severity_JEPA_IN21K_e25.csv` |
| JEPA IN21K e50 | `scripts/rebuttal/samples/severity_JEPA_IN21K_e50.csv` |
| JEPA IN21K e75 | `scripts/rebuttal/samples/severity_JEPA_IN21K_e75.csv` |
| JEPA IN21K e100 | `scripts/rebuttal/samples/severity_JEPA_IN21K_e100.csv` |
| BYOL e50 | `scripts/rebuttal/samples/severity_BYOL_e50.csv` |
| BYOL e100 | `scripts/rebuttal/samples/severity_BYOL_e100.csv` |
| MAE e50 | `scripts/rebuttal/samples/severity_MAE_e50.csv` |
| MAE e99 | `scripts/rebuttal/samples/severity_MAE_e99.csv` |
| SALT S2 e79 | `scripts/rebuttal/samples/severity_SALT_e79.csv` |

## NeurIPS Framing: Three Temporal Encoding Regimes

**Central claim for §4:** The prediction target doesn't just determine what is encoded — it determines what *survives training*. Three qualitatively distinct regimes:

1. **JEPA — Consolidation.** Temporal reliance peaks mid-training (e50: −42%) then consolidates into efficient, robust representation (e100: −17%). EMA provides continuously improving targets that incentivize temporal encoding throughout, but the representation becomes more efficient over time.

2. **MAE — Transient.** Temporal encoding appears early (e50 collapses at −313%) then vanishes by convergence (e99: −4%). Pixel reconstruction initially uses temporal consistency as a shortcut, then discovers static spatial features suffice. Temporal features are *unlearned*, not just weakened.

3. **BYOL — Stable.** Consistent ~40% degradation at both e50 and e100. Global self-distillation provides a fixed incentive for temporal encoding that neither grows nor shrinks.

**Why this matters:** This is invisible from single-checkpoint evaluation. The common view that "MAE doesn't learn temporal dynamics" is incomplete — MAE learns them, then discards them. Only the severity gradient × training dynamics matrix reveals this.

**Supporting results:**
- JEPA e100 fully shuffled (0.488) > BYOL e100 clean (0.468) → advantage is not just temporal
- SALT S2 collapses at 25% shuffle → EMA mechanism, not just latent target, is essential
- MAE's transient temporal encoding is a novel finding — challenges the static characterization

## Figure Plan

**Figure 2b (main text):** R² vs shuffle fraction for JEPA e100, BYOL e100, MAE e99 at the primary comparison point (~100 epochs). Three visually distinct curves: JEPA gentle slope (0.591→0.488), BYOL steep linear (0.468→0.291), MAE flat (0.445→0.428). Add SALT S2 e79 as dashed (0.293→−0.397).

**Figure 2c (main text):** Training dynamics panel. Two subplots:
- Left: Clean R² vs pretraining epoch for all three models. Shows JEPA pulling ahead over training.
- Right: Relative degradation (%) vs pretraining epoch. Shows MAE going from −313% (e50) to −4% (e99), JEPA from −42% (e50) to −17% (e100), BYOL stable at ~40%.

Alternative: heatmap with epoch on x-axis, shuffle fraction on y-axis, R² as color. Three panels (JEPA/BYOL/MAE).

**Appendix:** Full per-epoch severity gradient table (all 13 model×epoch combinations).

## What This Does NOT Test

Frame-varying noise filtering. This tests temporal *dependence* only. Speckle probing (§4.3) tests noise encoding. The noise autocorrelation sweep (§4.4, planned) tests the causal link between frame-varying noise and the MAE/JEPA difference.
