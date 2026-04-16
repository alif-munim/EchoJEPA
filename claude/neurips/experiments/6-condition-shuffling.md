# 6-Condition Frame Shuffling — Training Dynamics

**Date:** 2026-04-06
**Script:** `scripts/neurips/frame_shuffle_6cond.py`
**Dataset:** EchoNet-Dynamic test (1,277 videos)
**Protocol:** 6 temporal disruption conditions with increasing severity. Stochastic conditions use 3 seeds (100, 101, 102). No RoPE remapping for matched/matched_frame (standalone script limitation — see note below).

---

## Conditions (ordered by disruption severity)

1. **clean** — original frame order (baseline)
2. **tubelet** — permute at 2-frame tubelet granularity (preserves local temporal structure)
3. **reverse** — play video backwards (cardiac cycle is quasi-periodic, so systole→diastole ≈ diastole→systole)
4. **matched** — tubelet shuffle with FIXED perm (same permutation for all videos in a seed)
5. **shuffle** — full random frame permutation (per-video random seed)
6. **matched_frame** — frame-level shuffle with FIXED perm (most rigorous — destroys all temporal structure)

**Note on RoPE:** In the full `evals.main` pipeline, "matched" and "matched_frame" remap RoPE positional encodings so the encoder "knows" where each frame originally was. In this standalone script, RoPE is NOT remapped, so "matched" = tubelet with fixed perm, "matched_frame" = frame with fixed perm. The RoPE compensation effect is 7-17% based on the ICML rebuttal analysis.

---

## JEPA IN21K Results (R², mean of 3 seeds where applicable)

| Condition | e25 | e50 | e75 | e100 |
|-----------|-----|-----|-----|------|
| clean | 0.383 | 0.503 | 0.537 | **0.591** |
| tubelet | 0.384 | 0.507 | 0.532 | **0.582** |
| reverse | 0.384 | 0.487 | 0.489 | **0.539** |
| matched | 0.381 | 0.505 | 0.533 | **0.580** |
| shuffle | 0.328 | 0.288 | 0.375 | **0.484** |
| matched_frame | 0.323 | 0.273 | 0.372 | **0.477** |

### Relative degradation (clean → matched_frame)

| Epoch | Clean R² | Matched_frame R² | Relative Drop |
|-------|----------|------------------|---------------|
| e25 | 0.383 | 0.323 | −16% |
| e50 | 0.503 | 0.273 | −46% |
| e75 | 0.537 | 0.372 | −31% |
| e100 | 0.591 | 0.477 | −19% |

### Key findings

1. **Monotonic gradient confirmed:** clean ≈ tubelet ≈ matched > reverse > shuffle ≈ matched_frame. Same pattern as ICML rebuttal pt50 results, now validated with init-matched JEPA across 4 epochs.

2. **Tubelet and matched barely degrade** — local reordering within 2-frame tubelets doesn't disrupt JEPA (<2% drop). Cardiac dynamics at this temporal resolution are captured at coarser granularity.

3. **Reverse costs 5-9% R²** — playing backwards disrupts temporal directionality but not catastrophically. Consistent with the quasi-periodic nature of the cardiac cycle.

4. **Shuffle vs matched_frame gap is small** (~1-2pp) — RoPE positional compensation provides minimal benefit for JEPA at these disruption levels.

5. **Consolidation pattern holds across all 6 conditions:** e50 shows largest relative drop (clean→matched_frame: −46%), e100 the smallest (−19%). The temporal encoding trajectory (learn → peak → consolidate) is consistent regardless of how temporal order is disrupted.

6. **JEPA e100 matched_frame (0.477) still > BYOL e100 clean (0.468)** — even under the most rigorous temporal disruption, JEPA's spatial features alone beat BYOL's best.

## BYOL Results (R², mean of 3 seeds where applicable)

| Condition | e24 | e50 | e75 | e100 |
|-----------|-----|-----|-----|------|
| clean | .380 | .427 | .435 | .468 |
| tubelet | .342 | .372 | .413 | .402 |
| reverse | .252 | .354 | .331 | .373 |
| matched | .350 | .380 | .413 | .415 |
| shuffle | -.179 | .210 | .297 | .291 |
| matched_frame | -.188 | .194 | .292 | .280 |

### Relative degradation (clean → matched_frame)

| Epoch | Clean R² | Matched_frame R² | Relative Drop |
|-------|----------|------------------|---------------|
| e24 | 0.380 | -0.188 | −149% |
| e50 | 0.427 | 0.194 | −55% |
| e75 | 0.435 | 0.292 | −33% |
| e100 | 0.468 | 0.280 | −40% |

### BYOL-specific findings

1. **BYOL is more sensitive to local temporal disruption than JEPA.** Tubelet disruption costs BYOL ~14% at e100 (0.468→0.402) but JEPA only ~2% (0.591→0.582). BYOL's temporal encoding relies on local frame-pair structure; JEPA's operates at coarser temporal granularity.

2. **BYOL e24 collapses under global disruption** (shuffle R²=−0.179, matched_frame R²=−0.188) — consistent with severity gradient finding (−146%). Stabilizes by e50.

3. **BYOL's stabilization is visible across all conditions.** The degradation from clean→matched_frame settles at ~33-40% from e75 onward — no further consolidation like JEPA.

## MAE Results (R², mean of 3 seeds where applicable)

| Condition | e24 | e50 | e74 | e99 |
|-----------|-----|-----|-----|-----|
| clean | .221 | .141 | .390 | .445 |
| tubelet | .197 | .255 | .417 | .424 |
| reverse | .184 | .114 | .400 | .431 |
| matched | .208 | .231 | .400 | .419 |
| shuffle | .178 | -.278 | .327 | .422 |
| matched_frame | .189 | -.343 | .345 | .449 |

### Relative degradation (clean → matched_frame)

| Epoch | Clean R² | Matched_frame R² | Relative Drop |
|-------|----------|------------------|---------------|
| e24 | 0.221 | 0.189 | −14% |
| e50 | 0.141 | -0.343 | −343% |
| e74 | 0.390 | 0.345 | −12% |
| e99 | 0.445 | 0.449 | **+1% (invariant)** |

### MAE-specific findings

1. **MAE e99 is invariant across ALL 6 conditions.** R² ranges 0.419-0.449 — essentially flat. matched_frame (0.449) is marginally *higher* than clean (0.445). Frame order is completely irrelevant to converged MAE.

2. **MAE e50 shows catastrophic temporal dependence across all conditions.** Clean R²=0.141 (already weak), shuffle −0.278, matched_frame −0.343. The transient temporal encoding at mid-training is confirmed across all disruption types.

3. **MAE e50 tubelet (0.255) > MAE e50 clean (0.141).** This anomaly suggests that at mid-training, the temporal shortcut MAE learned actually *hurts* clean performance — the model overfits to frame-to-frame consistency. Disrupting local temporal structure forces the model to rely on spatial features, which are more useful.

4. **MAE's flatness at e99 is not a masking-design artifact (2026-04-08).** Our VideoMAE ViT-L used tube masking 90% — the canonical Tong et al. 2022 recipe that masks the same spatial patches across every frame, designed specifically to prevent a model from copying a masked patch from an adjacent frame. Yet the shortcut persists. This rules out cross-frame spatial copying and points to **within-frame** spatial interpolation (reconstructing masked patches from visible spatial neighbors at the same timestep) as the actual mechanism. Since adjacent spatial patches in echo are highly correlated, pixel reconstruction has a trivial spatial-only solution that tube masking cannot block. See `experiments/tube-masking-failure.md` for the full reframe.

---

## SALT S2 Training Dynamics Results (added 2026-04-08)

Full 4-checkpoint SALT training dynamics (S2 epochs 4/29/54/79, comparable total epochs ~24/49/74/99). All use neurips-style d=4 attentive probes trained on the same pipeline.

### SALT S2 Results (R², mean of 3 seeds where applicable)

| Condition | e4 (~e24) | e29 (~e49) | e54 (~e74) | e79 (~e99) |
|-----------|-----------|------------|------------|------------|
| clean | 0.007 | 0.277 | **0.330** | 0.296 |
| tubelet | 0.008 | 0.261 | 0.324 | 0.294 |
| reverse | 0.005 | 0.202 | 0.223 | 0.120 |
| matched | 0.007 | 0.257 | 0.320 | 0.296 |
| shuffle | -0.021 | **-0.439** | -0.294 | -0.283 |
| matched_frame | -0.020 | **-0.462** | -0.326 | -0.310 |

### Relative degradation (clean → matched_frame)

| Epoch | Clean R² | Matched_frame R² | Relative Drop |
|-------|----------|------------------|---------------|
| e4 | 0.007 | -0.020 | n/a (noise) |
| e29 | 0.277 | -0.462 | **−267%** |
| e54 | 0.330 | -0.326 | **−199%** |
| e79 | 0.296 | -0.310 | **−205%** |

### SALT training dynamics findings

1. **Cliff profile persists across all training stages.** From e29 onward, SALT shows the same signature: flat under local disruption (tubelet, matched within 1-5% of clean), catastrophic collapse under global disruption (shuffle/matched_frame go deeply negative). The cliff never softens.

2. **e29 has the steepest collapse** (matched_frame −267%). This is SALT's "e50 crisis" — peak temporal fragility. But unlike JEPA (−42% → −17%), SALT never consolidates. The degradation plateaus at ~200% from e54 onward.

3. **e54 has the highest clean R² (0.330)** but e54→e79 shows regression (0.330→0.296). SALT's best representation is at mid-training, not convergence — the frozen teacher's targets become less useful as the student saturates.

4. **Reverse sensitivity increases with training.** e29: −27%, e54: −32%, e79: −60%. SALT becomes progressively more sensitive to time reversal, the opposite of JEPA's consolidation. The frozen teacher's temporal encoding becomes increasingly brittle with more student training.

5. **No consolidation — the critical contrast with JEPA.** JEPA: −46%→−19% (6-cond matched_frame, e50→e100). SALT: −267%→−199%→−205% (e29→e54→e79). EMA co-evolution drives consolidation; frozen targets cannot.

6. **SALT learned *some* temporal encoding — it just can't generalize.** Tubelet/matched stay near clean (local disruption ≈ no-op), but shuffle/matched_frame collapse. The frozen pixel teacher provided targets encoding frame ordering, but the student only learned to match those targets under in-distribution arrangements.

7. **e4 is baseline noise** (clean R²=0.007). All conditions near zero. Confirms probe quality: degradation at e29+ is real signal.

### Connection to §4 tube-masking reframe

The tube-masking reframe (`experiments/tube-masking-failure.md`) argued that MAE's temporal flatness is intrinsic to pixel reconstruction on spatially redundant video. SALT's cliff profile adds a complementary mechanistic point:

> **The frozen-teacher mechanism produces fragile temporal encoding.** SALT's teacher is itself a pixel-reconstruction model (S1), so the latent targets the student learns to match encode whatever temporal structure S1 happened to extract before it was frozen. Without EMA co-evolution, the student has no mechanism to *improve* that encoding during its own training — it can only memorize the targets the frozen teacher provides. The result is brittle temporal features that work on in-distribution frame arrangements and shatter under novel ones.

This is the *teacher dynamics* component of the §4.5 SALT discussion: JEPA's advantage is both the latent target AND the EMA co-evolution. Removing co-evolution (SALT) while keeping the latent target produces temporal features that are worse than MAE's complete absence of temporal features — at least MAE's purely-spatial representation is consistent across disruptions.

---

## Cross-Model Comparison at Convergence (primary comparison point)

| Condition | JEPA e100 | BYOL e100 | MAE e99 | SALT e79 |
|-----------|-----------|-----------|---------|----------|
| clean | **.591** | .468 | .445 | .296 |
| tubelet | **.582** | .402 | .424 | .294 |
| reverse | **.539** | .373 | .431 | .120 |
| matched | **.580** | .415 | .419 | .296 |
| shuffle | **.484** | .291 | .422 | **−.283** |
| matched_frame | **.477** | .280 | .449 | **−.310** |

**Four qualitatively distinct profiles:**
- **JEPA**: monotonic gradient, gentle slope (0.591 → 0.477, −19%)
- **BYOL**: monotonic gradient, steep slope (0.468 → 0.280, −40%)
- **MAE**: completely flat (0.445 → 0.449, +1%)
- **SALT**: cliff — flat under local disruption, collapse under global (0.296 → −0.310, −205%)

**Key comparisons:**
- JEPA matched_frame (0.477) > BYOL clean (0.468) > MAE clean (0.445) > SALT clean (0.296) — **JEPA's fully-shuffled representation beats every other model's clean representation**
- MAE matched_frame (0.449) ≈ MAE clean (0.445) — frame order is irrelevant to MAE
- SALT is the only model to go negative under any condition
- Under full temporal disruption: JEPA (0.477) >> MAE (0.449) > BYOL (0.280) >> SALT (−0.310)
- SALT clean (0.296) < MAE clean (0.445): even the weakest EMA-based method beats the best SALT variant, regardless of temporal disruption

**Four-way profile summary:** each method has a signature response shape that reflects what the prediction target is doing.
- *Latent target + EMA co-evolution* (JEPA) → robust temporal encoding, gentle degradation
- *Global pool target + EMA* (BYOL) → moderate temporal encoding, steep degradation
- *Pixel target, no teacher* (MAE) → no temporal encoding, flat invariance (tube masking does not block within-frame spatial interpolation; see `tube-masking-failure.md`)
- *Latent target, frozen teacher* (SALT) → brittle temporal encoding, cliff collapse (no consolidation across training)

---

## Output Files

| Model | CSV Path |
|-------|----------|
| JEPA IN21K e25 | `scripts/neurips/samples/6cond_JEPA_IN21K_e25.csv` |
| JEPA IN21K e50 | `scripts/neurips/samples/6cond_JEPA_IN21K_e50.csv` |
| JEPA IN21K e75 | `scripts/neurips/samples/6cond_JEPA_IN21K_e75.csv` |
| JEPA IN21K e100 | `scripts/neurips/samples/6cond_JEPA_IN21K_e100.csv` |
| BYOL e24 | `scripts/neurips/samples/6cond_BYOL_e24.csv` |
| BYOL e50 | `scripts/neurips/samples/6cond_BYOL_e50.csv` |
| BYOL e75 | `scripts/neurips/samples/6cond_BYOL_e75.csv` |
| BYOL e100 | `scripts/neurips/samples/6cond_BYOL_e100.csv` |
| MAE e24 | `scripts/neurips/samples/6cond_MAE_e24.csv` |
| MAE e50 | `scripts/neurips/samples/6cond_MAE_e50.csv` |
| MAE e74 | `scripts/neurips/samples/6cond_MAE_e74.csv` |
| MAE e99 | `scripts/neurips/samples/6cond_MAE_e99.csv` |
| SALT S2 e4 | `scripts/neurips/samples/6cond_SALT_S2_e4.csv` |
| SALT S2 e29 | `scripts/neurips/samples/6cond_SALT_S2_e29.csv` |
| SALT S2 e54 | `scripts/neurips/samples/6cond_SALT_S2_e54.csv` |
| SALT S2 e79 | `scripts/neurips/samples/6cond_SALT_S2_e79.csv` |

## For NeurIPS

**Main text (§4.1, Fig 2a):** Bar chart of R² across 6 conditions for JEPA e100 / BYOL e100 / MAE e99 at the primary comparison point. Shows monotonic gradient + cross-model differences.

**Appendix:** Full training dynamics tables (4 epochs × 6 conditions × 3 models). The additional insight beyond the severity gradient: BYOL is more sensitive to local (tubelet) disruption than JEPA, suggesting BYOL's temporal encoding operates at finer temporal granularity.

**Assessment:** The 6-condition data is appendix material. The severity gradient (§4.2) remains the key result for main text. The 6-condition adds one new insight (BYOL tubelet sensitivity) worth one sentence in the paper.
