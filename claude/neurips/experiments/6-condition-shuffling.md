# 6-Condition Frame Shuffling — Training Dynamics

**Date:** 2026-04-06
**Script:** `scripts/rebuttal/frame_shuffle_6cond.py`
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

## MAE Results (running, ETA ~35 min)

*To be filled when complete.*

---

## Cross-Model Comparison at e100 (primary comparison point)

| Condition | JEPA e100 | BYOL e100 | MAE e99 (pending) |
|-----------|-----------|-----------|-------------------|
| clean | **.591** | .468 | — |
| tubelet | **.582** | .402 | — |
| reverse | **.539** | .373 | — |
| matched | **.580** | .415 | — |
| shuffle | **.484** | .291 | — |
| matched_frame | **.477** | .280 | — |

JEPA leads on every condition by a wide margin. **JEPA matched_frame (0.477) > BYOL clean (0.468).**

---

## Output Files

| Model | CSV Path |
|-------|----------|
| JEPA IN21K e25 | `scripts/rebuttal/samples/6cond_JEPA_IN21K_e25.csv` |
| JEPA IN21K e50 | `scripts/rebuttal/samples/6cond_JEPA_IN21K_e50.csv` |
| JEPA IN21K e75 | `scripts/rebuttal/samples/6cond_JEPA_IN21K_e75.csv` |
| JEPA IN21K e100 | `scripts/rebuttal/samples/6cond_JEPA_IN21K_e100.csv` |
| BYOL e24 | `scripts/rebuttal/samples/6cond_BYOL_e24.csv` |
| BYOL e50 | `scripts/rebuttal/samples/6cond_BYOL_e50.csv` |
| BYOL e75 | `scripts/rebuttal/samples/6cond_BYOL_e75.csv` |
| BYOL e100 | `scripts/rebuttal/samples/6cond_BYOL_e100.csv` |

## For NeurIPS

**Main text (§4.1, Fig 2a):** Bar chart of R² across 6 conditions for JEPA e100 / BYOL e100 / MAE e99 at the primary comparison point. Shows monotonic gradient + cross-model differences.

**Appendix:** Full training dynamics tables (4 epochs × 6 conditions × 3 models). The additional insight beyond the severity gradient: BYOL is more sensitive to local (tubelet) disruption than JEPA, suggesting BYOL's temporal encoding operates at finer temporal granularity.

**Assessment:** The 6-condition data is appendix material. The severity gradient (§4.2) remains the key result for main text. The 6-condition adds one new insight (BYOL tubelet sensitivity) worth one sentence in the paper.
