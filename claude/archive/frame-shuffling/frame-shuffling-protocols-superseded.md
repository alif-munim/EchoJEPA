# ⚠️ ARCHIVED — See `experiments/frame-shuffling-results.md` for current results

**Archived:** 2026-04-20. This document describes an older protocol. The current
consolidated reference is `claude/neurips/experiments/frame-shuffling-results.md`.

---

# Frame Shuffling Protocols — Definitive Reference

**Last updated:** 2026-04-20
**Purpose:** Clarify the different frame shuffling protocols, what each measures, which numbers appear in the paper, and what was run for extended training.

---

## The Six Conditions

| # | Name | What it does | RoPE positions | Disruption level |
|---|------|-------------|----------------|-----------------|
| 1 | **clean** | Original frame order | Match content | None |
| 2 | **tubelet** | Permute 2-frame tubelets (local reorder) | Unchanged | Minimal |
| 3 | **reverse** | Play backward (T→0) | Unchanged | Minimal |
| 4 | **matched** | Tubelet shuffle, fixed perm across all videos | In evals.main: remapped. In standalone: unchanged | Low |
| 5 | **shuffle** | Random frame permutation (per-video random seed) | Unchanged | High |
| 6 | **matched_frame** | Frame shuffle, fixed perm across all videos | In evals.main: remapped. In standalone: unchanged | Maximal |

**Key distinction:** "matched" and "matched_frame" have two implementations:
- **evals.main** (`src/datasets/video_dataset.py:375-412`): RoPE positional encodings are remapped to match shuffled content. The encoder "knows" where frames originally were but sees them in wrong order. This isolates learned temporal content from positional encoding artifacts. This is what the paper calls "time-aware shuffle."
- **Standalone scripts** (`frame_shuffle_6cond.py`, `frame_shuffle_severity.py`): RoPE is NOT remapped. "matched_frame" is just "fixed-permutation frame shuffle." The difference from "shuffle" is fixed vs random permutation, not RoPE remapping.

---

## Three Protocols Used

### Protocol A: ICML Rebuttal pt50 (2026-03-31)

**File:** `claude/neurips/experiments/frame-shuffling.md`
**Script:** `evals.main` with `FRAME_SHUFFLE` env vars
**Probes:** pt50 probes for all encoder epochs (same probe for all)
**RoPE remapping:** YES (evals.main handles it)
**Prediction averaging:** YES (num_segments=2)
**Single-epoch only:** pt50 for all three models

| Condition | JEPA | BYOL | MAE |
|-----------|------|------|-----|
| clean | **0.549** | 0.460 | 0.396 |
| matched_frame | **0.324** | 0.099 | 0.286 |

This is the source for the **Temporal-Spatial Decomposition** analysis in the paper (§4 discussion).
These are the numbers in `\cref{tab:shuffle}` in the main text.

### Protocol B: NeurIPS 6-condition Training Dynamics (2026-04-06)

**File:** `claude/neurips/experiments/6-condition-shuffling.md`
**Script:** `frame_shuffle_6cond.py` (standalone, single GPU)
**Probes:** Epoch-matched (each encoder epoch gets its own trained probe)
**RoPE remapping:** NO (standalone script limitation)
**Prediction averaging:** NO (single clip per video)
**Epochs:** 4 epochs per model (e24/e50/e74/e99 for MAE)

| Condition | MAE e24 | MAE e50 | MAE e74 | MAE e99 |
|-----------|---------|---------|---------|---------|
| clean | 0.221 | 0.141 | 0.390 | 0.445 |
| matched_frame | 0.189 | -0.343 | 0.345 | 0.449 |

**⚠️ "matched_frame" here is NOT time-aware shuffle.** It's a fixed-permutation frame shuffle without RoPE remapping. The name is misleading. The difference from "shuffle" is fixed vs random permutation, not RoPE compensation.

### Protocol C: NeurIPS Severity Gradient (2026-04-05)

**File:** `claude/neurips/experiments/severity-gradient.md`
**Script:** `frame_shuffle_severity.py` (standalone, single GPU)
**Probes:** Epoch-matched
**RoPE remapping:** NO
**Prediction averaging:** NO
**Fractions:** 0%, 25%, 50%, 75%, 100% × 3 seeds
**Epochs:** 4 epochs per model

| Fraction | MAE e24 | MAE e50 | MAE e74 | MAE e99 |
|----------|---------|---------|---------|---------|
| 0.00 | 0.221 | 0.141 | 0.390 | 0.445 |
| 1.00 | 0.176 | -0.301 | 0.330 | 0.428 |

The paper's training dynamics figure (`\cref{fig:dynamics}`) uses these numbers:
- Temporal Δ = clean R² − fully shuffled R² (100% fraction, 3-seed mean)
- MAE Δ: 0.045 (e24) → 0.443 (e50) → 0.060 (e74) → 0.016 (e99)

**Reproduced on HyperPod 2026-04-20 (job 215, e24 only before cancellation):**

| Fraction | Original | Reproduced |
|----------|----------|------------|
| 0.00 | 0.221 | 0.221 |
| 0.25 | 0.214 | 0.214 |
| 0.50 | 0.205 | 0.206 |
| 0.75 | 0.182 | 0.183 |
| 1.00 | 0.176 | 0.176 |

Exact match confirms checkpoint/probe integrity on HyperPod.

---

## Extended Training Results (2026-04-20)

### Protocol D: evals.main matched_frame with epoch-matched probes (job 216)

**Script:** `evals.main` with `FRAME_SHUFFLE_TYPE=matched_frame`
**Probes:** Epoch-matched (e25-e99 from ICML runs, e124-e194 from NeurIPS probe training)
**RoPE remapping:** YES (evals.main)
**Prediction averaging:** YES (num_segments=2)
**Epochs:** 8 MAE checkpoints (e25, e50, e75, e99, e124, e149, e174, e194)

| MAE Epoch | Clean R² | Matched_frame R² | Temporal Δ | Rel. Drop |
|-----------|----------|-------------------|------------|-----------|
| e25 | 0.225 | 0.257 | +0.033 | +15% |
| **e50** | **0.413** | **0.281** | **-0.132** | **-32%** |
| e75 | 0.435 | 0.356 | -0.080 | -18% |
| e99 | 0.467 | 0.440 | -0.027 | -6% |
| e124 | 0.469 | 0.428 | -0.041 | -9% |
| e149 | 0.527 | 0.491 | -0.035 | -7% |
| e174 | 0.500 | 0.448 | -0.052 | -10% |
| **e194** | **0.526** | **0.460** | **-0.065** | **-12%** |

### Protocol E: evals.main basic shuffle with epoch-matched probes (job 214)

**Script:** `evals.main` with `FRAME_SHUFFLE_TYPE=frame`
**Same as Protocol D but without RoPE remapping.**

| MAE Epoch | Clean R² | Shuffle R² | Temporal Δ |
|-----------|----------|------------|------------|
| e25 | 0.225 | 0.256 | +0.032 |
| e50 | 0.413 | 0.317 | -0.096 |
| e75 | 0.435 | 0.353 | -0.083 |
| e99 | 0.467 | 0.439 | -0.028 |
| e124 | 0.469 | 0.450 | -0.020 |
| e149 | 0.527 | 0.494 | -0.033 |
| e174 | 0.500 | 0.476 | -0.024 |
| e194 | 0.526 | 0.507 | -0.019 |

---

## Why the Numbers Differ Across Protocols

### Clean R² differences (e.g., MAE e50: 0.141 vs 0.413)

| Factor | Protocol B/C (standalone) | Protocol D/E (evals.main) |
|--------|--------------------------|---------------------------|
| Prediction averaging | NO (single center clip) | YES (num_segments=2, 2 temporal clips averaged) |
| Data augmentation | None | RandomResizedCrop, normalization |
| Frame sampling | Center clip, deterministic | Strategy E multi-segment |

Prediction averaging is the dominant factor. A single center clip is much noisier than averaging 2 clips. At e50 the encoder is still fragile — single-clip R²=0.141 reflects high variance, while 2-clip averaging stabilizes to 0.413.

### Matched_frame R² differences (e.g., MAE e99: 0.449 vs 0.440)

| Factor | Protocol B (standalone) | Protocol D (evals.main) |
|--------|------------------------|--------------------------|
| RoPE remapping | NO | YES |
| Permutation type | Fixed (same for all videos) | Fixed with RoPE remap |

With RoPE remapping, the encoder can partially compensate using positional information, which slightly changes the measured R². The direction of the effect varies by checkpoint and isn't always the same sign.

---

## What Appears in the Paper

| Paper location | Source protocol | Numbers |
|---------------|----------------|---------|
| `\cref{tab:shuffle}` (main text, 3 conditions) | **Protocol A** (ICML pt50, evals.main with RoPE remap) | MAE clean 0.396, matched_frame 0.286 |
| `\cref{fig:dynamics}` (training dynamics) | **Protocol C** (severity gradient, standalone, no pred avg) | MAE temporal Δ: 0.443 (e50) → 0.016 (e99) |
| `\cref{app:shuffle_full}` (appendix, 6 conditions) | **Protocol A** (ICML pt50, evals.main) | MAE shuffle 0.422, matched_frame 0.449 |
| `\cref{app:recon_trajectory}` (appendix, recon vs temporal) | **Protocol C** (severity gradient) | Same as fig:dynamics |

**⚠️ Inconsistency in current paper:** The main text table uses Protocol A (pt50 probes, evals.main), while the training dynamics figure uses Protocol C (epoch-matched probes, standalone). These are different protocols with different baselines. The paper text notes this implicitly ("The time-aware shuffle in \cref{tab:shuffle} additionally remaps positional encodings...") but should be more explicit.

---

## Extended Training: What to Use for the Paper

**For the training dynamics figure (extending from e99 to e194):**

Option 1: Run Protocol C (severity gradient standalone script) for e124-e194. Exact match with existing e24-e99 numbers. Requires running `frame_shuffle_severity.py` on HyperPod with the extended checkpoints. **This is the cleanest option — same protocol, extended.**

Option 2: Use Protocol D results (evals.main matched_frame). Different baseline from the existing figure (prediction averaging inflates clean R²). Would need to re-run e24-e99 with the same protocol to have a consistent trajectory. **Internally consistent but requires re-running everything.**

**Recommendation:** Option 1 — extend the severity gradient for e124-e194. The existing e24-e99 numbers are validated and the script is ready with the extended checkpoints registered.

---

## Lessons Learned (2026-04-19/20 HyperPod Runs)

1. **evals.main uses mp.Process** — must use `--ntasks-per-node=1` (no srun) on HyperPod
2. **Inference configs need 6 multihead_kwargs** matching probe checkpoint heads (1 entry → only head 0 → wrong R²)
3. **`echomae_l_mimic_ep99.pth` ≠ `videomae_l_mimic_ep99.pth`** — different files from different dates, encoder/probe mismatch gives garbage R²
4. **Prediction CSV only contains rank 0's subset** (N=160 not 1277) — R² in stdout is correct (all_reduce), but predictions CSV is incomplete
5. **The standalone severity gradient script reproduces exactly** — e24 numbers match to 3 decimal places on HyperPod
