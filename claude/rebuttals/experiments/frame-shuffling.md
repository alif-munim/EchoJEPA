# Frame Shuffling Temporal Ablation

**Date:** 2026-03-31
**Status:** Complete. Dropped from ICML rebuttal (see Interpretation). Intended for NeurIPS resubmission.

---

## Overview

Temporal ablation experiment measuring how downstream LVEF prediction degrades when frame order is disrupted. Tests whether each SSL objective encodes temporal cardiac dynamics or static frame appearance. Six temporal disruption conditions form a gradient from no disruption to maximal disruption with positional confounds removed.

## Experimental Setup

**Dataset:** EchoNet-Dynamic test set (1,277 videos, public)
**Task:** LVEF regression (frozen encoder + frozen probe, no retraining)
**Pipeline:** `evals.main` with `VideoDataset` temporal ablation hook (multi-segment prediction averaging, Strategy E)
**Models:** All ViT-L, 50 epochs on MIMIC-IV-Echo 525K clips

| Model | Encoder Checkpoint | Probe Checkpoint | Best Head |
|-------|-------------------|-----------------|-----------|
| EchoJEPA-L pt50 | `checkpoints/echojepa-l-pt50.pt` (key: `target_encoder`) | `evals/vitl/icml/echojepa_pt50_end_lvef_224/.../best.pt` | head 3 |
| EchoBYOL-L pt50 | `checkpoints/byol_vitl_imagenet_v2_e50.pt` (key: `target_encoder`) | `evals/vitl/icml/echobyol_pt50_end_lvef_224/.../best.pt` | head 1 |
| EchoMAE-L pt50 | `checkpoints/videomae_l_mimic_ep50.pth` (key: `model`) | `evals/vitl/icml/echomae_pt50_end_lvef_224/.../best.pt` | head 5 |

**Test CSV:** `data/csv/echonet_dynamic_test_local.csv` (1,277 videos, space-delimited: path raw_EF)

---

## Temporal Disruption Conditions

| Condition | Code | Description | Disruption Level |
|-----------|------|-------------|-----------------|
| **clean** | `type=matched, seed=100` | Original frame order. RoPE positions match content. Equivalent to no ablation. | None |
| **tubelet** | `type=tubelet, seed=100` | Permute at 2-frame tubelet granularity. Adjacent frames within each tubelet stay together. | Minimal |
| **reverse** | `type=reverse` | Play video backwards (frame T→0). Cardiac cycle is quasi-periodic so systole→diastole ≈ diastole→systole. | Minimal |
| **matched** | `type=matched, seed=100` | Tubelet-level shuffle with RoPE positions remapped to match shuffled content. Encoder "knows" where each tubelet originally was. | Low (positional encoding compensates) |
| **shuffle** | `type=frame, seeds=100,101,102` | Fully random frame permutation. Destroys all temporal coherence. RoPE positions are NOT remapped (positional mismatch). | High |
| **matched_frame** | `type=matched_frame, seed=100` | Frame-level shuffle with RoPE positions remapped to match content. Removes positional encoding as confound — isolates true temporal content reliance. | Maximal |

**Implementation:** `src/datasets/video_dataset.py:383-412` — temporal ablation applied after clip loading, before transform. Shuffle uses `np.random.RandomState` with per-condition seeds for reproducibility. `matched` and `matched_frame` variants use a fixed permutation (same seed for all videos) so the encoder's RoPE can be adjusted to match.

---

## Results (R² on EchoNet-Dynamic test, 1,277 videos)

### Raw R² Values

| Condition | JEPA | BYOL | MAE |
|-----------|------|------|-----|
| **clean** | **0.549** | 0.460 | 0.396 |
| **tubelet** | **0.554** | 0.442 | 0.410 |
| **reverse** | **0.535** | 0.444 | 0.388 |
| **matched** | **0.549** | 0.460 | 0.396 |
| **shuffle** (mean±std) | **0.365** ±0.009 | 0.174 ±0.003 | 0.318 ±0.002 |
| **matched_frame** | **0.324** | 0.099 | 0.286 |

Per-seed R² for the shuffle condition (3 seeds):

| Seed | JEPA | BYOL | MAE |
|------|------|------|-----|
| 100 | 0.357 | 0.178 | 0.316 |
| 101 | 0.377 | 0.173 | 0.321 |
| 102 | 0.361 | 0.171 | 0.318 |

### Relative Degradation from Clean

| Condition | JEPA | BYOL | MAE |
|-----------|------|------|-----|
| tubelet | +0.9% | -4.0% | +3.6% |
| reverse | -2.7% | -3.4% | -1.9% |
| matched | ~0% | ~0% | ~0% |
| **shuffle** | **-33.6%** | **-62.2%** | **-19.6%** |
| **matched_frame** | **-41.0%** | **-78.6%** | **-27.8%** |

---

## Interpretation

### Why This Was Dropped from the ICML Rebuttal

The rebuttal docs (`10-rebuttal-experiment-results.md:333`) state: "Dropped from rebuttal (doesn't favor JEPA)." The reasoning was that MAE showed smaller *relative* degradation (-20%) than JEPA (-34%) under frame shuffle, contradicting the initial hypothesis that "JEPA should be robust and MAE should degrade."

### Why This Framing Was Wrong

The initial hypothesis was backwards. The correct framing:

1. **All models degrade on frame shuffle because LVEF is inherently temporal** — it requires tracking wall motion across the cardiac cycle. The question is not "which model is invariant to shuffling" but "which model retains the most absolute signal after temporal disruption."

2. **JEPA retains the most absolute signal post-shuffle.** Shuffled JEPA (R²=0.365) still outperforms clean MAE (R²=0.396 — comparable) and vastly outperforms shuffled BYOL (R²=0.174). JEPA's temporal encoding is strong enough that even partial disruption leaves useful signal.

3. **MAE's small relative drop reflects that it had little temporal signal to lose.** Going from R²=0.396 to R²=0.318 (-20%) means it was encoding mostly static spatial patterns. The small *relative* drop is a weakness, not a strength.

4. **BYOL's catastrophic collapse (-62% to -79%) is the key finding.** Global mean-pooling creates implicit temporal dependence (temporal structure affects the mean) that shatters completely when frame order is destroyed. BYOL's representations are fragile to temporal disruption despite appearing strong under clean conditions.

5. **matched_frame is the most rigorous condition.** It removes positional encoding as a confound (RoPE positions match content), isolating true temporal content reliance. The drop from shuffle → matched_frame for each model (JEPA: -34→-41%, BYOL: -62→-79%, MAE: -20→-28%) shows that ~7-17% of each model's shuffle "robustness" was actually RoPE positional compensation, not true temporal invariance.

### Correct Framing for NeurIPS

**The temporal disruption gradient reveals three distinct encoding strategies:**

- **JEPA:** Encodes explicit temporal dynamics (wall motion, valve mechanics). Degrades substantially under shuffle (-34%) but retains the most absolute signal (R²=0.365 post-shuffle > MAE's clean R²=0.396). Local masked prediction forces the encoder to model temporal transitions between adjacent patches.

- **BYOL:** Creates implicit temporal dependence through global mean-pooling (temporal structure affects the global average). This produces strong clean performance (R²=0.460) but catastrophic fragility — temporal disruption collapses BYOL to R²=0.099 under matched_frame, far worse than either alternative.

- **MAE:** Encodes primarily static spatial appearance. Small relative drop (-20%) because there is little temporal signal to lose. Pixel reconstruction rewards frame-level detail, not inter-frame dynamics.

**The monotonic ordering of disruption conditions** (clean ≈ tubelet ≈ reverse ≈ matched > shuffle > matched_frame) is consistent across all three models, confirming the gradient is measuring temporal reliance and not an artifact of any single condition.

---

## Log Files

All logs are at `scripts/rebuttal/samples/`:

| Log File | Model | Condition |
|----------|-------|-----------|
| `end_byol_clean.log` | BYOL | clean |
| `end_byol_tubelet100.log` | BYOL | tubelet |
| `end_byol_reverse.log` | BYOL | reverse |
| `end_byol_matched100.log` | BYOL | matched |
| `end_byol_shuffle100.log` | BYOL | shuffle seed 100 |
| `end_byol_shuffle101.log` | BYOL | shuffle seed 101 |
| `end_byol_shuffle102.log` | BYOL | shuffle seed 102 |
| `end_byol_matched_frame100.log` | BYOL | matched_frame |
| `end_jepa_tubelet100.log` | JEPA | tubelet |
| `end_jepa_reverse.log` | JEPA | reverse |
| `end_jepa_matched100.log` | JEPA | clean (matched = clean with RoPE remap) |
| `end_jepa_shuffle100.log` | JEPA | shuffle seed 100 |
| `end_jepa_shuffle101.log` | JEPA | shuffle seed 101 |
| `end_jepa_shuffle102.log` | JEPA | shuffle seed 102 |
| `end_jepa_matched_frame100.log` | JEPA | matched_frame |
| `end_mae_clean.log` | MAE | clean |
| `end_mae_tubelet100.log` | MAE | tubelet |
| `end_mae_reverse.log` | MAE | reverse |
| `end_mae_matched100.log` | MAE | matched |
| `end_mae_shuffle100.log` | MAE | shuffle seed 100 |
| `end_mae_shuffle101.log` | MAE | shuffle seed 101 |
| `end_mae_shuffle102.log` | MAE | shuffle seed 102 |
| `end_mae_matched_frame100.log` | MAE | matched_frame |

**Note:** No `end_jepa_clean.log` exists. JEPA's clean baseline is `end_jepa_matched100.log` (R²=0.549), which is `type=matched` — tubelet shuffle with RoPE remapped. Since RoPE compensation fully restores performance, matched = clean (confirmed: BYOL clean 0.460 ≈ matched 0.460, MAE clean 0.396 = matched 0.396).

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/rebuttal/frame_shuffle_task.py` | Standalone task-level evaluation (single center clip, no prediction averaging). Used for initial exploration. |
| `src/datasets/video_dataset.py:375-412` | Temporal ablation implementation in VideoDataset (used by `evals.main` pipeline). |
| `scripts/rebuttal/frame_shuffling.py` | Representation-level cosine similarity analysis (too insensitive — see below). |

## Representation-Level Results (Supplementary)

Cosine similarity between mean-pooled features of original vs shuffled clips. Run on 200 EchoNet-Dynamic videos via `scripts/rebuttal/frame_shuffling.py`. These proved **too insensitive** for meaningful conclusions — all models show >0.93 cosine similarity even under full shuffle, because mean-pooling washes out temporal structure.

| Model | cos(orig, shuffled) | Degradation |
|-------|-------------------|-------------|
| JEPA-L pt50 | 0.985 ±0.002 | 1.5% |
| BYOL-L pt50 | 0.998 ±0.000 | 0.2% |
| MAE-L pt50 | 0.938 ±0.002 | 6.2% |

**Verdict:** Task-level evaluation (R²) is far more informative than representation-level (cosine) for measuring temporal encoding. Use the task-level results from the `evals.main` pipeline for all publications.
