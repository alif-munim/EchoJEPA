# ViT-B Temporal Shortcut Pilot Experiments

## Hypothesis

VideoMAE transiently learns temporal features mid-training (~e50) then abandons them by convergence (~e100) because tube masking allows spatial interpolation from adjacent frames — static spatial features suffice for pixel reconstruction. This "temporal shortcut" is a property of the masking strategy, not the reconstruction objective.

**Evidence from prior work:** Frame shuffling degradation on EchoMAE-L was -313% at e50 (temporal features present) vs -4% at e99 (temporal features abandoned). See `claude/rebuttals/experiments/frame-shuffling.md`.

## Experimental Design

Two VideoMAE ViT-B models trained in parallel, differing ONLY in masking strategy:

### Pilot 1: Standard MAE ViT-B (Baseline)

- **Masking:** Tube masking (standard VideoMAE) — same random spatial mask tiled across ALL 8 temporal positions. 90% mask ratio.
- **Init:** Random (no ImageNet init)
- **Epochs:** 100, checkpoints at e25/e50/e75/e100
- **Architecture:** ViT-B (86M), 12 blocks, 768 dim, 12 heads, patch 16, tubelet 2, decoder depth 4
- **Data:** MIMIC-IV-Echo 525K clips, 16 frames @ 8 fps
- **Training:** AdamW, base LR 1.5e-4 scaled to 6e-4 (eff BS 1024), cosine decay to 1e-5, warmup 10 epochs
- **Prediction:** Standard MAE tube masking allows transient temporal learning → spatial shortcut dominance by e100

### Pilot 2: Frame-Gap MAE ViT-B (Intervention)

- **Masking:** Frame-gap masking — 8 temporal positions split into context[t0:t3], gap[t3:t5], target[t5:t8]. Visible patches ONLY in context frames (~27% visible). Gap + target ALL masked (0% visible). Overall 90% mask ratio preserved.
- **Everything else:** Identical to Pilot 1
- **Prediction:** With temporal gap, spatial interpolation is impossible. Model MUST learn temporal dynamics to reconstruct target frames. Temporal features should be retained throughout training, including at e100.

### Frame-Gap Masking Details

```
Temporal positions:  t0  t1  t2 | t3  t4 | t5  t6  t7
                     context    |  gap   |  target
Visible patches:     ~27%      |   0%   |   0%
                     ↑ 157 of 588        ↑ all 980 masked

Total patches: 8 × 196 = 1568
Masked: 1411 (90%)  Visible: 157 (10%)
All visible patches in context frames only.
```

The key constraint: target frames (t5-t7) are separated from ANY visible patch by at minimum 2 temporal positions (the gap). The model cannot peek at spatially adjacent visible patches in nearby frames — it must reason about temporal dynamics.

## Evaluation Plan

**Frame Shuffling Severity Gradient** at e25/e50/e75/e100:

For each checkpoint:
1. Train LVEF regression probe (d=1 attentive, UHN data, frozen encoder)
2. Evaluate with frame shuffling at 5 severity levels: 0%, 25%, 50%, 75%, 100%
3. Measure R² degradation curve

**Expected results:**

| Epoch | Standard MAE | Frame-Gap MAE |
|-------|-------------|---------------|
| e25 | Moderate degradation | Moderate degradation |
| e50 | **Peak degradation** (temporal features present) | Strong degradation |
| e75 | Reduced degradation (abandoning temporal) | Strong degradation |
| e100 | **Minimal degradation** (temporal shortcut) | **Strong degradation** (temporal retained) |

If confirmed: the temporal shortcut is caused by tube masking, not the reconstruction objective. Frame-gap masking is a simple architectural fix that preserves MAE's temporal learning.

## Implementation

### Code Changes

1. **`s3_dataset.py`**: Added `FrameGapMaskingGenerator` class alongside existing `TubeMaskingGenerator`. Modified `VideoDataset` to accept `mask_type` and `temporal_gap` parameters.

2. **`run_mae_pretraining.py`**: Added `'frame_gap'` to `--mask_type` choices, added `--temporal_gap` argument. Both passed to `VideoDataset`.

### Jobs

| Pilot | Job ID | Node | Sbatch | S3 Output |
|-------|--------|------|--------|-----------|
| Standard MAE ViT-B | 570 | ip-10-0-50-83 | `scripts/videomae_pilot_standard_vitb.sbatch` | `s3://.../runs/mae_pilot_standard_vitb_570/` |
| Frame-Gap MAE ViT-B | 571 | ip-10-0-50-184 | `scripts/videomae_pilot_framegap_vitb.sbatch` | `s3://.../runs/mae_pilot_framegap_vitb_571/` |

### Training Config

| Parameter | Value |
|-----------|-------|
| Model | `pretrain_videomae_base_patch16_224` |
| Params | 86M encoder + decoder |
| Batch per GPU | 32 |
| Accum iter | 4 |
| Effective BS | 1024 |
| GPUs | 8 × H100 80GB |
| Peak LR | 6e-4 (1.5e-4 × 1024/256) |
| Warmup | 10 epochs (5130 steps) |
| Steps/epoch | 2052 |
| Total steps | 205,200 |
| GPU mem | ~22.8 GB |
| ETA/epoch | ~26 min |
| Total ETA | ~43 hours per model |
| Mask ratio | 0.9 |
| Save freq | every 25 epochs |

## Status

- [x] Implement `FrameGapMaskingGenerator`
- [x] Create sbatch scripts
- [x] Launch both jobs (2026-04-08)
- [ ] Training completes (~43h, ~2026-04-10)
- [ ] Train LVEF probes on e25/e50/e75/e100 × 2 models
- [ ] Run frame shuffling severity gradient evaluation
- [ ] Analyze temporal shortcut trajectory
