# MIMIC-IV-Echo Video Statistics

Statistics from 50 randomly sampled videos from `data/csv/mimic_annotations_s3.csv` (525,328 total clips). Measured with decord.

## Summary

| Metric | Value |
|---|---|
| **Total clips** | 525,328 |
| **Native FPS** | 30 fps (nearly all; some at 37) |
| **Resolution** | 224×224 (all pre-resized) |
| **Single-frame stills** | ~42% of clips |
| **Actual videos** | ~58% of clips |

## Frame Count Distribution (actual videos only, n≈29/50)

| Stat | Frames | Duration at 30fps |
|---|---|---|
| min | 43 | 1.4s |
| p25 | 62 | 2.1s |
| **median** | **74** | **2.5s** |
| p75 | 87 | 2.9s |
| p95 | 137 | 4.6s |
| max | 179 | 6.0s |

Histogram:
- [32, 48): 7%
- [48, 64): 28%
- **[64, 96): 48%**
- [96, 128): 10%
- [128, 200): 7%

## Frame Coverage at Different Sampling Rates

"Coverage" = percentage of actual videos with enough native frames.

### At current config: 8fps (frame_step=4)

| FPC | Native frames needed | Coverage | Temporal window |
|---|---|---|---|
| **16** | 64 | ~58% (median barely fits) | 2.0s |
| 32 | 128 | ~6% | 4.0s |
| 64 | 256 | 0% | 8.0s |

### At 15fps (frame_step=2)

| FPC | Native frames needed | Coverage | Temporal window |
|---|---|---|---|
| 16 | 32 | ~100% | 1.1s |
| **32** | **64** | **~58%** | **2.1s** |
| 64 | 128 | ~6% | 4.3s |

### At 24fps (frame_step=1)

| FPC | Native frames needed | Coverage | Temporal window |
|---|---|---|---|
| 16 | 16 | ~100% | 0.7s |
| **32** | **32** | **~100%** | **1.3s** |
| 48 | 48 | ~93% | 2.0s |
| 64 | 64 | ~60% | 2.7s |

## Key Implications for Training

### Why V-JEPA 2's cooldown recipe (4fps, 64 frames) is impossible
At 4fps, 64 frames requires 480 native frames (16 seconds). The longest MIMIC video is 179 frames (6 seconds). **0% coverage.**

### Recommended cooldown: 24fps, 32 frames
- All actual videos have ≥32 native frames (100% coverage, no padding)
- 1.3s covers 1–2 full cardiac cycles (60–100 bpm)
- 2× tokens (3136 vs 1568) — halve batch size to fit memory
- Higher temporal resolution captures fast valve motion

### About the single-frame stills (~42%)
These are Doppler traces, M-mode images, and static annotations. The training code pads them to the requested FPC by repeating the frame. This is harmless — the encoder learns that static content has temporally constant representations.

### Current pretrain (8fps, 16 frames)
Works but tight: median video (74 native frames) yields exactly 18 extractable frames at step=4. Videos shorter than 64 native frames (~42% of actual videos, at p25=62) require padding with repeated last frame.
