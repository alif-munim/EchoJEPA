# Issue 003: EchoFM temporal padding uses last-frame repetition

**Severity**: Moderate
**Discovered**: 2026-03-07
**Status**: Fixed

## Root Cause

`echofm_encoder.py` `_adapt_temporal()` padded short clips (16 frames → 32 frames) by repeating the last frame. For 16-frame V-JEPA clips, frames 17-32 were all identical copies of frame 16, creating a frozen-video artifact.

## Fix

Changed to `torch.linspace` repeat-interleave strategy (same as the downsampling path). Each original frame is evenly spread across the 32-frame target.

## Impact

Minor fairness concern in JEPA-vs-EchoFM comparison. EchoFM MIMIC embeddings already need re-extraction due to Issues 001 and 002, so this fix is automatically included.

## See Also

- `claude/dev/code-review.md` — Encoder Adapters section
