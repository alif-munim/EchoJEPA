# Issue 002: Encoder adapter normalization errors

**Severity**: HIGH
**Discovered**: 2026-03-07
**Status**: Fixed in code, MIMIC re-extraction needed

## Root Cause

Three encoder adapters in `evals/video_classification_frozen/modelcustom/` had incorrect input normalization. The shared `make_transforms(training=False)` pipeline applies ImageNet normalization to all inputs. Some external models expect different input ranges, requiring the adapter to undo ImageNet norm and apply model-specific normalization.

## Affected Models

### PanEcho (`panecho_encoder.py`)
**Bug**: `_preprocess()` applied ImageNet normalization a second time on already-ImageNet-normalized input (double normalization).
**Fix**: Removed redundant normalization in `_preprocess`.

### EchoPrime (`echo_prime_encoder.py`)
**Bug**: Expected input in [0, 255] range with EchoPrime-specific normalization, but received ImageNet-normalized input (~[-2, 2] range).
**Fix**: Added ImageNet de-normalization step before converting to [0, 255] and applying EchoPrime normalization.

### EchoFM (`echofm_encoder.py`)
**Bug**: Expected input in [0, 1] range (no normalization), but received ImageNet-normalized input.
**Fix**: Added ImageNet de-normalization to recover [0, 1] range.

## Models NOT affected
- **EchoJEPA-G/L** and **EchoMAE/VideoMAE**: Trained with ImageNet normalization. Input used as-is. Correct.

## Impact

All existing MIMIC embeddings for PanEcho, EchoPrime, and EchoFM were extracted with wrong normalization. These embeddings are corrupted and must be re-extracted.

Combined with Issue 001 (shuffle bug), all 7 MIMIC models need re-extraction. The 3 normalization-affected models need it for two reasons.

## See Also

- `claude/dev/code-review.md` — full adapter-by-adapter review with code references
