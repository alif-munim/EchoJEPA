# Issue 016: `vit_giant` vs `vit_giant_xformers` num_heads mismatch silently corrupts EchoJEPA-G

**Severity**: CRITICAL
**Discovered**: 2026-03-28 (Session 31, segmentation eval)
**Status**: **FIXED**
**File**: `evals/segmentation_frozen/eval.py`

## Summary

EchoJEPA-G checkpoint (trained with `vit_giant_xformers`, 22 heads) was loaded into a `vit_giant` architecture (16 heads). The weight shapes are identical, so `load_state_dict(strict=False)` succeeds without any warning. But the attention computation is completely wrong — every head gets the wrong feature subset, and RoPE rotation dimensions are mismatched. Result: 0.600 test Dice instead of expected ~0.82+.

## Root Cause

Four ViT-Giant factory functions exist in `src/models/vision_transformer.py`:

| Factory function           | num_heads | head_dim | RoPE   | Line |
|---------------------------|-----------|----------|--------|------|
| `vit_giant`               | **16**    | 88       | via `**kwargs` | 406 |
| `vit_giant_xformers`      | **22**    | 64       | via `**kwargs` | 303 |
| `vit_giant_rope`          | 16        | 88       | hardcoded | 420 |
| `vit_giant_xformers_rope` | 22        | 64       | hardcoded | 435 |

V-JEPA 2 pretraining uses `vit_giant_xformers` (22 heads, head_dim=64). The segmentation eval script's auto-detection logic defaulted to `vit_giant` (16 heads, head_dim=88).

**Why it loads without error**: The QKV weight shape is `[3 * embed_dim, embed_dim]` = `[4224, 1408]` regardless of `num_heads`. The `num_heads` parameter only affects how the QKV output is *reshaped* during the forward pass — it's not encoded in any weight shape. Both `strict=True` and `strict=False` pass with zero mismatched keys. The model has the same 484 parameter keys in both configurations.

**Why it silently corrupts attention**: During forward, the QKV output `[B, N, 4224]` is reshaped to `[B, N, 3, num_heads, head_dim]`:
- Correct (22 heads): `[B, N, 3, 22, 64]` — each head gets the 64-dim slice it was trained for
- Wrong (16 heads): `[B, N, 3, 16, 88]` — each head gets an 88-dim slice that mixes features from ~1.4 different trained heads

**Why RoPE is also wrong**: `RoPEAttention` computes rotation dimensions from `head_dim` (`src/models/utils/modules.py:286-288`):
```python
self.d_dim = int(2 * ((head_dim // 3) // 2))  # 64→20, 88→28
self.h_dim = int(2 * ((head_dim // 3) // 2))
self.w_dim = int(2 * ((head_dim // 3) // 2))
```
With head_dim=88, RoPE rotates 28 dims per axis (84 total) instead of the trained 20 dims per axis (60 total). The positional encoding is applied to wrong dimensions, further scrambling spatial structure.

## Symptoms

- EchoJEPA-G segmentation Dice: **0.600** (vs L's 0.818 and MAE's 0.790)
- G@224 and G@384 gave **identical** results (0.600 vs 0.596) — bizarre for a resolution change
- Feature norms 4× larger than L (118.5 vs 30.2)
- All 7 HP configs converged to the same narrow band (0.51-0.60) — not a hyperparameter problem

## Debugging Timeline

The bug was not obvious and required extensive investigation:

1. **Initial observation**: G stuck at ~0.54 while L reached 0.82. All 7 HP configs in the same range — ruled out hyperparameter issue.

2. **Resolution hypothesis**: G was annealed at 384px, so tried running at native resolution. Result: 0.596 ≈ 0.600. Identical results at different resolutions was the key red flag.

3. **Spatial coherence diagnostics**: Measured feature similarity on CAMUS echo data. Found "corners more similar than adjacent tokens" — but this was a **data artifact** (echo fan geometry has black corners), not a model artifact. User correctly identified this confound.

4. **"Deeper ViTs lose locality" hypothesis**: Proposed that 40 attention layers mix spatial info too much. User rejected this — DINOv2 ViT-g (same depth, same dim) achieves SOTA linear segmentation on ADE20K/VOC. The explanation was too general.

5. **RoPE buffer check**: Verified RoPE has no stored buffers (frequencies computed on-the-fly in `RoPEAttention.forward()`), so there's nothing to "drop" during loading. RoPE IS correctly enabled via `use_rope=True` kwarg.

6. **Architecture comparison**: Finally compared factory functions side-by-side. Found `vit_giant` has `num_heads=16` while `vit_giant_xformers` has `num_heads=22`. Same key count (484 vs 484), same weight shapes, completely different attention computation.

Key user insight that drove the diagnosis: "The `strict=False` loading with `vit_giant` is suspicious... If G was pretrained with RoPE, loading into a non-RoPE architecture with `strict=False` would silently drop the RoPE buffers." While the specific RoPE-buffer theory wasn't the cause, the user's instinct that the loading was silently wrong — and their rejection of premature conclusions — was critical to finding the real bug.

## Fix

Changed auto-detection default in `load_vjepa_encoder()` (`evals/segmentation_frozen/eval.py:132`):

```python
# Before (WRONG):
model_name = "vit_giant"          # 16 heads, head_dim=88

# After (CORRECT):
model_name = "vit_giant_xformers"  # 22 heads, head_dim=64
```

Also added `--model_name` CLI arg to allow explicit override.

## Verification

After fix, G epoch 1 Dice jumped from 0.395 → 0.640 (+0.245). Run still in progress.

## Broader Implications

This class of bug — **num_heads mismatch in ViT loading** — is undetectable by `load_state_dict` (even with `strict=True`) because `num_heads` only affects the reshape during forward, not weight shapes. Any code loading ViT checkpoints should explicitly verify the factory function matches the training config, not just check for key/shape match.

Other code paths that load G checkpoints (probe training via `evals/video_classification_frozen/`) use YAML configs that specify the correct model name, so they are not affected.

## See Also

- `src/models/vision_transformer.py:303-417` — factory function definitions
- `src/models/utils/modules.py:262-290` — RoPEAttention head_dim-dependent dimensions
