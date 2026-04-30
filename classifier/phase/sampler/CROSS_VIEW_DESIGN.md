# Cross-view JEPA — design note (provisional, Turn 1)

This note pins down the tensor shapes, mask semantics, and teacher-latent
indexing needed for the phase-matched cross-view objective. It is
**provisional** — Turn 3 will trace the full predictor/teacher forward
path end-to-end and amend this note before a single-batch smoke test.
No silent shape adaptation is allowed.

## Intraview JEPA (the baseline we're extending)

For one input clip `x` (shape `[B, T, H, W, C]`), V-JEPA's context encoder
emits patch tokens:

    h_ctx = encoder(x)              # [B, N, D_enc]
    N     = (T // tubelet) * (H // p) * (W // p)   # number of patches

`N` is the total spatiotemporal token count (patch-time × patch-h × patch-w).
For the standard ViT-L/16 pretrain config this is e.g. `N = 8 * 14 * 14 = 1568`
at `T=16, tubelet=2, H=W=224, p=16`.

The predictor takes a *context subset* of those tokens and predicts a
*target subset* of the teacher's latents on the *same clip*:

    h_tgt   = target_encoder(x)     # [B, N, D_enc]      (EMA of encoder)
    h_ctx_m = apply_masks(h_ctx, masks_x)   # [B * |m_x|, N_ctx, D_enc]
    h_pred  = predictor(
        x=h_ctx_m,
        masks_x=masks_x,            # list of LongTensor[B, N_ctx]
        masks_y=masks_y,            # list of LongTensor[B, N_tgt]
        ...
    )                               # [B * |m_x|, N_tgt, D_enc]
    h_gt    = apply_masks(h_tgt, masks_y)   # [B * |m_y|, N_tgt, D_enc]
    loss    = smooth_l1(h_pred, h_gt)

`masks_x` and `masks_y` are index lists into the `N` patch positions.
The predictor does NOT know about raw space/time — it just sees token
indices and a positional embedding (or RoPE) driven by those indices.

## Cross-view JEPA — proposed minimum-change design

Goal: keep the predictor module unchanged, reuse its existing
token-index-based forward, and simply swap *whose target latents we
supervise against*.

### Shape contract

Each cross-view training step processes a **phase-matched pair**
`(clip_a, clip_b)` where both tensors have identical shape
`[B, T, H, W, C]` produced by the anchor-centered loader at the
sampler's `frame_step`. `clip_a` is the context clip; `clip_b` is the
target clip.

    h_ctx_a = context_encoder(clip_a)       # [B, N, D_enc]
    h_tgt_b = target_encoder(clip_b)        # [B, N, D_enc]   (same EMA teacher)

### Mask sharing (Turn 1 decision: share masks)

`masks_x` and `masks_y` are the same lists used for the intraview loss
on `clip_a`. The predictor receives masked clip_a context tokens and is
asked to predict *clip_b* target-encoder latents at the **same patch
positions**:

    h_ctx_a_m = apply_masks(h_ctx_a, masks_x)   # [B*|m_x|, N_ctx, D_enc]
    h_pred_b  = predictor(
        x=h_ctx_a_m,
        masks_x=masks_x,
        masks_y=masks_y,
    )                                            # [B*|m_x|, N_tgt, D_enc]
    h_gt_b    = apply_masks(h_tgt_b, masks_y)    # [B*|m_y|, N_tgt, D_enc]
    loss_xv   = smooth_l1(h_pred_b, h_gt_b)

The interpretation: **at the same patch position, predict the
cross-view frame's latent**. Because both clips are centered on the same
cardiac phase at the same `frame_step`, patch-position `(t, y, x)` in
clip_a and clip_b refer to the same phase-aligned snapshot — not the
same physical space (that varies by view) but the same temporal slice
of the cycle. The model must produce the target-view latent from the
context-view context.

### Anchor-position guarantee

Anchor-centered loading puts the frame whose phase is closest to
`target_phi` at `anchor_pos` (≤ (T-1)/2 off center under boundary
clamp). Both clip_a and clip_b are anchor-centered, so clip_a's
`anchor_pos` and clip_b's `anchor_pos` are equal **up to boundary
clamping**. When `was_clamped=True` on either clip, the anchor is not
at center — the predictor still sees a well-formed token grid, but the
"same patch position" semantics are degraded on that side. Turn 2's
anchor-loading sanity check will measure `was_clamped` fraction and log
it per batch.

### Teacher-latent indexing

`h_tgt_b[b, i, :]` is the target-encoder latent for the i-th patch of
clip_b's token grid. `masks_y` already indexes into this grid for the
intraview objective. For cross-view we reuse the same mask indices,
so teacher latents are gathered via `apply_masks(h_tgt_b, masks_y)` —
no new indexing code. The predictor's `masks_y` argument is unchanged.

## Total loss

    L_total = L_intraview + lambda_crossview * L_crossview

- `L_intraview`: standard V-JEPA loss on `clip_a` only (one teacher
  forward, one predictor call).
- `L_crossview`: same predictor forward on `clip_a` context, but
  supervised against `clip_b` teacher latents.
- `lambda_crossview`: default 0.25. Disabled by
  `phase_multiview.use_crossview_loss=false`.

The intraview path on clip_a is unchanged from the baseline — the
cross-view path shares the context encoder forward output but reuses
the predictor with a different target tensor.

## What must hold at smoke-test time

Turn 3's one-batch smoke test will assert:

1. `h_ctx_a.shape == h_tgt_b.shape == [B, N, D_enc]`.
2. After `apply_masks(..., masks_x)`: shape `[B * |m_x|, N_ctx, D_enc]`.
3. After `apply_masks(..., masks_y)`: shape `[B * |m_y|, N_tgt, D_enc]`.
4. Predictor output shape matches `h_gt_b` shape, so `smooth_l1`
   broadcasts cleanly.
5. `masks_x` and `masks_y` are sampled once per step and reused for
   both the intraview loss (against `h_tgt_a`, gathered from clip_a's
   teacher encoder pass) and the cross-view loss (against `h_tgt_b`).

**If any of these fail we stop and widen this note**, not silently
reshape. This is why the first pass does not attempt raw ECG fusion,
phase-relation tokens, or independent masks — those all require
predictor-input shape changes and we want to confirm the vanilla shape
contract first.

## What is intentionally not in this design yet

- **Phase-relation token** as predictor conditioning. Hook is
  sketched in the config (`phase_multiview.predictor_phase_token`) but
  disabled by default; implementation deferred until shapes are proven.
- **Independent masks per clip**. Would need predictor input to
  disambiguate context-side from target-side positions; likely needs
  a separate `masks_y_target` argument on the predictor. Not needed
  for the first result.
- **Delta-phi / HR tokens**. The predictor already supports a
  `delta_phi` keyword for intraview; for cross-view the analogous
  quantity is the target-relative-to-context phase, which equals the
  intraview value when shared masks are used. Skip.
- **Teacher branch on context-view**. `L_intraview` already exercises
  `target_encoder(clip_a)`; the cross-view path only needs
  `target_encoder(clip_b)`. Two teacher forwards per step; acceptable.

## Open questions for Turn 3

- Exact predictor signature in `app/vjepa_multiview/train.py`: does the
  existing `predictor(...)` call in `app/vjepa/train.py` pass `has_cls`,
  `mask_index`, or anything else we need to carry over unchanged?
  Turn 3 will trace.
- How does masking interact with the anchor position? Specifically,
  are target patches at the anchor's (t, y, x) position guaranteed to
  be in `masks_y`, or is that a sampling-random property? If the
  anchor position is not always in `masks_y`, the "anchor-matched
  prediction" claim is weaker than "phase-aligned clip pair with
  shared masks". Log mask temporal indices relative to `anchor_pos`
  for the first few batches.
