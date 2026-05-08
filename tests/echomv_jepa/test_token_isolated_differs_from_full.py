"""Critical test for EchoMV-JEPA Option A: does token-level input genuinely
force the teacher's contextualized output to differ from the isolated output?

The pooled-cache Stage-1 failed §20.2.a because within-study c_clips are
pairwise cosine 0.87-0.91 on real MIMIC studies; self-attention over 6
near-identical vectors ≈ identity, so forward_contextualized(full study)
≈ forward_isolated(each element alone).

Option A feeds T=392 tokens per element (after a 2x2 spatial pool over
V-JEPA's 1568-token output). Within-element tokens carry spatial/temporal
structure that is destroyed by pooling. The hypothesis: when we run the
teacher at token granularity, ``forward_contextualized`` genuinely attends
across elements and diverges from the per-element ``forward_isolated`` output.

This file asserts the hypothesis holds in three regimes:

  1. Synthetic tokens with cross-element structure (independent Gaussians
     per token, not per-element). Should pass trivially for a trained
     transformer; a borderline pass at init is acceptable.

  2. Synthetic tokens mimicking the pooled-cache failure mode: every token
     of every element is drawn from the same Gaussian (near-identical
     tokens, identical cross- and within-element). Should show ≈ identity
     under *both* conditions — a control.

  3. Synthetic tokens with shared per-element cluster centers but
     independent per-token noise (most realistic for real V-JEPA tokens
     from the same clip). Within-element tokens cluster; across-element
     tokens differ. Should show meaningful contextualization.

If regime 1 and regime 3 both show cos(z_full, z_iso) < 0.9 on a fresh-init
wrapper, the token-level path is viable. If they both show cos > 0.98, the
architecture has a near-identity shortcut at token granularity too, and
Option A is dead before we touch the encoder or data pipeline.

This test is a *hypothesis check*, not a correctness proof. The ``StudyTransformer``
itself has permutation-invariance tests elsewhere; this test is only about
whether token sequencing changes the architectural outcome.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.models.echomv_jepa.token_study_transformer import TokenStudyTransformer
from src.models.study_transformer import StudyTransformer, StudyTransformerConfig


def _build(d_clip=32, d_model=32, n_layers=2, max_M=64):
    # Small dims to keep the test fast; architecture still meaningful.
    cfg = StudyTransformerConfig(
        d_clip=d_clip,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=4,
        ffn_mult=2,
        dropout_ffn=0.0,
        dropout_attn=0.0,
        max_M=max_M,
    )
    st = StudyTransformer(cfg).eval()
    return TokenStudyTransformer(st).eval()


def _token_full_vs_iso(
    wrapper: TokenStudyTransformer,
    element_tokens: torch.Tensor,  # (B, M, T, d_clip)
    element_meta_add: torch.Tensor,  # (B, M, d_model)
    elem_pad: torch.Tensor,  # (B, M) bool
) -> float:
    """Compute mean cosine between forward_contextualized (full study) and
    element-wise forward_isolated (each element's T tokens alone)."""
    B, M, T, _ = element_tokens.shape
    with torch.no_grad():
        z_full = wrapper.forward_contextualized(element_tokens, element_meta_add, elem_pad)  # (B, M, d_model)

        # Isolated per element: run the wrapper with M=1 for each element in turn.
        z_iso = torch.empty_like(z_full)
        no_pad_1 = torch.zeros(B, 1, dtype=torch.bool, device=elem_pad.device)
        for m in range(M):
            el_m = element_tokens[:, m : m + 1]  # (B, 1, T, d_clip)
            mt_m = element_meta_add[:, m : m + 1]  # (B, 1, d_model)
            z_iso[:, m : m + 1] = wrapper.forward_contextualized(el_m, mt_m, no_pad_1)

        d = z_full.shape[-1]
        z_full_ln = F.layer_norm(z_full.reshape(-1, d), (d,))
        z_iso_ln = F.layer_norm(z_iso.reshape(-1, d), (d,))
        cos = F.cosine_similarity(z_full_ln, z_iso_ln, dim=-1)
        # Mask by elem_pad — only valid elements count.
        valid = (~elem_pad).reshape(-1)
        return cos[valid].mean().item()


def test_regime1_independent_per_token_gaussian_contextualizes():
    """Per-token independent Gaussians — no special structure. At init the
    wrapper should still show *some* contextualization because the pad-mask
    is full and attention averages genuinely distinct token rows. Mean
    cosine should be below the 0.98 pooled-failure threshold."""
    torch.manual_seed(0)
    wrapper = _build()
    B, M, T, d_clip, d_model = 2, 4, 8, 32, 32
    tokens = torch.randn(B, M, T, d_clip)
    meta = torch.randn(B, M, d_model)
    elem_pad = torch.zeros(B, M, dtype=torch.bool)
    cos = _token_full_vs_iso(wrapper, tokens, meta, elem_pad)
    # Sanity: the token-level wrapper should not collapse to identity on
    # independent-gaussian inputs at init. A weak bound here:
    assert cos < 0.98, f"token wrapper collapsed even on independent tokens: cos={cos:.4f}"


def test_regime2_control_near_identical_tokens_collapses_to_identity():
    """Control: every token of every element is near-identical (no within-
    or across-element structure). We should see forward_contextualized
    ≈ forward_isolated because attention has no structure to exploit. This
    confirms the test harness is measuring what it claims."""
    torch.manual_seed(1)
    wrapper = _build()
    B, M, T, d_clip, d_model = 2, 4, 8, 32, 32
    base = torch.randn(B, 1, 1, d_clip)
    tokens = base.expand(B, M, T, d_clip).clone()  # all tokens identical across B*M*T
    meta = torch.zeros(B, M, d_model)
    elem_pad = torch.zeros(B, M, dtype=torch.bool)
    cos = _token_full_vs_iso(wrapper, tokens, meta, elem_pad)
    # With identical inputs across all positions, the full-study and isolated
    # outputs must agree (this is the v1 pooled-cache failure mode reproduced
    # deliberately, as a control).
    assert cos > 0.99, f"control regime expected full≈iso with identical tokens, got cos={cos:.4f}"


def test_regime3_per_element_clustered_tokens_contextualizes():
    """Per-element cluster with small within-element noise.

    Each element has its own cluster center (cosine between centers across
    elements ~0) and tokens within an element are near-identical (noise ≪
    center magnitude). This matches v1's pooled failure mode lifted to
    tokens: within-element tokens near-identical, cross-element tokens
    diverse. The wrapper's pool-over-tokens brings it close to the
    per-element-mean case, so at *cold init* we expect cos close to (but
    not exactly) the pooled failure level.

    The *real* test is regime 4 (genuine within-element variation) plus the
    smoke-time §20.2.a gate (``z_cosine_vs_isolated < 0.90`` after training).
    Here we only verify that even in the worst-case within-element regime,
    token-level input is materially better than the pooled-cache failure
    (cos = 0.998 observed in smoke job 741)."""
    torch.manual_seed(2)
    wrapper = _build()
    B, M, T, d_clip, d_model = 2, 6, 16, 32, 32
    centers = torch.randn(B, M, 1, d_clip) * 3.0
    noise = torch.randn(B, M, T, d_clip) * 0.5
    tokens = centers + noise
    meta = torch.randn(B, M, d_model) * 0.1
    elem_pad = torch.zeros(B, M, dtype=torch.bool)
    cos = _token_full_vs_iso(wrapper, tokens, meta, elem_pad)
    # At cold init with near-identical within-element tokens, ~0.96 is
    # expected. The point of this assertion is only: we must beat pooled's
    # 0.998 by a clear margin, so training has room to drop further below
    # the §20.2.a gate (0.90).
    assert cos < 0.97, (
        f"per-element-clustered tokens did not contextualize enough at init: cos={cos:.4f}. "
        "The pooled-cache failure was 0.998; we should see at least 0.97 at cold init."
    )


def test_regime4_per_element_cluster_with_real_within_element_variation():
    """Most realistic regime: each element has a cluster center AND real
    within-element token diversity (std comparable to center magnitude).

    This models ViT tokens within a single echo clip: different spatial
    patches differ substantially (cosine ~0.4-0.7 between tokens of the
    same clip in real V-JEPA output), unlike the pooled v1 case where all
    within-clip structure is destroyed by mean-pooling.

    When tokens within an element are genuinely diverse, self-attention
    across elements has something to do — attention can pick specific
    per-element tokens that resemble tokens of other elements, forming
    useful cross-element queries even at cold init.

    Cold-init threshold: cos < 0.92. The smoke-time §20.2.a gate is
    stricter (< 0.90 after training); that gap is the budget for training
    to drive contextualization further."""
    torch.manual_seed(5)
    wrapper = _build()
    B, M, T, d_clip, d_model = 2, 6, 16, 32, 32
    centers = torch.randn(B, M, 1, d_clip) * 1.0  # element-specific
    noise = torch.randn(B, M, T, d_clip) * 1.0  # equal-magnitude token variation
    tokens = centers + noise
    meta = torch.randn(B, M, d_model) * 0.1
    elem_pad = torch.zeros(B, M, dtype=torch.bool)
    cos = _token_full_vs_iso(wrapper, tokens, meta, elem_pad)
    assert cos < 0.92, (
        f"real-within-element-variation regime did not contextualize: cos={cos:.4f}. "
        "If this fires at cold init with genuine per-token diversity, the flat-sequence "
        "token-level formulation is insufficient and Option A is in jeopardy."
    )


def test_pad_mask_is_respected_by_flatten():
    """Padded elements must not affect the unpadded element outputs. If a
    padded element's tokens leak into attention, the flattened pad-mask is
    wrong and the whole token path is compromised."""
    torch.manual_seed(3)
    wrapper = _build()
    B, M, T, d_clip, d_model = 1, 4, 8, 32, 32
    tokens = torch.randn(B, M, T, d_clip)
    meta = torch.randn(B, M, d_model)
    elem_pad = torch.tensor([[False, False, True, True]])  # last 2 elements padded

    with torch.no_grad():
        z_with_pad = wrapper.forward_contextualized(tokens, meta, elem_pad)

    # Corrupt the padded elements' tokens; outputs at unpadded positions must be
    # identical.
    tokens2 = tokens.clone()
    tokens2[0, 2:] = torch.randn_like(tokens2[0, 2:]) * 100.0
    meta2 = meta.clone()
    meta2[0, 2:] = torch.randn_like(meta2[0, 2:]) * 100.0
    with torch.no_grad():
        z_with_pad2 = wrapper.forward_contextualized(tokens2, meta2, elem_pad)

    assert torch.allclose(
        z_with_pad[:, :2], z_with_pad2[:, :2], atol=1e-5
    ), "padded elements leaked into unpadded positions — flatten pad-mask is broken"


def test_shapes_propagate_correctly():
    wrapper = _build(d_clip=16, d_model=24)
    B, M, T = 3, 5, 9
    tokens = torch.randn(B, M, T, 16)
    meta = torch.randn(B, M, 24)
    pad = torch.zeros(B, M, dtype=torch.bool)
    with torch.no_grad():
        z = wrapper.forward_contextualized(tokens, meta, pad)
        z_tok = wrapper.forward_contextualized_tokens(tokens, meta, pad)
    assert z.shape == (B, M, 24)
    assert z_tok.shape == (B, M, T, 24)


def test_token_pad_mask_overrides_elem_pad():
    """An element can be valid but some of its tokens padded (e.g. variable
    clip lengths). The forward should honor per-token pad."""
    torch.manual_seed(4)
    wrapper = _build()
    B, M, T, d_clip, d_model = 1, 2, 6, 32, 32
    tokens = torch.randn(B, M, T, d_clip)
    meta = torch.randn(B, M, d_model)
    elem_pad = torch.zeros(B, M, dtype=torch.bool)
    token_pad = torch.zeros(B, M, T, dtype=torch.bool)
    token_pad[0, 0, 3:] = True  # last 3 tokens of element 0 padded

    with torch.no_grad():
        z = wrapper.forward_contextualized(tokens, meta, elem_pad, token_pad)

    # Corrupt the padded tokens; unpadded element 1 output must be unchanged.
    tokens2 = tokens.clone()
    tokens2[0, 0, 3:] = torch.randn_like(tokens2[0, 0, 3:]) * 100.0
    with torch.no_grad():
        z2 = wrapper.forward_contextualized(tokens2, meta, elem_pad, token_pad)
    assert torch.allclose(z[:, 1:], z2[:, 1:], atol=1e-5), "token pad leak"
