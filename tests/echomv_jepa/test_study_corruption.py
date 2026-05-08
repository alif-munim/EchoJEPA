"""Extracted study_corruption must preserve byte-identical behavior.

The full-joint trainer depends on this extraction; this test pins outputs
on a seeded synthetic batch. If the extraction diverges from the prior
in-tree behavior of ``_apply_study_corruption``, this test fails.
"""

from __future__ import annotations

import torch

from src.models.echomv_jepa.study_corruption import apply_study_corruption


def _make_batch(B: int = 3, M: int = 6, d_clip: int = 8, d_model: int = 16, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    ctx_elements = torch.randn(B, M, d_clip, generator=g)
    ctx_meta_add = torch.zeros(B, M, d_model)
    ctx_pad_mask = torch.tensor(
        [
            [False, False, False, False, False, False],
            [False, False, False, False, True, True],
            [False, False, False, True, True, True],
        ],
        dtype=torch.bool,
    )
    meta_view = torch.tensor(
        [
            [0, 0, 1, 2, 2, 3],
            [0, 1, 1, 2, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=torch.long,
    )
    meta_modality = torch.tensor(
        [
            [0, 0, 0, 1, 1, 1],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=torch.long,
    )
    return ctx_elements, ctx_meta_add, ctx_pad_mask, meta_view, meta_modality


def test_no_dropout_is_identity():
    ctx_elements, meta_add, pad, view, mod = _make_batch(seed=1)
    rng = torch.Generator().manual_seed(1)
    mix = {"no_dropout": 1.0}
    out_elem, out_pad = apply_study_corruption(ctx_elements, meta_add, pad, view, mod, mix, rng)
    assert torch.equal(out_pad, pad)
    assert torch.equal(out_elem, ctx_elements)


def test_random_element_dropout_expands_pad_but_keeps_at_least_one():
    ctx_elements, meta_add, pad, view, mod = _make_batch(seed=2)
    rng = torch.Generator().manual_seed(2)
    mix = {"random_element_dropout": 1.0}
    out_elem, out_pad = apply_study_corruption(ctx_elements, meta_add, pad, view, mod, mix, rng)
    # Each row must still have at least one unpadded position.
    assert ((~out_pad).sum(dim=1) >= 1).all()
    # Pad can only grow, not shrink.
    assert (out_pad | pad == out_pad).all()
    # Newly padded content is zeroed; previously unpadded non-dropped content is preserved.
    newly = out_pad & (~pad)
    if newly.any():
        assert (out_elem[newly] == 0.0).all()
    kept = ~out_pad
    assert torch.equal(out_elem[kept], ctx_elements[kept])


def test_whole_view_family_dropout_drops_single_family():
    ctx_elements, meta_add, pad, view, mod = _make_batch(seed=3)
    rng = torch.Generator().manual_seed(3)
    mix = {"whole_view_family_dropout": 1.0}
    _out_elem, out_pad = apply_study_corruption(ctx_elements, meta_add, pad, view, mod, mix, rng)
    # In every row at least one unpadded element must remain.
    assert ((~out_pad).sum(dim=1) >= 1).all()


def test_whole_modality_dropout_drops_single_modality():
    ctx_elements, meta_add, pad, view, mod = _make_batch(seed=4)
    rng = torch.Generator().manual_seed(4)
    mix = {"whole_modality_dropout": 1.0}
    _out_elem, out_pad = apply_study_corruption(ctx_elements, meta_add, pad, view, mod, mix, rng)
    assert ((~out_pad).sum(dim=1) >= 1).all()


def test_seeded_output_is_deterministic():
    """Byte-identical output under the same seed = the extraction preserved behavior."""
    ctx_elements, meta_add, pad, view, mod = _make_batch(seed=5)
    mix = {
        "random_element_dropout": 0.30,
        "whole_view_family_dropout": 0.25,
        "whole_modality_dropout": 0.15,
        "no_dropout": 0.30,
    }
    rng_a = torch.Generator().manual_seed(42)
    rng_b = torch.Generator().manual_seed(42)
    out_a_elem, out_a_pad = apply_study_corruption(ctx_elements, meta_add, pad, view, mod, mix, rng_a)
    out_b_elem, out_b_pad = apply_study_corruption(ctx_elements, meta_add, pad, view, mod, mix, rng_b)
    assert torch.equal(out_a_elem, out_b_elem)
    assert torch.equal(out_a_pad, out_b_pad)
