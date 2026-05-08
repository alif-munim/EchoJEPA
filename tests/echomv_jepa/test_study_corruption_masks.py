"""Arm C — study-level corruption mask tests."""

from __future__ import annotations

import torch

from app.echomv_jepa.train import _apply_study_corruption


def test_no_dropout_leaves_study_unchanged():
    B, M, d = 4, 6, 8
    elements = torch.randn(B, M, d)
    meta_add = torch.zeros(B, M, 12)
    pad = torch.zeros(B, M, dtype=torch.bool)
    view = torch.randint(0, 3, (B, M))
    modality = torch.randint(0, 2, (B, M))
    g = torch.Generator()
    g.manual_seed(0)
    out_el, out_pad = _apply_study_corruption(
        elements,
        meta_add,
        pad,
        view,
        modality,
        {"no_dropout": 1.0},
        g,
    )
    assert torch.equal(out_pad, pad)
    assert torch.equal(out_el, elements)


def test_random_element_dropout_rate_roughly_matches():
    torch.manual_seed(0)
    B, M, d = 64, 8, 8
    elements = torch.randn(B, M, d)
    meta_add = torch.zeros(B, M, 12)
    pad = torch.zeros(B, M, dtype=torch.bool)
    view = torch.randint(0, 3, (B, M))
    modality = torch.randint(0, 2, (B, M))
    g = torch.Generator()
    g.manual_seed(0)
    _, out_pad = _apply_study_corruption(
        elements,
        meta_add,
        pad,
        view,
        modality,
        {"random_element_dropout": 1.0},
        g,
    )
    # Drop probability is 0.25 per element inside the strategy; over 64*8=512
    # draws we expect roughly 128 drops ± 30.
    drops = (out_pad & ~pad).sum().item()
    assert 60 < drops < 250


def test_whole_view_family_dropout_drops_one_family_per_row():
    torch.manual_seed(1)
    B, M, d = 4, 6, 8
    elements = torch.randn(B, M, d)
    meta_add = torch.zeros(B, M, 12)
    pad = torch.zeros(B, M, dtype=torch.bool)
    # 2 distinct view_families per row, 2 elements each.
    view = torch.tensor([[0, 0, 1, 1, 2, 2]] * B)
    modality = torch.zeros(B, M, dtype=torch.long)
    g = torch.Generator()
    g.manual_seed(2)
    _, out_pad = _apply_study_corruption(
        elements,
        meta_add,
        pad,
        view,
        modality,
        {"whole_view_family_dropout": 1.0},
        g,
    )
    for b in range(B):
        dropped = out_pad[b] & ~pad[b]
        # At most M-1 elements dropped (guardrail), and at least some dropped.
        n_drop = int(dropped.sum().item())
        assert 0 < n_drop <= M - 1
        # All dropped elements share a single view family.
        if n_drop > 0:
            dropped_views = view[b, dropped].unique()
            assert dropped_views.numel() == 1


def test_corruption_guardrail_never_drops_all():
    """The guardrail ensures at least one element survives per row."""
    torch.manual_seed(2)
    B, M, d = 8, 3, 8
    elements = torch.randn(B, M, d)
    meta_add = torch.zeros(B, M, 12)
    pad = torch.zeros(B, M, dtype=torch.bool)
    # All elements share the same view and modality → whole_view_family would
    # try to drop all of them.
    view = torch.zeros(B, M, dtype=torch.long)
    modality = torch.zeros(B, M, dtype=torch.long)
    g = torch.Generator()
    g.manual_seed(3)
    _, out_pad = _apply_study_corruption(
        elements,
        meta_add,
        pad,
        view,
        modality,
        {"whole_view_family_dropout": 1.0},
        g,
    )
    # Every row must have at least one surviving element.
    surviving = (~out_pad).sum(dim=1)
    assert (surviving > 0).all(), f"some row had all elements dropped: {surviving}"
