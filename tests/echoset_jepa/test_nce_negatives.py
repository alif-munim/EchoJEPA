"""Prioritized NCE-negative tests (plan §5.2)."""

from __future__ import annotations

import torch

from app.echoset_jepa.train import _prioritized_neg_pool


def test_excludes_same_study_off_targets():
    view = torch.tensor([0, 0, 1, 1])
    mod = torch.tensor([0, 0, 0, 0])
    phase = torch.tensor([0, 0, 0, 0])
    study = torch.tensor([1, 1, 2, 2])   # rows 0,1 share study; 2,3 share study

    mask, _, _ = _prioritized_neg_pool(view, mod, phase, study, k_min=0)

    # Diagonal is positive (always True)
    assert torch.diag(mask).all()
    # Same-study off-diag is False (row 0, col 1) and (row 2, col 3)
    assert not mask[0, 1]
    assert not mask[1, 0]
    assert not mask[2, 3]
    assert not mask[3, 2]


def test_priority_1_used_when_sufficient():
    # 8 elements, all same view+modality+phase. Priority_1 is the whole batch.
    N = 8
    view = torch.zeros(N, dtype=torch.long)
    mod = torch.zeros(N, dtype=torch.long)
    phase = torch.zeros(N, dtype=torch.long)
    study = torch.arange(N, dtype=torch.long)  # every row is its own study

    mask, _, diag = _prioritized_neg_pool(view, mod, phase, study, k_min=4)
    # Every off-diag is True (same view, different study)
    off = mask & ~torch.eye(N, dtype=torch.bool)
    assert off.sum() == N * (N - 1)
    assert diag["fallback_fraction"] == 0.0


def test_falls_back_when_same_view_pool_is_empty():
    # Row 0 has a view that no other row shares; must fall back to same_modality
    view = torch.tensor([99, 1, 1, 1])
    mod = torch.tensor([0, 0, 0, 0])
    phase = torch.tensor([0, 0, 0, 0])
    study = torch.arange(4, dtype=torch.long)
    mask, same_v, diag = _prioritized_neg_pool(view, mod, phase, study, k_min=4)
    # Row 0 has same_view_count == 0
    assert same_v[0].item() == 0
    # But still has 3 valid negatives via same_modality fallback
    assert (mask[0] & ~torch.eye(4, dtype=torch.bool)[0]).sum().item() == 3
    assert diag["fallback_fraction"] > 0


def test_diagonal_always_positive():
    view = torch.randint(0, 5, (16,))
    mod = torch.randint(0, 5, (16,))
    phase = torch.randint(0, 5, (16,))
    study = torch.randint(0, 16, (16,))
    mask, _, _ = _prioritized_neg_pool(view, mod, phase, study, k_min=4)
    assert torch.diag(mask).all()
