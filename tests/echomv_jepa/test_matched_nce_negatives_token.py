"""Test matched NCE behavior on synthetic token-mode batches.

This exercises the existing ``prioritized_neg_pool`` + ``matched_nce``
composition (already unit-tested in test_matched_negatives.py for the pooled
path) specifically in the token-mode shapes used by Arm A:

  - targets flatten to (N, d_proj) where N = sum over batch of valid M_tgt
  - same-study off-targets must always be excluded from the negative pool
  - fallback_fraction is logged when < 50% of rows can find same-(v, m, p) negs

We don't re-test the math of matched_nce — that's covered in the pooled tests.
We only assert the same-study-exclusion invariant and the fallback_fraction
plumbing as seen from the token step's neg_mask diagnostics.
"""

from __future__ import annotations

import torch

from src.models.echomv_jepa import matched_nce, prioritized_neg_pool


def test_same_study_off_targets_excluded_in_token_shape():
    # Simulate a token-mode flat tgt batch: 3 studies, 2 targets each = N=6.
    view = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    modality = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.long)
    phase = torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.long)
    study = torch.tensor([7, 7, 11, 11, 13, 13], dtype=torch.long)
    mask, diag = prioritized_neg_pool(view, modality, phase, study, k_min=1)
    assert mask.shape == (6, 6)
    # Diagonal = True (positive).
    assert bool(mask.diagonal().all())
    # Same-study off-diagonals MUST be False. Row 0 is study 7; col 1 is also
    # study 7 (same study off-target) and must not appear in its negative set.
    assert bool(mask[0, 1].item()) is False
    assert bool(mask[1, 0].item()) is False
    assert bool(mask[2, 3].item()) is False
    # Cross-study, same (view, modality, phase) should be True where it
    # exists. Here row 0 (view=0 mod=0 phase=0 study=7) has no cross-study
    # match at priority 1 (study 11 has view=1, study 13 has view=2), so the
    # fallback ladder should open. Check that at least one negative is valid.
    assert bool((mask[0] & ~torch.eye(6, dtype=torch.bool)[0]).any())
    # fallback_fraction is a float in [0, 1].
    assert 0.0 <= diag["fallback_fraction"] <= 1.0


def test_matched_nce_gradient_flows_and_is_finite():
    torch.manual_seed(0)
    N, D = 12, 16
    h = torch.randn(N, D, requires_grad=True)
    z = torch.randn(N, D)
    mask = torch.ones(N, N, dtype=torch.bool)
    # Same-study pairs 0-1, 2-3, 4-5 excluded.
    for i in (0, 2, 4):
        mask[i, i + 1] = False
        mask[i + 1, i] = False
    loss = matched_nce(h, z, mask, tau=0.1)
    assert torch.isfinite(loss)
    loss.backward()
    assert h.grad is not None
    assert torch.isfinite(h.grad).all()
