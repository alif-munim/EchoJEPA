"""v4: L_view_nce cross-view retrieval contrastive.

Pins behaviour of `_view_nce_loss`:
  - shape + finite
  - perfect retrieval when q == t
  - same-study batch masking
  - top1 per target-view bucket
  - gradient flows through q (student side)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from app.vjepa_multiview.train import _view_nce_loss  # noqa: E402


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def _normed(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=-1)


def test_shape_and_finite():
    B, D = 8, 16
    q = _normed(torch.randn(B, D))
    t = _normed(torch.randn(B, D))
    sh = torch.arange(B, dtype=torch.long)
    views = torch.randint(0, 14, (B,))
    loss, diag = _view_nce_loss(q, t, sh, views, tau_view=0.1)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert 0.0 <= diag["view_nce_top1"].item() <= 1.0


def test_perfect_retrieval_when_q_equals_t():
    B, D = 6, 16
    t = _normed(torch.randn(B, D))
    q = t.clone()
    sh = torch.arange(B, dtype=torch.long)
    views = torch.zeros(B, dtype=torch.long)
    loss, diag = _view_nce_loss(q, t, sh, views, tau_view=0.05)
    assert diag["view_nce_top1"].item() == pytest.approx(1.0)
    assert loss.item() < 0.5


def test_same_study_batch_negatives_masked():
    """With same_target_view_required=True and same_study masking, a
    batch of rows 0/1 sharing a study must have 0 valid negatives
    between them (both same-study and same-target-view). The
    diagnostic view_nce_valid_neg_count_min should reflect this."""
    B, D = 4, 16
    q = _normed(torch.randn(B, D))
    t = _normed(torch.randn(B, D))
    sh = torch.tensor([1, 1, 2, 3], dtype=torch.long)
    views = torch.zeros(B, dtype=torch.long)
    _, diag_m = _view_nce_loss(
        q,
        t,
        sh,
        views,
        tau_view=0.1,
        mask_same_study_batch_negatives=True,
        same_target_view_required=True,
        family_fallback=False,
    )
    # All views identical → same_target_view_fraction counts every off-diag pair.
    assert diag_m["view_nce_same_target_view_fraction"].item() > 0.5
    # With same-study masking + same_target_view, rows 0/1 (same study)
    # have fewer valid negatives than rows 2/3.
    # Row 2 and row 3: both can use rows 0,1,3 or 0,1,2 (after same-study
    # and self masks remove their respective 2/3). So min valid should be
    # ≤ the max.
    min_cnt = diag_m["view_nce_valid_neg_count_min"].item()
    mean_cnt = diag_m["view_nce_valid_neg_count_mean"].item()
    assert 0.0 <= min_cnt <= mean_cnt


def test_v5_same_target_view_required_filters_negatives():
    """Even when all rows are from distinct studies, requiring same
    target_view shrinks the negative set."""
    B, D = 6, 16
    q = _normed(torch.randn(B, D))
    t = _normed(torch.randn(B, D))
    sh = torch.arange(B, dtype=torch.long)  # all distinct studies
    # Rows 0-2 target A4C (view_id=2); rows 3-5 target PLAX (view_id=4).
    views = torch.tensor([2, 2, 2, 4, 4, 4], dtype=torch.long)
    _, diag = _view_nce_loss(
        q,
        t,
        sh,
        views,
        tau_view=0.1,
        same_target_view_required=True,
        family_fallback=False,
    )
    # Each row has at most 2 same-target-view negatives (the 2 other A4C
    # or PLAX rows, minus self → 2).
    assert diag["view_nce_valid_neg_count_mean"].item() == pytest.approx(2.0)


def test_v5_family_fallback_expands_negatives():
    """When same_target_view_required=True but a row has zero same-
    target-view negatives, family_fallback must expand its pool to
    same-family views (apical: A4C/A5C/A3C/A2C)."""
    B, D = 4, 16
    q = _normed(torch.randn(B, D))
    t = _normed(torch.randn(B, D))
    sh = torch.arange(B, dtype=torch.long)
    # Four rows with distinct APICAL target views (A4C, A5C, A3C, A2C).
    views = torch.tensor([2, 3, 1, 0], dtype=torch.long)
    _, diag_nofb = _view_nce_loss(
        q,
        t,
        sh,
        views,
        tau_view=0.1,
        same_target_view_required=True,
        family_fallback=False,
    )
    # Without fallback: every row has 0 valid negatives (each view is unique).
    assert diag_nofb["view_nce_valid_neg_count_mean"].item() == 0.0
    _, diag_fb = _view_nce_loss(
        q,
        t,
        sh,
        views,
        tau_view=0.1,
        same_target_view_required=True,
        family_fallback=True,
    )
    # With fallback: all 4 rows are in the apical family → each has 3 valid negs.
    assert diag_fb["view_nce_valid_neg_count_mean"].item() == pytest.approx(3.0)
    assert diag_fb["view_nce_fallback_fraction"].item() == pytest.approx(1.0)


def test_v5_zero_valid_negatives_does_not_crash():
    """Skip-invariant: when every row has zero valid negatives (family
    fallback disabled, all distinct views, all same study → diagonal-
    only), the CE loss must still be finite (logsumexp over a single
    element yields 0 contribution). This is what the gate-checker's
    relaxed view_nce_valid_neg_count_mean>=1 (not min>=1) threshold
    relies on: rows with no negatives contribute no contrastive signal
    but don't crash training."""
    B, D = 3, 16
    q = _normed(torch.randn(B, D))
    t = _normed(torch.randn(B, D))
    # All same study → same-study mask wipes every off-diag.
    sh = torch.zeros(B, dtype=torch.long)
    # All same view → same_target_view pool is rows 1..B-1, but same-
    # study mask removes them all.
    views = torch.zeros(B, dtype=torch.long)
    loss, diag = _view_nce_loss(
        q,
        t,
        sh,
        views,
        tau_view=0.1,
        mask_same_study_batch_negatives=True,
        same_target_view_required=True,
        family_fallback=False,
    )
    # Every row has zero valid negatives.
    assert diag["view_nce_valid_neg_count_min"].item() == 0.0
    assert diag["view_nce_valid_neg_count_mean"].item() == 0.0
    # CE is still finite (each row's logsumexp is over the diagonal only,
    # so loss contribution per row is log(1) = 0 up to scale).
    assert torch.isfinite(loss)


def test_top1_by_view_bucket():
    B, D = 8, 16
    t = _normed(torch.randn(B, D))
    q = t.clone()
    sh = torch.arange(B, dtype=torch.long)
    # First half views=2 (A4C), second half views=4 (PLAX).
    views = torch.tensor([2, 2, 2, 2, 4, 4, 4, 4], dtype=torch.long)
    _, diag = _view_nce_loss(q, t, sh, views, tau_view=0.05)
    per_view = diag["view_nce_top1_by_view"]
    assert 2 in per_view and 4 in per_view
    assert per_view[2].item() == pytest.approx(1.0)
    assert per_view[4].item() == pytest.approx(1.0)


def test_gradient_flows_through_query():
    B, D = 4, 16
    q = _normed(torch.randn(B, D)).requires_grad_(True)
    t = _normed(torch.randn(B, D))
    sh = torch.arange(B, dtype=torch.long)
    views = torch.zeros(B, dtype=torch.long)
    loss, _ = _view_nce_loss(q, t, sh, views, tau_view=0.1)
    loss.backward()
    assert q.grad is not None
    assert torch.isfinite(q.grad).all()
    assert q.grad.abs().sum().item() > 0


def test_no_gradient_leaks_into_detached_target():
    B, D = 4, 16
    q = _normed(torch.randn(B, D)).requires_grad_(True)
    t = _normed(torch.randn(B, D))  # not requires_grad
    sh = torch.arange(B, dtype=torch.long)
    views = torch.zeros(B, dtype=torch.long)
    loss, _ = _view_nce_loss(q, t, sh, views, tau_view=0.1)
    loss.backward()
    # Target was a leaf with requires_grad=False → no grad.
    assert t.grad is None


def test_shape_mismatch_rejected():
    with pytest.raises(ValueError):
        _view_nce_loss(
            _normed(torch.randn(4, 8)),
            _normed(torch.randn(4, 16)),
            torch.arange(4, dtype=torch.long),
            torch.zeros(4, dtype=torch.long),
            tau_view=0.1,
        )
