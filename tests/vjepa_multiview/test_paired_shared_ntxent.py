"""Fix 3: paired NT-Xent has one positive per sample by construction.

Each row's positive is its own target vector (diagonal of the
z_src @ t_tgt.T matrix). Negatives are other rows' target vectors.
This removes the v1 silent-zero failure mode where a batch might
contain no same-study duplicates.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from app.vjepa_multiview.train import _paired_shared_ntxent  # noqa: E402


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def _normed(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=-1)


def test_paired_shape_and_finite():
    B, D = 8, 16
    z = _normed(torch.randn(B, D))
    t = _normed(torch.randn(B, D))
    loss, diag = _paired_shared_ntxent(z, t, tau=0.1)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert diag["paired_shared_top1"].item() >= 0.0
    assert diag["paired_shared_top1"].item() <= 1.0


def test_paired_positive_dominates_when_identical():
    """When z_src == t_tgt exactly, each row's positive is trivially
    its own target → top1 = 1.0, loss ≈ log(1)."""
    B, D = 6, 16
    z = _normed(torch.randn(B, D))
    t = z.clone()  # identical
    loss, diag = _paired_shared_ntxent(z, t, tau=0.05)
    assert diag["paired_shared_top1"].item() == pytest.approx(1.0)
    # Loss should be close to zero (but not exactly — softmax over B
    # identical candidates with one positive row).
    assert loss.item() < 0.1


def test_paired_grad_flows_to_z_src():
    B, D = 4, 8
    z = _normed(torch.randn(B, D)).requires_grad_(True)
    t = _normed(torch.randn(B, D))
    loss, _ = _paired_shared_ntxent(z, t, tau=0.1)
    loss.backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()


def test_shape_mismatch_rejected():
    with pytest.raises(ValueError):
        _paired_shared_ntxent(_normed(torch.randn(4, 8)), _normed(torch.randn(4, 16)), tau=0.1)
    with pytest.raises(ValueError):
        _paired_shared_ntxent(_normed(torch.randn(4, 8, 1)), _normed(torch.randn(4, 8)), tau=0.1)


def test_works_when_no_same_study_duplicates():
    """v1 NT-Xent would silently zero when no same-study positives
    existed. v2 always has a valid positive (the diagonal)."""
    B, D = 5, 8
    z = _normed(torch.randn(B, D))
    t = _normed(torch.randn(B, D))
    loss, diag = _paired_shared_ntxent(z, t, tau=0.1)
    # If this were v1 NT-Xent with distinct study hashes, loss would
    # be zero. Here it's non-zero because positives exist by design.
    assert loss.item() > 0.0
