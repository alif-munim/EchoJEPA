"""motion_delta_loss — same-view gating + zero-loss proxy."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from app.vjepa_multiview.token_relational_head import (  # noqa: E402
    DeltaTargetProjector,
    MotionDeltaHead,
)
from app.vjepa_multiview.token_relational_loss import motion_delta_loss  # noqa: E402


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def _fresh_heads(D, delta_dim):
    head = MotionDeltaHead(embed_dim=D, delta_dim=delta_dim)
    proj = DeltaTargetProjector(embed_dim=D, delta_dim=delta_dim)
    return head, proj


def test_delta_loss_all_same_view_positive():
    B, K, D = 4, 6, 32
    delta_dim = 16
    head, proj = _fresh_heads(D, delta_dim)
    z = torch.randn(B, K, D, requires_grad=True)
    h_a = torch.randn(B, K, D)
    h_pos = torch.randn(B, K, D)
    h_neg = torch.randn(B, K, D)
    src = torch.zeros(B, dtype=torch.long)
    tgt = torch.zeros(B, dtype=torch.long)  # exact same view
    dphi = torch.rand(B)
    out = motion_delta_loss(
        z, h_a, h_pos, h_neg, src, tgt, dphi, head, proj,
        tau=0.1, lambda_l1=1.0, lambda_nce=1.0,
    )
    assert out["delta_valid_rows"].item() == float(B)
    assert torch.isfinite(out["delta_loss"])
    assert out["delta_l1"].item() >= 0.0


def test_delta_loss_zero_when_no_same_view_rows():
    B, K, D = 4, 4, 16
    delta_dim = 8
    head, proj = _fresh_heads(D, delta_dim)
    z = torch.randn(B, K, D)
    h_a = torch.randn(B, K, D)
    h_pos = torch.randn(B, K, D)
    h_neg = torch.randn(B, K, D)
    src = torch.zeros(B, dtype=torch.long)  # A2C
    tgt = torch.ones(B, dtype=torch.long) * 4  # PLAX (cross-family)
    dphi = torch.rand(B)
    out = motion_delta_loss(
        z, h_a, h_pos, h_neg, src, tgt, dphi, head, proj,
        tau=0.1, lambda_l1=1.0, lambda_nce=1.0,
    )
    assert out["delta_valid_rows"].item() == 0.0
    assert float(out["delta_loss"].item()) == 0.0
    # Params still get a gradient via the zero-loss proxy (so DDP reducer is
    # stable across steps).
    out["delta_loss"].backward()
    any_grad = False
    for p in head.parameters():
        if p.grad is not None:
            any_grad = True
    assert any_grad, "motion_delta_head params must receive grad via zero-loss proxy"


def test_delta_loss_partial_same_view():
    B, K, D = 4, 4, 16
    delta_dim = 8
    head, proj = _fresh_heads(D, delta_dim)
    z = torch.randn(B, K, D)
    h_a = torch.randn(B, K, D)
    h_pos = torch.randn(B, K, D)
    h_neg = torch.randn(B, K, D)
    src = torch.tensor([0, 4, 0, 6], dtype=torch.long)
    tgt = torch.tensor([0, 0, 0, 0], dtype=torch.long)  # rows 0,2 same-view
    dphi = torch.rand(B)
    out = motion_delta_loss(
        z, h_a, h_pos, h_neg, src, tgt, dphi, head, proj,
    )
    assert out["delta_valid_rows"].item() == 2.0
    assert torch.isfinite(out["delta_loss"])


def test_delta_loss_teacher_tokens_are_detached():
    """h_* tensors must be detached inside the function; caller passes
    .detach() tensors but the function itself should not rely on them
    carrying grad."""
    B, K, D = 4, 4, 16
    delta_dim = 8
    head, proj = _fresh_heads(D, delta_dim)
    z = torch.randn(B, K, D, requires_grad=True)
    # Teacher tensors constructed with requires_grad=True to check we
    # don't accidentally propagate gradient back through them.
    h_a = torch.randn(B, K, D, requires_grad=True)
    h_pos = torch.randn(B, K, D, requires_grad=True)
    h_neg = torch.randn(B, K, D, requires_grad=True)
    src = torch.zeros(B, dtype=torch.long)
    tgt = torch.zeros(B, dtype=torch.long)
    dphi = torch.rand(B)
    out = motion_delta_loss(
        z, h_a, h_pos, h_neg, src, tgt, dphi, head, proj,
    )
    out["delta_loss"].backward()
    # None of the teacher tensors should have grad.
    assert h_a.grad is None
    assert h_pos.grad is None
    assert h_neg.grad is None
