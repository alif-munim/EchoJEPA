"""Fix 5b: _mean_shared_fused_target is deterministic + safe.

Shape, masking, gradient-free target, sensitivity to clip content,
all-invalid fallback.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from app.vjepa_multiview.factorized_head import FactorizedProjectionHead  # noqa: E402
from app.vjepa_multiview.train import _mean_shared_fused_target  # noqa: E402


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def _make_head(embed_dim=16, shared_dim=16):
    head = FactorizedProjectionHead(
        embed_dim=embed_dim,
        hidden_dim=32,
        shared_dim=shared_dim,
        phase_dim=shared_dim,
        view_dim=shared_dim,
    )
    # Freeze — this is how the EMA head is configured in training.
    for p in head.parameters():
        p.requires_grad = False
    return head


def test_output_shape():
    head = _make_head()
    B, N, D = 3, 4, 16
    pooled = torch.randn(B, N, D)
    mask = torch.ones(B, N, dtype=torch.bool)
    out, _ = _mean_shared_fused_target(pooled, mask, head)
    assert out.shape == (B, 16)
    assert torch.isfinite(out).all()


def test_mask_ignores_invalid_clips():
    """Invalid clips should not contribute to the mean."""
    head = _make_head()
    B, N, D = 2, 3, 16
    pooled = torch.randn(B, N, D)
    mask_full = torch.ones(B, N, dtype=torch.bool)
    mask_partial = mask_full.clone()
    mask_partial[0, 2] = False  # drop clip 2 for row 0
    out_full, _ = _mean_shared_fused_target(pooled, mask_full, head)
    out_partial, _ = _mean_shared_fused_target(pooled, mask_partial, head)
    # Row 1's mask is identical -> outputs for row 1 must match exactly.
    assert torch.allclose(out_full[1], out_partial[1])
    # Row 0's mask differs -> outputs must differ.
    assert not torch.allclose(out_full[0], out_partial[0])


def test_no_gradient_into_factorized_head_ema():
    head = _make_head()
    B, N, D = 2, 3, 16
    pooled = torch.randn(B, N, D, requires_grad=False)
    mask = torch.ones(B, N, dtype=torch.bool)
    out, _ = _mean_shared_fused_target(pooled, mask, head)
    # Output is detached by construction — no grad-fn.
    assert out.requires_grad is False
    assert out.grad_fn is None
    # head params must remain with no grad after a downstream backward.
    y = (out * 0.0).sum()  # need a leaf -> no backward possible; skip.
    # Verify all head parameters still have requires_grad=False.
    for p in head.parameters():
        assert p.requires_grad is False
        assert p.grad is None


def test_changing_valid_clip_changes_output():
    head = _make_head()
    B, N, D = 1, 2, 16
    pooled_a = torch.randn(B, N, D)
    pooled_b = pooled_a.clone()
    pooled_b[0, 0] += torch.randn(D)  # perturb first valid clip
    mask = torch.ones(B, N, dtype=torch.bool)
    out_a, _ = _mean_shared_fused_target(pooled_a, mask, head)
    out_b, _ = _mean_shared_fused_target(pooled_b, mask, head)
    assert not torch.allclose(out_a, out_b)


def test_all_invalid_row_returns_zero_vector():
    head = _make_head()
    B, N, D = 2, 3, 16
    pooled = torch.randn(B, N, D)
    mask = torch.ones(B, N, dtype=torch.bool)
    mask[0] = False  # row 0 has zero valid clips
    out, diag = _mean_shared_fused_target(pooled, mask, head)
    # Safe fallback: row 0 should be finite (we divide by clamped-to-1
    # denom, so 0 numerator / 1 = 0).
    assert torch.isfinite(out).all()
    assert out[0].abs().sum().item() == 0.0
    assert diag["fused_any_row_invalid"].item() == 1.0


def test_diagnostics_reflect_valid_counts():
    head = _make_head()
    B, N, D = 3, 4, 16
    pooled = torch.randn(B, N, D)
    mask = torch.tensor(
        [
            [True, True, True, True],
            [True, True, False, False],
            [True, False, False, False],
        ],
        dtype=torch.bool,
    )
    _, diag = _mean_shared_fused_target(pooled, mask, head)
    assert diag["fused_valid_views_mean"].item() == pytest.approx((4 + 2 + 1) / 3.0)
    assert diag["fused_valid_views_min"].item() == pytest.approx(1.0)


def test_shape_errors():
    head = _make_head()
    with pytest.raises(ValueError):
        _mean_shared_fused_target(torch.randn(2, 16), torch.ones(2, 3, dtype=torch.bool), head)
    with pytest.raises(ValueError):
        _mean_shared_fused_target(
            torch.randn(2, 3, 16),
            torch.ones(2, 4, dtype=torch.bool),
            head,
        )
