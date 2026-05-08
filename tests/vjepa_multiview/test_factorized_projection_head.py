"""Shape + gradient flow for FactorizedProjectionHead."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from app.vjepa_multiview.factorized_head import FactorizedProjectionHead  # noqa: E402


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def _make_head(embed_dim=64, shared_dim=32, phase_dim=32, view_dim=32):
    return FactorizedProjectionHead(
        embed_dim=embed_dim,
        hidden_dim=128,
        shared_dim=shared_dim,
        phase_dim=phase_dim,
        view_dim=view_dim,
    )


def test_forward_shapes():
    head = _make_head()
    pooled = torch.randn(4, 64)
    out = head(pooled)
    assert out["z_shared"].shape == (4, 32)
    assert out["z_phase"].shape == (4, 32)
    assert out["z_view"].shape == (4, 32)


def test_bad_input_shape_rejected():
    head = _make_head()
    with pytest.raises(ValueError):
        head(torch.randn(4, 64, 3))  # [B, D] required
    with pytest.raises(ValueError):
        head(torch.randn(4, 128))  # wrong embed_dim


def test_all_three_slots_receive_grad_when_all_used():
    head = _make_head()
    pooled = torch.randn(4, 64, requires_grad=False)
    out = head(pooled)
    loss = out["z_shared"].sum() + out["z_phase"].sum() + out["z_view"].sum()
    loss.backward()
    # Every trainable Linear should have non-None grad.
    for name, p in head.named_parameters():
        assert p.grad is not None, f"no grad on {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad on {name}"


def test_shared_only_loss_does_not_touch_phase_or_view_heads():
    """Loss that only touches z_shared must not populate z_phase/z_view grads.

    This is the guard for the plan's disjoint-slot invariant: if
    L_same_study_align only operates on z_shared, the phase-head and
    view-head parameters should see grad only via the losses that
    address them — not leak through the shared head.
    """
    head = _make_head()
    pooled = torch.randn(4, 64)
    out = head(pooled)
    loss = out["z_shared"].sum()
    loss.backward()
    for n, p in head.shared_mlp.named_parameters():
        assert p.grad is not None, f"shared head missing grad on {n}"
    for n, p in head.phase_mlp.named_parameters():
        assert p.grad is None, f"phase head leaked grad on {n}"
    for n, p in head.view_mlp.named_parameters():
        assert p.grad is None, f"view head leaked grad on {n}"


def test_heads_diverge_at_init():
    """Different RNG seeds per head — first-layer weights must differ."""
    head = _make_head()
    w_shared = head.shared_mlp[0].weight
    w_phase = head.phase_mlp[0].weight
    w_view = head.view_mlp[0].weight
    assert not torch.allclose(w_shared, w_phase)
    assert not torch.allclose(w_shared, w_view)
    assert not torch.allclose(w_phase, w_view)
