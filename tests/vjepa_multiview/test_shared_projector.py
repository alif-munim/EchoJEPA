"""Shape + grad flow for SharedProjector."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from app.vjepa_multiview.shared_projector import SharedProjector  # noqa: E402


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def test_forward_shape():
    p = SharedProjector(shared_dim=32, fused_dim=16, hidden_dim=64)
    z = torch.randn(4, 32)
    out = p(z)
    assert out.shape == (4, 16)
    assert torch.isfinite(out).all()


def test_rejects_wrong_dim():
    p = SharedProjector(shared_dim=32, fused_dim=16, hidden_dim=64)
    with pytest.raises(ValueError):
        p(torch.randn(4, 64))


def test_grad_flows():
    p = SharedProjector(shared_dim=32, fused_dim=16, hidden_dim=64)
    z = torch.randn(4, 32, requires_grad=True)
    p(z).sum().backward()
    assert z.grad is not None
    for name, param in p.named_parameters():
        assert param.grad is not None, f"no grad on {name}"
