"""CrossClipAdapter must be an exact identity at init (gamma=0)."""

from __future__ import annotations

import torch

from src.models.mcc_jepa import CrossClipAdapter


def test_adapter_gamma_initialized_to_zero():
    a = CrossClipAdapter(embed_dim=64, num_heads=4)
    assert float(a.gamma.item()) == 0.0


def test_adapter_is_identity_at_init():
    torch.manual_seed(0)
    a = CrossClipAdapter(embed_dim=64, num_heads=4)
    pred = torch.randn(2, 12, 64)
    src = torch.randn(2, 50, 64)
    out = a(pred, src)
    assert torch.allclose(out, pred, atol=0.0)


def test_adapter_source_proj_when_dims_differ():
    torch.manual_seed(0)
    a = CrossClipAdapter(embed_dim=64, num_heads=4, source_proj_dim=128)
    pred = torch.randn(2, 12, 64)
    src = torch.randn(2, 50, 128)
    out = a(pred, src)
    # identity at gamma=0 regardless of source dim
    assert torch.allclose(out, pred, atol=0.0)


def test_adapter_diverges_from_identity_when_gamma_nonzero():
    torch.manual_seed(0)
    a = CrossClipAdapter(embed_dim=64, num_heads=4)
    with torch.no_grad():
        a.gamma.fill_(0.5)
    pred = torch.randn(2, 12, 64)
    src = torch.randn(2, 50, 64)
    out = a(pred, src)
    assert not torch.allclose(out, pred)
