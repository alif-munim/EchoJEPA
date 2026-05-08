"""Anchor loss behaves correctly and is differentiable on the online side."""

from __future__ import annotations

import torch

from src.models.echomv_jepa.full_joint_losses import (
    anchor_cosine_to_e100,
    anchor_loss,
    pool_tokens_mean,
)


def test_anchor_loss_zero_when_tokens_identical():
    torch.manual_seed(0)
    online = torch.randn(4, 50, 64)
    anchor = online.clone()
    loss = anchor_loss(online, anchor)
    assert torch.isfinite(loss)
    assert loss.item() < 1e-5


def test_anchor_loss_positive_when_tokens_differ():
    torch.manual_seed(1)
    online = torch.randn(4, 50, 64, requires_grad=True)
    anchor = torch.randn(4, 50, 64)
    loss = anchor_loss(online, anchor)
    assert loss.item() > 1e-3
    # Gradient must flow to online tokens.
    loss.backward()
    assert online.grad is not None
    assert online.grad.abs().sum() > 0


def test_anchor_cosine_diagnostic_matches():
    torch.manual_seed(2)
    online = torch.randn(4, 50, 64)
    anchor = online.clone()
    cos = anchor_cosine_to_e100(online, anchor)
    assert cos > 0.999


def test_pool_tokens_mean_shape():
    x = torch.randn(3, 20, 16)
    p = pool_tokens_mean(x)
    assert p.shape == (3, 16)
