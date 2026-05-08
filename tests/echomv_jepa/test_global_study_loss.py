"""Global study loss is finite and behaves like cosine-regress."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.echomv_jepa.full_joint_losses import (
    LossRamp,
    LossWeights,
    assemble_total_loss,
    global_study_loss,
    single_view_to_study_loss,
)


class _Ident(nn.Module):
    def forward(self, x):
        return x


def test_global_study_loss_is_zero_when_inputs_match():
    torch.manual_seed(0)
    h = torch.randn(4, 16)
    z = h.clone()
    loss = global_study_loss(h, z, _Ident(), _Ident())
    assert torch.isfinite(loss)
    assert loss.item() < 1e-5


def test_global_study_loss_positive_when_inputs_differ():
    torch.manual_seed(1)
    h = torch.randn(4, 16)
    z = torch.randn(4, 16)
    loss = global_study_loss(h, z, _Ident(), _Ident())
    assert loss.item() > 1e-3


def test_single_view_to_study_loss_alias():
    torch.manual_seed(2)
    h = torch.randn(4, 16)
    z = torch.randn(4, 16)
    a = global_study_loss(h, z, _Ident(), _Ident())
    b = single_view_to_study_loss(h, z, _Ident(), _Ident())
    assert torch.allclose(a, b)


def test_assemble_total_loss_weighted_sum():  # noqa: D401
    torch.manual_seed(3)
    losses = {
        "clip": torch.tensor(1.0),
        "study": torch.tensor(2.0),
        "nce": torch.tensor(3.0),
        "cov": torch.tensor(4.0),
        "anchor": torch.tensor(5.0),
        "sv": torch.tensor(6.0),
    }
    weights = LossWeights()
    total, applied = assemble_total_loss(losses, weights)
    # 1.0*1 + 0.1*2 + 0.005*3 + 0.001*4 + 0.05*5 + 0.02*6
    expected = 1.0 + 0.2 + 0.015 + 0.004 + 0.25 + 0.12
    assert abs(total.item() - expected) < 1e-5
    # Effective weights surfaced back to the caller.
    assert applied["lambda_clip_t"] == 1.0
    assert applied["lambda_study_t"] == 0.1
    assert applied["lambda_anchor_t"] == 0.05
    assert applied["lambda_sv_t"] == 0.02


def test_loss_ramp_linear_then_saturates():
    ramp = LossRamp(target_weight=0.3, warmup_steps=100)
    assert ramp.value_at(0) == 0.0
    assert abs(ramp.value_at(50) - 0.15) < 1e-6
    assert ramp.value_at(100) == 0.3
    assert ramp.value_at(200) == 0.3
