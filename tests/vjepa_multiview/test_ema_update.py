"""Fix 5: momentum_update_ema_ copies online weights toward EMA with
the specified momentum, and EMA parameters never receive gradient.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from app.vjepa_multiview.train import momentum_update_ema_  # noqa: E402


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def _make_mlp(in_dim=8, out_dim=8):
    return nn.Sequential(nn.Linear(in_dim, 16), nn.GELU(), nn.Linear(16, out_dim))


def test_ema_no_op_when_online_equals_ema():
    online = _make_mlp()
    ema = copy.deepcopy(online)
    momentum_update_ema_(online, ema, m=0.99)
    for p_o, p_e in zip(online.parameters(), ema.parameters()):
        assert torch.allclose(p_o, p_e)


def test_ema_interpolates():
    """After one step with momentum m, ema_p should equal m*ema_p +
    (1-m)*online_p (applied to the *old* ema value)."""
    online = _make_mlp()
    ema = _make_mlp()  # different init
    old_ema = [p.detach().clone() for p in ema.parameters()]
    old_online = [p.detach().clone() for p in online.parameters()]
    m = 0.9
    momentum_update_ema_(online, ema, m=m)
    for p_new_ema, p_old_ema, p_old_online in zip(ema.parameters(), old_ema, old_online):
        expected = m * p_old_ema + (1 - m) * p_old_online
        assert torch.allclose(p_new_ema, expected, atol=1e-6)


def test_ema_params_never_require_grad_after_first_update():
    """Typical flow: ema copy is deepcopied, requires_grad is set to
    False on every param, then momentum_update_ema_ is called in a
    loop. This test pins that the update doesn't re-enable grad."""
    online = _make_mlp()
    ema = copy.deepcopy(online)
    for p in ema.parameters():
        p.requires_grad = False
    momentum_update_ema_(online, ema, m=0.99)
    for p in ema.parameters():
        assert p.requires_grad is False


def test_ddp_wrapped_online_works_via_module_attr():
    """When the online module is DDP-wrapped, the helper should strip
    the ``.module`` attribute so parameter iteration lines up."""

    class _FakeDDP(nn.Module):
        def __init__(self, inner: nn.Module):
            super().__init__()
            self.module = inner

    online = _make_mlp()
    ema = copy.deepcopy(online)
    for p in ema.parameters():
        p.requires_grad = False
    wrapped = _FakeDDP(online)
    # Perturb online weights so the update has something to do.
    with torch.no_grad():
        for p in online.parameters():
            p.add_(torch.randn_like(p) * 0.01)
    before = [p.detach().clone() for p in ema.parameters()]
    momentum_update_ema_(wrapped, ema, m=0.9)
    for p_before, p_after in zip(before, ema.parameters()):
        # Expect nontrivial update from the perturbation.
        assert not torch.allclose(p_before, p_after)


def test_no_backward_path_through_ema():
    """Forward through the EMA module, backprop a loss, and confirm
    no EMA parameter receives grad."""
    online = _make_mlp()
    ema = copy.deepcopy(online)
    for p in ema.parameters():
        p.requires_grad = False
    x = torch.randn(4, 8, requires_grad=True)
    y = ema(x)
    y.sum().backward()
    for p in ema.parameters():
        assert p.grad is None
