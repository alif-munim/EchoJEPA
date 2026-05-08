"""Teacher + anchor networks must have requires_grad=False."""

from __future__ import annotations

import torch.nn as nn

from src.models.echomv_jepa.clip_ema import assert_no_grad, freeze


def test_assert_no_grad_passes_on_frozen():
    mod = nn.Linear(4, 4)
    freeze(mod)
    # No exception.
    assert_no_grad(mod, "mod")


def test_assert_no_grad_fails_on_trainable():
    mod = nn.Linear(4, 4)  # grads on by default
    try:
        assert_no_grad(mod, "mod")
        raise RuntimeError("should have failed")
    except AssertionError:
        pass


def test_freeze_sets_eval():
    mod = nn.Linear(4, 4)
    freeze(mod)
    assert not mod.training
    for p in mod.parameters():
        assert not p.requires_grad
