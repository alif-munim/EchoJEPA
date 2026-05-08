"""EMA helpers for the clip-level encoder in full-joint EchoMV-JEPA.

Mirrors the V-JEPA clip-encoder EMA pattern in ``app/vjepa/train.py:287–315,
400–405, 800–809`` but as standalone callables so the full-joint trainer
can use them without importing the vanilla V-JEPA train script.
"""

from __future__ import annotations

from typing import Iterable, Iterator

import torch
import torch.nn as nn


def step_clip_ema(teacher: nn.Module, student: nn.Module, tau: float) -> None:
    """In-place EMA update: ``teacher <- tau * teacher + (1 - tau) * student``.

    Uses the fused ``_foreach`` primitives to stay fast at ViT-L scale.
    """
    tpars = [p.data for p in teacher.parameters()]
    spars = [p.data for p in student.parameters()]
    if not tpars:
        return
    torch._foreach_mul_(tpars, tau)
    torch._foreach_add_(tpars, spars, alpha=1.0 - tau)


def clip_ema_schedule(tau_start: float, tau_end: float, total_steps: int, start_step: int = 0) -> Iterator[float]:
    """Linear τ schedule from ``tau_start`` at step 0 to ``tau_end`` at
    ``total_steps``. Resume-aware: pass ``start_step`` when continuing a run.
    """
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive; got {total_steps}")
    delta = (tau_end - tau_start) / float(total_steps)
    for i in range(start_step, total_steps + 1):
        yield tau_start + i * delta


@torch.no_grad()
def ema_delta_norm(teacher: nn.Module, student: nn.Module) -> float:
    """Return L2 norm of (teacher - student) parameter-vector difference,
    for the ``ema_{clip,study}_delta`` diagnostic."""
    diffs = []
    for pt, ps in zip(teacher.parameters(), student.parameters()):
        diffs.append((pt.data - ps.data).float().flatten())
    if not diffs:
        return 0.0
    return float(torch.cat(diffs).norm().item())


def assert_no_grad(module: nn.Module, name: str = "module") -> None:
    """Helper for tests: every parameter must have ``requires_grad=False``."""
    bad = [n for n, p in module.named_parameters() if p.requires_grad]
    if bad:
        raise AssertionError(f"{name} has grad-enabled params: {bad[:8]}...")


def freeze(module: nn.Module) -> nn.Module:
    """Freeze every parameter in ``module``. Also puts the module in ``.eval()``."""
    for p in module.parameters():
        p.requires_grad_(False)
    module.eval()
    return module


def iter_trainable(*modules: nn.Module) -> Iterable[nn.Parameter]:
    """Yield trainable parameters from one or more modules."""
    for m in modules:
        for p in m.parameters():
            if p.requires_grad:
                yield p
