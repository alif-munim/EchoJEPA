"""Student + EMA teacher projectors for EchoSet-JEPA.

Two independent 2-layer MLPs (d_model -> d_hidden -> d_proj) with GELU. The
teacher is an EMA-updated copy of the student; see plan §4.4 and §4.5.
"""

from __future__ import annotations

import copy
from typing import Iterable

import torch
import torch.nn as nn


class StudyProjector(nn.Module):
    def __init__(self, d_model: int = 512, d_hidden: int = 1024, d_proj: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_proj),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EMAProjectorPair(nn.Module):
    """A student projector paired with an EMA teacher copy.

    The teacher's parameters never receive gradients; they are updated by
    :py:meth:`update_teacher` after every optimizer step.
    """

    def __init__(self, d_model: int = 512, d_hidden: int = 1024, d_proj: int = 256) -> None:
        super().__init__()
        self.student = StudyProjector(d_model, d_hidden, d_proj)
        self.teacher = copy.deepcopy(self.student)
        for p in self.teacher.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update_teacher(self, tau: float) -> None:
        """EMA update: ``teacher <- tau * teacher + (1 - tau) * student``."""
        for ts, ss in zip(self.teacher.parameters(), self.student.parameters()):
            ts.data.mul_(tau).add_(ss.data, alpha=1.0 - tau)

    def student_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.student(x)

    @torch.no_grad()
    def teacher_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.teacher(x)


def cosine_schedule(step: int, total: int, tau_start: float, tau_end: float) -> float:
    """Cosine EMA schedule over ``total`` steps; clamped to ``[tau_start, tau_end]``."""
    import math

    if total <= 0:
        return tau_end
    t = min(max(step, 0), total) / total
    return tau_end + 0.5 * (tau_start - tau_end) * (1.0 + math.cos(math.pi * t))


__all__ = ["StudyProjector", "EMAProjectorPair", "cosine_schedule"]
