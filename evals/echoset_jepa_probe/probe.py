"""Shared study-level probe head for all EchoSet-JEPA downstream evals (plan §8.6).

Same architecture for every method (EchoSet-JEPA, Controls B1/B2/D/E). Control
A is per-clip + arithmetic mean and uses its own existing d=1 attentive probe.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class StudyProbeHead(nn.Module):
    """``LN -> Linear(d_model -> 256) -> GELU -> Linear(256 -> n_targets)``.

    Applied to:
      - EchoSet-JEPA: ``[STUDY]`` token output
      - Control B1: 2-layer late-fusion pool output
      - Control B2: ``[STUDY]`` token of its own study transformer
      - Control D: ``[STUDY]`` token (metadata-only)
      - Control E: pooled-over-elements vector
    """

    def __init__(self, d_in: int, n_targets: int, d_hidden: int = 256, dropout: float = 0.0) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(d_in)
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_hidden, n_targets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.drop(self.act(self.fc1(self.ln(x)))))


__all__ = ["StudyProbeHead"]
