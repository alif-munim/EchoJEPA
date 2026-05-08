"""Shared projector for L_fused alignment.

Small MLP mapping the student's z_shared slot to the fused-teacher
target space. Used only on the sparse `L_fused` path (Bernoulli(p_fused)
steps). The target side comes from MultiViewTeacherFusion.

    L_fused = SmoothL1(P_shared(z_shared),  sg(t_fused_shared))

Both source and target live in `fused_dim`. When fused_dim == shared_dim
this collapses to an almost-identity adapter, but a non-trivial MLP lets
the student's z_shared live in a different coordinate system from the
fused target without being forced into alignment directly through the
encoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SharedProjector(nn.Module):
    def __init__(
        self,
        shared_dim: int = 256,
        fused_dim: int = 256,
        hidden_dim: int = 512,
        init_std: float = 0.02,
    ):
        super().__init__()
        self.shared_dim = int(shared_dim)
        self.fused_dim = int(fused_dim)
        self.hidden_dim = int(hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(shared_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, fused_dim),
        )

        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=init_std)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, z_shared: torch.Tensor) -> torch.Tensor:
        if z_shared.dim() != 2:
            raise ValueError(f"z_shared must be [B, D]; got {tuple(z_shared.shape)}")
        if z_shared.shape[-1] != self.shared_dim:
            raise ValueError(f"z_shared last dim must be {self.shared_dim}; got {z_shared.shape[-1]}")
        return self.mlp(z_shared)
