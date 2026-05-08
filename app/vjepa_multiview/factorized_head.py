"""Factorized projection head for Privileged Multi-View EchoJEPA.

Consumes pooled encoder output [B, D] and produces three disjoint slot
vectors via independent MLPs initialized from distinct truncated-normal
seeds. The slots are intended to carry disjoint signal:

    z_shared : phase-invariant, view-invariant study physiology
    z_phase  : cardiac-cycle / motion state
    z_view   : view-local residual (optional)

Loss assignment (see plan):
    L_same_study_align : z_shared only
    L_phase_rel        : z_phase only
    L_pair_view_pred   : [z_shared, z_phase]
    L_fused (sparse)   : z_shared only  (through a separate SharedProjector)

Every parameter receives grad on every step provided at least one of
lambda_pair / lambda_shared / lambda_phase is non-zero — the DDP reducer
therefore sees all three heads with the default hybrid config.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FactorizedProjectionHead(nn.Module):
    def __init__(
        self,
        embed_dim: int = 1024,
        hidden_dim: int = 1024,
        shared_dim: int = 256,
        phase_dim: int = 256,
        view_dim: int = 256,
        init_std: float = 0.02,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.hidden_dim = int(hidden_dim)
        self.shared_dim = int(shared_dim)
        self.phase_dim = int(phase_dim)
        self.view_dim = int(view_dim)

        self.shared_mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, shared_dim),
        )
        self.phase_mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, phase_dim),
        )
        self.view_mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, view_dim),
        )

        self._init_std = float(init_std)
        # Re-initialize each sub-MLP with its own RNG stream so the three
        # heads start from distinct truncated-normal draws. This is the
        # slot-separation guard: if all three heads start identical the
        # only signal pushing them apart is the loss-disjointness — which
        # is weak early in training.
        for idx, mlp in enumerate((self.shared_mlp, self.phase_mlp, self.view_mlp)):
            g = torch.Generator().manual_seed(0xFACE0000 + idx)
            for m in mlp.modules():
                if isinstance(m, nn.Linear):
                    w = torch.empty_like(m.weight)
                    # trunc_normal_ doesn't take a generator; approximate
                    # via normal_(generator=g) + clip to ±2σ.
                    w.normal_(0.0, self._init_std, generator=g)
                    w.clamp_(-2.0 * self._init_std, 2.0 * self._init_std)
                    m.weight.data.copy_(w)
                    if m.bias is not None:
                        m.bias.data.zero_()

    def forward(self, pooled: torch.Tensor) -> dict[str, torch.Tensor]:
        """pooled: [B, embed_dim]. Returns dict with z_shared/z_phase/z_view.

        Caller L2-normalizes as needed. No normalization applied here so
        the same head can serve both InfoNCE (needs normed) and SmoothL1
        regression (doesn't).
        """
        if pooled.dim() != 2:
            raise ValueError(f"pooled must be [B, D]; got {tuple(pooled.shape)}")
        if pooled.shape[-1] != self.embed_dim:
            raise ValueError(f"pooled last dim must be embed_dim={self.embed_dim}; " f"got {pooled.shape[-1]}")
        return {
            "z_shared": self.shared_mlp(pooled),
            "z_phase": self.phase_mlp(pooled),
            "z_view": self.view_mlp(pooled),
        }
