"""Control E — Identity / no-cross-clip encoder (plan §7).

Stage-2 is replaced with per-element LayerNorm + mean pool over elements. No
attention. Downstream probe trained on the pooled vector.

Floor for a K-averaged study representation at matched probe capacity.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from evals.echoset_jepa_probe.probe import StudyProbeHead


class IdentityStudyEncoder(nn.Module):
    """Mean-pool over context elements after per-element LayerNorm."""

    def __init__(self, d_clip: int = 1024) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(d_clip)

    def forward(self, ctx_elements: torch.Tensor, ctx_pad_mask: torch.Tensor) -> torch.Tensor:
        # (B, M, d_clip) → (B, d_clip)
        x = self.ln(ctx_elements)
        mask = (~ctx_pad_mask).float().unsqueeze(-1)        # (B, M, 1) True=valid
        num = (x * mask).sum(dim=1)
        den = mask.sum(dim=1).clamp(min=1.0)
        return num / den


def build_control_e(d_clip: int, n_targets: int) -> nn.Module:
    class E(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = IdentityStudyEncoder(d_clip=d_clip)
            self.head = StudyProbeHead(d_clip, n_targets)

        def forward(self, ctx_elements, ctx_pad_mask):
            return self.head(self.enc(ctx_elements, ctx_pad_mask))

    return E()


__all__ = ["IdentityStudyEncoder", "build_control_e"]
