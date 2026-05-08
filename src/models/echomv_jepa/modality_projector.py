"""Per-modality EMA projector pair (EchoMV-JEPA Stage-1m).

One ``EMAProjectorPair`` per modality id. Routes each row by its target-slot
modality id. A fallback projector handles any modality id out of range.

Falls back to a single shared projector when ``num_modalities == 1`` — which
matches the Stage-1 / Stage-0 behavior. This keeps the code path identical;
Stage-1m is just "instantiate with num_modalities = len(MODALITY_VOCAB)".
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.study_projectors import EMAProjectorPair


class ModalityProjectorPair(nn.Module):
    """A bank of EMA projector pairs, indexed by modality id.

    Parameters
    ----------
    num_modalities : int
        Number of modality ids (``len(MODALITY_VOCAB)`` in Stage-1m; 1 in Stage-1).
    d_model, d_hidden, d_proj : int
        Per-pair dims, matching ``EMAProjectorPair``.
    """

    def __init__(
        self,
        num_modalities: int = 1,
        d_model: int = 512,
        d_hidden: int = 1024,
        d_proj: int = 256,
    ) -> None:
        super().__init__()
        if num_modalities < 1:
            raise ValueError("num_modalities must be >= 1")
        self.num_modalities = num_modalities
        self.d_proj = d_proj
        self.pairs = nn.ModuleList(
            [EMAProjectorPair(d_model=d_model, d_hidden=d_hidden, d_proj=d_proj) for _ in range(num_modalities)]
        )

    # ----- routing helpers ------------------------------------------------

    def _route(self, x: torch.Tensor, modality_ids: torch.Tensor, *, use_teacher: bool) -> torch.Tensor:
        """Apply the per-modality projector to each row of ``x``.

        x             : (N, d_model)
        modality_ids  : (N,) long
        returns       : (N, d_proj)
        """
        out = x.new_empty(x.shape[0], self.d_proj)
        for m in range(self.num_modalities):
            idx = (modality_ids == m).nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                continue
            pair = self.pairs[m]
            fn = pair.teacher_forward if use_teacher else pair.student_forward
            out[idx] = fn(x[idx])
        # Any modality ids outside [0, num_modalities) get routed to pair 0.
        oob = (modality_ids < 0) | (modality_ids >= self.num_modalities)
        if oob.any():
            idx = oob.nonzero(as_tuple=False).squeeze(-1)
            pair = self.pairs[0]
            fn = pair.teacher_forward if use_teacher else pair.student_forward
            out[idx] = fn(x[idx])
        return out

    # ----- public API: mirrors EMAProjectorPair for drop-in compatibility -

    def student_forward(self, x: torch.Tensor, modality_ids: torch.Tensor) -> torch.Tensor:
        return self._route(x, modality_ids, use_teacher=False)

    @torch.no_grad()
    def teacher_forward(self, x: torch.Tensor, modality_ids: torch.Tensor) -> torch.Tensor:
        return self._route(x, modality_ids, use_teacher=True)

    @torch.no_grad()
    def update_teacher(self, tau: float) -> None:
        for pair in self.pairs:
            pair.update_teacher(tau)

    @property
    def student(self) -> nn.ModuleList:
        """Expose all student projectors for DDP/optimizer wrapping."""
        return nn.ModuleList([pair.student for pair in self.pairs])


__all__ = ["ModalityProjectorPair"]
