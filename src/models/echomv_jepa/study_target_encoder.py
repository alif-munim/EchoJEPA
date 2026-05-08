"""EMA full-study target study encoder (EchoMV-JEPA Stage-1).

This is the defining new module of EchoMV-JEPA. It owns an EMA copy of a
``StudyTransformer`` and exposes:

- ``forward_contextualized(full_elements, full_meta_add, full_pad_mask)`` →
  per-element contextualized hidden states over the full unmasked study.
- ``select_at_targets(z_per_element, target_idx)`` → gather at target indices.
- ``update_teacher(student, tau)`` → EMA update.

Teacher-only: no gradient ever flows through this module.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from src.models.study_transformer import StudyTransformer

from .ema import ema_update_


class StudyTransformerEMA(nn.Module):
    """Teacher wrapper around a :class:`StudyTransformer`.

    Parameters
    ----------
    student : StudyTransformer
        The online student to copy from.
    """

    def __init__(self, student: StudyTransformer) -> None:
        super().__init__()
        self.teacher = copy.deepcopy(student)
        for p in self.teacher.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update_teacher(self, student: StudyTransformer, tau: float) -> None:
        ema_update_(self.teacher, student, tau)

    @torch.no_grad()
    def forward_contextualized(
        self,
        full_elements: torch.Tensor,  # (B, M, d_clip)
        full_meta_add: torch.Tensor,  # (B, M, d_model)
        full_pad_mask: torch.Tensor,  # (B, M) bool: True = pad
    ) -> torch.Tensor:
        """Return per-element contextualized hidden states, (B, M, d_model)."""
        return self.teacher.forward_contextualized(full_elements, full_meta_add, full_pad_mask)

    @torch.no_grad()
    def forward_isolated(
        self,
        elements: torch.Tensor,  # (B, M, d_clip) — treat each row independently
        meta_add: torch.Tensor,  # (B, M, d_model)
    ) -> torch.Tensor:
        """Run the teacher on each element alone (no cross-element context).

        Used by the ``z_cosine_vs_isolated`` diagnostic (§15.1a). Runs the
        teacher with a single-element sequence per row; returns (B, M, d_model).
        Does not apply pad masking — callers should pass only unpadded rows or
        mask results downstream.
        """
        B, M, _ = elements.shape
        # Reshape to (B*M, 1, *) and run the teacher on one element at a time.
        el = elements.reshape(B * M, 1, elements.shape[-1])
        ma = meta_add.reshape(B * M, 1, meta_add.shape[-1])
        pad = torch.zeros(B * M, 1, dtype=torch.bool, device=elements.device)
        out = self.teacher.forward_contextualized(el, ma, pad)  # (B*M, 1, d_model)
        return out.reshape(B, M, -1)


__all__ = ["StudyTransformerEMA"]
