"""Token-level wrapper over StudyTransformer for EchoMV-JEPA Option A.

The pooled-cache Stage-1 runs the study transformer on ``(B, M, d_clip)``
element vectors — one 1024-d vector per element. Within-study c_clip vectors
are near-identical (cosine 0.87–0.91 on real MIMIC studies), so self-attention
over 6 near-identical vectors collapses to identity and the contextualization
gate fails.

Option A feeds tokens, not pooled vectors. Each element provides ``T_e`` tokens
(e.g. 392 after a 2x2 spatial pool over the V-JEPA 1568-token output). The
study transformer attends at token granularity: ``M * T_e`` tokens per study.
The final per-element target is the mean of the contextualized tokens belonging
to that element, so the downstream loss and §15.1a diagnostics (which operate
per-element) stay unchanged.

The wrapper is the smallest possible additive layer over ``StudyTransformer``:
  - flatten (B, M, T, d_clip) → (B, M*T, d_clip) before the transformer,
  - broadcast per-element meta across the T tokens of that element,
  - construct the flat pad mask from element-level pad + token-level pad,
  - run ``StudyTransformer.forward_contextualized``,
  - reshape (B, M*T, d_model) back to (B, M, T, d_model) and mean-pool over T.

The core invariant: the underlying ``StudyTransformer`` is unchanged. If we
ever want per-token targets, we expose ``forward_contextualized_tokens`` that
skips the final pool.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from src.models.study_transformer import StudyTransformer


class TokenStudyTransformer(nn.Module):
    """Token-level adapter over a :class:`StudyTransformer`.

    The wrapped transformer must have ``d_clip`` matching the token embedding
    dim produced by the upstream online encoder (e.g. 1024 for ViT-L).
    """

    def __init__(self, st: StudyTransformer) -> None:
        super().__init__()
        self.st = st

    @staticmethod
    def _flatten_pad(elem_pad: torch.Tensor, token_pad: Optional[torch.Tensor], M: int, T: int) -> torch.Tensor:
        """Construct a flat ``(B, M*T)`` pad mask.

        ``elem_pad`` is (B, M) — a padded element implies all its T tokens
        are padded. ``token_pad`` is optional (B, M, T) — finer-grained per-
        token pad (e.g. variable clip length). If None, all tokens of an
        unpadded element are valid.
        """
        B = elem_pad.shape[0]
        out = elem_pad.unsqueeze(-1).expand(B, M, T).contiguous()  # (B, M, T)
        if token_pad is not None:
            out = out | token_pad
        return out.reshape(B, M * T)

    def forward_contextualized(
        self,
        element_tokens: torch.Tensor,  # (B, M, T, d_clip)
        element_meta_add: torch.Tensor,  # (B, M, d_model) — one meta vec per element
        elem_pad_mask: torch.Tensor,  # (B, M) bool: True = pad
        token_pad_mask: Optional[torch.Tensor] = None,  # (B, M, T) bool, optional
    ) -> torch.Tensor:
        """Run the wrapped transformer at token granularity.

        Returns ``(B, M, d_model)`` — per-element contextualized hidden states
        formed by mean-pooling the T contextualized tokens of each element.
        """
        B, M, T, d_clip = element_tokens.shape
        d_model = element_meta_add.shape[-1]

        # Broadcast the single meta vec across the T tokens of its element.
        meta_flat = element_meta_add.unsqueeze(2).expand(B, M, T, d_model).reshape(B, M * T, d_model)
        tokens_flat = element_tokens.reshape(B, M * T, d_clip)
        pad_flat = self._flatten_pad(elem_pad_mask, token_pad_mask, M, T)

        out_flat = self.st.forward_contextualized(tokens_flat, meta_flat, pad_flat)  # (B, M*T, d_model)
        out = out_flat.reshape(B, M, T, d_model)

        # Mean over valid tokens per element. For entirely padded elements the
        # mean is over zeros (contextualized output at padded positions is
        # unused downstream because the caller selects only at unpadded
        # target slots), but guard against nan by masking.
        if token_pad_mask is not None:
            token_valid = (~token_pad_mask).float().unsqueeze(-1)  # (B, M, T, 1)
            counts = token_valid.sum(dim=2).clamp(min=1.0)  # (B, M, 1)
            pooled = (out * token_valid).sum(dim=2) / counts  # (B, M, d_model)
        else:
            pooled = out.mean(dim=2)
        return pooled

    def forward_contextualized_tokens(
        self,
        element_tokens: torch.Tensor,
        element_meta_add: torch.Tensor,
        elem_pad_mask: torch.Tensor,
        token_pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Same forward but returns the un-pooled token outputs, (B, M, T, d_model)."""
        B, M, T, d_clip = element_tokens.shape
        d_model = element_meta_add.shape[-1]
        meta_flat = element_meta_add.unsqueeze(2).expand(B, M, T, d_model).reshape(B, M * T, d_model)
        tokens_flat = element_tokens.reshape(B, M * T, d_clip)
        pad_flat = self._flatten_pad(elem_pad_mask, token_pad_mask, M, T)
        out_flat = self.st.forward_contextualized(tokens_flat, meta_flat, pad_flat)
        return out_flat.reshape(B, M, T, d_model)

    def forward_with_study_token(
        self,
        element_tokens: torch.Tensor,  # (B, M, T, d_clip)
        element_meta_add: torch.Tensor,  # (B, M, d_model)
        elem_pad_mask: torch.Tensor,  # (B, M) bool: True = pad
        token_pad_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Token-level wrapper over ``StudyTransformer.forward_with_study_token``.

        Returns ``(h_per_element, h_study)`` with shapes
        ``(B, M, d_model)`` and ``(B, d_model)``. The per-element output is
        mean-pooled across the T contextualized tokens belonging to that
        element (masking padded tokens if ``token_pad_mask`` is given).
        """
        B, M, T, d_clip = element_tokens.shape
        d_model = element_meta_add.shape[-1]
        meta_flat = element_meta_add.unsqueeze(2).expand(B, M, T, d_model).reshape(B, M * T, d_model)
        tokens_flat = element_tokens.reshape(B, M * T, d_clip)
        pad_flat = self._flatten_pad(elem_pad_mask, token_pad_mask, M, T)
        h_per_token_flat, h_study = self.st.forward_with_study_token(tokens_flat, meta_flat, pad_flat)
        h_per_token = h_per_token_flat.reshape(B, M, T, d_model)
        if token_pad_mask is not None:
            token_valid = (~token_pad_mask).float().unsqueeze(-1)
            counts = token_valid.sum(dim=2).clamp(min=1.0)
            h_per_elem = (h_per_token * token_valid).sum(dim=2) / counts
        else:
            h_per_elem = h_per_token.mean(dim=2)
        return h_per_elem, h_study


__all__ = ["TokenStudyTransformer"]
