"""Permutation-invariant study transformer for EchoSet-JEPA.

Inputs: context element vectors with meta tokens + [STUDY] token + [MASK]_t
slots carrying target meta tokens. No positional embeddings; the dataloader
is expected to permute element order every step (see plan §4.3).

Output: per-token hidden states at d_model. Callers read:
  - ``h_study`` = output at the [STUDY] position (position 0)
  - ``h_mask``  = outputs at the mask positions (one per target)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class StudyTransformerConfig:
    d_clip: int = 1024
    d_model: int = 512
    n_layers: int = 4
    n_heads: int = 8
    ffn_mult: int = 4
    dropout_ffn: float = 0.1
    dropout_attn: float = 0.0
    max_M: int = 64


class _StudyBlock(nn.Module):
    def __init__(self, cfg: StudyTransformerConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=cfg.d_model,
            num_heads=cfg.n_heads,
            dropout=cfg.dropout_attn,
            batch_first=True,
        )
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.ffn_mult * cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout_ffn),
            nn.Linear(cfg.ffn_mult * cfg.d_model, cfg.d_model),
            nn.Dropout(cfg.dropout_ffn),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        y = self.ln1(x)
        out, _ = self.attn(y, y, y, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + out
        x = x + self.ffn(self.ln2(x))
        return x


class StudyTransformer(nn.Module):
    def __init__(self, cfg: StudyTransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.clip_in = nn.Linear(cfg.d_clip, cfg.d_model)
        self.study_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        nn.init.trunc_normal_(self.study_token, std=0.02)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.blocks = nn.ModuleList([_StudyBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm_out = nn.LayerNorm(cfg.d_model)

    def forward(
        self,
        ctx_elements: torch.Tensor,  # (B, M_ctx, d_clip)
        ctx_meta_add: torch.Tensor,  # (B, M_ctx, d_model)
        ctx_pad_mask: torch.Tensor,  # (B, M_ctx) bool: True = pad
        tgt_meta_add: torch.Tensor,  # (B, M_tgt, d_model) — target meta tokens at [MASK]
        tgt_pad_mask: torch.Tensor,  # (B, M_tgt) bool: True = pad
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the study transformer.

        Returns ``(h_study, h_mask)`` where
          - ``h_study`` has shape ``(B, d_model)``
          - ``h_mask``  has shape ``(B, M_tgt, d_model)``.
        """
        B = ctx_elements.shape[0]
        M_ctx = ctx_elements.shape[1]
        M_tgt = tgt_meta_add.shape[1]

        # Context token = Linear(LN-ed clip agg) + meta additive
        x_ctx = self.clip_in(ctx_elements) + ctx_meta_add
        x_mask = self.mask_token.expand(B, M_tgt, -1) + tgt_meta_add
        x_study = self.study_token.expand(B, 1, -1)

        # Padding masks: [STUDY] is never padded
        study_pad = torch.zeros(B, 1, dtype=torch.bool, device=ctx_elements.device)
        full_pad = torch.cat([study_pad, ctx_pad_mask, tgt_pad_mask], dim=1)

        x = torch.cat([x_study, x_ctx, x_mask], dim=1)

        for blk in self.blocks:
            x = blk(x, key_padding_mask=full_pad)
        x = self.norm_out(x)

        h_study = x[:, 0, :]
        h_mask = x[:, 1 + M_ctx :, :]
        return h_study, h_mask

    def forward_contextualized(
        self,
        elements: torch.Tensor,  # (B, M, d_clip)
        meta_add: torch.Tensor,  # (B, M, d_model)
        pad_mask: torch.Tensor,  # (B, M) bool: True = pad
    ) -> torch.Tensor:
        """Run the transformer over a full set of elements (no mask slots).

        Used by EchoMV-JEPA's EMA teacher: all elements are treated as context
        so every position gets contextualized by every other position. Returns
        ``(B, M, d_model)`` — the per-element contextualized hidden states.
        Callers gather at target indices after this call.
        """
        out, _ = self.forward_with_study_token(elements, meta_add, pad_mask)
        return out

    def forward_with_study_token(
        self,
        elements: torch.Tensor,
        meta_add: torch.Tensor,
        pad_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Same as ``forward_contextualized`` but also returns ``h_study``.

        Returns ``(h_per_element, h_study)`` with shapes
        ``(B, M, d_model)`` and ``(B, d_model)``.
        """
        B = elements.shape[0]
        x_elem = self.clip_in(elements) + meta_add
        x_study = self.study_token.expand(B, 1, -1)
        study_pad = torch.zeros(B, 1, dtype=torch.bool, device=elements.device)
        full_pad = torch.cat([study_pad, pad_mask], dim=1)
        x = torch.cat([x_study, x_elem], dim=1)
        for blk in self.blocks:
            x = blk(x, key_padding_mask=full_pad)
        x = self.norm_out(x)
        h_study = x[:, 0, :]
        h_per_elem = x[:, 1:, :]
        return h_per_elem, h_study


__all__ = ["StudyTransformer", "StudyTransformerConfig"]
