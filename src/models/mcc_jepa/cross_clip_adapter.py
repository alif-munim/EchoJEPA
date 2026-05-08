"""Zero-gated cross-clip residual adapter for target-anchored MCC-JEPA.

Injected after the V-JEPA predictor's output projection. At initialization
``gamma = 0``, so ``pred = pred_B`` exactly; training can grow ``gamma`` only
if cross-clip source context helps.

Reference: ``src/models/utils/modules.CrossAttention``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.utils.modules import CrossAttention


class CrossClipAdapter(nn.Module):
    """Residual cross-attention from target-B mask tokens onto source-A tokens.

    Forward:
        ``pred = pred_B + gamma * LN_q(pred_B) · CrossAttn · LN_kv(A_source)``
    ``gamma`` is a learned scalar initialized to zero.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        source_proj_dim: int | None = None,
        gamma_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        if source_proj_dim and source_proj_dim != embed_dim:
            self.source_proj: nn.Module = nn.Linear(source_proj_dim, embed_dim, bias=True)
        else:
            self.source_proj = nn.Identity()
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.xattn = CrossAttention(embed_dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.gamma = nn.Parameter(torch.full((1,), float(gamma_init)))

    def forward(
        self,
        pred_b_tokens: torch.Tensor,
        a_source_tokens: torch.Tensor,
        a_source_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        :param pred_b_tokens: [B, N_tgt, D] predictor output on target B.
        :param a_source_tokens: [B, N_A, D_src] source-clip-A tokens (encoder out).
        :param a_source_mask: optional [B, N_A] boolean mask; True means "ignore".
        :return: pred_b_tokens + gamma * cross_attn(q=pred_b_tokens, kv=a_source_tokens)
        """
        a = self.source_proj(a_source_tokens)
        q = self.norm_q(pred_b_tokens)
        kv = self.norm_kv(a)
        delta = self.xattn(q, kv, attn_mask=a_source_mask)
        return pred_b_tokens + self.gamma * delta

    @torch.no_grad()
    def diag(self, pred_b_tokens: torch.Tensor, a_source_tokens: torch.Tensor) -> dict:
        """Return diagnostics: gamma value and the norm of the cross-attention delta."""
        a = self.source_proj(a_source_tokens)
        q = self.norm_q(pred_b_tokens)
        kv = self.norm_kv(a)
        delta = self.xattn(q, kv, attn_mask=None)
        return {
            "gamma": float(self.gamma.detach().item()),
            "cross_attn_norm": float(delta.norm().item()),
            "delta_mean_abs": float(delta.abs().mean().item()),
        }
