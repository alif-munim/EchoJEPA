"""Multi-view teacher fusion for sparse privileged supervision.

Fuses N same-study teacher clip latents into a bottlenecked "shared
study" target vector. Invoked only on Bernoulli(p_fused) steps —
callers must gate externally.

Input per clip v ∈ 1..N:
    pooled_v  : [B, D]  teacher pooled latent (already detached by caller)
    view_id_v : [B]     integer view id for clip v
    phase_v   : [B]     relative cardiac phase at anchor of clip v

Output:
    t_fused_shared : [B, fused_dim]  detached fused study target

The fusion is a small cross-attention: learned slot query attends over
the stack of (N clips × B) conditioned tokens. Caller holds teacher
parameters frozen (EMA-updated elsewhere); this module's own parameters
are trainable and live on the student side — same layout as
PhaseRelationalHead.target_proj.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from app.vjepa_multiview.phase_relational_head import NUM_VIEWS, _PhaseMLP


class MultiViewTeacherFusion(nn.Module):
    def __init__(
        self,
        embed_dim: int = 1024,
        fused_dim: int = 256,
        hidden_dim: int = 1024,
        num_views: int = NUM_VIEWS,
        view_embedding_dim: int = 64,
        n_phase_freqs: int = 4,
        num_heads: int = 8,
        init_std: float = 0.02,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.fused_dim = int(fused_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_views = int(num_views)
        self.view_embedding_dim = int(view_embedding_dim)
        self.n_phase_freqs = int(n_phase_freqs)

        self.view_embed = nn.Embedding(num_views, view_embedding_dim)
        nn.init.trunc_normal_(self.view_embed.weight, std=init_std)

        phase_out_dim = 2 * view_embedding_dim
        self.phase_mlp = _PhaseMLP(n_phase_freqs, phase_out_dim)

        # Project each teacher clip token (pooled + view_emb + phase_emb)
        # to the fused-dim space.
        token_in_dim = embed_dim + view_embedding_dim + phase_out_dim
        self.token_proj = nn.Sequential(
            nn.Linear(token_in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, fused_dim),
        )

        self.query = nn.Parameter(torch.zeros(1, 1, fused_dim))
        nn.init.trunc_normal_(self.query, std=init_std)

        self.attn = nn.MultiheadAttention(
            embed_dim=fused_dim,
            num_heads=num_heads,
            batch_first=True,
            bias=True,
        )
        self.norm = nn.LayerNorm(fused_dim)
        self.out_proj = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.GELU(),
            nn.Linear(fused_dim, fused_dim),
        )

    def forward(
        self,
        pooled_nv: torch.Tensor,  # [B, N, embed_dim]
        view_ids_nv: torch.Tensor,  # [B, N]  long
        phase_nv: torch.Tensor,  # [B, N]  float
        key_padding_mask: torch.Tensor | None = None,  # [B, N] bool, True=pad
    ) -> torch.Tensor:
        """Fuse N teacher clips into a single [B, fused_dim] study target.

        Caller ensures `pooled_nv` is detached from the teacher ENCODER's
        graph (no gradient flows back to the EMA-updated teacher weights).
        The fusion module's *own* trainable parameters still receive
        gradient through the output — this is the signal that trains the
        cross-view fusion to be a useful study summary.

        If the caller wants a hard stop-grad on the target side of the
        outer loss (BYOL-style), they should wrap the return value in
        ``.detach()`` at the call site. This module does not impose that
        — doing so would freeze the fusion weights entirely.
        """
        if pooled_nv.dim() != 3:
            raise ValueError(f"pooled_nv must be [B, N, D]; got {tuple(pooled_nv.shape)}")
        B, N, D = pooled_nv.shape
        if D != self.embed_dim:
            raise ValueError(f"pooled_nv last dim must be embed_dim={self.embed_dim}; got {D}")
        if view_ids_nv.shape != (B, N) or phase_nv.shape != (B, N):
            raise ValueError(
                f"view_ids_nv/phase_nv must be [B, N]; got " f"{tuple(view_ids_nv.shape)} / {tuple(phase_nv.shape)}"
            )

        v_emb = self.view_embed(view_ids_nv)  # [B, N, V]
        # _PhaseMLP expects [B] — run per-clip then stack back to [B, N, 2V].
        phase_flat = self.phase_mlp(phase_nv.reshape(B * N).to(pooled_nv.dtype)).reshape(B, N, -1)

        tokens = torch.cat([pooled_nv, v_emb, phase_flat], dim=-1)  # [B, N, token_in]
        tokens = self.token_proj(tokens)  # [B, N, fused]

        q = self.query.expand(B, -1, -1)  # [B, 1, fused]
        attn_out, _ = self.attn(
            q,
            tokens,
            tokens,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )  # [B, 1, fused]
        fused = self.norm(attn_out.squeeze(1))  # [B, fused]
        fused = fused + self.out_proj(fused)  # residual
        return fused
