"""Token-level phase-relational heads for EchoJEPA-TokenRel / -Motion.

Preserves V4's phase-discrimination mechanism but moves it from pooled
clip embeddings to per-token spatiotemporal outputs. Cross-view rows use
token-set matching (logsumexp over target tokens) because token-index
correspondence is not anatomically meaningful across different echo
views.

Modules:
  TokenRelationalHead    — query/target projection, Fourier(Δφ) + view
                           embedding conditioning broadcast over tokens.
  MotionDeltaHead        — predicts per-token latent delta between
                           anchor and positive; same-view rows only.
  DeltaTargetProjector   — projects (teacher_tokens_pos − teacher_tokens_a)
                           into delta_dim for the InfoNCE target.

All heads are discarded after pretraining; downstream probes load only
``target_encoder``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

# Reuse VIEW_ID_MAP / NUM_VIEWS / _PhaseMLP style.
from app.vjepa_multiview.phase_relational_head import NUM_VIEWS, _PhaseMLP  # noqa: F401


def subsample_tokens(
    tokens: torch.Tensor,
    k: int,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Subsample ``k`` token indices from ``tokens`` ([B, N, D]).

    Returns ``(subsampled, indices)`` where:
      subsampled: [B, K, D]
      indices:    [K] long (shared across batch rows)

    Using a single shared index set across batch rows keeps downstream
    same-view delta-token pairing consistent between student and teacher
    forwards — both have to index the same token positions for the delta
    target to align. The subsampling is randomized per call.

    If ``k >= N`` we return the full token sequence unchanged.
    """
    if tokens.dim() != 3:
        raise ValueError(f"tokens must be [B, N, D]; got {tuple(tokens.shape)}")
    B, N, D = tokens.shape
    if k >= N:
        idx = torch.arange(N, device=tokens.device, dtype=torch.long)
        return tokens, idx
    # Shared index across batch rows — simplest correct choice for
    # same-view MotionDelta pairing. If we ever want per-row indices we
    # must use ``gather`` + broadcast-matched delta indices on teacher.
    if generator is None:
        idx = torch.randperm(N, device=tokens.device)[:k]
    else:
        idx = torch.randperm(N, generator=generator, device=tokens.device)[:k]
    idx = idx.long()
    return tokens.index_select(dim=1, index=idx), idx


class TokenRelationalHead(nn.Module):
    """Per-token query/target projection with (src_view, tgt_view, Δφ)
    conditioning broadcast over tokens.

    Inputs to ``.query``:
      z_tokens:        [B, K, embed_dim]  student clip_a tokens
      src_view_ids:    [B] long           anchor view id
      tgt_view_ids:    [B] long           target (clip_b_pos) view id
      delta_phase:     [B] float          signed Δφ mod 1

    Returns: [B, K, rel_dim] — caller L2-normalizes before contrastive.

    Input to ``.target``:
      h_tokens:        [B, K, embed_dim]  teacher target tokens
                       (caller detaches before calling)

    Returns: [B, K, rel_dim].

    Design notes:
    - Source projection is on tokens (per-position MLP; broadcasts
      naturally via nn.Linear over the last dim).
    - View + Δφ conditioning is computed once per row [B, cond_dim] and
      broadcast by expanding across the K dim; the relation MLP then
      fuses per-token ``concat(token_proj, cond_vec)``.
    - Separate source and target view embeddings so the head's
      conditioning is asymmetric (like V4's PhaseRelationalHead).
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        rel_dim: int = 256,
        hidden_dim: int = 1024,
        num_views: int = NUM_VIEWS,
        view_embedding_dim: int = 64,
        n_phase_freqs: int = 4,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.rel_dim = int(rel_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_views = int(num_views)
        self.view_embedding_dim = int(view_embedding_dim)
        self.n_phase_freqs = int(n_phase_freqs)

        self.view_embed_src = nn.Embedding(num_views, view_embedding_dim)
        self.view_embed_tgt = nn.Embedding(num_views, view_embedding_dim)
        nn.init.trunc_normal_(self.view_embed_src.weight, std=0.02)
        nn.init.trunc_normal_(self.view_embed_tgt.weight, std=0.02)

        phase_out_dim = 2 * view_embedding_dim
        self.phase_mlp = _PhaseMLP(n_phase_freqs, phase_out_dim)

        cond_dim = 2 * view_embedding_dim + phase_out_dim

        # Token projector: per-position linear + GELU + linear -> rel_dim.
        self.token_proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, rel_dim),
        )

        # Relation MLP: fuses per-token projection with broadcasted
        # conditioning. Input is concat([token_proj, cond]).
        self.relation_mlp = nn.Sequential(
            nn.Linear(rel_dim + cond_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, rel_dim),
        )

        # Target projector: teacher tokens -> rel_dim. No conditioning
        # on target — the contrast decides the relation, not the
        # projection itself.
        self.target_proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, rel_dim),
        )

    def query(
        self,
        z_tokens: torch.Tensor,
        src_view_ids: torch.Tensor,
        tgt_view_ids: torch.Tensor,
        delta_phase: torch.Tensor,
    ) -> torch.Tensor:
        if z_tokens.dim() != 3:
            raise ValueError(f"z_tokens must be [B, K, D]; got {tuple(z_tokens.shape)}")
        B, K, _ = z_tokens.shape
        src = self.token_proj(z_tokens)  # [B, K, rel_dim]
        emb_src = self.view_embed_src(src_view_ids)  # [B, V]
        emb_tgt = self.view_embed_tgt(tgt_view_ids)  # [B, V]
        phase = self.phase_mlp(delta_phase.to(z_tokens.dtype))  # [B, 2V]
        cond = torch.cat([emb_src, emb_tgt, phase], dim=-1)  # [B, cond_dim]
        cond = cond.unsqueeze(1).expand(B, K, -1)  # [B, K, cond_dim]
        fused = torch.cat([src, cond], dim=-1)
        return self.relation_mlp(fused)  # [B, K, rel_dim]

    def target(self, h_tokens: torch.Tensor) -> torch.Tensor:
        if h_tokens.dim() != 3:
            raise ValueError(f"h_tokens must be [B, K, D]; got {tuple(h_tokens.shape)}")
        return self.target_proj(h_tokens)

    def forward(
        self,
        z_tokens: torch.Tensor,
        src_view_ids: torch.Tensor,
        tgt_view_ids: torch.Tensor,
        delta_phase: torch.Tensor,
        y_pos_tokens: torch.Tensor,
        y_hard_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Unified DDP-safe forward so the reducer sees all head params
        on every step (token_proj + relation_mlp + view embeds +
        phase_mlp + target_proj)."""
        q = self.query(z_tokens, src_view_ids, tgt_view_ids, delta_phase)
        y_pos = self.target(y_pos_tokens)
        y_hard = self.target(y_hard_tokens)
        return q, y_pos, y_hard


class MotionDeltaHead(nn.Module):
    """Predicts per-token latent motion delta for same-view rows.

    Inputs:
      z_tokens:        [B, K, embed_dim]  student clip_a tokens
      src_view_ids:    [B] long
      delta_phase:     [B] float

    Returns: [B, K, delta_dim] — caller decides normalization.

    Same interface semantics as TokenRelationalHead but without a
    target-view branch: MotionDelta is defined only for exact same-view
    rows, so the conditioning depends only on the shared view and Δφ.
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        delta_dim: int = 256,
        hidden_dim: int = 1024,
        num_views: int = NUM_VIEWS,
        view_embedding_dim: int = 64,
        n_phase_freqs: int = 4,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.delta_dim = int(delta_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_views = int(num_views)
        self.view_embedding_dim = int(view_embedding_dim)
        self.n_phase_freqs = int(n_phase_freqs)

        self.view_embed = nn.Embedding(num_views, view_embedding_dim)
        nn.init.trunc_normal_(self.view_embed.weight, std=0.02)
        phase_out_dim = 2 * view_embedding_dim
        self.phase_mlp = _PhaseMLP(n_phase_freqs, phase_out_dim)

        cond_dim = view_embedding_dim + phase_out_dim

        self.token_proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, delta_dim),
        )
        self.relation_mlp = nn.Sequential(
            nn.Linear(delta_dim + cond_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, delta_dim),
        )

    def forward(
        self,
        z_tokens: torch.Tensor,
        src_view_ids: torch.Tensor,
        delta_phase: torch.Tensor,
    ) -> torch.Tensor:
        if z_tokens.dim() != 3:
            raise ValueError(f"z_tokens must be [B, K, D]; got {tuple(z_tokens.shape)}")
        B, K, _ = z_tokens.shape
        src = self.token_proj(z_tokens)  # [B, K, delta_dim]
        emb = self.view_embed(src_view_ids)  # [B, V]
        phase = self.phase_mlp(delta_phase.to(z_tokens.dtype))  # [B, 2V]
        cond = torch.cat([emb, phase], dim=-1).unsqueeze(1).expand(B, K, -1)
        fused = torch.cat([src, cond], dim=-1)
        return self.relation_mlp(fused)


class DeltaTargetProjector(nn.Module):
    """Projects per-token raw deltas (teacher_pos − teacher_anchor) into
    ``delta_dim`` to serve as the MotionDelta target.

    Input/output shapes: [B, K, embed_dim] -> [B, K, delta_dim].
    Caller detaches the raw delta before calling this projector.
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        delta_dim: int = 256,
        hidden_dim: int = 1024,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.delta_dim = int(delta_dim)
        self.hidden_dim = int(hidden_dim)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, delta_dim),
        )

    def forward(self, delta_tokens: torch.Tensor) -> torch.Tensor:
        if delta_tokens.dim() != 3:
            raise ValueError(f"delta_tokens must be [B, K, D]; got {tuple(delta_tokens.shape)}")
        return self.proj(delta_tokens)


__all__ = [
    "TokenRelationalHead",
    "MotionDeltaHead",
    "DeltaTargetProjector",
    "subsample_tokens",
]
