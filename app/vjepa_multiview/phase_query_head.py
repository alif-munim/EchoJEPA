"""Conditional query head for L_phase_rel on the factorized z_phase slot.

Mirrors PhaseRelationalHead's ``.query`` semantics (source_proj +
view embeddings + Fourier Δφ MLP + relation MLP), but takes the
factorized ``z_phase`` slot as input instead of the full pooled
encoder latent. Target projection is done by a separate small MLP
applied to the teacher's factorized ``z_phase`` slot — symmetric
to how PhaseRelationalHead's ``.target`` works.

This is Fix 2 from the v2 plan: the v1 path used
``q := z_phase`` directly, which dropped the source-view /
target-view / Δφ conditioning that made EchoJEPA-Rel work.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from app.vjepa_multiview.phase_relational_head import NUM_VIEWS, _PhaseMLP


class PhaseQueryHead(nn.Module):
    """Relation-conditioned query for z_phase-based InfoNCE.

    Inputs to ``.query`` (exactly these four):
        z_phase        : [B, phase_dim]  factorized student z_phase slot
        src_view_ids   : [B]             long
        tgt_view_ids   : [B]             long
        delta_phase    : [B]             float, Δφ (src -> tgt)

    Input to ``.target`` (applied identically to positive and hard-neg
    teacher z_phase slots):
        z_phase_detached : [B, phase_dim]  teacher-side z_phase (stop-grad)

    Head is discarded after pretraining, same as PhaseRelationalHead.
    """

    def __init__(
        self,
        phase_dim: int = 256,
        rel_dim: int = 256,
        hidden_dim: int = 512,
        num_views: int = NUM_VIEWS,
        view_embedding_dim: int = 64,
        n_phase_freqs: int = 4,
        init_std: float = 0.02,
    ):
        super().__init__()
        self.phase_dim = int(phase_dim)
        self.rel_dim = int(rel_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_views = int(num_views)
        self.view_embedding_dim = int(view_embedding_dim)
        self.n_phase_freqs = int(n_phase_freqs)

        self.src_view_embed = nn.Embedding(num_views, view_embedding_dim)
        self.tgt_view_embed = nn.Embedding(num_views, view_embedding_dim)
        nn.init.trunc_normal_(self.src_view_embed.weight, std=init_std)
        nn.init.trunc_normal_(self.tgt_view_embed.weight, std=init_std)

        phase_out_dim = 2 * view_embedding_dim
        self.phase_mlp = _PhaseMLP(n_phase_freqs, phase_out_dim)

        # Source projection: z_phase -> rel_dim.
        self.source_proj = nn.Sequential(
            nn.Linear(phase_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, rel_dim),
        )

        # Relation MLP fuses projected source with view+phase conditioning.
        relation_in_dim = rel_dim + 2 * view_embedding_dim + phase_out_dim
        self.relation_mlp = nn.Sequential(
            nn.Linear(relation_in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, rel_dim),
        )

        # Target projector for teacher z_phase -> rel_dim.
        self.target_proj = nn.Sequential(
            nn.Linear(phase_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, rel_dim),
        )

    def forward(
        self,
        z_phase: torch.Tensor,
        src_view_ids: torch.Tensor,
        tgt_view_ids: torch.Tensor,
        delta_phase: torch.Tensor,
        y_pos_phase: torch.Tensor,
        y_neg_phase: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single unified forward so DDP sees every head parameter
        on every step (source_proj + relation_mlp + view embeds +
        phase_mlp + target_proj). Caller is responsible for
        ``F.normalize`` and for detaching y_pos_phase / y_neg_phase
        before passing them in."""
        q_pre = self.query(z_phase, src_view_ids, tgt_view_ids, delta_phase)
        y_pos_pre = self.target(y_pos_phase)
        y_hard_pre = self.target(y_neg_phase)
        return q_pre, y_pos_pre, y_hard_pre

    def query(
        self,
        z_phase: torch.Tensor,
        src_view_ids: torch.Tensor,
        tgt_view_ids: torch.Tensor,
        delta_phase: torch.Tensor,
    ) -> torch.Tensor:
        if z_phase.dim() != 2:
            raise ValueError(f"z_phase must be [B, D]; got {tuple(z_phase.shape)}")
        if z_phase.shape[-1] != self.phase_dim:
            raise ValueError(f"z_phase last dim must be {self.phase_dim}; got {z_phase.shape[-1]}")
        src = self.source_proj(z_phase)
        emb_s = self.src_view_embed(src_view_ids)
        emb_t = self.tgt_view_embed(tgt_view_ids)
        phi = self.phase_mlp(delta_phase.to(z_phase.dtype))
        fused = torch.cat([src, emb_s, emb_t, phi], dim=-1)
        return self.relation_mlp(fused)

    def target(self, z_phase_detached: torch.Tensor) -> torch.Tensor:
        if z_phase_detached.dim() != 2:
            raise ValueError(f"z_phase_detached must be [B, D]; got {tuple(z_phase_detached.shape)}")
        return self.target_proj(z_phase_detached)
