"""Conditional view predictor for Privileged Multi-View EchoJEPA.

Given the student's factorized slots (z_shared, z_phase), source and
target view IDs, and the source→target phase displacement Δφ, predict
the teacher's embedding for the target clip.

    q = P_view(z_shared, z_phase, src_id, tgt_id, Δφ)
    L_pair_view_pred = SmoothL1(q, sg(y_tgt))

The target y_tgt is produced outside this module (usually the teacher's
t_shared for the target clip). `sg(·)` = stop-grad applied by the caller.

Source and target view embeddings are independent nn.Embeddings so the
predictor cannot trivially collapse by relying on "view identity" alone.
Δφ is encoded via the same Fourier-MLP recipe used in
phase_relational_head._PhaseMLP, reused for consistency with the existing
method.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from app.vjepa_multiview.phase_relational_head import NUM_VIEWS, _PhaseMLP


class ConditionalViewPredictor(nn.Module):
    """Takes (z_shared, z_phase, z_view, src_view, tgt_view, Δφ) and
    predicts a target-view-specific latent. Used by L_pair_view
    (Fix 4): the view slot is consumed here, which is what makes the
    z_view head receive gradient in training.

    If ``use_z_view=False``, the view slot is dropped from the input
    and the predictor becomes a (shared + phase)-only model. That
    mode is available for ablation and for configurations that keep
    the factorized head but freeze z_view.
    """

    def __init__(
        self,
        shared_dim: int = 256,
        phase_dim: int = 256,
        view_dim: int = 256,
        target_dim: int = 256,
        hidden_dim: int = 512,
        num_views: int = NUM_VIEWS,
        view_embedding_dim: int = 64,
        n_phase_freqs: int = 4,
        init_std: float = 0.02,
        use_z_view: bool = True,
    ):
        super().__init__()
        self.shared_dim = int(shared_dim)
        self.phase_dim = int(phase_dim)
        self.view_dim = int(view_dim)
        self.target_dim = int(target_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_views = int(num_views)
        self.view_embedding_dim = int(view_embedding_dim)
        self.n_phase_freqs = int(n_phase_freqs)
        self.use_z_view = bool(use_z_view)

        self.src_view_embed = nn.Embedding(num_views, view_embedding_dim)
        self.tgt_view_embed = nn.Embedding(num_views, view_embedding_dim)
        nn.init.trunc_normal_(self.src_view_embed.weight, std=init_std)
        nn.init.trunc_normal_(self.tgt_view_embed.weight, std=init_std)

        phase_out_dim = 2 * view_embedding_dim
        self.phase_mlp = _PhaseMLP(n_phase_freqs, phase_out_dim)

        in_dim = self.shared_dim + self.phase_dim + 2 * view_embedding_dim + phase_out_dim
        if self.use_z_view:
            in_dim += self.view_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, target_dim),
        )

    def forward(
        self,
        z_shared: torch.Tensor,
        z_phase: torch.Tensor,
        src_view_ids: torch.Tensor,
        tgt_view_ids: torch.Tensor,
        delta_phase: torch.Tensor,
        z_view: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Returns [B, target_dim]. Caller L2-normalizes if using cosine loss.

        ``z_view`` is required when the predictor was built with
        ``use_z_view=True``; otherwise it is ignored. This keeps a
        single call site while allowing the z_view slot to be an
        ablation toggle.
        """
        if z_shared.dim() != 2 or z_phase.dim() != 2:
            raise ValueError(f"z_shared/z_phase must be [B, D]; got {tuple(z_shared.shape)} / {tuple(z_phase.shape)}")
        if z_shared.shape[0] != z_phase.shape[0]:
            raise ValueError(f"Batch size mismatch: z_shared {z_shared.shape[0]} vs z_phase {z_phase.shape[0]}")
        if z_shared.shape[-1] != self.shared_dim:
            raise ValueError(f"z_shared last dim must be {self.shared_dim}; got {z_shared.shape[-1]}")
        if z_phase.shape[-1] != self.phase_dim:
            raise ValueError(f"z_phase last dim must be {self.phase_dim}; got {z_phase.shape[-1]}")
        src = self.src_view_embed(src_view_ids)  # [B, V]
        tgt = self.tgt_view_embed(tgt_view_ids)  # [B, V]
        phi = self.phase_mlp(delta_phase.to(z_shared.dtype))  # [B, 2V]
        parts = [z_shared, z_phase]
        if self.use_z_view:
            if z_view is None:
                raise ValueError(
                    "ConditionalViewPredictor was built with use_z_view=True " "but forward() received z_view=None."
                )
            if z_view.dim() != 2 or z_view.shape[-1] != self.view_dim:
                raise ValueError(f"z_view must be [B, {self.view_dim}]; got {tuple(z_view.shape)}")
            parts.append(z_view)
        parts.extend([src, tgt, phi])
        fused = torch.cat(parts, dim=-1)
        return self.mlp(fused)
