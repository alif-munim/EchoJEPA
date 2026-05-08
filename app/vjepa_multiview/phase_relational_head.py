"""Pooled relation-conditioned head for phase_relational pretraining.

Discarded after pretraining. The head consumes POOLED latents (no token
grids) and emits projected vectors; callers L2-normalize before InfoNCE.

The head signature accepts only the four visual/relational inputs:
    (c_a_pool, view_a_ids, view_b_pos_ids, delta_phase_pos)

It does NOT accept absolute phase, HR, quality tier, RR status, phase
error, view confidence, view_b_neg_id, Δφ_neg, or any study/patient/
dicom ID. This is a structural guard against metadata-shortcut leakage —
the smoke test asserts `_build_predictor_inputs` returns exactly 3
tensors and this head's `.query` cannot accept anything else.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

VIEW_ID_MAP: dict[str, int] = {
    "A2C": 0,
    "A3C": 1,
    "A4C": 2,
    "A5C": 3,
    "PLAX": 4,
    "PSAX-AV": 5,
    "PSAX-MV": 6,
    "PSAX-PM": 7,
    "PSAX-AP": 8,
    "Subcostal": 9,
    "IVC": 10,
    "SSN": 11,
    "TEE": 12,
    "UNKNOWN": 13,
}
NUM_VIEWS: int = 14

# Coarse view family integer IDs. Mirrors classifier/phase/sampler's
# VIEW_FAMILIES but expressed as int IDs so contrastive / retrieval
# code can vectorize family membership without string lookup.
#   0 = apical       (A2C, A3C, A4C, A5C)
#   1 = parasternal_long  (PLAX)
#   2 = parasternal_short (PSAX-AV, PSAX-MV, PSAX-PM, PSAX-AP)
#   3 = other        (Subcostal, IVC, SSN, TEE)
#   4 = unknown      (UNKNOWN)
VIEW_FAMILY_ID: dict[int, int] = {
    VIEW_ID_MAP["A2C"]: 0,
    VIEW_ID_MAP["A3C"]: 0,
    VIEW_ID_MAP["A4C"]: 0,
    VIEW_ID_MAP["A5C"]: 0,
    VIEW_ID_MAP["PLAX"]: 1,
    VIEW_ID_MAP["PSAX-AV"]: 2,
    VIEW_ID_MAP["PSAX-MV"]: 2,
    VIEW_ID_MAP["PSAX-PM"]: 2,
    VIEW_ID_MAP["PSAX-AP"]: 2,
    VIEW_ID_MAP["Subcostal"]: 3,
    VIEW_ID_MAP["IVC"]: 3,
    VIEW_ID_MAP["SSN"]: 3,
    VIEW_ID_MAP["TEE"]: 3,
    VIEW_ID_MAP["UNKNOWN"]: 4,
}


def family_of(view_id: int) -> int:
    """Integer family ID for a view integer ID. Unknown family = 4."""
    return VIEW_FAMILY_ID.get(int(view_id), 4)


def view_to_id(v) -> int:
    """Canonicalize a view string to an integer id. Unknown → 13 (UNKNOWN).

    Handles None, empty string, the common "SUBCOSTAL" case variant, and
    arbitrary strings not in the canonical list.
    """
    if v is None:
        return VIEW_ID_MAP["UNKNOWN"]
    s = str(v)
    if s == "":
        return VIEW_ID_MAP["UNKNOWN"]
    if s.upper() == "SUBCOSTAL":
        s = "Subcostal"
    # Exact-match on canonical key; fall back to UNKNOWN.
    if s in VIEW_ID_MAP:
        return VIEW_ID_MAP[s]
    # Case-variant fallback for fully-uppercase / lowercase forms of
    # canonical strings (e.g. "plax" or "A4c").
    upper = s.upper()
    for k, v in VIEW_ID_MAP.items():
        if k.upper() == upper:
            return v
    return VIEW_ID_MAP["UNKNOWN"]


def pool_tokens(x: torch.Tensor) -> torch.Tensor:
    """Mean-pool token embeddings over the sequence dim.

    x: [B, N, D] -> [B, D]. Caller decides grad / detach.
    """
    return x.mean(dim=1)


class _PhaseMLP(nn.Module):
    """Fourier encoding of Δφ followed by a 2-layer MLP.

    Mirrors the phase-conditioning pattern in src/models/predictor.py.
    """

    def __init__(self, n_freqs: int, out_dim: int):
        super().__init__()
        self.register_buffer(
            "freqs",
            torch.arange(1, n_freqs + 1, dtype=torch.float32),
            persistent=False,
        )
        self.mlp = nn.Sequential(
            nn.Linear(2 * n_freqs, out_dim, bias=True),
            nn.GELU(),
            nn.Linear(out_dim, out_dim, bias=True),
        )

    def forward(self, delta_phase: torch.Tensor) -> torch.Tensor:
        """delta_phase: [B] in any real-valued range (will be mapped mod 1
        implicitly via sin/cos). Returns [B, out_dim]."""
        x = delta_phase[:, None] * 2.0 * math.pi * self.freqs[None, :]
        feat = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        return self.mlp(feat)


class PhaseRelationalHead(nn.Module):
    """Query/target projection head for phase_relational InfoNCE.

    Inputs to ``.query`` (ONLY these four):
      c_a_pool        : [B, embed_dim]  pooled student context tokens on clip_a
      view_a_ids      : [B] long        integer view id of the anchor clip
      view_b_pos_ids  : [B] long        integer view id of the POSITIVE target
      delta_phase_pos : [B] float       Δφ between anchor and positive target

    Input to ``.target`` (applied identically to both positive and
    same-study wrong-phase targets):
      pool_detached   : [B, embed_dim]  pooled teacher tokens (detached)

    The head is discarded after pretraining — no downstream code depends
    on it.
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

        # View embeddings — separate for source and target.
        self.view_embed_a = nn.Embedding(num_views, view_embedding_dim)
        self.view_embed_b_pos = nn.Embedding(num_views, view_embedding_dim)
        nn.init.trunc_normal_(self.view_embed_a.weight, std=0.02)
        nn.init.trunc_normal_(self.view_embed_b_pos.weight, std=0.02)

        # Fourier encoding of Δφ; emits a vector the same size as the
        # concatenated view embeddings so the relation MLP sees balanced
        # feature widths.
        phase_out_dim = 2 * view_embedding_dim
        self.phase_mlp = _PhaseMLP(n_phase_freqs, phase_out_dim)

        # Source projection: pooled student latent → rel_dim.
        self.source_proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, rel_dim),
        )

        # Relation MLP: fuses source-projected latent with view +
        # phase conditioning into the final query.
        relation_in_dim = rel_dim + 2 * view_embedding_dim + phase_out_dim
        self.relation_mlp = nn.Sequential(
            nn.Linear(relation_in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, rel_dim),
        )

        # Target projector: pooled teacher latent → rel_dim.
        # Receives stopgrad input from the caller (detached pooled
        # teacher features). The projector itself is trainable.
        self.target_proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, rel_dim),
        )

    def forward(
        self,
        c_a_pool: torch.Tensor,
        view_a_ids: torch.Tensor,
        view_b_pos_ids: torch.Tensor,
        delta_phase_pos: torch.Tensor,
        y_pos_pool: torch.Tensor,
        y_neg_pool: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single unified forward so DDP's reducer sees every parameter
        of the head on every step (source_proj + relation_mlp + view
        embeds + phase_mlp + target_proj). Calling ``query`` / ``target``
        as separate DDP forwards would touch disjoint parameter subsets
        and would either need ``find_unused_parameters=True`` (slower
        reducer) or break ``static_graph=False`` reducer invariants.

        Returns the three pre-normalization projected vectors in one
        call. Caller is responsible for ``F.normalize`` on each and for
        detaching ``y_pos_pool`` / ``y_neg_pool`` *before* calling, so
        gradients don't flow into the teacher encoder.
        """
        q_pre = self.query(c_a_pool, view_a_ids, view_b_pos_ids, delta_phase_pos)
        y_pos_pre = self.target(y_pos_pool)
        y_hard_pre = self.target(y_neg_pool)
        return q_pre, y_pos_pre, y_hard_pre

    def query(
        self,
        c_a_pool: torch.Tensor,
        view_a_ids: torch.Tensor,
        view_b_pos_ids: torch.Tensor,
        delta_phase_pos: torch.Tensor,
    ) -> torch.Tensor:
        """Produce the relation-conditioned query vector.

        Returns [B, rel_dim]. Caller applies F.normalize.
        """
        if c_a_pool.dim() != 2:
            raise ValueError(f"c_a_pool must be [B, D]; got {tuple(c_a_pool.shape)}")
        src = self.source_proj(c_a_pool)  # [B, rel_dim]
        emb_a = self.view_embed_a(view_a_ids)  # [B, V]
        emb_b = self.view_embed_b_pos(view_b_pos_ids)  # [B, V]
        phase = self.phase_mlp(delta_phase_pos.to(c_a_pool.dtype))  # [B, 2V]
        fused = torch.cat([src, emb_a, emb_b, phase], dim=-1)
        return self.relation_mlp(fused)  # [B, rel_dim]

    def target(self, pool_detached: torch.Tensor) -> torch.Tensor:
        """Project a detached pooled teacher latent.

        Returns [B, rel_dim]. Applied identically to both the positive
        and the same-study wrong-phase target. Caller applies
        F.normalize.
        """
        if pool_detached.dim() != 2:
            raise ValueError(f"pool_detached must be [B, D]; got {tuple(pool_detached.shape)}")
        return self.target_proj(pool_detached)
