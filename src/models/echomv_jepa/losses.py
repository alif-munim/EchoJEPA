"""EchoMV-JEPA loss primitives.

Reuses the v1 prioritized-negative-pool and cosine-regress math. Kept as small
pure functions so they are easy to unit-test and so the training loop stays
compact. Identical numerics to ``app.echoset_jepa.train._cosine_regress`` and
``_nce_loss``; any drift would make Stage-0 → Stage-1 comparison unfair.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def layernorm_cosine(h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Per-row LN on both, then cosine similarity. Returns ``(N,)``."""
    h = F.layer_norm(h, h.shape[-1:])
    z = F.layer_norm(z, z.shape[-1:])
    return F.cosine_similarity(h, z, dim=-1)


def cosine_regress(h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """L_study_jepa: 1 - mean cosine(LN(h), LN(z))."""
    return (1.0 - layernorm_cosine(h, z)).mean()


def prioritized_neg_pool(
    tgt_view: torch.Tensor,
    tgt_modality: torch.Tensor,
    tgt_phase: torch.Tensor,
    tgt_study_id: torch.Tensor,
    k_min: int = 4,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Build an ``(N, N)`` bool mask of valid negatives.

    4-tier priority ladder identical to v1's ``_prioritized_neg_pool``:
      1. same (view, modality, phase) from other studies
      2. same (view, modality) from other studies
      3. same modality from other studies
      4. any batch row from other studies

    Diagonal is True (positive). Same-study off-targets are always False.
    """
    N = tgt_view.shape[0]
    device = tgt_view.device
    same_study = tgt_study_id.unsqueeze(0) == tgt_study_id.unsqueeze(1)
    excl = same_study.clone()
    excl.fill_diagonal_(False)

    same_v = tgt_view.unsqueeze(0) == tgt_view.unsqueeze(1)
    same_m = tgt_modality.unsqueeze(0) == tgt_modality.unsqueeze(1)
    same_p = tgt_phase.unsqueeze(0) == tgt_phase.unsqueeze(1)

    pri_1 = same_v & same_m & same_p & ~excl
    pri_2 = same_v & same_m & ~excl
    pri_3 = same_m & ~excl
    pri_4 = ~excl

    eye = torch.eye(N, dtype=torch.bool, device=device)

    mask = torch.zeros_like(pri_1)
    fallback_level = torch.zeros(N, dtype=torch.long, device=device)
    for lvl, pri in enumerate([pri_1, pri_2, pri_3, pri_4], start=1):
        off_diag_count = (mask & ~eye).sum(dim=1)
        needs = off_diag_count < k_min
        if not needs.any():
            break
        promote = needs.unsqueeze(1) & pri
        mask = mask | promote
        fallback_level = torch.where(needs, torch.full_like(fallback_level, lvl), fallback_level)

    mask = mask | eye

    same_view_count = (mask & same_v & ~eye).sum(dim=1).float()
    same_modality_count = (mask & same_m & ~eye).sum(dim=1).float()
    fallback_frac = (fallback_level > 2).float().mean().item()

    diag: Dict[str, float] = {
        "valid_neg_count_same_view_mean": same_view_count.mean().item(),
        "valid_neg_count_same_view_min": float(same_view_count.min().item()) if N else 0.0,
        "valid_neg_count_same_modality_mean": same_modality_count.mean().item(),
        "fallback_fraction": fallback_frac,
    }
    return mask, diag


def matched_nce(
    h: torch.Tensor,
    z: torch.Tensor,
    neg_mask: torch.Tensor,
    tau: float = 0.1,
) -> torch.Tensor:
    """InfoNCE with an explicit (N, N) mask of valid negatives (§9.2)."""
    h = F.layer_norm(h, h.shape[-1:])
    z = F.layer_norm(z, z.shape[-1:])
    logits = h @ z.t() / tau
    logits = logits.masked_fill(~neg_mask, float("-inf"))
    labels = torch.arange(h.shape[0], device=h.device)
    return F.cross_entropy(logits, labels)


def covariance_penalty(
    h: torch.Tensor,
    var_floor: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Off-diagonal covariance penalty + optional variance-floor hinge.

    Given ``h`` of shape ``(N, D)``, return (L_cov, L_var) where

      L_cov = mean(offdiag(Cov(h))^2)
      L_var = mean(ReLU(var_floor - std(h, dim=0))^2)  # only if var_floor > 0

    The caller composes them: ``L_total += lambda_cov * (L_cov + L_var)``.
    Using the *mean* (not sum) keeps the scale approximately scale-free in D.
    ``L_var`` is zero when ``var_floor <= 0``.
    """
    if h.shape[0] < 2:
        z = torch.zeros((), device=h.device)
        return z, z
    h_c = h - h.mean(dim=0, keepdim=True)
    cov = (h_c.t() @ h_c) / max(h.shape[0] - 1, 1)
    off = cov - torch.diag(torch.diagonal(cov))
    l_cov = (off * off).mean()
    if var_floor > 0.0:
        std = h.std(dim=0)
        l_var = torch.clamp(var_floor - std, min=0.0).pow(2).mean()
    else:
        l_var = torch.zeros((), device=h.device)
    return l_cov, l_var


def matched_rank_metrics(
    h: torch.Tensor,
    z: torch.Tensor,
    neg_mask: torch.Tensor,
) -> Dict[str, float]:
    """Retrieval-style ranking diagnostics over matched negatives.

    For each row i, rank row i's positive (``z_i``) among all ``z_j`` where
    ``neg_mask[i, j]`` is True (diagonal is always True == positive).

    Returns:
      - matched_rank_top1           : fraction with rank 0
      - matched_rank_top5           : fraction with rank < 5
      - pos_minus_hardneg_gap_mean  : mean (pos_cos - max_neg_cos) per row
    """
    N = h.shape[0]
    if N < 2:
        return {
            "matched_rank_top1": float("nan"),
            "matched_rank_top5": float("nan"),
            "pos_minus_hardneg_gap_mean": float("nan"),
        }
    h_ln = F.layer_norm(h, h.shape[-1:])
    z_ln = F.layer_norm(z, z.shape[-1:])
    sim = h_ln @ z_ln.t()  # (N, N)
    # For "rank of positive among valid negatives+positive", mask invalid cols
    # to -inf so they don't displace the positive.
    masked = sim.masked_fill(~neg_mask, float("-inf"))
    pos = masked.diagonal()  # (N,)
    # Count how many valid columns have a higher score than the positive.
    strictly_higher = (masked > pos.unsqueeze(1)) & neg_mask
    # Diagonal is its own positive; subtracting the diagonal is unnecessary
    # because ``masked > pos`` is always False on the diagonal.
    ranks = strictly_higher.sum(dim=1)  # (N,) long
    top1 = (ranks == 0).float().mean().item()
    top5 = (ranks < 5).float().mean().item()

    # Hardest negative: highest valid off-diagonal cosine.
    eye = torch.eye(N, dtype=torch.bool, device=h.device)
    neg_only = neg_mask & ~eye
    sim_neg = sim.masked_fill(~neg_only, float("-inf"))
    hardneg, _ = sim_neg.max(dim=1)  # (N,)
    # If no valid negative for a row, hardneg is -inf; mask those rows.
    valid_rows = neg_only.any(dim=1)
    if valid_rows.any():
        gap = (pos - hardneg)[valid_rows].mean().item()
    else:
        gap = float("nan")
    return {
        "matched_rank_top1": top1,
        "matched_rank_top5": top5,
        "pos_minus_hardneg_gap_mean": gap,
    }


__all__ = [
    "layernorm_cosine",
    "cosine_regress",
    "prioritized_neg_pool",
    "matched_nce",
    "covariance_penalty",
    "matched_rank_metrics",
]
