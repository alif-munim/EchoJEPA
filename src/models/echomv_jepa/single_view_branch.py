"""DDP-safe single-view-to-study sampler.

Replaces the v1 Bernoulli-per-step gate (which fired inconsistently
under DDP because each rank drew independently) with a deterministic
per-row subset:

  * Pick a fraction ``p_rows`` of each batch to run through the SV branch.
  * For each chosen row, prefer the listed view families (default:
    "apical" first, then RV-focused if labeled, then "parasternal_long",
    "parasternal_short"). The MIMIC manifest only carries family-level
    labels, not A4C/A2C, so "A4C" maps to the "apical" family.

Output is a NEW pad mask of shape ``(B, M_full)`` where only the chosen
view family's elements remain unpadded for rows in the SV subset.
Non-SV rows are marked invalid in the returned ``sv_row_mask`` so the
caller can compute L_sv only on rows that have a valid single-view
context.

Returns ``(sv_pad_mask, sv_row_mask, stats)`` where stats counts
view-family usage and the number of rows per family.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from src.models.meta_embeddings import VIEW_FAMILY_VOCAB

DEFAULT_PREFERENCE_ORDER: List[str] = [
    "apical",  # A4C / A2C / A3C / A5C
    "parasternal_long",  # PLAX
    "parasternal_short",  # PSAX
    "subcostal",
    "suprasternal",
]


def _name_to_view_id(name: str) -> int:
    try:
        return VIEW_FAMILY_VOCAB.index(name)
    except ValueError:
        return VIEW_FAMILY_VOCAB.index("unknown")


def sample_single_view_rows(
    full_pad_mask: torch.Tensor,  # (B, M_full) bool
    meta_view: torch.Tensor,  # (B, M_full) long (view-family ids)
    *,
    p_rows: float = 0.25,
    preference_order: Optional[List[str]] = None,
    generator: Optional[torch.Generator] = None,
    min_rows: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Build a single-view-subset pad mask for a fraction of batch rows.

    Args:
      full_pad_mask: (B, M_full) bool (True = pad). Unchanged on rows
          not chosen for SV.
      meta_view: (B, M_full) long. View-family vocab ids (see
          ``VIEW_FAMILY_VOCAB``).
      p_rows: fraction of rows to include in the SV branch each step.
          Deterministic: rows are picked via a CPU generator, so under
          DDP each rank gets a different per-rank set but the per-rank
          fraction is stable at ``p_rows``.
      preference_order: list of view-family names to prefer, in order.
          Falls back to the first available family for that row.
      generator: optional CPU generator for row selection.

    Returns:
      sv_pad_mask: (B, M_full) bool — only the chosen family's elements
          are unpadded for SV rows; all elements padded for non-SV rows
          (forcing the caller to skip them with ``sv_row_mask``).
      sv_row_mask: (B,) bool — True for rows that made it into the SV
          branch with ≥ 1 valid unpadded element; False otherwise.
      stats: dict with
          ``sv_valid_fraction``   — (sv_row_mask True) / B
          ``sv_num_rows``         — int
          ``sv_family_counts``    — family → int mapping
          ``a4c_sv_count``        — int (apical family only)
    """
    if preference_order is None:
        preference_order = DEFAULT_PREFERENCE_ORDER
    preferred_ids = [_name_to_view_id(n) for n in preference_order]
    B, M = full_pad_mask.shape
    device = full_pad_mask.device

    if generator is None:
        generator = torch.Generator(device="cpu").manual_seed(0)

    # Deterministic row selection: ~p_rows fraction of rows, but always at
    # least ``min_rows`` when B > 0 (otherwise the branch never fires for
    # small per-rank batches, which happened on v2 smoke where B=2 rounded
    # 0.25·2 = 0.5 → 0).
    n_sv = int(round(p_rows * B))
    if B > 0 and p_rows > 0.0:
        n_sv = max(n_sv, min_rows)
    n_sv = max(n_sv, 0)
    n_sv = min(n_sv, B)
    if n_sv == 0:
        sv_pad_mask = torch.ones_like(full_pad_mask)
        sv_row_mask = torch.zeros(B, dtype=torch.bool, device=device)
        return (
            sv_pad_mask,
            sv_row_mask,
            {
                "sv_valid_fraction": 0.0,
                "sv_num_rows": 0,
                "sv_family_counts": {},
                "a4c_sv_count": 0,
            },
        )

    perm = torch.randperm(B, generator=generator)
    chosen = perm[:n_sv].tolist()

    sv_pad_mask = torch.ones_like(full_pad_mask)
    sv_row_mask = torch.zeros(B, dtype=torch.bool, device=device)
    family_counts: Dict[str, int] = {}
    a4c_count = 0

    for b in chosen:
        unpadded = (~full_pad_mask[b]).nonzero(as_tuple=False).squeeze(-1)
        if unpadded.numel() == 0:
            continue
        row_views = meta_view[b, unpadded]

        # Walk preference order; take the first family that has ≥ 1 element in this row.
        chosen_family_id: Optional[int] = None
        for fid in preferred_ids:
            if (row_views == fid).any():
                chosen_family_id = int(fid)
                break
        if chosen_family_id is None:
            # Fall back to *any* available family (pick the most common).
            uniq, counts = torch.unique(row_views, return_counts=True)
            chosen_family_id = int(uniq[counts.argmax()].item())

        keep_idx = unpadded[row_views == chosen_family_id]
        if keep_idx.numel() == 0:
            continue
        sv_pad_mask[b, keep_idx] = False
        sv_row_mask[b] = True
        family_name = VIEW_FAMILY_VOCAB[chosen_family_id]
        family_counts[family_name] = family_counts.get(family_name, 0) + 1
        if family_name == "apical":
            a4c_count += 1

    stats = {
        "sv_valid_fraction": float(sv_row_mask.float().mean().item()),
        "sv_num_rows": int(sv_row_mask.sum().item()),
        "sv_family_counts": family_counts,
        "a4c_sv_count": a4c_count,
    }
    return sv_pad_mask, sv_row_mask, stats


__all__ = ["sample_single_view_rows", "DEFAULT_PREFERENCE_ORDER"]
