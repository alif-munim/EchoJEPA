"""Cross-rank study-level InfoNCE with bucket-aware negative selection.

The trainer calls this helper after ``all_gather``ing the student/teacher
projected study embeddings + per-study bookkeeping across all DDP ranks.
The logic here is a pure function over gathered tensors so it's easy to
test without a DDP runtime.

Negatives are drawn preferentially from studies that match the anchor
on ``(view_count_bucket, modality_count_bucket, clip_count_bucket)``.
If fewer than ``min_negatives`` preferred-bucket negatives exist,
fallback to any-other-patient negatives.

Excludes:
  * same-row self-match on the diagonal
  * same-study (shouldn't happen across ranks; safety net)
  * same-patient (if provided)

Returns:
  loss:         scalar InfoNCE loss, reduced over local rows
  diagnostics:  dict of scalars
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def cross_rank_study_nce_loss(
    h_student_local: torch.Tensor,  # (B_local, D) — student [STUDY] proj, grad
    z_teacher_global: torch.Tensor,  # (N_global, D) — teacher [STUDY] proj, no grad
    local_offset: int,  # where h_student_local sits inside the global
    study_id_global: torch.Tensor,  # (N_global,) long
    patient_id_global: Optional[torch.Tensor],  # (N_global,) long or None
    view_count_bucket_global: torch.Tensor,  # (N_global,) long
    modality_count_bucket_global: torch.Tensor,  # (N_global,) long
    clip_count_bucket_global: torch.Tensor,  # (N_global,) long
    *,
    tau: float = 0.1,
    match_view_bucket: bool = True,
    match_modality_bucket: bool = True,
    match_clip_bucket: bool = True,
    min_negatives: int = 1,
    exclude_same_patient: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute cross-rank study InfoNCE.

    Row i of ``h_student_local`` matches column ``local_offset + i`` of
    ``z_teacher_global`` as its positive. All other columns are candidate
    negatives, subject to bucket matching + same-patient exclusion.
    """
    B_local, D = h_student_local.shape
    N_global = z_teacher_global.shape[0]
    device = h_student_local.device

    # Local row i ↔ global column (local_offset + i)
    pos_idx = torch.arange(B_local, device=device) + local_offset

    # LN on both sides (symmetry with the v1 study loss).
    h = F.layer_norm(h_student_local, h_student_local.shape[-1:])
    z = F.layer_norm(z_teacher_global.to(device), z_teacher_global.shape[-1:])

    logits = (h @ z.t()) / tau  # (B_local, N_global)

    # Build mask of VALID negatives per row.
    # Start: all columns are candidate negatives.
    neg_mask = torch.ones(B_local, N_global, dtype=torch.bool, device=device)
    # Exclude self + same-study.
    row_study = study_id_global[pos_idx]  # (B_local,)
    same_study = study_id_global.unsqueeze(0).eq(row_study.unsqueeze(1))  # (B_local, N_global)
    neg_mask &= ~same_study
    # Exclude same-patient if provided.
    fallback_count = 0
    if exclude_same_patient and patient_id_global is not None:
        row_patient = patient_id_global[pos_idx]
        same_patient = patient_id_global.unsqueeze(0).eq(row_patient.unsqueeze(1))
        neg_mask &= ~same_patient

    # Preferred-bucket mask: same bucket on all enabled axes.
    preferred_mask = torch.ones_like(neg_mask)
    if match_view_bucket:
        row_b = view_count_bucket_global[pos_idx]
        preferred_mask &= view_count_bucket_global.unsqueeze(0).eq(row_b.unsqueeze(1))
    if match_modality_bucket:
        row_b = modality_count_bucket_global[pos_idx]
        preferred_mask &= modality_count_bucket_global.unsqueeze(0).eq(row_b.unsqueeze(1))
    if match_clip_bucket:
        row_b = clip_count_bucket_global[pos_idx]
        preferred_mask &= clip_count_bucket_global.unsqueeze(0).eq(row_b.unsqueeze(1))

    preferred_and_valid = neg_mask & preferred_mask
    n_preferred = preferred_and_valid.sum(dim=1)  # (B_local,)
    # Fallback: if preferred pool < min_negatives, use the full valid pool for that row.
    use_fallback = n_preferred < min_negatives
    final_mask = torch.where(use_fallback.unsqueeze(1), neg_mask, preferred_and_valid)
    fallback_count = int(use_fallback.sum().item())

    # Build label-and-mask logits.
    # The positive column MUST remain in the mask.
    pos_col = pos_idx
    final_mask[torch.arange(B_local, device=device), pos_col] = True

    # Mask out non-selected columns with -inf.
    logits = logits.masked_fill(~final_mask, float("-inf"))

    # Labels point to the positive column in the global space.
    loss = F.cross_entropy(logits, pos_col)

    # Diagnostics.
    with torch.no_grad():
        valid_neg = (final_mask.sum(dim=1) - 1).float()  # subtract the positive
        # Rank metrics (how high the positive ranks among valid negatives).
        pos_logit = logits[torch.arange(B_local, device=device), pos_col]
        better = (logits > pos_logit.unsqueeze(1)).sum(dim=1)
        rank = (better + 1).float()
        rank_top1 = (rank <= 1).float().mean().item()
        rank_top5 = (rank <= 5).float().mean().item()
        # Hardest negative logit (largest non-positive in the kept set).
        tmp = logits.clone()
        tmp[torch.arange(B_local, device=device), pos_col] = float("-inf")
        hard_neg = tmp.max(dim=1).values
        pos_minus_hardneg = (pos_logit - hard_neg).mean().item()

    diagnostics = {
        "study_nce_pool_size": float(valid_neg.mean().item()),
        "study_nce_valid_neg_mean": float(valid_neg.mean().item()),
        "study_nce_valid_neg_min": float(valid_neg.min().item()) if B_local > 0 else 0.0,
        "study_nce_fallback_fraction": float(fallback_count) / max(B_local, 1),
        "study_matched_rank_top1_global": float(rank_top1),
        "study_matched_rank_top5_global": float(rank_top5),
        "pos_minus_hardneg_gap_global": float(pos_minus_hardneg),
    }
    return loss, diagnostics


__all__ = ["cross_rank_study_nce_loss"]
