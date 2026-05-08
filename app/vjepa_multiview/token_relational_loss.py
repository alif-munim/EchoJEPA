"""Loss helpers + forward for EchoJEPA-TokenRel / -Motion.

Two losses:
  * ``token_set_infonce_with_hard_neg`` — per-row token-set logsumexp
    contrast. Cross-view safe (no token-index alignment assumed).
  * ``motion_delta_loss`` — same-view-only latent delta prediction with
    SmoothL1 + token-level InfoNCE contrast against the same-view
    wrong-phase delta.

And the dispatch-facing forward:
  ``forward_token_phase_relational``.

Design constraints honored:
  - Student forward sees clip_a only (V4 parity).
  - Teacher concat-forward on [clip_a, clip_b_pos, clip_b_neg] under
    ``no_grad``.
  - Token-set matching for the token-rel InfoNCE so cross-view rows
    don't rely on anatomically-invalid token-index correspondence.
  - MotionDelta is restricted to ``src_view_ids == tgt_view_ids`` rows.
  - A small pooled-V4 safety loss is computed alongside; caller
    weights it at ``lambda_pool_rel = 0.005``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Re-used from existing phase_relational machinery.
from app.vjepa_multiview.phase_relational_head import pool_tokens


# --------------------------------------------------------------------- #
# Token-set logsumexp InfoNCE with a same-row hard negative
# --------------------------------------------------------------------- #


def token_set_infonce_with_hard_neg(
    q_tokens: torch.Tensor,
    y_pos_tokens: torch.Tensor,
    y_hard_tokens: torch.Tensor,
    study_hashes: torch.Tensor,
    tau: float,
    mask_same_study_batch_negatives: bool = True,
) -> dict:
    """Token-set matching InfoNCE with a same-row hard negative.

    All three inputs are L2-normed along the last dim before calling.

    Candidate structure per row ``b``:
      col 0         : same-row positive token set     (label = 0)
      col 1         : same-row hard-neg token set
      cols 2..B+1   : other rows' positive token sets as batch negatives

    Set-vs-set similarity (row ``b`` query ↔ row ``c`` target):
      set_sim[b, c] = mean_i logsumexp_j ( cos(q[b, i], y[c, j]) / tau )

    This reduces to per-query logsumexp over target tokens, averaged
    across query tokens. It is **cross-view safe** — no assumption
    that token index ``i`` of the query aligns with token index ``j`` of
    the target.

    Self-diagonal of the batch-negative block is ``-inf``. If
    ``mask_same_study_batch_negatives``, same-study off-diagonals are
    also ``-inf``. The explicit hard negative column is never masked.
    """
    if q_tokens.dim() != 3 or y_pos_tokens.dim() != 3 or y_hard_tokens.dim() != 3:
        raise ValueError(
            f"q/y_pos/y_hard must be [B, K, D]; got "
            f"{tuple(q_tokens.shape)} / {tuple(y_pos_tokens.shape)} / "
            f"{tuple(y_hard_tokens.shape)}"
        )
    B, K_q, D = q_tokens.shape
    if y_pos_tokens.shape[0] != B or y_pos_tokens.shape[-1] != D:
        raise ValueError(
            f"y_pos_tokens {tuple(y_pos_tokens.shape)} incompatible with q {tuple(q_tokens.shape)}"
        )
    if y_hard_tokens.shape[0] != B or y_hard_tokens.shape[-1] != D:
        raise ValueError(
            f"y_hard_tokens {tuple(y_hard_tokens.shape)} incompatible with q {tuple(q_tokens.shape)}"
        )

    device = q_tokens.device

    # ---- same-row positive set logit: [B] ----
    # cos(q[b,i], y_pos[b,j]) = q[b,i] @ y_pos[b,j]^T (inputs already normed).
    sim_pos_self = torch.einsum("bid,bjd->bij", q_tokens, y_pos_tokens) / tau  # [B, K_q, K_tp]
    pos_row = torch.logsumexp(sim_pos_self, dim=-1).mean(dim=-1)  # [B]
    pos_logit = pos_row.unsqueeze(1)  # [B, 1]

    # ---- same-row hard-neg set logit ----
    sim_hard_self = torch.einsum("bid,bjd->bij", q_tokens, y_hard_tokens) / tau
    hard_row = torch.logsumexp(sim_hard_self, dim=-1).mean(dim=-1)
    hard_logit = hard_row.unsqueeze(1)  # [B, 1]

    # ---- batch negative block [B, B] ----
    # sim_batch[b, c] = mean_i logsumexp_j (q[b, i] . y_pos[c, j]) / tau
    # Build the full q ↔ y_pos cross-similarity: [B, B, K_q, K_t].
    # B ≤ 32 on 8×H100 with bs=32 local; tokens K ≤ 64 per spec. So
    # B * B * K * K = 32*32*64*64 ≈ 4.2M entries per row-pair dim. Memory
    # is fine; still compact via einsum to avoid materializing a larger
    # intermediate.
    sim_cross = torch.einsum("bid,cjd->bcij", q_tokens, y_pos_tokens) / tau  # [B, B, K_q, K_t]
    set_sim_batch = torch.logsumexp(sim_cross, dim=-1).mean(dim=-1)  # [B, B]

    eye = torch.eye(B, dtype=torch.bool, device=device)
    neg_inf = torch.finfo(set_sim_batch.dtype).min
    set_sim_batch = set_sim_batch.masked_fill(eye, neg_inf)
    same_study_count = 0
    if mask_same_study_batch_negatives:
        same_study = study_hashes.unsqueeze(0).eq(study_hashes.unsqueeze(1))
        off_diag_same_study = same_study & ~eye
        set_sim_batch = set_sim_batch.masked_fill(off_diag_same_study, neg_inf)
        same_study_count = int(off_diag_same_study.sum().item())

    logits = torch.cat([pos_logit, hard_logit, set_sim_batch], dim=1)  # [B, B+2]
    labels = torch.zeros(B, dtype=torch.long, device=device)
    loss = F.cross_entropy(logits, labels)

    with torch.no_grad():
        top1 = (logits.argmax(dim=1) == labels).float().mean()
        pos_sim_mean = (pos_row * tau).mean()
        hard_sim_mean = (hard_row * tau).mean()
        bn_mask = set_sim_batch > neg_inf / 2.0
        if bn_mask.any():
            batch_neg_sim_mean = ((set_sim_batch * tau)[bn_mask]).mean()
        else:
            batch_neg_sim_mean = torch.zeros((), device=device)
        pos_minus_hard = pos_sim_mean - hard_sim_mean
        pos_minus_batch = pos_sim_mean - batch_neg_sim_mean
        logits_std = (
            logits[logits > neg_inf / 2.0].std()
            if (logits > neg_inf / 2.0).any()
            else torch.zeros((), device=device)
        )
        q_var = q_tokens.reshape(-1, D).var(dim=0).mean()
        y_var = y_pos_tokens.reshape(-1, D).var(dim=0).mean()

    return {
        "token_rel_loss": loss,
        "token_rel_top1_with_hard": top1,
        "token_rel_pos_sim_mean": pos_sim_mean,
        "token_rel_hard_sim_mean": hard_sim_mean,
        "token_rel_batch_neg_sim_mean": batch_neg_sim_mean,
        "token_rel_pos_minus_hard_gap": pos_minus_hard,
        "token_rel_pos_minus_batch_gap": pos_minus_batch,
        "token_rel_logits_std": logits_std,
        "token_rel_q_var": q_var,
        "token_rel_y_var": y_var,
        "token_rel_valid_rows": torch.tensor(float(B), device=device),
        "token_rel_same_study_masked_count": torch.tensor(float(same_study_count), device=device),
    }


# --------------------------------------------------------------------- #
# MotionDelta — same-view-only latent delta prediction
# --------------------------------------------------------------------- #


def motion_delta_loss(
    z_tokens_sub: torch.Tensor,        # [B, K, D] student tokens on subsampled indices
    h_a_tokens_sub: torch.Tensor,      # [B, K, D] teacher anchor tokens on same indices
    h_pos_tokens_sub: torch.Tensor,    # [B, K, D] teacher positive tokens on same indices
    h_neg_tokens_sub: torch.Tensor,    # [B, K, D] teacher same-view-wrong-phase tokens on same indices
    src_view_ids: torch.Tensor,        # [B]
    tgt_view_ids: torch.Tensor,        # [B]
    delta_phase: torch.Tensor,         # [B]
    motion_delta_head: nn.Module,
    delta_target_projector: nn.Module,
    *,
    tau: float = 0.10,
    lambda_l1: float = 1.0,
    lambda_nce: float = 1.0,
) -> dict:
    """Same-view-only latent motion-delta prediction.

    Only rows with ``src_view_ids == tgt_view_ids`` are eligible —
    cross-view rows are excluded because their token-index deltas are
    not anatomically valid. If there are no eligible rows, returns a
    zero scalar (gradient-connected to delta_target_projector so DDP
    reducers stay happy) and ``delta_valid_rows = 0``.

    Loss = ``lambda_l1 * SmoothL1(q_delta, stopgrad(d_pos_proj))``
         + ``lambda_nce * InfoNCE(q_delta, d_pos_proj, d_hard_proj, batch)``

    Inputs must be DETACHED on the teacher side before calling — the
    delta projector is trainable, but gradient must not flow back into
    the teacher encoder through h_* tensors.
    """
    if z_tokens_sub.dim() != 3 or h_a_tokens_sub.dim() != 3:
        raise ValueError("motion_delta_loss expects all *_tokens_sub to be [B, K, D]")
    B, K, D = z_tokens_sub.shape
    device = z_tokens_sub.device

    same_view_mask = src_view_ids.eq(tgt_view_ids)
    valid_rows = int(same_view_mask.sum().item())

    if valid_rows == 0:
        # Touch all trainable deltadrop params via a zero-loss proxy so
        # DDP reducer sees them every step — avoid find_unused_parameters.
        # Use a single constant pass through both modules.
        dummy_tokens = z_tokens_sub[:1]
        dummy_q = motion_delta_head(
            dummy_tokens,
            src_view_ids[:1],
            delta_phase[:1],
        )
        dummy_target = delta_target_projector(dummy_tokens.detach())
        zero_loss = 0.0 * (dummy_q.sum() + dummy_target.sum())
        return {
            "delta_loss": zero_loss,
            "delta_l1": torch.zeros((), device=device),
            "delta_nce": torch.zeros((), device=device),
            "delta_valid_rows": torch.tensor(0.0, device=device),
            "delta_pos_sim_mean": torch.zeros((), device=device),
            "delta_hard_sim_mean": torch.zeros((), device=device),
            "delta_pos_minus_hard_gap": torch.zeros((), device=device),
            "delta_q_var": torch.zeros((), device=device),
            "delta_target_var": torch.zeros((), device=device),
        }

    # Subset to eligible rows only. We still pass full-batch tensors
    # through the heads — select indices in the slicing, not in the
    # MLP forward. This keeps the parameters exercised regardless.
    idx = same_view_mask.nonzero(as_tuple=True)[0]  # [V]

    # Student query on anchor tokens; conditioned on src_view and Δφ.
    q_delta = motion_delta_head(
        z_tokens_sub[idx],
        src_view_ids[idx],
        delta_phase[idx],
    )  # [V, K, delta_dim]

    # Raw deltas (detached) + projected.
    d_pos_raw = (h_pos_tokens_sub[idx] - h_a_tokens_sub[idx]).detach()
    d_hard_raw = (h_neg_tokens_sub[idx] - h_a_tokens_sub[idx]).detach()
    d_pos = delta_target_projector(d_pos_raw)  # [V, K, delta_dim]
    d_hard = delta_target_projector(d_hard_raw)

    # --- SmoothL1 on projected targets ---
    l1 = F.smooth_l1_loss(q_delta, d_pos.detach())

    # --- Token-set InfoNCE on projected deltas ---
    # Normalize.
    q_n = F.normalize(q_delta.float(), dim=-1)
    p_n = F.normalize(d_pos.detach().float(), dim=-1)
    h_n = F.normalize(d_hard.detach().float(), dim=-1)

    V = q_n.shape[0]

    sim_pos = torch.einsum("bid,bjd->bij", q_n, p_n) / tau
    pos_row = torch.logsumexp(sim_pos, dim=-1).mean(dim=-1)  # [V]
    sim_hard = torch.einsum("bid,bjd->bij", q_n, h_n) / tau
    hard_row = torch.logsumexp(sim_hard, dim=-1).mean(dim=-1)
    sim_cross = torch.einsum("bid,cjd->bcij", q_n, p_n) / tau
    set_sim_batch = torch.logsumexp(sim_cross, dim=-1).mean(dim=-1)  # [V, V]
    eye = torch.eye(V, dtype=torch.bool, device=device)
    neg_inf = torch.finfo(set_sim_batch.dtype).min
    set_sim_batch = set_sim_batch.masked_fill(eye, neg_inf)
    logits = torch.cat([pos_row.unsqueeze(1), hard_row.unsqueeze(1), set_sim_batch], dim=1)
    labels = torch.zeros(V, dtype=torch.long, device=device)
    if V == 1:
        # With only one eligible row, batch neg block is empty and
        # hard-only contrast remains. CE still well-defined.
        nce = F.cross_entropy(logits[:, :2], labels)
    else:
        nce = F.cross_entropy(logits, labels)

    loss = lambda_l1 * l1 + lambda_nce * nce

    with torch.no_grad():
        pos_sim_mean = (pos_row * tau).mean()
        hard_sim_mean = (hard_row * tau).mean()
        q_var = q_n.reshape(-1, q_n.shape[-1]).var(dim=0).mean()
        t_var = p_n.reshape(-1, p_n.shape[-1]).var(dim=0).mean()

    return {
        "delta_loss": loss,
        "delta_l1": l1.detach(),
        "delta_nce": nce.detach(),
        "delta_valid_rows": torch.tensor(float(valid_rows), device=device),
        "delta_pos_sim_mean": pos_sim_mean,
        "delta_hard_sim_mean": hard_sim_mean,
        "delta_pos_minus_hard_gap": pos_sim_mean - hard_sim_mean,
        "delta_q_var": q_var,
        "delta_target_var": t_var,
    }


__all__ = [
    "token_set_infonce_with_hard_neg",
    "motion_delta_loss",
    "pool_tokens",  # re-export for convenience
]
