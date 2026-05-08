"""EchoMV-JEPA Stage-1 training loop.

This extends ``app.echoset_jepa.train`` with:

1. An EMA copy of the full ``StudyTransformer`` as the target encoder
   (``StudyTransformerEMA``). Target embeddings are selected AFTER full-study
   encoding — the defining JEPA-faithful addition. See plan §5, §7.5.

2. A per-step teacher-contextualization diagnostic suite (§15.1a):
   - ``z_cosine_vs_v1``       — falsification probe vs v1's pre-context target.
   - ``z_cosine_vs_isolated`` — cosine vs teacher run on each target element alone.
   - ``z_cosine_vs_peer_drop``— every N steps, drop one peer context element.

3. Optional per-modality projector (Stage-1m): ``ModalityProjectorPair`` routes
   each target row by its modality id. ``projector.num_modalities == 1``
   reproduces Stage-1 exactly.

4. Tiny matched NCE (Stage-1b): ``λ_nce ∈ {0.005, 0.01}`` by default. Stage-1
   runs with ``λ_nce = 0.0``.

The training loop is deliberately structured as a pure ``training_step``
function plus a ``main`` wrapper, paralleling ``app/echoset_jepa/train.py`` so
the two code paths can be compared line-by-line during review.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch

from src.models.echomv_jepa import (
    ModalityProjectorPair,
    StudyTransformerEMA,
    TokenStudyTransformer,
    cosine_regress,
    covariance_penalty,
    layernorm_cosine,
    matched_nce,
    matched_rank_metrics,
    prioritized_neg_pool,
)
from src.models.meta_embeddings import MODALITY_VOCAB, MetaDropout, MetaEmbeddings
from src.models.study_projectors import cosine_schedule
from src.models.study_transformer import StudyTransformer, StudyTransformerConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------


@dataclass
class StepOutput:
    loss: torch.Tensor
    loss_regress: torch.Tensor
    loss_nce: torch.Tensor
    diagnostics: Dict[str, float]


def _route_project(
    proj: Any,  # EMAProjectorPair | ModalityProjectorPair
    x: torch.Tensor,  # (N, d_model)
    modality_ids: torch.Tensor,  # (N,) long
    *,
    use_teacher: bool,
) -> torch.Tensor:
    """Call the correct projector variant based on whether it's per-modality."""
    if isinstance(proj, ModalityProjectorPair):
        fn = proj.teacher_forward if use_teacher else proj.student_forward
        return fn(x, modality_ids)
    fn = proj.teacher_forward if use_teacher else proj.student_forward
    return fn(x)


def training_step_echomv(
    batch: Dict[str, torch.Tensor],
    st: StudyTransformer,
    meta: MetaEmbeddings,
    proj: Any,  # EMAProjectorPair | ModalityProjectorPair
    teacher_st: StudyTransformerEMA,
    *,
    lambda_nce: float = 0.0,
    tau_nce: float = 0.1,
    lambda_cov: float = 0.0,
    var_floor: float = 0.0,
    p_target_self_mask: float = 0.0,
    include_target_phase: bool = True,
    include_target_quality: bool = False,
    diag_peer_drop_every_n_steps: int = 50,
    diag_extra_every_n_steps: int = 25,
    global_step: int = 0,
) -> StepOutput:
    """Single forward + loss for EchoMV-JEPA Stage-1 / 1b / 1m / Arm A / Arm B.

    Loss:
      L_total = L_cosine
              + lambda_nce * L_matched_nce      (Arm A; default 0)
              + lambda_cov * (L_cov + L_var)    (Arm A; default 0)

    Arm B:
      p_target_self_mask > 0 replaces a fraction of target-element rows in the
      teacher's input with the student's learned mask_token (+ target meta)
      BEFORE the teacher's forward pass. The teacher still sees all context
      elements unchanged; only target visual content is masked. This forces
      the teacher to rely on cross-element context for the target embedding.

    Contract (required batch keys):
      - All EchoSet v1 keys (see ``app/echoset_jepa/train.training_step``).
      - ``full_elements``, ``full_pad_mask``   — full unmasked study (ctx ∥ tgt).
      - ``full_meta_{view,modality,phase,quality}`` — long tensors (B, M).
      - ``target_idx_in_full``                 — (B, M_tgt) long.
    """
    # --- student path: masked/incomplete study ----------------------------
    ctx_meta_add = meta.encode_context(
        batch["ctx_meta_view"],
        batch["ctx_meta_modality"],
        batch["ctx_meta_phase"],
        batch["ctx_meta_quality"],
    )
    tgt_meta_add = meta.encode_target_slot(
        batch["tgt_meta_view"],
        batch["tgt_meta_modality"],
        phase_ids=batch.get("tgt_meta_phase"),
        include_phase=include_target_phase,
        include_quality=include_target_quality,
        quality_ids=batch.get("tgt_meta_quality"),
    )

    h_study, h_mask = st(
        ctx_elements=batch["ctx_elements"],
        ctx_meta_add=ctx_meta_add,
        ctx_pad_mask=batch["ctx_pad_mask"],
        tgt_meta_add=tgt_meta_add,
        tgt_pad_mask=batch["tgt_pad_mask"],
    )  # h_mask: (B, M_tgt, d_model)

    # --- teacher path: full unmasked study, no meta dropout ---------------
    with torch.no_grad():
        prev_training = meta.training
        meta.eval()  # disables per-field dropout
        # Teacher sees true phase & quality; target-slot exclusions don't apply
        # because the teacher encodes EVERY position the same way.
        full_meta_add = meta.encode_context(
            batch["full_meta_view"],
            batch["full_meta_modality"],
            batch["full_meta_phase"],
            batch["full_meta_quality"],
        )
        meta.train(prev_training)

        # Arm B — teacher target self-masking (pooled path).
        # Replace the visual content at target-element positions with zeros with
        # probability ``p_target_self_mask``. We replace the raw element vector
        # (pre-clip_in) with zeros; the teacher's ``clip_in(0) = bias_only``
        # plays the role of a learned mask-content vector. Meta is preserved.
        # Only target rows are affected; context rows and the student path are
        # untouched (anti-leak rule 1 still holds — student sees no target
        # visual content even when p_target_self_mask = 0).
        teacher_full_elements = batch["full_elements"]
        teacher_selfmask_rate = 0.0
        if p_target_self_mask > 0.0:
            B = batch["full_elements"].shape[0]
            M_tgt_local = batch["tgt_pad_mask"].shape[1]
            drop_b = (
                torch.rand(B, M_tgt_local, device=batch["full_elements"].device) < p_target_self_mask
            )  # (B, M_tgt) bool
            # Zero out only rows that are both "to-drop" and unpadded targets.
            drop_b = drop_b & (~batch["tgt_pad_mask"])
            teacher_selfmask_rate = drop_b.float().mean().item()
            # Scatter zero-content into the full_elements at target indices.
            tgt_idx_pre = batch["target_idx_in_full"]  # (B, M_tgt)
            idx_exp_pre = tgt_idx_pre.unsqueeze(-1).expand(-1, -1, batch["full_elements"].shape[-1])
            mask_val = torch.zeros_like(
                torch.gather(batch["full_elements"], dim=1, index=idx_exp_pre)
            )  # (B, M_tgt, d_clip)
            # Only replace where drop_b is True.
            rows_at_tgt = torch.gather(batch["full_elements"], dim=1, index=idx_exp_pre)
            replaced = torch.where(drop_b.unsqueeze(-1), mask_val, rows_at_tgt)
            teacher_full_elements = batch["full_elements"].clone()
            teacher_full_elements.scatter_(dim=1, index=idx_exp_pre, src=replaced)

        z_per_element = teacher_st.forward_contextualized(
            teacher_full_elements,
            full_meta_add,
            batch["full_pad_mask"],
        )  # (B, M, d_model)

        # Gather teacher hidden states at target positions
        tgt_idx = batch["target_idx_in_full"]  # (B, M_tgt)
        idx_exp = tgt_idx.unsqueeze(-1).expand(-1, -1, z_per_element.shape[-1])
        z_at_targets = torch.gather(z_per_element, dim=1, index=idx_exp)  # (B, M_tgt, d_model)

    # --- project, select valid targets, compute loss ---------------------
    valid = ~batch["tgt_pad_mask"]  # (B, M_tgt)
    h_flat_pre = h_mask[valid]  # (N, d_model)
    z_flat_pre = z_at_targets[valid]  # (N, d_model)
    v_flat = batch["tgt_meta_view"][valid]
    m_flat = batch["tgt_meta_modality"][valid]
    p_flat = batch["tgt_meta_phase"][valid]
    s_flat = batch["study_id_int"].unsqueeze(1).expand_as(batch["tgt_meta_view"])[valid]

    # Student + teacher project (routed by modality for Stage-1m)
    h_flat = _route_project(proj, h_flat_pre, m_flat, use_teacher=False)  # (N, d_proj)
    with torch.no_grad():
        z_flat = _route_project(proj, z_flat_pre, m_flat, use_teacher=True).detach()

    loss_reg = cosine_regress(h_flat, z_flat)

    if lambda_nce > 0.0 and h_flat.shape[0] > 1:
        neg_mask, nce_diag = prioritized_neg_pool(v_flat, m_flat, p_flat, s_flat)
        loss_nce = matched_nce(h_flat, z_flat, neg_mask, tau=tau_nce)
    else:
        nce_diag = {}
        loss_nce = torch.zeros((), device=h_flat.device)

    # Arm A — covariance regularizer on student projections.
    if lambda_cov > 0.0 and h_flat.shape[0] > 1:
        l_cov, l_var = covariance_penalty(h_flat, var_floor=var_floor)
        loss_cov = l_cov + l_var
    else:
        loss_cov = torch.zeros((), device=h_flat.device)

    loss = loss_reg + lambda_nce * loss_nce + lambda_cov * loss_cov

    # --- diagnostics ------------------------------------------------------
    with torch.no_grad():
        # Representation health
        var_t = h_flat.std(dim=0).mean().item() if h_flat.shape[0] > 1 else 0.0
        if h_flat.shape[0] > 1:
            cov = h_flat - h_flat.mean(0, keepdim=True)
            cov = cov.t() @ cov / max(h_flat.shape[0] - 1, 1)
            cov_off = cov.fill_diagonal_(0.0).abs().mean().item()
        else:
            cov_off = 0.0

        # Teacher contextualization diagnostics (§15.1a)
        # 1. z_cosine_vs_v1: cosine between full-study teacher target and v1's pre-context target
        z_v1_pre = st.clip_in(batch["tgt_elements"])[valid]  # (N, d_model)
        z_v1 = _route_project(proj, z_v1_pre, m_flat, use_teacher=True)  # (N, d_proj)
        z_cosine_vs_v1 = layernorm_cosine(z_flat, z_v1).mean().item()

        # 2. z_cosine_vs_isolated: teacher on each target element alone
        # Encode target elements in isolation using the teacher, with true meta.
        # We reuse full_meta_add rows at target indices via the same gather.
        tgt_meta_add_true = torch.gather(full_meta_add, dim=1, index=idx_exp)  # (B, M_tgt, d_model)
        z_iso_per = teacher_st.forward_isolated(batch["tgt_elements"], tgt_meta_add_true)
        z_iso_flat_pre = z_iso_per[valid]  # (N, d_model)
        z_iso = _route_project(proj, z_iso_flat_pre, m_flat, use_teacher=True)
        z_cosine_vs_isolated = layernorm_cosine(z_flat, z_iso).mean().item()

        # 3. z_cosine_vs_peer_drop: every N steps, drop one random peer context elem
        z_cosine_vs_peer_drop = float("nan")
        if diag_peer_drop_every_n_steps > 0 and (global_step % diag_peer_drop_every_n_steps == 0):
            B, M = batch["full_pad_mask"].shape
            # Build a "drop one context" mask per row: find unpadded non-target positions,
            # randomly select one to mark padded. If no eligible context, skip that row.
            drop_mask = batch["full_pad_mask"].clone()
            # context positions are indices not in target_idx_in_full and not already padded
            device = drop_mask.device
            is_tgt = torch.zeros_like(drop_mask)
            is_tgt.scatter_(1, batch["target_idx_in_full"], True)
            eligible = (~drop_mask) & (~is_tgt)  # (B, M) bool
            elig_count = eligible.sum(dim=1)  # (B,)
            if (elig_count > 0).any():
                # random index within the eligible set
                rnd = torch.rand(B, M, device=device)
                rnd = rnd.masked_fill(~eligible, -1.0)
                chosen = rnd.argmax(dim=1)  # (B,) — -1 row will pick padded pos (skipped)
                rows_ok = elig_count > 0
                drop_mask[rows_ok, chosen[rows_ok]] = True
                z_pd_per = teacher_st.forward_contextualized(
                    batch["full_elements"],
                    full_meta_add,
                    drop_mask,
                )
                z_pd_at_tgt = torch.gather(z_pd_per, dim=1, index=idx_exp)
                z_pd_flat_pre = z_pd_at_tgt[valid]
                z_pd = _route_project(proj, z_pd_flat_pre, m_flat, use_teacher=True)
                z_cosine_vs_peer_drop = layernorm_cosine(z_flat, z_pd).mean().item()

        # --- New Part 2 diagnostics --------------------------------------
        # Cadence-gated; computed every ``diag_extra_every_n_steps``.
        student_context_delta = float("nan")
        target_meta_only_cos = float("nan")
        target_meta_only_gap = float("nan")
        matched_rank_top1 = float("nan")
        matched_rank_top5 = float("nan")
        pos_minus_hardneg_gap = float("nan")

        cadence_ok = diag_extra_every_n_steps > 0 and (global_step % diag_extra_every_n_steps == 0)
        if cadence_ok and h_flat.shape[0] > 1:
            # (A) student_context_delta — shuffle context across studies and rerun the
            # *student* forward; measure 1 - cos(h_actual, h_shuffled). A healthy student
            # should produce different target predictions when given another study's context.
            B = batch["ctx_elements"].shape[0]
            if B > 1:
                perm = torch.randperm(B, device=batch["ctx_elements"].device)
                # If the permutation is identity (rare), rotate by 1.
                if int(perm[0].item()) == 0 and B > 1 and int(perm[1].item()) == 1:
                    perm = torch.roll(torch.arange(B, device=perm.device), shifts=1)
                ctx_shuf = batch["ctx_elements"][perm]
                ctx_meta_add_shuf = ctx_meta_add[perm]
                ctx_pad_shuf = batch["ctx_pad_mask"][perm]
                _, h_mask_shuf = st(
                    ctx_elements=ctx_shuf,
                    ctx_meta_add=ctx_meta_add_shuf,
                    ctx_pad_mask=ctx_pad_shuf,
                    tgt_meta_add=tgt_meta_add,
                    tgt_pad_mask=batch["tgt_pad_mask"],
                )
                h_shuf_flat_pre = h_mask_shuf[valid]
                h_shuf_flat = _route_project(proj, h_shuf_flat_pre, m_flat, use_teacher=False)
                cos_shuf = layernorm_cosine(h_flat, h_shuf_flat).mean().item()
                student_context_delta = 1.0 - cos_shuf

            # (B) target_meta_only_gap — run the student with NO context (only [STUDY]
            # + target mask slots). A healthy study-level objective should have
            # cos(h_actual, z_t) strictly greater than cos(h_meta, z_t).
            empty_ctx = batch["ctx_elements"][:, :0]  # (B, 0, d_clip)
            empty_meta_add = ctx_meta_add[:, :0]
            empty_pad = batch["ctx_pad_mask"][:, :0]
            _, h_mask_meta = st(
                ctx_elements=empty_ctx,
                ctx_meta_add=empty_meta_add,
                ctx_pad_mask=empty_pad,
                tgt_meta_add=tgt_meta_add,
                tgt_pad_mask=batch["tgt_pad_mask"],
            )
            h_meta_flat_pre = h_mask_meta[valid]
            h_meta_flat = _route_project(proj, h_meta_flat_pre, m_flat, use_teacher=False)
            cos_actual = layernorm_cosine(h_flat, z_flat).mean().item()
            cos_meta = layernorm_cosine(h_meta_flat, z_flat).mean().item()
            target_meta_only_cos = cos_meta
            target_meta_only_gap = cos_actual - cos_meta

            # (C) matched_rank metrics — rank of positive vs matched negatives.
            if h_flat.shape[0] > 1:
                neg_mask_diag, _ = prioritized_neg_pool(v_flat, m_flat, p_flat, s_flat)
                rank_stats = matched_rank_metrics(h_flat, z_flat, neg_mask_diag)
                matched_rank_top1 = rank_stats["matched_rank_top1"]
                matched_rank_top5 = rank_stats["matched_rank_top5"]
                pos_minus_hardneg_gap = rank_stats["pos_minus_hardneg_gap_mean"]

    diagnostics: Dict[str, float] = {
        "loss_regress": loss_reg.item(),
        "loss_nce": loss_nce.item(),
        "loss_cov": loss_cov.item(),
        "var_t": var_t,
        "cov_off": cov_off,
        "z_cosine_vs_v1": z_cosine_vs_v1,
        "z_cosine_vs_isolated": z_cosine_vs_isolated,
        "z_cosine_vs_peer_drop": z_cosine_vs_peer_drop,
        "student_context_delta": student_context_delta,
        "target_meta_only_cos": target_meta_only_cos,
        "target_meta_only_gap": target_meta_only_gap,
        "matched_rank_top1": matched_rank_top1,
        "matched_rank_top5": matched_rank_top5,
        "pos_minus_hardneg_gap": pos_minus_hardneg_gap,
        "teacher_selfmask_rate": teacher_selfmask_rate,
        **nce_diag,
    }
    return StepOutput(loss=loss, loss_regress=loss_reg, loss_nce=loss_nce, diagnostics=diagnostics)


# ---------------------------------------------------------------------------
# Token-level training step (Option A: on-the-fly encoding, no pooled cache)
# ---------------------------------------------------------------------------


def _encode_clips_to_tokens(
    online_encoder,  # OnlineVJepaEncoder
    clips: torch.Tensor,  # (B, M, 3, T_frames, H, W)
) -> torch.Tensor:
    """Flatten (B, M, 3, T, H, W) → (B*M, 3, T, H, W), encode, reshape to (B, M, T_tok, d_clip)."""
    B, M, C, Tf, H, W = clips.shape
    flat = clips.reshape(B * M, C, Tf, H, W)
    toks = online_encoder.forward_tokens(flat)  # (B*M, T_tok, d_clip)
    T_tok, d_clip = toks.shape[-2], toks.shape[-1]
    return toks.reshape(B, M, T_tok, d_clip)


def training_step_echomv_tokens(
    batch: Dict[str, torch.Tensor],
    st: StudyTransformer,
    meta: MetaEmbeddings,
    proj: Any,  # EMAProjectorPair | ModalityProjectorPair
    teacher_st: StudyTransformerEMA,
    online_encoder,  # OnlineVJepaEncoder
    *,
    lambda_nce: float = 0.0,
    tau_nce: float = 0.1,
    lambda_cov: float = 0.0,
    var_floor: float = 0.0,
    p_target_token_mask: float = 0.0,
    include_target_phase: bool = True,
    include_target_quality: bool = False,
    diag_peer_drop_every_n_steps: int = 50,
    diag_extra_every_n_steps: int = 25,
    global_step: int = 0,
) -> StepOutput:
    """Option A: token-level, on-the-fly clip encoding.

    Differences from ``training_step_echomv``:
      - Batch carries ``ctx_clips`` / ``tgt_clips`` / ``full_clips`` in shape
        ``(B, M_*, 3, T_frames, H, W)`` instead of pooled element vectors.
      - An ``OnlineVJepaEncoder`` encodes clips → ``(B, M_*, T_tok, d_clip)``
        inside ``torch.no_grad`` (the encoder is frozen).
      - Student and teacher study transformers are wrapped in
        ``TokenStudyTransformer`` to consume the token sequence. The
        per-element output is the mean of T_tok contextualized tokens.
      - Downstream loss and §15.1a diagnostics are identical to the pooled
        path because the wrapper returns per-element ``(B, M, d_model)``.

    Contract (required batch keys):
      - ``ctx_clips, tgt_clips, full_clips`` — pixel tensors.
      - ``ctx_pad_mask, tgt_pad_mask, full_pad_mask`` — element-level pads.
      - ``ctx_meta_*, tgt_meta_*, full_meta_*`` — element-level meta ids.
      - ``target_idx_in_full`` — (B, M_tgt) long.
      - ``study_id_int`` — (B,) long.
    """
    B, M_tgt = batch["tgt_pad_mask"].shape
    _, M_ctx = batch["ctx_pad_mask"].shape
    _, M_full = batch["full_pad_mask"].shape

    # --- encode clips to tokens (frozen, no grad) -------------------------
    # The student also encodes (same tokens since encoder is frozen); we
    # reuse one encoding pass and feed both paths.
    with torch.no_grad():
        full_tokens = _encode_clips_to_tokens(online_encoder, batch["full_clips"])  # (B, M_full, T, d_clip)
        # Split into ctx/tgt using index order (ctx comes first by collate convention).
        ctx_tokens = full_tokens[:, :M_ctx]
        tgt_tokens = full_tokens[:, M_ctx : M_ctx + M_tgt]

    # --- student path: masked/incomplete study (tokens) -------------------
    ctx_meta_add = meta.encode_context(
        batch["ctx_meta_view"],
        batch["ctx_meta_modality"],
        batch["ctx_meta_phase"],
        batch["ctx_meta_quality"],
    )
    tgt_meta_add = meta.encode_target_slot(
        batch["tgt_meta_view"],
        batch["tgt_meta_modality"],
        phase_ids=batch.get("tgt_meta_phase"),
        include_phase=include_target_phase,
        include_quality=include_target_quality,
        quality_ids=batch.get("tgt_meta_quality"),
    )

    # The student in token mode sees the *masked* study: context clip tokens
    # for ctx slots + zero-token "mask" for target slots. We emulate this by
    # running the TokenStudyTransformer on the full token stack but with a
    # zero-filled token block at target positions and the target-slot meta
    # added. This mirrors what ``StudyTransformer.forward`` does at the
    # pooled layer (mask_token + tgt_meta_add at target positions).
    tok_st = TokenStudyTransformer(st)

    # Build student input: concat (ctx_tokens, zero_token_block_for_targets)
    # along the element axis. Meta is ctx_meta_add ∥ tgt_meta_add.
    if M_tgt > 0:
        zero_tgt_tokens = torch.zeros_like(tgt_tokens)
        student_tokens = torch.cat([ctx_tokens, zero_tgt_tokens], dim=1)  # (B, M_full, T, d_clip)
    else:
        student_tokens = ctx_tokens
    student_meta = torch.cat([ctx_meta_add, tgt_meta_add], dim=1) if M_tgt > 0 else ctx_meta_add
    student_pad = batch["full_pad_mask"]
    h_per_elem = tok_st.forward_contextualized(student_tokens, student_meta, student_pad)
    # h_per_elem: (B, M_full, d_model). Gather at target indices.
    tgt_idx = batch["target_idx_in_full"]
    idx_exp_d = tgt_idx.unsqueeze(-1).expand(-1, -1, h_per_elem.shape[-1])
    h_mask = torch.gather(h_per_elem, dim=1, index=idx_exp_d)  # (B, M_tgt, d_model)

    # --- teacher path: full unmasked study (tokens) -----------------------
    with torch.no_grad():
        prev_training = meta.training
        meta.eval()
        full_meta_add = meta.encode_context(
            batch["full_meta_view"],
            batch["full_meta_modality"],
            batch["full_meta_phase"],
            batch["full_meta_quality"],
        )
        meta.train(prev_training)

        # Arm B — teacher target token self-masking (token path).
        # Draw a per-row Bernoulli(p_target_token_mask) mask over target
        # element tokens. Replace masked token content with zero; meta is
        # broadcast at the TokenStudyTransformer wrapper layer, so we preserve
        # meta by only zeroing token *content*. Context tokens are never
        # masked here; the student path is untouched.
        teacher_full_tokens = full_tokens
        teacher_selfmask_rate = 0.0
        if p_target_token_mask > 0.0 and M_tgt > 0:
            T_tok = full_tokens.shape[2]
            d_clip_enc = full_tokens.shape[3]
            # (B, M_tgt, T) mask over tokens of target elements
            drop_b = torch.rand(B, M_tgt, T_tok, device=full_tokens.device) < p_target_token_mask
            # Unpadded target rows only
            drop_b = drop_b & (~batch["tgt_pad_mask"]).unsqueeze(-1)
            teacher_selfmask_rate = drop_b.float().mean().item()

            # Build teacher_full_tokens by scattering zero into target rows.
            teacher_full_tokens = full_tokens.clone()
            # tgt_idx is (B, M_tgt); expand to (B, M_tgt, T, d) for scatter.
            idx_full = tgt_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, T_tok, d_clip_enc)
            # Pull current target-row tokens, apply mask, scatter back.
            rows_at_tgt = torch.gather(teacher_full_tokens, dim=1, index=idx_full)
            rows_masked = torch.where(drop_b.unsqueeze(-1), torch.zeros_like(rows_at_tgt), rows_at_tgt)
            teacher_full_tokens.scatter_(dim=1, index=idx_full, src=rows_masked)

        teacher_tok = TokenStudyTransformer(teacher_st.teacher)
        z_per_element = teacher_tok.forward_contextualized(
            teacher_full_tokens, full_meta_add, batch["full_pad_mask"]
        )  # (B, M_full, d_model)
        idx_exp_t = tgt_idx.unsqueeze(-1).expand(-1, -1, z_per_element.shape[-1])
        z_at_targets = torch.gather(z_per_element, dim=1, index=idx_exp_t)

    # --- project, select valid targets, compute loss ---------------------
    valid = ~batch["tgt_pad_mask"]
    h_flat_pre = h_mask[valid]
    z_flat_pre = z_at_targets[valid]
    v_flat = batch["tgt_meta_view"][valid]
    m_flat = batch["tgt_meta_modality"][valid]
    p_flat = batch["tgt_meta_phase"][valid]
    s_flat = batch["study_id_int"].unsqueeze(1).expand_as(batch["tgt_meta_view"])[valid]

    h_flat = _route_project(proj, h_flat_pre, m_flat, use_teacher=False)
    with torch.no_grad():
        z_flat = _route_project(proj, z_flat_pre, m_flat, use_teacher=True).detach()

    loss_reg = cosine_regress(h_flat, z_flat)

    if lambda_nce > 0.0 and h_flat.shape[0] > 1:
        neg_mask, nce_diag = prioritized_neg_pool(v_flat, m_flat, p_flat, s_flat)
        loss_nce = matched_nce(h_flat, z_flat, neg_mask, tau=tau_nce)
    else:
        nce_diag = {}
        loss_nce = torch.zeros((), device=h_flat.device)

    if lambda_cov > 0.0 and h_flat.shape[0] > 1:
        l_cov, l_var = covariance_penalty(h_flat, var_floor=var_floor)
        loss_cov = l_cov + l_var
    else:
        loss_cov = torch.zeros((), device=h_flat.device)

    loss = loss_reg + lambda_nce * loss_nce + lambda_cov * loss_cov

    # --- diagnostics (§15.1a, token-level) -------------------------------
    with torch.no_grad():
        var_t = h_flat.std(dim=0).mean().item() if h_flat.shape[0] > 1 else 0.0
        if h_flat.shape[0] > 1:
            cov = h_flat - h_flat.mean(0, keepdim=True)
            cov = cov.t() @ cov / max(h_flat.shape[0] - 1, 1)
            cov_off = cov.fill_diagonal_(0.0).abs().mean().item()
        else:
            cov_off = 0.0

        # z_cosine_vs_v1: "v1" at token granularity = mean-pool tokens per element,
        # then apply student's clip_in linear + EMA projector. This is the pooled-
        # cache stale Stage-0 baseline lifted to the token setting.
        tgt_tokens_mean = tgt_tokens.mean(dim=2)  # (B, M_tgt, d_clip)
        z_v1_pre = st.clip_in(tgt_tokens_mean)[valid]  # (N, d_model)
        z_v1 = _route_project(proj, z_v1_pre, m_flat, use_teacher=True)
        z_cosine_vs_v1 = layernorm_cosine(z_flat, z_v1).mean().item()

        # z_cosine_vs_isolated: teacher on each target element's tokens alone.
        # We run the teacher element-by-element (M_tgt passes) over the per-
        # element token block only.
        z_iso_per = torch.empty_like(z_at_targets)
        tgt_meta_add_true = torch.gather(full_meta_add, dim=1, index=idx_exp_t)
        no_pad_1 = torch.zeros(B, 1, dtype=torch.bool, device=full_tokens.device)
        for m in range(M_tgt):
            el_tok = tgt_tokens[:, m : m + 1]  # (B, 1, T, d_clip)
            el_meta = tgt_meta_add_true[:, m : m + 1]
            z_iso_per[:, m : m + 1] = teacher_tok.forward_contextualized(el_tok, el_meta, no_pad_1)
        z_iso_flat_pre = z_iso_per[valid]
        z_iso = _route_project(proj, z_iso_flat_pre, m_flat, use_teacher=True)
        z_cosine_vs_isolated = layernorm_cosine(z_flat, z_iso).mean().item()

        # z_cosine_vs_peer_drop: drop one random non-target peer, rerun teacher.
        z_cosine_vs_peer_drop = float("nan")
        if diag_peer_drop_every_n_steps > 0 and (global_step % diag_peer_drop_every_n_steps == 0):
            drop_mask = batch["full_pad_mask"].clone()
            device = drop_mask.device
            is_tgt = torch.zeros_like(drop_mask)
            is_tgt.scatter_(1, batch["target_idx_in_full"], True)
            eligible = (~drop_mask) & (~is_tgt)
            elig_count = eligible.sum(dim=1)
            if (elig_count > 0).any():
                rnd = torch.rand(B, M_full, device=device)
                rnd = rnd.masked_fill(~eligible, -1.0)
                chosen = rnd.argmax(dim=1)
                rows_ok = elig_count > 0
                drop_mask[rows_ok, chosen[rows_ok]] = True
                z_pd_per = teacher_tok.forward_contextualized(full_tokens, full_meta_add, drop_mask)
                z_pd_at_tgt = torch.gather(z_pd_per, dim=1, index=idx_exp_t)
                z_pd_flat_pre = z_pd_at_tgt[valid]
                z_pd = _route_project(proj, z_pd_flat_pre, m_flat, use_teacher=True)
                z_cosine_vs_peer_drop = layernorm_cosine(z_flat, z_pd).mean().item()

        # --- New Part 2 diagnostics (token path) -------------------------
        student_context_delta = float("nan")
        target_meta_only_cos = float("nan")
        target_meta_only_gap = float("nan")
        matched_rank_top1 = float("nan")
        matched_rank_top5 = float("nan")
        pos_minus_hardneg_gap = float("nan")

        cadence_ok = diag_extra_every_n_steps > 0 and (global_step % diag_extra_every_n_steps == 0)
        if cadence_ok and h_flat.shape[0] > 1:
            # (A) student_context_delta — shuffle context-token block across
            # the batch; keep target-slot zero-content and target meta fixed.
            if B > 1:
                perm = torch.randperm(B, device=ctx_tokens.device)
                if int(perm[0].item()) == 0 and B > 1 and int(perm[1].item()) == 1:
                    perm = torch.roll(torch.arange(B, device=perm.device), shifts=1)
                ctx_tok_shuf = ctx_tokens[perm]
                ctx_meta_add_shuf = ctx_meta_add[perm]
                ctx_pad_shuf = batch["ctx_pad_mask"][perm]
                # Rebuild student input with shuffled ctx.
                if M_tgt > 0:
                    student_tokens_shuf = torch.cat([ctx_tok_shuf, zero_tgt_tokens], dim=1)
                else:
                    student_tokens_shuf = ctx_tok_shuf
                student_meta_shuf = (
                    torch.cat([ctx_meta_add_shuf, tgt_meta_add], dim=1) if M_tgt > 0 else ctx_meta_add_shuf
                )
                # Rebuild full_pad_mask for shuffled ctx (use shuffled ctx_pad,
                # unchanged tgt_pad, concat).
                student_pad_shuf = torch.cat([ctx_pad_shuf, batch["tgt_pad_mask"]], dim=1)
                h_per_elem_shuf = tok_st.forward_contextualized(
                    student_tokens_shuf, student_meta_shuf, student_pad_shuf
                )
                h_mask_shuf = torch.gather(h_per_elem_shuf, dim=1, index=idx_exp_d)
                h_shuf_flat_pre = h_mask_shuf[valid]
                h_shuf_flat = _route_project(proj, h_shuf_flat_pre, m_flat, use_teacher=False)
                cos_shuf = layernorm_cosine(h_flat, h_shuf_flat).mean().item()
                student_context_delta = 1.0 - cos_shuf

            # (B) target_meta_only_gap — run student with NO context tokens at all.
            empty_ctx_tok = ctx_tokens[:, :0]
            empty_meta_add = ctx_meta_add[:, :0]
            empty_pad = batch["ctx_pad_mask"][:, :0]
            if M_tgt > 0:
                student_tokens_meta = torch.cat([empty_ctx_tok, zero_tgt_tokens], dim=1)
            else:
                student_tokens_meta = empty_ctx_tok
            student_meta_meta = torch.cat([empty_meta_add, tgt_meta_add], dim=1) if M_tgt > 0 else empty_meta_add
            student_pad_meta = torch.cat([empty_pad, batch["tgt_pad_mask"]], dim=1)
            h_per_elem_meta = tok_st.forward_contextualized(student_tokens_meta, student_meta_meta, student_pad_meta)
            # tgt_idx is built off the full concat — with empty ctx the target
            # positions are simply [0, M_tgt) in the meta-only stream.
            meta_tgt_idx = torch.arange(M_tgt, dtype=torch.long, device=ctx_tokens.device).unsqueeze(0).expand(B, -1)
            meta_idx_exp = meta_tgt_idx.unsqueeze(-1).expand(-1, -1, h_per_elem_meta.shape[-1])
            h_mask_meta = torch.gather(h_per_elem_meta, dim=1, index=meta_idx_exp)
            h_meta_flat_pre = h_mask_meta[valid]
            h_meta_flat = _route_project(proj, h_meta_flat_pre, m_flat, use_teacher=False)
            cos_actual = layernorm_cosine(h_flat, z_flat).mean().item()
            cos_meta = layernorm_cosine(h_meta_flat, z_flat).mean().item()
            target_meta_only_cos = cos_meta
            target_meta_only_gap = cos_actual - cos_meta

            # (C) matched_rank metrics
            if h_flat.shape[0] > 1:
                neg_mask_diag, _ = prioritized_neg_pool(v_flat, m_flat, p_flat, s_flat)
                rank_stats = matched_rank_metrics(h_flat, z_flat, neg_mask_diag)
                matched_rank_top1 = rank_stats["matched_rank_top1"]
                matched_rank_top5 = rank_stats["matched_rank_top5"]
                pos_minus_hardneg_gap = rank_stats["pos_minus_hardneg_gap_mean"]

    diagnostics: Dict[str, float] = {
        "loss_regress": loss_reg.item(),
        "loss_nce": loss_nce.item(),
        "loss_cov": loss_cov.item(),
        "var_t": var_t,
        "cov_off": cov_off,
        "z_cosine_vs_v1": z_cosine_vs_v1,
        "z_cosine_vs_isolated": z_cosine_vs_isolated,
        "z_cosine_vs_peer_drop": z_cosine_vs_peer_drop,
        "student_context_delta": student_context_delta,
        "target_meta_only_cos": target_meta_only_cos,
        "target_meta_only_gap": target_meta_only_gap,
        "matched_rank_top1": matched_rank_top1,
        "matched_rank_top5": matched_rank_top5,
        "pos_minus_hardneg_gap": pos_minus_hardneg_gap,
        "teacher_selfmask_rate": teacher_selfmask_rate,
        **nce_diag,
    }
    return StepOutput(loss=loss, loss_regress=loss_reg, loss_nce=loss_nce, diagnostics=diagnostics)


# ---------------------------------------------------------------------------
# Arm C — Global study-token JEPA (pooled path)
# ---------------------------------------------------------------------------


# Study-level element corruption moved to
# ``src/models/echomv_jepa/study_corruption.py`` so the full-joint trainer
# can reuse it without importing Arm A/B/C's train module. Re-exported here
# under the original private name for back-compatibility with in-file
# callers.
from src.models.echomv_jepa.study_corruption import apply_study_corruption as _apply_study_corruption  # noqa: E402,F401


def training_step_echomv_global(
    batch: Dict[str, torch.Tensor],
    st: StudyTransformer,
    meta: MetaEmbeddings,
    proj: Any,  # EMAProjectorPair | ModalityProjectorPair
    teacher_st: StudyTransformerEMA,
    *,
    lambda_nce: float = 0.0,
    tau_nce: float = 0.1,
    lambda_cov: float = 0.0,
    var_floor: float = 0.0,
    corruption_mix: Optional[Dict[str, float]] = None,
    diag_extra_every_n_steps: int = 25,
    global_step: int = 0,
) -> StepOutput:
    """Arm C — global study-token JEPA (pooled element inputs).

    Predicts the teacher's [STUDY]-token readout h_study from a corrupted
    student view of the same study. No per-element masking; corruption
    happens at element granularity.

    Contract (required batch keys, same as pooled echomv_collate):
      - ``full_elements``, ``full_meta_{view,modality,phase,quality}``
      - ``full_pad_mask``, ``study_id_int``, ``n_elements``
    """
    if corruption_mix is None:
        corruption_mix = {
            "random_element_dropout": 0.30,
            "whole_view_family_dropout": 0.25,
            "whole_modality_dropout": 0.15,
            "no_dropout": 0.30,
        }

    full_el = batch["full_elements"]
    full_pad = batch["full_pad_mask"]
    B = full_el.shape[0]

    # --- teacher path: full unmasked study -------------------------------
    with torch.no_grad():
        prev_training = meta.training
        meta.eval()
        full_meta_add = meta.encode_context(
            batch["full_meta_view"],
            batch["full_meta_modality"],
            batch["full_meta_phase"],
            batch["full_meta_quality"],
        )
        meta.train(prev_training)
        _, z_study = teacher_st.teacher.forward_with_study_token(
            full_el,
            full_meta_add,
            full_pad,
        )  # (B, d_model)

    # --- student path: corrupted study -----------------------------------
    # Encode meta (with dropout in train mode — intended for the student).
    ctx_meta_add = meta.encode_context(
        batch["full_meta_view"],
        batch["full_meta_modality"],
        batch["full_meta_phase"],
        batch["full_meta_quality"],
    )
    # Local CPU RNG seeded by step for reproducibility (torch.multinomial
    # does not accept a CUDA generator against CPU weights).
    g = torch.Generator()  # CPU device by default
    g.manual_seed(global_step * 7919 + 13)
    ctx_el_corrupt, ctx_pad_corrupt = _apply_study_corruption(
        full_el,
        ctx_meta_add,
        full_pad,
        batch["full_meta_view"],
        batch["full_meta_modality"],
        corruption_mix,
        g,
    )
    _, h_study = st.forward_with_study_token(ctx_el_corrupt, ctx_meta_add, ctx_pad_corrupt)

    # Route through student projector (assume shared num_modalities=1 for Arm C;
    # modality-routing on [STUDY] is ill-defined since [STUDY] has no modality id).
    if isinstance(proj, ModalityProjectorPair):
        # Use pair 0 as the canonical study projector.
        h_t = proj.pairs[0].student_forward(h_study)
        with torch.no_grad():
            z_t = proj.pairs[0].teacher_forward(z_study).detach()
    else:
        h_t = proj.student_forward(h_study)
        with torch.no_grad():
            z_t = proj.teacher_forward(z_study).detach()

    loss_reg = cosine_regress(h_t, z_t)

    # Study-level matched NCE: "matched" here means same study-size bucket.
    # For the first smoke we use a simpler setup — all other rows are valid
    # negatives (same-patient exclusion is TODO; study_id_int is unique per
    # study but shared patients are rare in a random batch).
    nce_diag: Dict[str, float] = {}
    if lambda_nce > 0.0 and B > 1:
        neg_mask = torch.ones(B, B, dtype=torch.bool, device=h_t.device)
        # Same-study exclusion: diagonal is positive, no two rows share a
        # study_id_int inside one batch (collate guarantees).
        loss_nce = matched_nce(h_t, z_t, neg_mask, tau=tau_nce)
        nce_diag = {"study_nce_pool_size": float(B - 1)}
    else:
        loss_nce = torch.zeros((), device=h_t.device)

    if lambda_cov > 0.0 and B > 1:
        l_cov, l_var = covariance_penalty(h_t, var_floor=var_floor)
        loss_cov = l_cov + l_var
    else:
        loss_cov = torch.zeros((), device=h_t.device)

    loss = loss_reg + lambda_nce * loss_nce + lambda_cov * loss_cov

    with torch.no_grad():
        var_t = h_t.std(dim=0).mean().item() if B > 1 else 0.0
        if B > 1:
            cov_m = h_t - h_t.mean(0, keepdim=True)
            cov_m = cov_m.t() @ cov_m / max(B - 1, 1)
            cov_off = cov_m.fill_diagonal_(0.0).abs().mean().item()
        else:
            cov_off = 0.0

        study_context_delta = float("nan")
        metadata_only_study_gap = float("nan")
        study_matched_rank_top1 = float("nan")
        study_matched_rank_top5 = float("nan")

        cadence_ok = diag_extra_every_n_steps > 0 and (global_step % diag_extra_every_n_steps == 0)
        if cadence_ok and B > 1:
            # (A) study_context_delta: shuffle corrupted study across batch.
            perm = torch.randperm(B, device=full_el.device)
            if int(perm[0].item()) == 0 and B > 1 and int(perm[1].item()) == 1:
                perm = torch.roll(torch.arange(B, device=perm.device), shifts=1)
            _, h_study_shuf = st.forward_with_study_token(
                ctx_el_corrupt[perm],
                ctx_meta_add[perm],
                ctx_pad_corrupt[perm],
            )
            if isinstance(proj, ModalityProjectorPair):
                h_shuf = proj.pairs[0].student_forward(h_study_shuf)
            else:
                h_shuf = proj.student_forward(h_study_shuf)
            study_context_delta = 1.0 - layernorm_cosine(h_t, h_shuf).mean().item()

            # (B) metadata_only_study_gap: student sees *only* [STUDY] + all
            # elements padded → forward returns a pure-meta h_study.
            all_padded = torch.ones_like(ctx_pad_corrupt)
            _, h_study_meta = st.forward_with_study_token(
                ctx_el_corrupt,
                ctx_meta_add,
                all_padded,
            )
            if isinstance(proj, ModalityProjectorPair):
                h_meta = proj.pairs[0].student_forward(h_study_meta)
            else:
                h_meta = proj.student_forward(h_study_meta)
            cos_actual = layernorm_cosine(h_t, z_t).mean().item()
            cos_meta = layernorm_cosine(h_meta, z_t).mean().item()
            metadata_only_study_gap = cos_actual - cos_meta

            # (C) study_matched_rank
            rank_stats = matched_rank_metrics(h_t, z_t, torch.ones(B, B, dtype=torch.bool, device=h_t.device))
            study_matched_rank_top1 = rank_stats["matched_rank_top1"]
            study_matched_rank_top5 = rank_stats["matched_rank_top5"]

    diagnostics: Dict[str, float] = {
        "loss_regress": loss_reg.item(),
        "loss_nce": loss_nce.item(),
        "loss_cov": loss_cov.item(),
        "var_t": var_t,
        "cov_off": cov_off,
        "study_context_delta": study_context_delta,
        "metadata_only_study_gap": metadata_only_study_gap,
        "study_matched_rank_top1": study_matched_rank_top1,
        "study_matched_rank_top5": study_matched_rank_top5,
        **nce_diag,
    }
    return StepOutput(loss=loss, loss_regress=loss_reg, loss_nce=loss_nce, diagnostics=diagnostics)


# ---------------------------------------------------------------------------
# Entry point (scaffold)
# ---------------------------------------------------------------------------


def main(args=None, resume_preempt: bool = False) -> None:
    """EchoMV-JEPA Stage-1 training entry point.

    Mirrors the structure of ``app.echoset_jepa.train.main``. The only
    structural differences: construct ``StudyTransformerEMA`` alongside the
    student ``StudyTransformer``, optionally use ``ModalityProjectorPair``
    (Stage-1m), update the teacher each step, and use the echomv collate.
    """
    import os

    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]
    except KeyError:
        pass

    import numpy as np
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp

    try:
        mp.set_sharing_strategy("file_system")
    except Exception:
        pass
    from torch.nn.parallel import DistributedDataParallel
    from torch.utils.data import DataLoader

    from src.datasets.echomv_jepa_dataset import EchoMVJEPADataset, echomv_collate
    from src.datasets.echomv_jepa_pixel_dataset import (
        EchoMVJEPAPixelDataset,
        echomv_pixel_collate,
    )
    from src.models.study_projectors import EMAProjectorPair
    from src.utils.distributed import init_distributed
    from src.utils.logging import CSVLogger, get_logger

    log = get_logger(__name__, force=True)

    cfg = args if isinstance(args, dict) else {}
    exp = cfg.get("experiment", {})

    # Dispatch to the full-joint trainer when requested. Full joint trains
    # the V-JEPA clip encoder online alongside a study transformer and uses
    # a completely separate training loop; we avoid shoe-horning it into
    # the cached-pooled main() below.
    trainer_name = str(cfg.get("trainer", exp.get("trainer", "stage1")))
    if trainer_name == "full_joint":
        from app.echomv_jepa.train_full_joint import main as full_joint_main

        return full_joint_main(args=args, resume_preempt=resume_preempt)

    st_cfg_dict = exp.get("study_transformer", {})
    masking_cfg = exp.get("masking", {})
    sampler_cfg = exp.get("sampler", {})
    loss_cfg = exp.get("loss", {})
    ema_cfg = exp.get("ema", {})
    optim_cfg = exp.get("optim", {})
    clip_enc_cfg = exp.get("clip_encoder", {})
    target_meta_cfg = exp.get("target_meta", {})
    elements_cfg = exp.get("elements", {})
    logging_cfg = exp.get("logging", {})
    collapse_cfg = exp.get("collapse_monitor", {})
    projector_cfg = exp.get("projector", {})
    diag_cfg = exp.get("diagnostics", {})
    folder = cfg.get("folder", "./echomv_jepa_run")
    os.makedirs(folder, exist_ok=True)

    seed = int(cfg.get("seed", 0))
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True

    world_size, rank = init_distributed()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    log.info("[rank %d/%d] init on %s", rank, world_size, device)

    # --- models ----------------------------------------------------------
    st_cfg = StudyTransformerConfig(
        d_clip=int(st_cfg_dict.get("d_clip", 1024)),
        d_model=int(st_cfg_dict.get("d_model", 512)),
        n_layers=int(st_cfg_dict.get("n_layers", 4)),
        n_heads=int(st_cfg_dict.get("n_heads", 8)),
        ffn_mult=int(st_cfg_dict.get("ffn_mult", 4)),
        dropout_ffn=float(st_cfg_dict.get("dropout_ffn", 0.1)),
        dropout_attn=float(st_cfg_dict.get("dropout_attn", 0.0)),
        max_M=int(st_cfg_dict.get("max_M", 64)),
    )
    meta_dropout = MetaDropout(
        view=float(exp.get("meta_dropout", {}).get("view", 0.15)),
        modality=float(exp.get("meta_dropout", {}).get("modality", 0.10)),
        phase=float(exp.get("meta_dropout", {}).get("phase", 0.30)),
        quality=float(exp.get("meta_dropout", {}).get("quality", 0.30)),
    )
    st = StudyTransformer(st_cfg).to(device)
    meta = MetaEmbeddings(d_model=st_cfg.d_model, dropout=meta_dropout).to(device)

    # Projector: single shared (Stage-1 / 1b) or per-modality (Stage-1m)
    num_modalities = int(projector_cfg.get("num_modalities", 1))
    d_hidden = int(projector_cfg.get("d_hidden", 1024))
    d_proj = int(projector_cfg.get("d_proj", 256))
    if num_modalities <= 1:
        proj = EMAProjectorPair(d_model=st_cfg.d_model, d_hidden=d_hidden, d_proj=d_proj).to(device)
    else:
        if num_modalities != len(MODALITY_VOCAB):
            log.warning(
                "projector.num_modalities=%d but MODALITY_VOCAB has %d entries; proceeding.",
                num_modalities,
                len(MODALITY_VOCAB),
            )
        proj = ModalityProjectorPair(
            num_modalities=num_modalities,
            d_model=st_cfg.d_model,
            d_hidden=d_hidden,
            d_proj=d_proj,
        ).to(device)

    # Full-study EMA target encoder — built AFTER the student so the initial
    # copy matches student weights one-to-one.
    teacher_st = StudyTransformerEMA(st).to(device)

    if world_size > 1:
        st = DistributedDataParallel(st, device_ids=[device.index or 0])
        meta = DistributedDataParallel(meta, device_ids=[device.index or 0])

    # --- data ------------------------------------------------------------
    k_sample = sampler_cfg.get("sample_manifest")
    if not k_sample:
        raise ValueError("config must set sampler.sample_manifest")

    clip_source = clip_enc_cfg.get("source", "cached")
    use_token_mode = clip_source == "online"
    online_encoder = None
    if use_token_mode:
        # Option A: online token-level encoding (no c_clip cache).
        from src.models.echomv_jepa.online_encoder import OnlineVJepaEncoder

        enc_config_path = clip_enc_cfg.get("config_path")
        if not enc_config_path:
            raise ValueError("clip_encoder.source=online requires clip_encoder.config_path")
        online_encoder = OnlineVJepaEncoder(
            config_path=str(enc_config_path),
            device=device,
            token_spatial_pool=int(clip_enc_cfg.get("token_spatial_pool", 2)),
            spatial_hw=int(clip_enc_cfg.get("spatial_hw", 14)),
            temporal_tubelets=int(clip_enc_cfg.get("temporal_tubelets", 8)),
        )
        log.info(
            "[rank %d] online encoder: tokens_per_clip=%d d_clip=(set on first forward)",
            rank,
            online_encoder.tokens_per_clip,
        )
        # Build transform used for the raw clip pipeline. We use the same
        # default transform cache_cclip.py uses (ImageNet-normalized 224x224).
        from evals.video_classification_frozen.utils import make_transforms

        enc_data_cfg = online_encoder._data_cfg  # noqa: SLF001
        DEFAULT_NORM = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        transform = make_transforms(
            training=False,
            num_views_per_clip=1,
            random_horizontal_flip=False,
            random_resize_aspect_ratio=(0.75, 4 / 3),
            random_resize_scale=(0.08, 1.0),
            reprob=0.0,
            auto_augment=False,
            motion_shift=False,
            crop_size=enc_data_cfg.get("resolution", 224),
            normalize=enc_data_cfg.get("normalization") or DEFAULT_NORM,
        )
        dataset = EchoMVJEPAPixelDataset(
            k_sample_manifest_path=k_sample,
            meta=meta.module if isinstance(meta, DistributedDataParallel) else meta,
            frames_per_clip=int(enc_data_cfg.get("frames_per_clip", 16)),
            frame_step=int(enc_data_cfg.get("frame_step", 2)),
            resolution=int(enc_data_cfg.get("resolution", 224)),
            transform=transform,
            strategy_weights=masking_cfg.get("strategy_weights"),
            seed=seed + rank,
        )
    else:
        cache_prefix = clip_enc_cfg.get("cache_local_prefix") or clip_enc_cfg.get("cache_s3_prefix", "")
        if not cache_prefix:
            raise ValueError("clip_encoder.source=cached requires cache_local_prefix")
        dataset = EchoMVJEPADataset(
            k_sample_manifest_path=k_sample,
            cache_prefix=cache_prefix,
            meta=meta.module if isinstance(meta, DistributedDataParallel) else meta,
            element_agg=elements_cfg.get("element_agg", "mean"),
            strategy_weights=masking_cfg.get("strategy_weights"),
            seed=seed + rank,
        )

    ddp_sampler = None
    if world_size > 1:
        from torch.utils.data.distributed import DistributedSampler

        ddp_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=seed)

    batch_size = int(optim_cfg.get("batch_studies_per_gpu", 32))
    loader_collate = echomv_pixel_collate if use_token_mode else echomv_collate
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=ddp_sampler,
        shuffle=(ddp_sampler is None),
        collate_fn=loader_collate,
        num_workers=int(cfg.get("num_workers", 4)),
        drop_last=True,
        persistent_workers=False,
        pin_memory=True,
    )

    # --- optimizer -------------------------------------------------------
    params = list(st.parameters()) + list(meta.parameters())
    if isinstance(proj, ModalityProjectorPair):
        for pair in proj.pairs:
            params += list(pair.student.parameters())
    else:
        params += list(proj.student.parameters())
    optimizer = torch.optim.AdamW(
        params,
        lr=float(optim_cfg.get("lr", 5e-4)),
        betas=tuple(optim_cfg.get("betas", (0.9, 0.95))),
        weight_decay=float(optim_cfg.get("weight_decay", 0.05)),
    )
    warmup_steps = int(optim_cfg.get("warmup_steps", 2000))
    main_steps = int(optim_cfg.get("main_steps", 50000))
    cooldown_steps = int(optim_cfg.get("cooldown_steps", 5000))
    total_steps = warmup_steps + main_steps + cooldown_steps
    peak_lr = float(optim_cfg.get("lr", 5e-4))

    def _lr_at(step: int) -> float:
        if step < warmup_steps:
            return peak_lr * (step + 1) / max(warmup_steps, 1)
        if step < warmup_steps + main_steps:
            return peak_lr
        cool_i = step - warmup_steps - main_steps
        frac = cool_i / max(cooldown_steps, 1)
        return peak_lr * max(0.0, 1.0 - frac)

    tau_start = float(ema_cfg.get("tau_start", 0.996))
    tau_end = float(ema_cfg.get("tau_end", 0.9999))

    # --- csv logger ------------------------------------------------------
    schema = logging_cfg.get("csv_schema") or [
        "step",
        "loss",
        "loss_regress",
        "loss_nce",
        "var_t",
        "cov_off",
        "z_cosine_vs_v1",
        "z_cosine_vs_isolated",
        "z_cosine_vs_peer_drop",
        "valid_neg_count_same_view_mean",
        "valid_neg_count_same_view_min",
        "valid_neg_count_same_modality_mean",
        "fallback_fraction",
        "mask_strategy",
        "M_elements_mean",
    ]
    csv = None
    if rank == 0:
        csv_cols = [("%d", "step")]
        for k in schema:
            if k == "step":
                continue
            if k == "mask_strategy":
                csv_cols.append(("%s", k))
            else:
                csv_cols.append(("%.6f", k))
        csv = CSVLogger(os.path.join(folder, "train_log.csv"), *csv_cols)

    log_every = int(logging_cfg.get("log_every_steps", 50))
    ckpt_every = int(cfg.get("checkpoint_every_steps", 2500))
    diag_peer_drop_every = int(diag_cfg.get("peer_drop_every_n_steps", 50))

    # --- falsification-probe halt bookkeeping (§15.1a) -------------------
    halt_cos_threshold = float(diag_cfg.get("halt_z_cosine_vs_v1_threshold", 0.98))
    halt_cos_consec = int(diag_cfg.get("halt_z_cosine_vs_v1_consec_steps", 5000))
    halt_cos_count = 0

    # --- resume ----------------------------------------------------------
    ckpt_dir = os.path.join(folder, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    latest_path = os.path.join(ckpt_dir, "latest.pt")
    global_step = 0
    start_epoch = 0
    if resume_preempt and os.path.exists(latest_path):
        log.info("[rank %d] resuming from %s", rank, latest_path)
        sd = torch.load(latest_path, map_location="cpu")
        _load_state(st, sd.get("study_transformer"))
        _load_state(meta, sd.get("meta_embeddings"))
        _load_state_proj(proj, sd.get("projector"))
        _load_state(teacher_st.teacher, sd.get("teacher_study_transformer"))
        optimizer.load_state_dict(sd["optimizer"])
        global_step = int(sd.get("global_step", 0))
        start_epoch = int(sd.get("epoch", 0))

    var_t_below_floor_count = 0
    var_t_floor = float(collapse_cfg.get("var_t_floor", 0.3))
    halt_below_for = int(collapse_cfg.get("halt_if_below_for_steps", 500))

    st.train()
    meta.train()
    if isinstance(proj, ModalityProjectorPair):
        for pair in proj.pairs:
            pair.student.train()
    else:
        proj.student.train()

    num_epochs = int(optim_cfg.get("num_epochs", 10_000_000))
    for epoch in range(start_epoch, num_epochs):
        if ddp_sampler is not None:
            ddp_sampler.set_epoch(epoch)
        for batch in loader:
            if global_step >= total_steps:
                break

            batch = {
                k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()
            }

            for pg in optimizer.param_groups:
                pg["lr"] = _lr_at(global_step)

            common_step_kwargs = dict(
                lambda_nce=float(loss_cfg.get("lambda_nce", 0.0)),
                tau_nce=float(loss_cfg.get("tau_nce", 0.1)),
                lambda_cov=float(loss_cfg.get("lambda_cov", 0.0)),
                var_floor=float(loss_cfg.get("var_floor", 0.0)),
                diag_extra_every_n_steps=int(diag_cfg.get("extra_every_n_steps", 25)),
                global_step=global_step,
            )
            objective = exp.get("objective", "element")
            if objective == "study":
                # Arm C — global study-token (pooled path only in this first
                # implementation; token mode is a follow-up).
                out = training_step_echomv_global(
                    batch,
                    st.module if isinstance(st, DistributedDataParallel) else st,
                    meta.module if isinstance(meta, DistributedDataParallel) else meta,
                    proj,
                    teacher_st,
                    corruption_mix=exp.get("study_corruption", None),
                    **common_step_kwargs,
                )
            elif use_token_mode:
                out = training_step_echomv_tokens(
                    batch,
                    st.module if isinstance(st, DistributedDataParallel) else st,
                    meta.module if isinstance(meta, DistributedDataParallel) else meta,
                    proj,
                    teacher_st,
                    online_encoder,
                    p_target_token_mask=float(exp.get("teacher_selfmask", {}).get("p_target_token_mask", 0.0)),
                    include_target_phase=bool(target_meta_cfg.get("include_target_phase", True)),
                    include_target_quality=bool(target_meta_cfg.get("target_quality_token_ablation", False)),
                    diag_peer_drop_every_n_steps=diag_peer_drop_every,
                    **common_step_kwargs,
                )
            else:
                out = training_step_echomv(
                    batch,
                    st.module if isinstance(st, DistributedDataParallel) else st,
                    meta.module if isinstance(meta, DistributedDataParallel) else meta,
                    proj,
                    teacher_st,
                    p_target_self_mask=float(exp.get("teacher_selfmask", {}).get("p_target_self_mask", 0.0)),
                    include_target_phase=bool(target_meta_cfg.get("include_target_phase", True)),
                    include_target_quality=bool(target_meta_cfg.get("target_quality_token_ablation", False)),
                    diag_peer_drop_every_n_steps=diag_peer_drop_every,
                    **common_step_kwargs,
                )
            optimizer.zero_grad(set_to_none=True)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            # EMA update of both the projector and the full-study teacher
            tau = cosine_schedule(global_step, total_steps, tau_start, tau_end)
            if isinstance(proj, ModalityProjectorPair):
                proj.update_teacher(tau)
            else:
                proj.update_teacher(tau)
            teacher_st.update_teacher(
                st.module if isinstance(st, DistributedDataParallel) else st,
                tau,
            )

            # Collapse monitor (var_t)
            if out.diagnostics["var_t"] < var_t_floor:
                var_t_below_floor_count += 1
            else:
                var_t_below_floor_count = 0
            if var_t_below_floor_count >= halt_below_for:
                log.error(
                    "var_t < %.2f for %d steps — halting per collapse monitor",
                    var_t_floor,
                    var_t_below_floor_count,
                )
                _save(
                    ckpt_dir, "halt_collapse.pt", st, meta, proj, teacher_st, optimizer, epoch, global_step, rank=rank
                )
                return

            # Falsification probe halt (§15.1a). Arm C (study-level) does not
            # emit z_cosine_vs_v1; treat absence as NaN which never triggers
            # the halt and resets the consecutive-step counter.
            zc = out.diagnostics.get("z_cosine_vs_v1", float("nan"))
            if zc > halt_cos_threshold:
                halt_cos_count += 1
            else:
                halt_cos_count = 0
            if halt_cos_count >= halt_cos_consec:
                log.error(
                    "z_cosine_vs_v1 > %.3f for %d steps — halting per falsification probe",
                    halt_cos_threshold,
                    halt_cos_count,
                )
                _save(
                    ckpt_dir,
                    "halt_falsification.pt",
                    st,
                    meta,
                    proj,
                    teacher_st,
                    optimizer,
                    epoch,
                    global_step,
                    rank=rank,
                )
                return

            if rank == 0 and (global_step % log_every == 0):
                m_mean = float(batch["n_elements"].float().mean().item())
                strat = batch["mask_strategies"][0] if batch.get("mask_strategies") else "-"
                row = [global_step]
                for k in schema:
                    if k == "step":
                        continue
                    if k == "mask_strategy":
                        row.append(strat)
                    elif k == "loss":
                        row.append(float(out.loss.item()))
                    elif k == "M_elements_mean":
                        row.append(m_mean)
                    else:
                        row.append(float(out.diagnostics.get(k, float("nan"))))
                csv.log(*row)
                log.info(
                    "step=%d loss=%.4f reg=%.4f nce=%.4f var=%.3f cov=%.4f " "z_v1=%.3f z_iso=%.3f M=%.1f strat=%s",
                    global_step,
                    out.loss.item(),
                    out.diagnostics.get("loss_regress", float("nan")),
                    out.diagnostics.get("loss_nce", float("nan")),
                    out.diagnostics.get("var_t", float("nan")),
                    out.diagnostics.get("cov_off", float("nan")),
                    out.diagnostics.get("z_cosine_vs_v1", float("nan")),
                    out.diagnostics.get("z_cosine_vs_isolated", float("nan")),
                    m_mean,
                    strat,
                )

            if rank == 0 and ckpt_every > 0 and global_step > 0 and global_step % ckpt_every == 0:
                _save(
                    ckpt_dir,
                    f"step{global_step}.pt",
                    st,
                    meta,
                    proj,
                    teacher_st,
                    optimizer,
                    epoch,
                    global_step,
                    rank=rank,
                )
                _save(ckpt_dir, "latest.pt", st, meta, proj, teacher_st, optimizer, epoch, global_step, rank=rank)

            global_step += 1
        if global_step >= total_steps:
            break

    if rank == 0:
        _save(ckpt_dir, "final.pt", st, meta, proj, teacher_st, optimizer, epoch, global_step, rank=rank)
    if world_size > 1:
        dist.barrier()
    log.info("[rank %d] training complete at step %d", rank, global_step)


def _unwrap(m):
    from torch.nn.parallel import DistributedDataParallel

    return m.module if isinstance(m, DistributedDataParallel) else m


def _load_state(m, sd):
    if sd is None:
        return
    _unwrap(m).load_state_dict(sd, strict=False)


def _load_state_proj(proj, sd):
    if sd is None:
        return
    if isinstance(proj, ModalityProjectorPair):
        for i, pair in enumerate(proj.pairs):
            key = f"pair_{i}"
            if key in sd:
                pair.student.load_state_dict(sd[key]["student"], strict=False)
                pair.teacher.load_state_dict(sd[key]["teacher"], strict=False)
    else:
        proj.student.load_state_dict(sd.get("student", {}), strict=False)
        proj.teacher.load_state_dict(sd.get("teacher", {}), strict=False)


def _proj_state_dict(proj):
    if isinstance(proj, ModalityProjectorPair):
        return {
            f"pair_{i}": {"student": p.student.state_dict(), "teacher": p.teacher.state_dict()}
            for i, p in enumerate(proj.pairs)
        }
    return {"student": proj.student.state_dict(), "teacher": proj.teacher.state_dict()}


def _save(
    ckpt_dir: str, name: str, st, meta, proj, teacher_st, optimizer, epoch: int, global_step: int, rank: int
) -> None:
    if rank != 0:
        return
    import os

    path = os.path.join(ckpt_dir, name)
    torch.save(
        {
            "study_transformer": _unwrap(st).state_dict(),
            "meta_embeddings": _unwrap(meta).state_dict(),
            "projector": _proj_state_dict(proj),
            "teacher_study_transformer": teacher_st.teacher.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
        },
        path,
    )


__all__ = [
    "training_step_echomv",
    "StepOutput",
    "main",
]
