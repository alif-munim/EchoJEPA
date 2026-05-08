"""Loss primitives specific to the full-joint Global Study-Token run.

Reuses the existing ``losses.py`` (cosine_regress, matched_nce,
covariance_penalty, matched_rank_metrics) and adds:
  * ``clip_vjepa_lp_loss`` — V-JEPA L_p between predicted masked tokens
    and teacher targets (wraps the ``apply_masks`` + Lp pattern from
    ``app/vjepa/train.py:771–779``).
  * ``anchor_loss``        — LN-cosine distance between a pooled ``f_theta``
    token vector and the same pool from the frozen ``f_0`` anchor. Keeps
    the trainable clip encoder from drifting too far from e100.
  * ``global_study_loss``  — LN-cosine distance between the student's
    projected ``[STUDY]`` and the teacher's projected ``[STUDY]``, with
    stop-grad on the teacher side.
  * ``single_view_to_study_loss`` — same contract as ``global_study_loss``
    but the student input is a single-view subset of the study's clips.
  * ``assemble_total_loss`` — weighted sum with λ-warmup support.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.masks.utils import apply_masks

# ---------------------------------------------------------------------- #
# Clip-level V-JEPA loss
# ---------------------------------------------------------------------- #


def clip_vjepa_lp_loss(
    z: List[List[torch.Tensor]],
    h: List[torch.Tensor],
    masks_pred: List[List[torch.Tensor]],
    loss_exp: float = 1.0,
) -> torch.Tensor:
    """V-JEPA L_p loss over predictor outputs vs teacher targets.

    Mirrors the nested ``loss_fn`` in ``app/vjepa/train.py:771–779`` exactly.

    Args:
      z:           list (over fpc) of list (over mask-generator) of
                   ``(B, N_tgt, D)`` predicted target tokens.
      h:           list (over fpc) of teacher tokens ``(B, N_full, D)``.
      masks_pred:  list (over fpc) of list (over mask-gen) of
                   ``(B, N_tgt)`` target token indices.
      loss_exp:    L_p exponent. 1.0 = L1 (V-JEPA default).
    """
    h_gathered = [apply_masks(hi, mi, concat=False) for hi, mi in zip(h, masks_pred)]
    total, n = z[0][0].new_zeros(()), 0
    for zi, hi in zip(z, h_gathered):
        for zij, hij in zip(zi, hi):
            total = total + torch.mean(torch.abs(zij - hij) ** loss_exp) / loss_exp
            n += 1
    return total / max(n, 1)


def clip_vjepa_true_loss(
    clips: torch.Tensor,  # (N, 3, T, H, W) — selected clips (one per study)
    encoder: torch.nn.Module,  # f_theta (MultiSeqWrapper) — online, trainable
    target_encoder: torch.nn.Module,  # fbar_theta — EMA, no grad
    predictor: torch.nn.Module,  # PredictorMultiSeqWrapper — trainable
    masks_enc: torch.Tensor,  # (N, N_ctx) long
    masks_pred: torch.Tensor,  # (N, N_tgt) long
    *,
    loss_exp: float = 1.0,
) -> torch.Tensor:
    """Standard V-JEPA latent prediction loss on a single-clip-per-study batch.

    Replicates the vanilla V-JEPA pipeline exactly:
      1. Teacher (EMA) encodes full clips, no grad, normalized.
      2. Student encodes only the ``masks_enc`` visible tokens.
      3. Predictor takes student tokens + mask query positions and predicts
         latents at ``masks_pred`` positions.
      4. Loss is L_p between predictor output and LN-normalized teacher tokens
         at the same mask positions.

    All inputs are assumed on-device. The wrapper shapes (MultiSeqWrapper +
    PredictorMultiSeqWrapper) match ``init_video_model`` output.
    """
    # Wrap into list-over-fpc = [tensor] / list-of-list for the wrappers.
    clips_l = [clips]
    masks_enc_l = [[masks_enc]]
    masks_pred_l = [[masks_pred]]

    # --- teacher ---
    with torch.no_grad():
        h = target_encoder(clips_l)  # list-over-fpc of (N, N_full, D)
        h = [torch.nn.functional.layer_norm(hi, (hi.size(-1),)) for hi in h]

    # --- student ---
    z_visible = encoder(clips_l, masks_enc_l)  # list-of-list
    z_pred = predictor(z_visible, masks_enc_l, masks_pred_l, delta_phi=None)

    return clip_vjepa_lp_loss(z_pred, h, masks_pred_l, loss_exp=loss_exp)


# ---------------------------------------------------------------------- #
# Anchor loss
# ---------------------------------------------------------------------- #


def pool_tokens_mean(tokens: torch.Tensor) -> torch.Tensor:
    """Mean-pool over the token axis. ``tokens`` is ``(N, T_tok, D)``;
    returns ``(N, D)``."""
    return tokens.mean(dim=1)


def anchor_loss(
    online_tokens: torch.Tensor,  # (N, T, D) — f_theta output (grad)
    anchor_tokens: torch.Tensor,  # (N, T, D) — f_0 output (no grad)
) -> torch.Tensor:
    """``1 − mean cos(LN(pool(f_theta)), stopgrad(LN(pool(f_0))))``.

    Computed on a subsample of clips (caller slices). Keeps the trainable
    encoder from drifting away from e100's feature geometry.
    """
    h = pool_tokens_mean(online_tokens)
    z = pool_tokens_mean(anchor_tokens).detach()
    h = F.layer_norm(h, h.shape[-1:])
    z = F.layer_norm(z, z.shape[-1:])
    return (1.0 - F.cosine_similarity(h, z, dim=-1)).mean()


@torch.no_grad()
def anchor_cosine_to_e100(
    online_tokens: torch.Tensor,
    anchor_tokens: torch.Tensor,
) -> float:
    """Diagnostic value of the anchor cosine (no-grad)."""
    h = pool_tokens_mean(online_tokens)
    z = pool_tokens_mean(anchor_tokens)
    h = F.layer_norm(h, h.shape[-1:])
    z = F.layer_norm(z, z.shape[-1:])
    return float(F.cosine_similarity(h, z, dim=-1).mean().item())


# ---------------------------------------------------------------------- #
# Study-level losses (global and single-view)
# ---------------------------------------------------------------------- #


def global_study_loss(
    h_study: torch.Tensor,  # (B, d_model) — student [STUDY] output (grad)
    z_study: torch.Tensor,  # (B, d_model) — teacher [STUDY] output (no grad)
    p_student_proj: nn.Module,
    p_teacher_proj: nn.Module,
) -> torch.Tensor:
    """L_global_study_jepa.

    Predicts the teacher's projected [STUDY] from the student's projected
    [STUDY] using cosine regress on layer-normalized vectors. Teacher's
    output is detached; gradient flows only through the student and its
    projector.
    """
    h = p_student_proj(h_study)
    with torch.no_grad():
        z = p_teacher_proj(z_study).detach()
    h = F.layer_norm(h, h.shape[-1:])
    z = F.layer_norm(z, z.shape[-1:])
    return (1.0 - F.cosine_similarity(h, z, dim=-1)).mean()


def single_view_to_study_loss(
    h_study_sv: torch.Tensor,  # (B, d_model) — student [STUDY] from single-view input
    z_study_full: torch.Tensor,  # (B, d_model) — teacher [STUDY] from full study
    p_student_proj: nn.Module,
    p_teacher_proj: nn.Module,
) -> torch.Tensor:
    """Same contract as ``global_study_loss`` but the student saw only
    one view family. Teacher always saw the full K-clip study.
    """
    return global_study_loss(h_study_sv, z_study_full, p_student_proj, p_teacher_proj)


# ---------------------------------------------------------------------- #
# Loss assembler
# ---------------------------------------------------------------------- #


@dataclass
class LossWeights:
    lambda_clip: float = 1.0  # legacy lightweight clip consistency (back-compat)
    lambda_clip_vjepa_true: float = 1.0
    lambda_clip_consistency: float = 0.1
    lambda_study: float = 0.1
    lambda_nce: float = 0.005
    lambda_cov: float = 0.001
    lambda_anchor: float = 0.05
    lambda_sv: float = 0.02


@dataclass
class LossRamp:
    """Optional linear warmup for one loss weight.

    ``warmup_steps=0`` disables ramping (weight stays at its full value).
    Otherwise the weight grows linearly from 0 at step 0 to
    ``target_weight`` at step ``warmup_steps``. Resume-aware: pass the
    global step.
    """

    target_weight: float = 0.0
    warmup_steps: int = 0

    def value_at(self, step: int) -> float:
        if self.warmup_steps <= 0:
            return self.target_weight
        if step >= self.warmup_steps:
            return self.target_weight
        return self.target_weight * float(step) / float(self.warmup_steps)


@dataclass
class LossDecay:
    """Cosine (or linear) decay from ``start_weight`` to ``final_weight``.

    Used for the anchor loss: we want strong retention at the beginning
    (to keep the trainable clip encoder close to e100) and weak retention
    at the end (so the study objective can actually shape the encoder).
    Resume-aware via ``global_step``.
    """

    start_weight: float = 0.05
    final_weight: float = 0.005
    decay_steps: int = 10000
    schedule: str = "cosine"  # "cosine" | "linear" | "constant_start"

    def value_at(self, step: int) -> float:
        if self.decay_steps <= 0 or self.schedule == "constant_start":
            return self.start_weight
        if step >= self.decay_steps:
            return self.final_weight
        t = float(step) / float(self.decay_steps)
        if self.schedule == "cosine":
            # cosine from 1 (start) → 0 (final), value = final + 0.5*(start-final)*(1+cos(πt))
            import math

            return self.final_weight + 0.5 * (self.start_weight - self.final_weight) * (1.0 + math.cos(math.pi * t))
        if self.schedule == "linear":
            return self.start_weight + (self.final_weight - self.start_weight) * t
        raise ValueError(f"unknown schedule: {self.schedule!r}")


def assemble_total_loss(
    losses: Dict[str, torch.Tensor],
    weights: LossWeights,
    *,
    lambda_study_ramp: Optional[LossRamp] = None,
    lambda_sv_ramp: Optional[LossRamp] = None,
    lambda_anchor_decay: Optional[LossDecay] = None,
    global_step: int = 0,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """Weighted sum of per-component losses.

    ``losses`` must carry the keys that exist for this step. Missing keys
    are skipped (e.g. ``L_single_view_to_study`` only present on SV steps).

    Returns ``(total_loss, applied_weights_dict)`` so the caller can log
    the *effective* per-step λ (after warmup / decay).
    """
    l_clip = losses.get("clip")
    l_study = losses.get("study")
    l_nce = losses.get("nce")
    l_cov = losses.get("cov")
    l_anchor = losses.get("anchor")
    l_sv = losses.get("sv")
    l_clip_vjepa_true = losses.get("clip_vjepa_true")
    l_clip_consistency = losses.get("clip_consistency")

    device = next(v.device for v in losses.values() if isinstance(v, torch.Tensor))
    total = torch.zeros((), device=device)
    applied: Dict[str, float] = {}

    # Clip losses (true V-JEPA + optional consistency)
    if l_clip_vjepa_true is not None:
        total = total + weights.lambda_clip_vjepa_true * l_clip_vjepa_true
        applied["lambda_clip_vjepa_true_t"] = weights.lambda_clip_vjepa_true
    if l_clip_consistency is not None:
        total = total + weights.lambda_clip_consistency * l_clip_consistency
        applied["lambda_clip_consistency_t"] = weights.lambda_clip_consistency
    if l_clip is not None:
        # Back-compat: only used if neither of the new clip losses is present.
        if l_clip_vjepa_true is None and l_clip_consistency is None:
            total = total + weights.lambda_clip * l_clip
            applied["lambda_clip_t"] = weights.lambda_clip

    if l_study is not None:
        lam = weights.lambda_study
        if lambda_study_ramp is not None:
            lam = lambda_study_ramp.value_at(global_step)
        total = total + lam * l_study
        applied["lambda_study_t"] = lam
    if l_nce is not None:
        total = total + weights.lambda_nce * l_nce
        applied["lambda_nce_t"] = weights.lambda_nce
    if l_cov is not None:
        total = total + weights.lambda_cov * l_cov
        applied["lambda_cov_t"] = weights.lambda_cov
    if l_anchor is not None:
        if lambda_anchor_decay is not None:
            lam = lambda_anchor_decay.value_at(global_step)
        else:
            lam = weights.lambda_anchor
        total = total + lam * l_anchor
        applied["lambda_anchor_t"] = lam
    if l_sv is not None:
        lam = weights.lambda_sv
        if lambda_sv_ramp is not None:
            lam = lambda_sv_ramp.value_at(global_step)
        total = total + lam * l_sv
        applied["lambda_sv_t"] = lam
    return total, applied


__all__ = [
    "clip_vjepa_lp_loss",
    "pool_tokens_mean",
    "anchor_loss",
    "anchor_cosine_to_e100",
    "global_study_loss",
    "single_view_to_study_loss",
    "LossWeights",
    "LossRamp",
    "LossDecay",
    "assemble_total_loss",
]
