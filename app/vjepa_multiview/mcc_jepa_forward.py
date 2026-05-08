"""Target-anchored (and pure) Masked Cross-Clip V-JEPA forward pass.

This module is deliberately small: it reuses the encoder, EMA target
encoder, predictor, and ``_jepa_loss_fn`` already defined in
``app/vjepa_multiview/train.py``. The only new component is a
``CrossClipAdapter`` (``src/models/mcc_jepa/``) that wraps the predictor's
output with a ``gamma``-gated residual cross-attention onto source-clip A.

Two modes:
  * ``pure``:            student context = clip_A only; target = masked B.
                         Diagnostic only; no L_vjepa_self.
  * ``target_anchored``: student context = visible B; adapter injects A.
                         Computes L_vjepa_self + lambda_mcc * L_mcc.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from app.vjepa_multiview.train import PairBatch, _jepa_loss_fn

logger = logging.getLogger(__name__)


def _flatten_z(z: list[list[torch.Tensor]]) -> list[torch.Tensor]:
    """Flatten the predictor's list-of-list output to a single list over
    (fpc_i, mask_i) pairs, preserving order."""
    out: list[torch.Tensor] = []
    for fi in z:
        out.extend(fi)
    return out


def _apply_adapter_to_predictor_out(
    z_pred: list[list[torch.Tensor]],
    a_source_tokens: list[torch.Tensor],
    adapter: nn.Module,
) -> list[list[torch.Tensor]]:
    """Apply ``adapter(pred_B_tokens, A_source_tokens)`` for each
    (fpc, mask-generator) entry in z_pred.

    ``a_source_tokens`` is list-over-fpc from ``encoder(clip_a)`` (no mask).
    The adapter is applied once per mask-generator index sharing the same
    source A tokens — source A is mask-independent.
    """
    out: list[list[torch.Tensor]] = []
    for fi, a_i in zip(z_pred, a_source_tokens):
        row: list[torch.Tensor] = []
        for z_ij in fi:
            b = z_ij.size(0)
            # z_ij has leading dim B * len(masks_x); a_i is [B, N_A, D].
            # Broadcast a_i across mask-generator repetitions.
            if a_i.size(0) != b:
                repeat = b // a_i.size(0)
                a_rep = a_i.repeat_interleave(repeat, dim=0)
            else:
                a_rep = a_i
            row.append(adapter(z_ij, a_rep))
        out.append(row)
    return out


def forward_mcc_jepa(
    pair: PairBatch,
    encoder: nn.Module,
    target_encoder: nn.Module,
    predictor: nn.Module,
    adapter: Optional[nn.Module],
    *,
    mode: str = "target_anchored",
    lambda_mcc: float = 0.2,
    lambda_vjepa: float = 1.0,
    loss_exp: float = 1.0,
) -> dict:
    """Compute MCC-JEPA loss on clip B, using clip A as cross-clip source.

    :param pair: PairBatch where ``clip_a`` is the source clip and
        ``clip_b`` is the target (masks are on clip_b).
    :param encoder: student online encoder.
    :param target_encoder: EMA teacher (no grad).
    :param predictor: V-JEPA predictor.
    :param adapter: CrossClipAdapter; required when ``mode=='target_anchored'``.
    :param mode: ``'pure'`` or ``'target_anchored'``.
    :param lambda_mcc: weight on the cross-clip MCC loss.
    :param lambda_vjepa: weight on the B-only V-JEPA self loss (target_anchored only).
    :param loss_exp: L_p exponent (V-JEPA default 1.0 = L1).
    :return: dict with ``total_loss``, ``loss_mcc``, ``loss_vjepa_self``,
        ``pred_delta_from_A`` diagnostic, and ``mode`` label.
    """
    if mode not in ("pure", "target_anchored"):
        raise ValueError(f"mode must be 'pure' or 'target_anchored'; got {mode!r}")

    # Teacher on full clip B (no masks; predictor indexes target slots).
    with torch.no_grad():
        h_b = target_encoder(pair.clip_b)
        h_b = [F.layer_norm(hi, (hi.size(-1),)) for hi in h_b]

    device = pair.clip_b[0].device

    # Source A tokens used by both modes (encoder, no mask). Student grads
    # flow through here so the encoder learns to emit useful source tokens.
    z_a_source = encoder(pair.clip_a)  # list-over-fpc of [B, N_A, D_enc]

    if mode == "pure":
        # Student context = full A; target mask indices live on B.
        # masks_x for the predictor must index positions inside A (we use
        # the full-token identity positions: 0..N_A-1). Build per-fpc
        # identity mask lists that match V-JEPA's collator shape.
        B, N_A = z_a_source[0].size(0), z_a_source[0].size(1)
        full_idx = [[torch.arange(N_A, device=device).unsqueeze(0).expand(B, -1).contiguous()] for _ in z_a_source]
        z = predictor([a for a in z_a_source], full_idx, pair.masks_pred, delta_phi=None)
        loss_mcc = _jepa_loss_fn(z, h_b, pair.masks_pred, loss_exp=loss_exp)
        loss_vjepa_self = torch.zeros((), device=device)
        pred_delta_from_A = torch.zeros((), device=device)
        total = lambda_mcc * loss_mcc
        return {
            "total_loss": total,
            "loss_mcc": loss_mcc,
            "loss_vjepa_self": loss_vjepa_self,
            "pred_delta_from_A": pred_delta_from_A,
            "mode": mode,
        }

    # target_anchored --------------------------------------------------------
    if adapter is None:
        raise ValueError("target_anchored mode requires an adapter instance")

    # Student on visible B (standard V-JEPA context).
    z_b_visible = encoder(pair.clip_b, pair.masks_enc)
    # Predictor on B-visible, returning predictions at masks_pred positions.
    z_pred_base = predictor(z_b_visible, pair.masks_enc, pair.masks_pred, delta_phi=None)

    # L_vjepa_self: plain B->B V-JEPA at lambda_vjepa.
    loss_vjepa_self = _jepa_loss_fn(z_pred_base, h_b, pair.masks_pred, loss_exp=loss_exp)

    # Apply cross-clip adapter.
    z_pred_anchored = _apply_adapter_to_predictor_out(z_pred_base, z_a_source, adapter)
    loss_mcc = _jepa_loss_fn(z_pred_anchored, h_b, pair.masks_pred, loss_exp=loss_exp)

    # Diagnostic: how much does the adapter move the prediction? (1 - cos).
    with torch.no_grad():
        z_base_flat = torch.cat([t.flatten() for t in _flatten_z(z_pred_base)]).detach()
        z_anch_flat = torch.cat([t.flatten() for t in _flatten_z(z_pred_anchored)]).detach()
        cos = torch.nn.functional.cosine_similarity(z_base_flat.unsqueeze(0), z_anch_flat.unsqueeze(0)).squeeze()
        pred_delta_from_A = (1.0 - cos).detach()

    total = lambda_vjepa * loss_vjepa_self + lambda_mcc * loss_mcc
    return {
        "total_loss": total,
        "loss_mcc": loss_mcc,
        "loss_vjepa_self": loss_vjepa_self,
        "pred_delta_from_A": pred_delta_from_A,
        "mode": mode,
    }
