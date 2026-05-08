"""EchoMV-JEPA dataset + collate.

Extends EchoSet-JEPA v1's per-study dataset with the tensors needed by the
full-study EMA target encoder:

- ``full_elements``, ``full_meta_{view,modality,phase,quality}``, ``full_pad_mask``
  — the concatenation of context and target in a fixed order (ctx first, then
  target) after padding.
- ``target_idx_in_full``, ``context_idx_in_full`` — gather indices into the
  ``full_*`` tensors. Because we pad ctx and tgt independently and concatenate
  along the element axis, the target block always lives at offsets
  ``[max_ctx, max_ctx + max_tgt)``.

The per-study dataset itself is identical to
:class:`src.datasets.echoset_jepa_collate.EchoSetStudyDataset` — we re-export
it so configs can point at the same class and new code paths only diverge at
the collate boundary.
"""

from __future__ import annotations

from typing import Dict, List

import torch

from src.datasets.echoset_jepa_collate import (  # re-export
    EchoSetStudyDataset,
    echoset_collate,
)

EchoMVJEPADataset = EchoSetStudyDataset  # alias — same per-study contract


def echomv_collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate that produces the EchoSet v1 batch plus EchoMV full-study tensors.

    Output adds these keys on top of :func:`echoset_collate`:
      - ``full_elements``       : (B, M_ctx + M_tgt, d_clip)
      - ``full_meta_view``      : (B, M_ctx + M_tgt) long
      - ``full_meta_modality``  : (B, M_ctx + M_tgt) long
      - ``full_meta_phase``     : (B, M_ctx + M_tgt) long
      - ``full_meta_quality``   : (B, M_ctx + M_tgt) long
      - ``full_pad_mask``       : (B, M_ctx + M_tgt) bool (True = pad)
      - ``target_idx_in_full``  : (B, M_tgt) long
      - ``context_idx_in_full`` : (B, M_ctx) long
    """
    out = echoset_collate(batch)

    ctx = out["ctx_elements"]  # (B, max_ctx, d_clip)
    tgt = out["tgt_elements"]  # (B, max_tgt, d_clip)
    B, max_ctx, _ = ctx.shape
    max_tgt = tgt.shape[1]

    out["full_elements"] = torch.cat([ctx, tgt], dim=1)
    out["full_meta_view"] = torch.cat([out["ctx_meta_view"], out["tgt_meta_view"]], dim=1)
    out["full_meta_modality"] = torch.cat([out["ctx_meta_modality"], out["tgt_meta_modality"]], dim=1)
    out["full_meta_phase"] = torch.cat([out["ctx_meta_phase"], out["tgt_meta_phase"]], dim=1)
    out["full_meta_quality"] = torch.cat([out["ctx_meta_quality"], out["tgt_meta_quality"]], dim=1)
    out["full_pad_mask"] = torch.cat([out["ctx_pad_mask"], out["tgt_pad_mask"]], dim=1)

    ctx_idx = torch.arange(max_ctx, dtype=torch.long).unsqueeze(0).expand(B, -1)
    tgt_idx = torch.arange(max_tgt, dtype=torch.long).unsqueeze(0).expand(B, -1) + max_ctx
    out["context_idx_in_full"] = ctx_idx.contiguous()
    out["target_idx_in_full"] = tgt_idx.contiguous()
    return out


__all__ = ["EchoMVJEPADataset", "echomv_collate", "EchoSetStudyDataset"]
