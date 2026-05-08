"""Study-level element corruption for EchoMV-JEPA.

Applies one of four corruption strategies to a study's element sequence,
sampled per row from a weight mix:

  * ``random_element_dropout``     drop each unpadded element with p=0.25
  * ``whole_view_family_dropout``  drop all elements sharing one view_family
  * ``whole_modality_dropout``     drop all elements sharing one modality
  * ``no_dropout``                 pass through

Extracted from ``app/echomv_jepa/train.py`` verbatim (Arm C smoke used the
same behavior; the full-joint variant reuses it unchanged). The original
train.py re-imports this function so Arm A/B/C remain byte-identical.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch


def apply_study_corruption(
    ctx_elements: torch.Tensor,  # (B, M_full, d_clip)
    ctx_meta_add: torch.Tensor,  # (B, M_full, d_model)  # accepted for API parity
    ctx_pad_mask: torch.Tensor,  # (B, M_full) bool
    meta_view: torch.Tensor,  # (B, M_full) long
    meta_modality: torch.Tensor,  # (B, M_full) long
    mix: Dict[str, float],
    rng: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample a corruption strategy per row and apply it.

    Returns ``(ctx_elements_corrupted, ctx_pad_mask_corrupted)``. Meta is
    left unchanged; padded positions get zero content at the transformer's
    input. Always keeps at least one unpadded element per row (guardrail).

    ``rng`` must be a CPU ``torch.Generator`` — ``torch.multinomial`` does
    not accept a CUDA generator against CPU weights. Small random draws
    here are cheap on CPU.
    """
    del ctx_meta_add  # unused; retained for caller API parity
    B, M_full, _ = ctx_elements.shape
    device = ctx_elements.device
    out_pad = ctx_pad_mask.clone()
    strategies = list(mix.keys())
    weights = torch.tensor([mix[s] for s in strategies], dtype=torch.float32)  # CPU
    for b in range(B):
        strat_idx = int(torch.multinomial(weights, 1, generator=rng).item())
        strat = strategies[strat_idx]
        if strat == "no_dropout":
            continue
        unpadded = (~ctx_pad_mask[b]).nonzero(as_tuple=False).squeeze(-1)
        if unpadded.numel() == 0:
            continue
        if strat == "random_element_dropout":
            r = torch.rand(unpadded.shape[0], generator=rng)  # CPU
            drop_idx = unpadded[(r < 0.25).to(device)]
        elif strat == "whole_view_family_dropout":
            views = meta_view[b, unpadded].unique()
            if views.numel() == 0:
                continue
            pick_cpu = torch.randint(0, views.numel(), (1,), generator=rng)
            pick = views[pick_cpu.to(device)]
            drop_idx = unpadded[meta_view[b, unpadded] == pick]
        elif strat == "whole_modality_dropout":
            mods = meta_modality[b, unpadded].unique()
            if mods.numel() == 0:
                continue
            pick_cpu = torch.randint(0, mods.numel(), (1,), generator=rng)
            pick = mods[pick_cpu.to(device)]
            drop_idx = unpadded[meta_modality[b, unpadded] == pick]
        else:
            continue
        if drop_idx.numel() == unpadded.numel():
            drop_idx = drop_idx[:-1]
        out_pad[b, drop_idx] = True
    elem_out = ctx_elements.clone()
    newly_padded = out_pad & (~ctx_pad_mask)
    if newly_padded.any():
        elem_out[newly_padded] = 0.0
    return elem_out, out_pad
