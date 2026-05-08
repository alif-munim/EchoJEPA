"""Sample N_per_study clips per study for the true V-JEPA clip loss.

The study-level forward encodes all K clips; running the full V-JEPA
loss on all K is expensive (K ≤ 8 × predictor + teacher). We run it on
a small subset per study each step. Default: 1 clip per study, sampled
uniformly from valid (non-padded) clips.

Returns:
  * ``sel_clips``:   (B, 3, T, H, W) on the same device as ``full_clips``
  * ``sel_idx_bm``:  (B, 2) — (batch_idx, elem_idx) for each selection,
                     useful for pairing with study-level outputs
  * ``valid_mask``:  (B,) bool — False if the row had no valid clip
                     (should be rare; degenerate all-padded row)
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def sample_clip_subset(
    full_clips: torch.Tensor,  # (B, M_full, 3, T, H, W)
    pad_mask: torch.Tensor,  # (B, M_full) bool (True = pad)
    n_per_study: int = 1,
    policy: str = "random_valid",
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``n_per_study`` clips per study.

    Args:
      full_clips: pixel tensor (B, M, 3, T, H, W).
      pad_mask: per-element pad mask.
      n_per_study: number of clips to draw per study. 1 by default;
          raising above 1 multiplies the V-JEPA loss compute.
      policy: ``"random_valid"`` (default) samples uniformly from valid
          elements. ``"first_valid"`` takes the first valid element
          (deterministic; useful for smoke/tests).

    Returns a tuple ``(sel_clips, sel_idx_bm, valid_mask)``:
      * ``sel_clips``: (B * n_per_study, 3, T, H, W)
      * ``sel_idx_bm``: (B * n_per_study, 2) long, columns = [batch_row, elem_idx]
      * ``valid_mask``: (B * n_per_study,) bool — True if selection has a
        valid clip; False only for all-padded rows (sentinel clip is
        the first element regardless of pad; caller should skip these).
    """
    if n_per_study < 1:
        raise ValueError(f"n_per_study must be >= 1; got {n_per_study}")
    if policy not in ("random_valid", "first_valid"):
        raise ValueError(f"unknown policy: {policy!r}")

    B, M, _C, _T, _H, _W = full_clips.shape
    device = full_clips.device
    if generator is None:
        generator = torch.Generator(device="cpu")

    sel_batch: list[int] = []
    sel_elem: list[int] = []
    sel_valid: list[bool] = []

    for b in range(B):
        valid_elems = (~pad_mask[b]).nonzero(as_tuple=False).squeeze(-1)
        if valid_elems.numel() == 0:
            # All-padded row: pick element 0 as sentinel but flag invalid.
            for _ in range(n_per_study):
                sel_batch.append(b)
                sel_elem.append(0)
                sel_valid.append(False)
            continue
        if policy == "first_valid":
            for i in range(n_per_study):
                e = int(valid_elems[min(i, valid_elems.numel() - 1)].item())
                sel_batch.append(b)
                sel_elem.append(e)
                sel_valid.append(True)
        else:  # random_valid
            # Sample with replacement if n_per_study > valid count
            idx = torch.randint(0, valid_elems.numel(), (n_per_study,), generator=generator)
            for i in range(n_per_study):
                e = int(valid_elems[idx[i]].item())
                sel_batch.append(b)
                sel_elem.append(e)
                sel_valid.append(True)

    batch_idx = torch.tensor(sel_batch, dtype=torch.long, device=device)
    elem_idx = torch.tensor(sel_elem, dtype=torch.long, device=device)
    sel_idx_bm = torch.stack([batch_idx, elem_idx], dim=1)
    sel_clips = full_clips[batch_idx, elem_idx]
    valid_mask = torch.tensor(sel_valid, dtype=torch.bool, device=device)
    return sel_clips, sel_idx_bm, valid_mask


__all__ = ["sample_clip_subset"]
