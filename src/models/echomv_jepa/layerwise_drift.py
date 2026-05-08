"""Richer drift diagnostics for the trainable clip encoder vs frozen e100.

v1 logs a single scalar (``anchor_cosine_to_e100``) over the final pooled
token vector. This module adds per-block cosine similarities + covariance
diagnostics so we can see *where* in the encoder the drift is happening.

The helper hooks a forward pass on both the online encoder and the frozen
anchor and captures block outputs by installing ``register_forward_hook``
on ``encoder.backbone.blocks[i]``. Hooks are installed + removed inside
the function; no persistent state.

All diagnostics are no-grad and mean-pool the token axis before comparing.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _install_block_hooks(module: nn.Module, indices: List[int]) -> Dict[int, List[torch.Tensor]]:
    """Install forward hooks on the listed blocks; return a dict that
    accumulates their outputs. The caller is expected to remove the
    hooks when done."""
    buffers: Dict[int, List[torch.Tensor]] = {i: [] for i in indices}
    handles = []
    backbone = module.backbone if hasattr(module, "backbone") else module
    blocks = getattr(backbone, "blocks", None)
    if blocks is None:
        return buffers  # no-op
    for i in indices:
        if i >= len(blocks):
            continue
        blk = blocks[i]

        def _hook(mod, inp, out, _i=i):
            t = out if isinstance(out, torch.Tensor) else out[0]
            buffers[_i].append(t.detach())

        handles.append(blk.register_forward_hook(_hook))
    return buffers, handles  # type: ignore[return-value]


def _remove_hooks(handles) -> None:
    for h in handles:
        h.remove()


def _cosine_pooled(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean-pool token axis, LN, cosine, reduce over batch to a scalar."""
    if a.numel() == 0 or b.numel() == 0:
        return float("nan")
    ap = a.mean(dim=1)  # (N, D)
    bp = b.mean(dim=1)
    ap = F.layer_norm(ap, ap.shape[-1:])
    bp = F.layer_norm(bp, bp.shape[-1:])
    return float(F.cosine_similarity(ap, bp, dim=-1).mean().item())


@torch.no_grad()
def compute_layerwise_cosine(
    online: nn.Module,  # f_theta (MultiSeqWrapper or similar)
    anchor: nn.Module,  # f_0 (same arch, different weights)
    clips: torch.Tensor,  # (N, 3, T, H, W)
    block_indices: Optional[List[int]] = None,
    *,
    view_mask: Optional[torch.Tensor] = None,  # (N,) bool — e.g., A4C-only subset
) -> Dict[str, float]:
    """Run both encoders on ``clips`` once, with forward hooks on the
    specified blocks, and return mean cosine similarity between online
    and anchor outputs at each block.

    Returns dict with keys ``block_{i}`` for each block index + ``top_block``
    for the final block, + ``pooled_final`` for the encoder output post-norm,
    + optional ``a4c_pooled`` when ``view_mask`` is provided.
    """
    if clips.numel() == 0:
        ks = [f"block_{i}" for i in (block_indices or [])] + ["top_block", "pooled_final"]
        if view_mask is not None:
            ks.append("a4c_pooled")
        return {k: float("nan") for k in ks}

    if block_indices is None:
        # Default: evenly spaced across 24 blocks for ViT-L
        backbone = online.backbone if hasattr(online, "backbone") else online
        n_blocks = len(getattr(backbone, "blocks", range(24)))
        block_indices = list(sorted({0, n_blocks // 4, n_blocks // 2, 3 * n_blocks // 4, n_blocks - 1}))

    on_buffers, on_handles = _install_block_hooks(online, block_indices)
    an_buffers, an_handles = _install_block_hooks(anchor, block_indices)
    try:
        on_final = online([clips])
        an_final = anchor([clips])
    finally:
        _remove_hooks(on_handles)
        _remove_hooks(an_handles)

    out: Dict[str, float] = {}
    for i in block_indices:
        if not on_buffers[i] or not an_buffers[i]:
            out[f"block_{i}"] = float("nan")
            continue
        out[f"block_{i}"] = _cosine_pooled(on_buffers[i][-1], an_buffers[i][-1])

    # Top block alias = final block index.
    out["top_block"] = out.get(f"block_{block_indices[-1]}", float("nan"))

    on_tok = on_final[0] if isinstance(on_final, list) else on_final
    an_tok = an_final[0] if isinstance(an_final, list) else an_final
    out["pooled_final"] = _cosine_pooled(on_tok, an_tok)

    if view_mask is not None and view_mask.any():
        m = view_mask.to(on_tok.device)
        out["a4c_pooled"] = _cosine_pooled(on_tok[m], an_tok[m])

    return out


@torch.no_grad()
def compute_clip_cov_var(tokens: torch.Tensor) -> Dict[str, float]:
    """Simple covariance + variance diagnostic on pooled clip vectors.

    ``tokens``: (N, T_tok, D). Returns ``clip_var`` (mean std across dims)
    and ``clip_cov_off`` (mean absolute off-diagonal cov).
    """
    if tokens.numel() == 0 or tokens.size(0) < 2:
        return {"clip_var": 0.0, "clip_cov_off": 0.0}
    pooled = tokens.mean(dim=1)  # (N, D)
    var = pooled.std(dim=0).mean().item()
    x = pooled - pooled.mean(dim=0, keepdim=True)
    cov = x.t() @ x / max(pooled.size(0) - 1, 1)
    cov_off = cov.fill_diagonal_(0.0).abs().mean().item()
    return {"clip_var": float(var), "clip_cov_off": float(cov_off)}


__all__ = ["compute_layerwise_cosine", "compute_clip_cov_var"]
