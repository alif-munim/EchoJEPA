"""V-JEPA clip encoder + predictor loaded from the e100 checkpoint.

Builds a trainable online encoder (``f_theta``) with weights initialized
from ``jepa_in21k_vitl_e100.pt``. The frozen anchor ``f_0`` is a deepcopy
of the freshly-loaded encoder (pre-training), also loaded from e100.

Also provides a layer-wise LR decay param-groups helper. The inner
encoder is ``MultiSeqWrapper(VisionTransformer)``; we index
``encoder.backbone.blocks`` for depth.
"""

from __future__ import annotations

import copy
import gc
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from app.vjepa.utils import init_video_model
from src.utils.checkpoint_loader import robust_checkpoint_loader

logger = logging.getLogger(__name__)


@dataclass
class ClipBackbonePack:
    """Return bundle from ``build_clip_encoder_from_e100``.

    ``encoder``, ``target_encoder``, ``anchor`` all share the e100 init
    but are independent parameter sets (separate deepcopy'd state).
    ``predictor`` is loaded from e100's predictor state dict and is
    trainable.
    """

    encoder: nn.Module
    target_encoder: nn.Module
    anchor: nn.Module
    predictor: nn.Module
    embed_dim: int


def build_clip_encoder_from_e100(
    *,
    ckpt_path: str,
    device: torch.device,
    model_name: str = "vit_large",
    crop_size: int = 224,
    max_num_frames: int = 16,
    tubelet_size: int = 2,
    patch_size: int = 16,
    pred_depth: int = 12,
    pred_embed_dim: int = 384,
    pred_num_heads: int = 12,
    use_mask_tokens: bool = True,
    num_mask_tokens: int = 10,
    use_rope: bool = True,
    use_sdpa: bool = True,
    use_activation_checkpointing: bool = True,
    uniform_power: bool = True,
    zero_init_mask_tokens: bool = True,
) -> ClipBackbonePack:
    """Init a V-JEPA (encoder, predictor) pair and load e100 weights into:
    * ``encoder``        trainable (``f_theta``)
    * ``target_encoder`` EMA teacher (no grad) — deepcopy of encoder
    * ``anchor``         frozen e100 copy (no grad) — deepcopy of encoder
    """
    encoder, predictor = init_video_model(
        device=device,
        patch_size=patch_size,
        max_num_frames=max_num_frames,
        tubelet_size=tubelet_size,
        model_name=model_name,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_embed_dim=pred_embed_dim,
        pred_num_heads=pred_num_heads,
        uniform_power=uniform_power,
        use_mask_tokens=use_mask_tokens,
        num_mask_tokens=num_mask_tokens,
        zero_init_mask_tokens=zero_init_mask_tokens,
        use_sdpa=use_sdpa,
        use_rope=use_rope,
        use_activation_checkpointing=use_activation_checkpointing,
    )
    logger.info(f"FULL-JOINT: FORCE-LOADING encoder + predictor from {ckpt_path}")
    ckpt = robust_checkpoint_loader(ckpt_path, map_location=torch.device("cpu"))
    enc_sd: Dict[str, Any] = ckpt.get("encoder", ckpt)
    enc_sd = {k.replace("module.", ""): v for k, v in enc_sd.items()}
    pe_key = "backbone.patch_embed.proj.weight"
    if pe_key in enc_sd and enc_sd[pe_key].ndim == 4:
        pe = enc_sd[pe_key]
        enc_sd[pe_key] = pe.unsqueeze(2).repeat(1, 1, tubelet_size, 1, 1) / float(tubelet_size)
    msg = encoder.load_state_dict(enc_sd, strict=False)
    logger.info(f"  encoder load: {msg}")
    if "predictor" in ckpt:
        pred_sd = {k.replace("module.", ""): v for k, v in ckpt["predictor"].items()}
        msg = predictor.load_state_dict(pred_sd, strict=False)
        logger.info(f"  predictor load: {msg}")
    del ckpt
    gc.collect()

    target_encoder = copy.deepcopy(encoder).to(device)
    anchor = copy.deepcopy(encoder).to(device)
    for p in target_encoder.parameters():
        p.requires_grad_(False)
    for p in anchor.parameters():
        p.requires_grad_(False)
    target_encoder.eval()
    anchor.eval()

    embed_dim = int(encoder.backbone.embed_dim)
    return ClipBackbonePack(
        encoder=encoder,
        target_encoder=target_encoder,
        anchor=anchor,
        predictor=predictor,
        embed_dim=embed_dim,
    )


def layerwise_param_groups(
    encoder: nn.Module,
    predictor: nn.Module,
    *,
    base_lr: float,
    weight_decay: float,
    n_blocks: Optional[int] = None,
    min_scale: float = 0.1,
    mid_scale: float = 0.3,
    top_scale: float = 1.0,
) -> List[Dict[str, Any]]:
    """Build AdamW param groups with per-depth LR decay on the encoder.

    Scheme (for n=24 layers):
      - blocks 0 .. n/4 - 1          → ``min_scale``  (tight, slow)
      - blocks n/4 .. 3n/4 - 1       → ``mid_scale``
      - blocks 3n/4 .. n-1 + norm    → ``top_scale``  (fast adapt)
      - patch_embed + pos_embed      → ``min_scale``
      - predictor                    → ``top_scale``

    Bias / 1-D params are pulled out into no-weight-decay groups.
    """
    backbone = encoder.backbone if hasattr(encoder, "backbone") else encoder
    blocks: nn.Module = getattr(backbone, "blocks")
    depth = len(blocks) if n_blocks is None else n_blocks

    def _scale_for_block(i: int) -> float:
        if i < depth // 4:
            return min_scale
        if i < 3 * depth // 4:
            return mid_scale
        return top_scale

    groups: List[Dict[str, Any]] = []

    def _add_group(params, lr_scale: float, wd_exclude: bool):
        params = [p for p in params if p.requires_grad]
        if not params:
            return
        groups.append(
            {
                "params": params,
                "lr": base_lr * lr_scale,
                "weight_decay": 0.0 if wd_exclude else weight_decay,
                "WD_exclude": wd_exclude,
                "lr_scale": lr_scale,
            }
        )

    # patch_embed + pos_embed → min_scale, wd ON for patch_embed, OFF for pos_embed
    pe = getattr(backbone, "patch_embed", None)
    if pe is not None:
        wd_params = [p for n, p in pe.named_parameters() if p.requires_grad and p.ndim > 1 and "bias" not in n]
        nwd_params = [p for n, p in pe.named_parameters() if p.requires_grad and (p.ndim <= 1 or "bias" in n)]
        _add_group(wd_params, min_scale, wd_exclude=False)
        _add_group(nwd_params, min_scale, wd_exclude=True)
    for extra_name in ("pos_embed", "cls_token"):
        extra = getattr(backbone, extra_name, None)
        if extra is not None and hasattr(extra, "requires_grad"):
            _add_group([extra], min_scale, wd_exclude=True)

    # Blocks with depth-dependent scale
    for i, blk in enumerate(blocks):
        scale = _scale_for_block(i)
        wd_params = [p for n, p in blk.named_parameters() if p.requires_grad and p.ndim > 1 and "bias" not in n]
        nwd_params = [p for n, p in blk.named_parameters() if p.requires_grad and (p.ndim <= 1 or "bias" in n)]
        _add_group(wd_params, scale, wd_exclude=False)
        _add_group(nwd_params, scale, wd_exclude=True)

    # Final norm on backbone → top_scale
    norm = getattr(backbone, "norm", None)
    if norm is not None:
        for p in norm.parameters():
            if p.requires_grad:
                _add_group([p], top_scale, wd_exclude=True)

    # Predictor → top_scale
    wd_params = [p for n, p in predictor.named_parameters() if p.requires_grad and p.ndim > 1 and "bias" not in n]
    nwd_params = [p for n, p in predictor.named_parameters() if p.requires_grad and (p.ndim <= 1 or "bias" in n)]
    _add_group(wd_params, top_scale, wd_exclude=False)
    _add_group(nwd_params, top_scale, wd_exclude=True)

    return groups
