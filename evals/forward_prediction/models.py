"""Load frozen encoder + predictor + target_encoder from JEPA checkpoint.

Provides the three networks needed for prediction-error–based evaluation:
  - encoder: processes visible (context) tokens
  - predictor: predicts target representations from encoder output + mask positions
  - target_encoder: EMA copy providing ground-truth targets (layer-normed)
"""

import logging
from functools import partial

import torch
import torch.nn as nn

import src.models.vision_transformer as vit
from src.models.predictor import vit_predictor

logger = logging.getLogger(__name__)


def _strip_prefix(state_dict):
    """Strip 'module.' and 'backbone.' prefixes from checkpoint keys."""
    out = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "").replace("backbone.", "")
        out[k] = v
    return out


def _load_with_logging(model, state_dict, name):
    """Load state dict with shape-mismatch handling and logging."""
    cleaned = _strip_prefix(state_dict)
    model_sd = model.state_dict()
    for k, v in model_sd.items():
        if k not in cleaned:
            logger.warning(f"[{name}] key '{k}' not in checkpoint")
        elif cleaned[k].shape != v.shape:
            logger.warning(f"[{name}] shape mismatch for '{k}': ckpt {cleaned[k].shape} vs model {v.shape}")
            cleaned[k] = v
    msg = model.load_state_dict(cleaned, strict=False)
    logger.info(f"[{name}] loaded with msg: {msg}")
    return model


def load_jepa_models(
    checkpoint_path,
    img_size=224,
    num_frames=16,
    patch_size=16,
    tubelet_size=2,
    model_name="vit_giant_xformers",
    predictor_depth=12,
    predictor_embed_dim=384,
    use_rope=True,
    uniform_power=True,
    device="cuda",
):
    """Load encoder, predictor, and target_encoder from a JEPA checkpoint.

    Args:
        checkpoint_path: path to .pt checkpoint
        img_size: input resolution
        num_frames: frames per clip
        patch_size: spatial patch size
        tubelet_size: temporal patch size
        model_name: encoder architecture name
        predictor_depth: predictor transformer depth
        predictor_embed_dim: predictor hidden dimension
        use_rope: whether encoder/predictor use RoPE
        uniform_power: sincos pos embed power distribution
        device: target device

    Returns:
        (encoder, predictor, target_encoder) — all frozen, in eval mode
    """
    logger.info(f"Loading JEPA checkpoint from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    # --- Encoder ---
    encoder = vit.__dict__[model_name](
        img_size=img_size,
        num_frames=num_frames,
        patch_size=patch_size,
        tubelet_size=tubelet_size,
        uniform_power=uniform_power,
        use_rope=use_rope,
    )
    _load_with_logging(encoder, ckpt["encoder"], "encoder")

    # --- Target encoder (same architecture, EMA weights) ---
    target_encoder = vit.__dict__[model_name](
        img_size=img_size,
        num_frames=num_frames,
        patch_size=patch_size,
        tubelet_size=tubelet_size,
        uniform_power=uniform_power,
        use_rope=use_rope,
    )
    _load_with_logging(target_encoder, ckpt["target_encoder"], "target_encoder")

    # --- Predictor ---
    embed_dim = encoder.embed_dim
    predictor = vit_predictor(
        img_size=img_size,
        num_frames=num_frames,
        patch_size=patch_size,
        tubelet_size=tubelet_size,
        embed_dim=embed_dim,
        predictor_embed_dim=predictor_embed_dim,
        depth=predictor_depth,
        num_heads=predictor_embed_dim // 64,  # 384/64=6 heads
        use_mask_tokens=True,
        num_mask_tokens=10,
        uniform_power=uniform_power,
        use_rope=use_rope,
    )
    _load_with_logging(predictor, ckpt["predictor"], "predictor")

    # Freeze all and move to device
    for model in (encoder, predictor, target_encoder):
        model.eval()
        model.requires_grad_(False)
        model.to(device)

    logger.info(
        f"Loaded JEPA models: encoder={sum(p.numel() for p in encoder.parameters()) / 1e6:.0f}M, "
        f"predictor={sum(p.numel() for p in predictor.parameters()) / 1e6:.0f}M, "
        f"target_encoder={sum(p.numel() for p in target_encoder.parameters()) / 1e6:.0f}M"
    )

    del ckpt
    return encoder, predictor, target_encoder
