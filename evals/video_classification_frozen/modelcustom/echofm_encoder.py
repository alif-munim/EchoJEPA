# evals/video_classification_frozen/modelcustom/echofm_encoder.py

"""
Custom model wrapper for EchoFM to work with V-JEPA 2 eval system.

EchoFM: ViT-L MAE pretrained on 290K echo clips with triplet loss.
- Architecture: ViT-L/16, t_patch_size=4, 32 frames, sep_pos_embed, cls_embed
- embed_dim: 1024
- Checkpoint: pretrain format (no decoder keys in released weights)
- Normalization: trained on [0, 1] range (T.ToTensor only, no ImageNet norm)

Reference: https://github.com/SekeunKim/EchoFM
"""

import logging
import os
import sys
from typing import Any, List, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _collect_leaf_tensors(x: Any) -> List[torch.Tensor]:
    """Flatten nested list/tuple structures into a list of leaf tensors [B, C, T, H, W]."""
    leaves: List[torch.Tensor] = []

    def rec(z: Any):
        if torch.is_tensor(z):
            leaves.append(z)
        elif isinstance(z, (list, tuple)):
            for zz in z:
                rec(zz)
        else:
            raise TypeError(f"Unsupported clip container type: {type(z)}")

    rec(x)
    if len(leaves) == 0:
        raise ValueError("No tensors found in clip container.")
    return leaves


class EchoFMEncoder(nn.Module):
    """
    EchoFM encoder that runs the ViT blocks without masking.

    Built from EchoFM's MaskedAutoencoderViT but with a clean forward
    pass (no masking, no decoder). Returns spatial-temporal tokens.
    """

    def __init__(self, mae_model, embed_dim=1024):
        super().__init__()
        # Copy encoder components from the MAE model
        self.patch_embed = mae_model.patch_embed
        self.blocks = mae_model.blocks
        self.norm = mae_model.norm
        self.embed_dim = embed_dim

        # Position embeddings
        self.sep_pos_embed = mae_model.sep_pos_embed
        self.cls_embed = mae_model.cls_embed
        self.input_size = mae_model.input_size

        if self.sep_pos_embed:
            self.pos_embed_spatial = mae_model.pos_embed_spatial
            self.pos_embed_temporal = mae_model.pos_embed_temporal
            if self.cls_embed:
                self.pos_embed_class = mae_model.pos_embed_class

        if self.cls_embed:
            self.cls_token = mae_model.cls_token

        # Freeze
        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass without masking.

        x: [B, C, T, H, W]
        returns: [B, N_tokens, D] (spatial-temporal tokens, no CLS)
        """
        # Patch embed: [B, C, T, H, W] -> [B, T_patches, L_patches, D]
        x = self.patch_embed(x)
        N, T, L, C = x.shape
        x = x.reshape(N, T * L, C)

        # CLS token
        if self.cls_embed:
            cls_tokens = self.cls_token.expand(N, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        # Position embeddings
        if self.sep_pos_embed:
            pos_embed = self.pos_embed_spatial.repeat(
                1, self.input_size[0], 1
            ) + torch.repeat_interleave(
                self.pos_embed_temporal,
                self.input_size[1] * self.input_size[2],
                dim=1,
            )
            pos_embed = pos_embed.expand(N, -1, -1)
            if self.cls_embed:
                pos_embed = torch.cat(
                    [self.pos_embed_class.expand(N, -1, -1), pos_embed], dim=1
                )
        else:
            pos_embed = self.pos_embed[:, :, :].expand(N, -1, -1)

        x = x + pos_embed

        # Check if attention requires T-shaped input
        requires_t_shape = (
            len(self.blocks) > 0
            and hasattr(self.blocks[0].attn, "requires_t_shape")
            and self.blocks[0].attn.requires_t_shape
        )
        if requires_t_shape:
            x = x.view([N, T, L, C])

        # Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        if requires_t_shape:
            x = x.view([N, T * L + (1 if self.cls_embed else 0), C])

        x = self.norm(x)

        # Remove CLS token, return spatial-temporal tokens
        if self.cls_embed:
            x = x[:, 1:, :]

        return x  # [B, T*L, D]


class EchoFMWrapper(nn.Module):
    """
    Wraps EchoFMEncoder to satisfy V-JEPA2 eval API.

    Output: List of [B, N_tokens, D] tensors (one per clip).
    """

    def __init__(self, encoder: EchoFMEncoder, num_frames=32, t_patch_size=4):
        super().__init__()
        self.encoder = encoder
        self.embed_dim = encoder.embed_dim
        self.target_frames = num_frames
        self.t_patch_size = t_patch_size

        # ImageNet normalization constants used by the shared make_transforms
        # pipeline. EchoFM was trained on [0, 1] range data (T.Resize + T.ToTensor
        # only), so we must undo the ImageNet normalization before inference.
        inet_mean = torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1, 1)
        inet_std = torch.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1, 1)
        self.register_buffer("inet_mean", inet_mean, persistent=False)
        self.register_buffer("inet_std", inet_std, persistent=False)

        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def _adapt_temporal(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adapt input to EchoFM's expected 32 frames.
        x: [B, C, T, H, W]
        """
        T = x.shape[2]
        if T == self.target_frames:
            return x
        if T > self.target_frames:
            # Subsample
            idx = torch.linspace(0, T - 1, self.target_frames, device=x.device).round().long()
            return x.index_select(2, idx)
        # Pad by repeating last frame
        pad_size = self.target_frames - T
        pad = x[:, :, -1:, :, :].expand(-1, -1, pad_size, -1, -1)
        return torch.cat([x, pad], dim=2)

    @torch.no_grad()
    def _adapt_spatial(self, x: torch.Tensor) -> torch.Tensor:
        """Resize to 224x224 if needed."""
        H, W = x.shape[-2], x.shape[-1]
        if H == 224 and W == 224:
            return x
        B, C, T = x.shape[:3]
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        return x.reshape(B, T, C, 224, 224).permute(0, 2, 1, 3, 4)

    def forward(
        self,
        clips: Union[torch.Tensor, List[Any]],
        clip_indices=None,
    ) -> List[torch.Tensor]:
        if torch.is_tensor(clips):
            leaves = [clips]
        else:
            leaves = _collect_leaf_tensors(clips)

        B = int(leaves[0].shape[0])
        for t in leaves:
            if int(t.shape[0]) != B:
                raise ValueError("All clip leaves must share the same batch dimension.")

        # Flatten leaves: [L*B, C, T, H, W]
        x_flat = torch.cat(leaves, dim=0)

        # Undo ImageNet normalization → [0, 1] range expected by EchoFM
        x_flat = x_flat * self.inet_std + self.inet_mean

        # Adapt to EchoFM's expected input
        x_flat = self._adapt_temporal(x_flat)
        x_flat = self._adapt_spatial(x_flat)

        # Forward
        feats = self.encoder.forward_features(x_flat.float())  # [L*B, N_tokens, D]

        # Reshape back to per-leaf
        L = len(leaves)
        N_tokens = feats.shape[1]
        D = feats.shape[2]
        feats = feats.view(L, B, N_tokens, D).permute(1, 0, 2, 3).contiguous()
        return [feats[:, i, :, :] for i in range(L)]


def init_module(
    resolution: int,
    frames_per_clip: int,
    checkpoint: str,
    model_kwargs: dict,
    wrapper_kwargs: dict,
):
    """
    V-JEPA2 eval entrypoint for EchoFM.
    """
    logger.info("=" * 60)
    logger.info("Initializing EchoFM encoder")
    logger.info(f"  Resolution: {resolution}")
    logger.info(f"  Frames per clip: {frames_per_clip}")
    logger.info(f"  Checkpoint: {checkpoint}")
    logger.info("=" * 60)

    enc_cfg = model_kwargs.get("encoder", model_kwargs)
    num_frames = enc_cfg.get("num_frames", 32)
    t_patch_size = enc_cfg.get("t_patch_size", 4)
    embed_dim = enc_cfg.get("embed_dim", 1024)
    depth = enc_cfg.get("depth", 24)
    num_heads = enc_cfg.get("num_heads", 16)
    patch_size = enc_cfg.get("patch_size", 16)

    # Import EchoFM with namespace isolation (has its own util/ that conflicts with vjepa2)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    echofm_root = os.path.join(current_dir, "EchoFM")
    if not os.path.exists(echofm_root):
        raise FileNotFoundError(f"EchoFM source not found at {echofm_root}")

    # Save and swap vjepa2's modules to avoid collision
    vjepa_modules = {k: v for k, v in sys.modules.items() if k == "src" or k.startswith("src.")}
    original_path = list(sys.path)
    for k in list(vjepa_modules.keys()):
        sys.modules.pop(k, None)
    sys.path.insert(0, echofm_root)

    try:
        # Force reimport of EchoFM modules
        for mod_name in list(sys.modules.keys()):
            if mod_name == "EchoFM" or mod_name.startswith("EchoFM."):
                sys.modules.pop(mod_name, None)

        from EchoFM.models_mae import MaskedAutoencoderViT

        mae_model = MaskedAutoencoderViT(
            img_size=224,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            num_frames=num_frames,
            t_patch_size=t_patch_size,
            sep_pos_embed=True,
            cls_embed=True,
        )

        # Load checkpoint
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

        logger.info(f"Loading checkpoint: {checkpoint}")
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model", ckpt)

        # Filter to encoder-only keys (no decoder)
        encoder_keys = {
            k: v
            for k, v in state_dict.items()
            if not any(
                k.startswith(p)
                for p in ("decoder", "mask_token", "decoder_embed", "decoder_cls_token", "decoder_prj_cls_token")
            )
        }

        msg = mae_model.load_state_dict(encoder_keys, strict=False)
        missing_non_decoder = [k for k in msg.missing_keys if "decoder" not in k and "mask_token" not in k]
        logger.info(
            f"Loaded {len(encoder_keys)} keys. "
            f"Missing (non-decoder): {len(missing_non_decoder)}, "
            f"Unexpected: {len(msg.unexpected_keys)}"
        )
        if missing_non_decoder:
            logger.warning(f"Missing encoder keys: {missing_non_decoder[:5]}")

    finally:
        # Restore vjepa2 modules
        echofm_modules = {k: v for k, v in sys.modules.items() if k == "EchoFM" or k.startswith("EchoFM.")}
        for k in list(echofm_modules.keys()):
            sys.modules.pop(k, None)
        sys.path = original_path
        sys.modules.update(vjepa_modules)

    # Wrap encoder
    encoder = EchoFMEncoder(mae_model, embed_dim=embed_dim)
    wrapper = EchoFMWrapper(encoder, num_frames=num_frames, t_patch_size=t_patch_size)

    num_params = sum(p.numel() for p in wrapper.parameters()) / 1e6
    logger.info(f"EchoFM encoder ready: {num_params:.1f}M parameters (frozen)")
    logger.info(f"  embed_dim={embed_dim}, tokens={num_frames // t_patch_size * 196}")
    logger.info("=" * 60)

    return wrapper
