# evals/video_classification_frozen/modelcustom/panecho_encoder.py

"""
Custom model wrapper for PanEcho to work with V-JEPA 2 eval system.

Key fixes:
1. Applies ImageNet normalization internally (PanEcho requirement)
2. Upscales spatial dimensions to 224x224 if needed (PanEcho trained on 224x224)
"""

# --- SSL FIX (Required for weight downloads) ---
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
# -----------------------------------------------

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import importlib
import importlib.util

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class PanEchoWrapper(nn.Module):
    """
    PanEcho wrapper for V-JEPA2 eval system.

    Handles:
    - Spatial upscaling to 224x224 if needed (PanEcho trained resolution)
    - V-JEPA2 input/output format conversion

    ImageNet normalization is handled by the shared make_transforms pipeline.

    Output embedding: 768 dimensions (ConvNeXt-Tiny + Transformer)
    """

    def __init__(self, panecho_model, embed_dim=768):
        super().__init__()
        self.panecho_model = panecho_model
        self.embed_dim = embed_dim

        # NOTE: ImageNet normalization is already applied by the shared
        # make_transforms pipeline (eval_transform). Do NOT re-normalize here.

        # Freeze model
        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        Preprocess input tensor for PanEcho.

        The shared make_transforms pipeline already applies ImageNet
        normalization, so this method only handles spatial resizing.

        Args:
            x: [B, C, T, H, W] tensor, already ImageNet-normalized

        Returns:
            [B, C, T, 224, 224] tensor
        """
        B, C, T, H, W = x.shape

        # Upscale spatial dimensions to 224x224 if needed
        if H != 224 or W != 224:
            x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
            x = x.reshape(B, T, C, 224, 224).permute(0, 2, 1, 3, 4)

        return x.float()

    @torch.no_grad()
    def forward(self, x, clip_indices=None):
        """
        Process clips and return embeddings in V-JEPA2 format.
        
        Expected V-JEPA eval input:
            x: list[num_segments][num_views] of tensors shaped [B, C, T, H, W]
            
        Returns:
            list of length N (=num_segments*num_views), each tensor [B, 1, D]
        """

        # Case 1: nested list-of-lists (standard V-JEPA eval pathway)
        if isinstance(x, list):
            # Flatten tokens while preserving batch dimension in each tensor
            tokens = [dij for di in x for dij in di]  # length N, each [B,C,T,H,W]
            if len(tokens) == 0:
                raise ValueError("Empty clip list received.")

            # Stack along a new token dimension => [B, N, C, T, H, W]
            x_tok = torch.stack(tokens, dim=1)

        # Case 2: already a tensor
        elif torch.is_tensor(x):
            if x.ndim == 5:
                # [B,C,T,H,W] -> [B,1,C,T,H,W]
                x_tok = x.unsqueeze(1)
            elif x.ndim == 6:
                # assume [B,N,C,T,H,W]
                x_tok = x
            else:
                raise ValueError(f"Unsupported tensor input shape: {tuple(x.shape)}")
        else:
            raise ValueError(f"Unsupported input type: {type(x)}")

        B, N, C, T, H, W = x_tok.shape

        # Flatten batch and clips: [B*N, C, T, H, W]
        x_flat = x_tok.reshape(B * N, C, T, H, W)

        # Preprocess: upscale to 224x224 and apply ImageNet normalization
        x_flat = self._preprocess(x_flat)

        # Run PanEcho backbone
        out = self.panecho_model(x_flat)

        # Handle different output formats
        if isinstance(out, dict):
            if "embedding" in out:
                emb = out["embedding"]
            elif "last_hidden_state" in out:
                hs = out["last_hidden_state"]
                emb = hs[:, 0] if hs.ndim == 3 else hs
            else:
                # Take first value from dict
                emb = next(iter(out.values()))
        else:
            emb = out

        # Ensure [B*N, D] shape
        if emb.ndim == 3 and emb.shape[1] == 1:
            emb = emb[:, 0]  # [B*N, D]
        
        if emb.ndim == 1:
            emb = emb.unsqueeze(0)  # [1, D]

        if emb.ndim != 2 or emb.shape[1] != self.embed_dim:
            raise ValueError(
                f"Unexpected PanEcho embedding shape: {tuple(emb.shape)}, "
                f"expected [*, {self.embed_dim}]"
            )

        # Reshape to [B, N, D]
        emb = emb.float().reshape(B, N, self.embed_dim)

        # Return list-of-tokens, each [B, 1, D], as expected by V-JEPA2 eval
        return [emb[:, i:i+1, :] for i in range(N)]


def init_module(
    resolution: int,
    frames_per_clip: int,
    checkpoint: str,
    model_kwargs: dict,
    wrapper_kwargs: dict,
    device=None,
):
    """
    V-JEPA2 eval entrypoint for PanEcho.
    
    Args:
        resolution: Input resolution (will be upscaled to 224 internally if needed)
        frames_per_clip: Number of frames per clip
        checkpoint: Not used (PanEcho loads from hub)
        model_kwargs: Not used
        wrapper_kwargs: Additional wrapper parameters
        device: Target device
    """
    logger.info("=" * 60)
    logger.info("Initializing PanEcho encoder")
    logger.info(f"  Input resolution: {resolution} (will upscale to 224 if needed)")
    logger.info(f"  Frames per clip: {frames_per_clip}")
    logger.info("=" * 60)

    if resolution != 224:
        logger.warning(
            f"PanEcho was trained on 224x224; got resolution={resolution}. "
            f"Will upscale internally."
        )

    current_dir = os.path.dirname(os.path.abspath(__file__))
    panecho_root = os.path.join(current_dir, "PanEcho")
    if not os.path.exists(panecho_root):
        raise FileNotFoundError(f"PanEcho source not found at {panecho_root}")

    # Context swap to avoid src/ namespace collisions with V-JEPA2
    vjepa_modules = {
        k: v for k, v in sys.modules.items() 
        if k == "src" or k.startswith("src.")
    }
    original_path = list(sys.path)
    for k in list(vjepa_modules.keys()):
        sys.modules.pop(k, None)
    sys.path.insert(0, panecho_root)

    try:
        importlib.invalidate_caches()
        hubconf_path = os.path.join(panecho_root, "hubconf.py")
        spec = importlib.util.spec_from_file_location("hubconf", hubconf_path)
        hubconf = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(hubconf)

        panecho_model = hubconf.PanEcho(
            pretrained=True,
            clip_len=frames_per_clip,
            backbone_only=True,
        )

        panecho_model.eval()
        for p in panecho_model.parameters():
            p.requires_grad_(False)

        if device is not None:
            panecho_model.to(device)

    finally:
        # Restore V-JEPA2 src/ modules
        panecho_modules = {
            k: v for k, v in sys.modules.items() 
            if k == "src" or k.startswith("src.")
        }
        for k in list(panecho_modules.keys()):
            sys.modules.pop(k, None)
        sys.path = original_path
        sys.modules.update(vjepa_modules)

    num_params = sum(p.numel() for p in panecho_model.parameters()) / 1e6
    logger.info(f"PanEcho loaded successfully: {num_params:.1f}M parameters (frozen)")
    logger.info("=" * 60)

    return PanEchoWrapper(panecho_model, embed_dim=768)