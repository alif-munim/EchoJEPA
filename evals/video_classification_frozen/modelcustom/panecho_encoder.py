"""
Custom model wrapper for PanEcho to work with V-JEPA 2 eval system.
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
import sys, os, importlib, importlib.util

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class PanEchoWrapper(nn.Module):
    def __init__(self, panecho_model, embed_dim=768):
        super().__init__()
        self.panecho_model = panecho_model
        self.embed_dim = embed_dim

    @torch.no_grad()
    def forward(self, x, clip_indices=None):
        """
        Expected V-JEPA eval input:
          x: list[num_segments][num_views] of tensors shaped [B, C, T, H, W]
        Return:
          list length N (=num_segments*num_views) of tensors [B, 1, D]
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
        x_flat = x_tok.reshape(B * N, C, T, H, W)

        out = self.panecho_model(x_flat)

        # Normalize outputs to [B*N, D]
        if isinstance(out, dict):
            if "embedding" in out:
                emb = out["embedding"]
            elif "last_hidden_state" in out:
                hs = out["last_hidden_state"]
                emb = hs[:, 0] if hs.ndim == 3 else hs
            else:
                emb = next(iter(out.values()))
        else:
            emb = out

        if emb.ndim == 3 and emb.shape[1] == 1:
            emb = emb[:, 0]  # [B*N, D]

        if emb.ndim != 2 or emb.shape[1] != self.embed_dim:
            raise ValueError(f"Unexpected PanEcho embedding shape: {tuple(emb.shape)}")

        emb = emb.reshape(B, N, self.embed_dim)  # [B, N, D]

        # Return list-of-tokens, each [B,1,D], as expected by your eval loop
        return [emb[:, i:i+1, :] for i in range(N)]


def init_module(
    resolution: int,
    frames_per_clip: int,
    checkpoint: str,
    model_kwargs: dict,
    wrapper_kwargs: dict,
    device=None,          # IMPORTANT: accept device for compatibility
):
    logger.info("Loading PanEcho model from local source...")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    panecho_root = os.path.join(current_dir, "PanEcho")
    if not os.path.exists(panecho_root):
        raise FileNotFoundError(f"PanEcho source not found at {panecho_root}")

    # Context swap to avoid src/ namespace collisions (kept from your version)
    vjepa_modules = {k: v for k, v in sys.modules.items() if k == "src" or k.startswith("src.")}
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
        # restore vjepa src/ modules
        panecho_modules = {k: v for k, v in sys.modules.items() if k == "src" or k.startswith("src.")}
        for k in list(panecho_modules.keys()):
            sys.modules.pop(k, None)
        sys.path = original_path
        sys.modules.update(vjepa_modules)

    logger.info("PanEcho loaded successfully.")
    return PanEchoWrapper(panecho_model, embed_dim=768)
