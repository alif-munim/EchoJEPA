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
import sys
import os
import importlib
import importlib.util

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class PanEchoWrapper(nn.Module):
    def __init__(self, panecho_model):
        super().__init__()
        self.panecho_model = panecho_model
        self.embed_dim = 768

    def forward(self, x):
        # x shape: [B, C, T, H, W] -> [B, 768]
        embeddings = self.panecho_model(x)
        return embeddings.unsqueeze(1)

def init_module(
    resolution: int,
    frames_per_clip: int,
    checkpoint: str,
    model_kwargs: dict,
    wrapper_kwargs: dict,
):
    logger.info("Loading PanEcho model from local source...")
    
    # 1. Locate local PanEcho directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    panecho_root = os.path.join(current_dir, "PanEcho")
    
    if not os.path.exists(panecho_root):
        raise FileNotFoundError(f"PanEcho source not found at {panecho_root}")

    # 2. Context Swap: V-JEPA src -> PanEcho src
    # We must remove ALL 'src.*' modules to prevent collision
    vjepa_modules = {k: v for k, v in sys.modules.items() if k == 'src' or k.startswith('src.')}
    original_path = list(sys.path)
    
    # Unload V-JEPA
    for k in vjepa_modules:
        del sys.modules[k]
        
    # Prepend PanEcho path
    sys.path.insert(0, panecho_root)
    
    try:
        # 3. Explicitly load PanEcho's 'src' package
        # This ensures sys.modules['src'] is correctly populated with PanEcho's version
        importlib.invalidate_caches()
        import src 
        
        # 4. Load hubconf and instantiate model
        hubconf_path = os.path.join(panecho_root, "hubconf.py")
        spec = importlib.util.spec_from_file_location("hubconf", hubconf_path)
        hubconf = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(hubconf)
        
        # 'pretrained=True' triggers weight download (handled by SSL fix)
        panecho_model = hubconf.PanEcho(pretrained=True, clip_len=frames_per_clip)
        
    except Exception as e:
        logger.error(f"Failed to load PanEcho: {e}")
        # Print path info for debugging
        logger.error(f"Current sys.path: {sys.path}")
        raise e
        
    finally:
        # 5. Restore V-JEPA Environment
        
        # Remove PanEcho modules from cache
        panecho_modules = {k: v for k, v in sys.modules.items() if k == 'src' or k.startswith('src.')}
        for k in panecho_modules:
            del sys.modules[k]
            
        # Restore Path
        sys.path = original_path
        
        # Restore V-JEPA modules
        sys.modules.update(vjepa_modules)

    logger.info(f"PanEcho loaded successfully. Embed dim: {768}")
    return PanEchoWrapper(panecho_model)