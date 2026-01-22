"""
Custom model wrapper for PanEcho to work with V-JEPA 2 eval system.
"""
# --- SSL FIX START ---
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
# --- SSL FIX END ---

import logging
import torch
import torch.nn as nn
import sys
import os
import importlib.util

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class PanEchoWrapper(nn.Module):
    def __init__(self, panecho_model):
        super().__init__()
        self.panecho_model = panecho_model
        # PanEcho output dim
        self.embed_dim = 768

    def forward(self, x):
        # x shape: [B, C, T, H, W]
        # PanEcho expects standard video input
        # Returns [B, 768]
        embeddings = self.panecho_model(x)
        # Reshape for AttentiveClassifier: [B, 1, 768]
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
    # Assumes PanEcho is at: evals/video_classification_frozen/modelcustom/PanEcho
    current_dir = os.path.dirname(os.path.abspath(__file__))
    panecho_root = os.path.join(current_dir, "PanEcho")
    
    if not os.path.exists(panecho_root):
        raise FileNotFoundError(f"PanEcho source not found at {panecho_root}")

    # 2. Handle 'src' Namespace Collision
    # Both V-JEPA and PanEcho have a 'src' folder. We must force Python
    # to use PanEcho's 'src' temporarily.
    
    # Save V-JEPA's 'src' if it's already loaded
    vjepa_src = sys.modules.get('src')
    if 'src' in sys.modules:
        del sys.modules['src']
    
    # Prepend PanEcho to sys.path
    sys.path.insert(0, panecho_root)
    
    try:
        # 3. Manually load hubconf.py from PanEcho
        hubconf_path = os.path.join(panecho_root, "hubconf.py")
        spec = importlib.util.spec_from_file_location("panecho_hubconf", hubconf_path)
        hubconf = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(hubconf)
        
        # 4. Instantiate Model
        # This triggers 'from src.models import ...' inside PanEcho, 
        # which now resolves correctly to PanEcho/src
        panecho_model = hubconf.PanEcho(pretrained=True, clip_len=frames_per_clip)
        
    finally:
        # 5. Cleanup: Restore V-JEPA environment
        sys.path.pop(0) # Remove PanEcho from path
        
        # Remove PanEcho's 'src' from cache
        if 'src' in sys.modules:
            del sys.modules['src']
            
        # Restore V-JEPA's 'src'
        if vjepa_src:
            sys.modules['src'] = vjepa_src

    logger.info(f"PanEcho loaded successfully. Embed dim: 768")
    return PanEchoWrapper(panecho_model)