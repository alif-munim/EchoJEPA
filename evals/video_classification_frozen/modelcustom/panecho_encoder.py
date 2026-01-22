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

    def forward(self, x, clip_indices=None):
        # Helper: Recursively stack lists into a single Tensor
        def flatten_to_tensor(input_data):
            if isinstance(input_data, torch.Tensor):
                return input_data
            if isinstance(input_data, list):
                # Recursively process elements
                processed = [flatten_to_tensor(d) for d in input_data]
                # Stack: If [Tensor, Tensor] -> Stacked Tensor
                return torch.stack(processed)
            raise ValueError(f"Unsupported type: {type(input_data)}")

        # 1. Robustly Handle List Input (Collated views / Nested lists)
        if isinstance(x, list):
            try:
                x = flatten_to_tensor(x)
            except Exception as e:
                raise ValueError(f"Received list input but failed to flatten: {e}")

        # 2. Handle 6D Input [B, N, C, T, H, W] (Multi-clip/Multi-view)
        # PanEcho expects [Batch, C, T, H, W]. We treat (B*N) as the batch.
        if x.ndim == 6:
            B, N, C, T, H, W = x.shape
            x = x.reshape(B * N, C, T, H, W)
            
            # Forward pass -> [B*N, D]
            emb = self.panecho_model(x)
            
            # Reshape back to [B, N, D] so AttentiveClassifier sees N tokens
            return emb.view(B, N, self.embed_dim)

        # 3. Handle 5D Input [B, C, T, H, W] (Single clip)
        if x.ndim == 5:
            # Forward pass -> [B, D]
            emb = self.panecho_model(x)
            # Add token dimension -> [B, 1, D]
            return emb.unsqueeze(1)
            
        raise ValueError(f"Unexpected input shape: {x.shape}")

def init_module(
    resolution: int,
    frames_per_clip: int,
    checkpoint: str,
    model_kwargs: dict,
    wrapper_kwargs: dict,
):
    logger.info("Loading PanEcho model from local source...")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    panecho_root = os.path.join(current_dir, "PanEcho")
    
    if not os.path.exists(panecho_root):
        raise FileNotFoundError(f"PanEcho source not found at {panecho_root}")

    # Context Swap: V-JEPA src -> PanEcho src
    vjepa_modules = {k: v for k, v in sys.modules.items() if k == 'src' or k.startswith('src.')}
    original_path = list(sys.path)
    
    # Unload V-JEPA modules
    for k in vjepa_modules:
        del sys.modules[k]
    
    # Inject PanEcho path
    sys.path.insert(0, panecho_root)
    
    try:
        importlib.invalidate_caches()
        # Force load PanEcho's src package
        import src 
        
        hubconf_path = os.path.join(panecho_root, "hubconf.py")
        spec = importlib.util.spec_from_file_location("hubconf", hubconf_path)
        hubconf = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(hubconf)
        
        # Instantiate Model
        panecho_model = hubconf.PanEcho(pretrained=True, clip_len=frames_per_clip)
        
    except Exception as e:
        logger.error(f"Failed to load PanEcho: {e}")
        raise e
        
    finally:
        # Cleanup and Restore V-JEPA environment
        panecho_modules = {k: v for k, v in sys.modules.items() if k == 'src' or k.startswith('src.')}
        for k in panecho_modules:
            del sys.modules[k]
            
        sys.path = original_path
        sys.modules.update(vjepa_modules)

    logger.info(f"PanEcho loaded successfully. Embed dim: {768}")
    return PanEchoWrapper(panecho_model)