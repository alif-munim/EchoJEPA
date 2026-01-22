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

        # 1. Handle List Input (Collated views / Nested lists)
        if isinstance(x, list):
            try:
                x = flatten_to_tensor(x)
            except Exception as e:
                raise ValueError(f"Received list input but failed to flatten: {e}")

        # 2. Universal Flattening for High-Dimension Inputs (e.g. 7D from multi-segment)
        # We assume the LAST 4 dimensions are always [C, T, H, W]
        # Everything before that is "Batch" or "View" structure.
        if x.ndim >= 5:
            shape = x.shape
            
            # Separate batch structure from image dimensions
            batch_dims = shape[:-4]   # e.g. [2, 1, 1]
            spatial_dims = shape[-4:] # [3, 16, 224, 224]
            
            # Flatten all batch dims into one: [Total_Batch, 3, 16, 224, 224]
            flat_batch_size = 1
            for d in batch_dims:
                flat_batch_size *= d
            
            x_flat = x.view(flat_batch_size, *spatial_dims)
            
            # Forward pass -> [Total_Batch, 768] (Tensor)
            # NOTE: Because we pass backbone_only=True below, this returns a Tensor, not a dict.
            emb_flat = self.panecho_model(x_flat)
            
            # Reshape back to [Batch, Num_Tokens, D] for AttentiveClassifier
            # We treat the first dimension as the true batch size, and the rest as tokens (clips)
            true_batch_size = batch_dims[0]
            num_tokens = flat_batch_size // true_batch_size
            
            return emb_flat.view(true_batch_size, num_tokens, self.embed_dim)

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
    
    for k in vjepa_modules:
        del sys.modules[k]
    
    sys.path.insert(0, panecho_root)
    
    try:
        importlib.invalidate_caches()
        import src 
        
        hubconf_path = os.path.join(panecho_root, "hubconf.py")
        spec = importlib.util.spec_from_file_location("hubconf", hubconf_path)
        hubconf = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(hubconf)
        
        # Instantiate Model with backbone_only=True
        # This ensures output is a Tensor, not a Dict
        panecho_model = hubconf.PanEcho(
            pretrained=True, 
            clip_len=frames_per_clip,
            backbone_only=True  # <--- CRITICAL FIX
        )
        
    except Exception as e:
        logger.error(f"Failed to load PanEcho: {e}")
        raise e
        
    finally:
        # Cleanup
        panecho_modules = {k: v for k, v in sys.modules.items() if k == 'src' or k.startswith('src.')}
        for k in panecho_modules:
            del sys.modules[k]
            
        sys.path = original_path
        sys.modules.update(vjepa_modules)

    logger.info(f"PanEcho loaded successfully. Embed dim: {768}")
    return PanEchoWrapper(panecho_model)