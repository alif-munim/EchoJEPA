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
                processed = [flatten_to_tensor(d) for d in input_data]
                return torch.stack(processed)
            raise ValueError(f"Unsupported type: {type(input_data)}")

        # 1. Robustly Handle List Input
        if isinstance(x, list):
            try:
                x = flatten_to_tensor(x)
            except Exception as e:
                raise ValueError(f"Received list input but failed to flatten: {e}")

        # 2. Flatten for PanEcho [Batch*Views, C, T, H, W]
        # We assume the LAST 4 dimensions are always [C, T, H, W]
        # Everything before that is Batch/Views
        if x.ndim >= 5:
            shape = x.shape
            batch_dims = shape[:-4]   # e.g., (B, N) or (B, N, M)
            spatial_dims = shape[-4:] # (C, T, H, W)
            
            # Flatten batch dims
            flat_batch_size = 1
            for d in batch_dims:
                flat_batch_size *= d
            
            x_flat = x.view(flat_batch_size, *spatial_dims)
            
            # Forward pass
            output = self.panecho_model(x_flat)
            
            # Handle Dictionary Output
            if isinstance(output, dict):
                if 'embedding' in output:
                    emb_flat = output['embedding']
                elif 'last_hidden_state' in output:
                    state = output['last_hidden_state']
                    emb_flat = state[:, 0] if state.ndim == 3 else state
                else:
                    emb_flat = next(iter(output.values()))
            else:
                emb_flat = output
            
            # --- CRITICAL FIX: Return List[Tensor] ---
            # V-JEPA eval loop iterates over the output.
            # If we return a Tensor [B, N, D], iteration yields B tensors of shape [N, D] (2D) -> CRASH.
            # We MUST return a List of length N, where each item is [B, 1, D] (3D).
            
            true_batch_size = batch_dims[0]
            num_tokens = flat_batch_size // true_batch_size
            
            # Reshape to [B, N, D]
            emb_3d = emb_flat.view(true_batch_size, num_tokens, self.embed_dim)
            
            # Split into list of [B, 1, D] tensors
            # This ensures that when eval.py iterates, it gets a Batch of data for one view.
            return [emb_3d[:, i:i+1, :] for i in range(num_tokens)]

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

    # Context Swap
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
        
        panecho_model = hubconf.PanEcho(
            pretrained=True, 
            clip_len=frames_per_clip,
            backbone_only=True
        )
        
    except Exception as e:
        logger.error(f"Failed to load PanEcho: {e}")
        raise e
        
    finally:
        panecho_modules = {k: v for k, v in sys.modules.items() if k == 'src' or k.startswith('src.')}
        for k in panecho_modules:
            del sys.modules[k]
        sys.path = original_path
        sys.modules.update(vjepa_modules)

    logger.info(f"PanEcho loaded successfully. Embed dim: {768}")
    return PanEchoWrapper(panecho_model)