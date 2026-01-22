"""  
Custom model wrapper for PanEcho to work with V-JEPA 2 evaluation system.  
"""  
  
import logging  
import torch  
import torch.nn as nn  
  
logging.basicConfig()  
logger = logging.getLogger()  
logger.setLevel(logging.INFO)  
  
  
class PanEchoWrapper(nn.Module):  
    """Wrapper to make PanEcho compatible with V-JEPA 2 eval system."""  
      
    def __init__(self, panecho_model):  
        super().__init__()  
        self.panecho_model = panecho_model  
        self.embed_dim = 768  # PanEcho's embedding dimension  
          
    def forward(self, x):  
        """  
        Args:  
            x: Video clips of shape [batch_size, 3, num_frames, height, width]  
        Returns:  
            Features of shape [batch_size, 1, embed_dim] for AttentiveClassifier  
        """  
        # Get PanEcho embeddings: [batch_size, 768]  
        embeddings = self.panecho_model(x)  
          
        # Add token dimension for AttentiveClassifier compatibility  
        # Shape: [batch_size, 1, 768]  
        return embeddings.unsqueeze(1)  
  
  
def init_module(  
    resolution: int,  
    frames_per_clip: int,  
    checkpoint: str,  
    # --  
    model_kwargs: dict,  
    wrapper_kwargs: dict,  
):  
    """Initialize PanEcho model for V-JEPA 2 evaluation."""  
    logger.info("Loading PanEcho model via PyTorch Hub")  
      
    # Load PanEcho backbone (embeddings only)  
    panecho_model = torch.hub.load(  
        'CarDS-Yale/PanEcho',   
        'PanEcho',   
        force_reload=True,   
        backbone_only=True,  
        clip_len=frames_per_clip  
    )  
      
    # Wrap PanEcho for V-JEPA 2 compatibility  
    model = PanEchoWrapper(panecho_model)  
      
    logger.info(f"PanEcho wrapper initialized with embed_dim: {model.embed_dim}")  
    return model