"""EchoJEPA / V-JEPA encoder adapter for EchoBench."""

import torch

from echobench.adapters.base import BaseAdapter


class EchoJEPAAdapter(BaseAdapter):
    """Adapter for EchoJEPA (V-JEPA 2) ViT encoders.

    Loads a ViT encoder from an EchoJEPA checkpoint and wraps it for
    the EchoBench evaluation interface.

    Args:
        checkpoint: path to .pt checkpoint file
        model_name: ViT variant (e.g., "vit_large", "vit_giant_xformers")
        checkpoint_key: key in checkpoint dict ("target_encoder" or "encoder")
        device: torch device
        resolution: input spatial resolution (default 224)
        frames: number of input frames (default 16)
    """

    def __init__(
        self,
        checkpoint,
        model_name="vit_large",
        checkpoint_key="target_encoder",
        device="cuda",
        resolution=224,
        frames=16,
    ):
        super().__init__()
        import src.models.vision_transformer as vit

        self.encoder = vit.__dict__[model_name](
            img_size=resolution,
            num_frames=frames,
            patch_size=16,
            tubelet_size=2,
            uniform_power=True,
            use_rope=True,
        )
        self.embed_dim = self.encoder.embed_dim

        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = ckpt[checkpoint_key]
        state = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state.items()}

        # Handle shape mismatches (e.g., RoPE buffers)
        model_sd = self.encoder.state_dict()
        for k in list(state.keys()):
            if k in model_sd and state[k].shape != model_sd[k].shape:
                del state[k]

        self.encoder.load_state_dict(state, strict=False)
        self.to(device)
        self.freeze()

    def forward(self, x):
        """[B, C, T, H, W] -> [B, N, D] token embeddings."""
        return self.encoder(x)
