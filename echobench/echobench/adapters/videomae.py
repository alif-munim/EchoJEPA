"""VideoMAE encoder adapter for EchoBench."""

import torch

from echobench.adapters.base import BaseAdapter


class VideoMAEAdapter(BaseAdapter):
    """Adapter for VideoMAE (EchoMAE) encoders.

    Loads a VideoMAE-Large checkpoint and wraps it for EchoBench evaluation.

    Args:
        checkpoint: path to .pth checkpoint file
        device: torch device
        resolution: input spatial resolution (default 224)
        frames: number of input frames (default 16)
    """

    def __init__(
        self,
        checkpoint,
        device="cuda",
        resolution=224,
        frames=16,
    ):
        super().__init__()
        from evals.video_classification_frozen.modelcustom.videomae_encoder import (
            _convert_pretrain_to_finetune_state_dict,
            _import_modeling_finetune,
        )

        mf = _import_modeling_finetune()
        self.encoder = mf.vit_large_patch16_224(
            img_size=resolution,
            all_frames=frames,
            tubelet_size=2,
            num_classes=1000,
        )
        self.embed_dim = self.encoder.embed_dim

        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt)
        state = _convert_pretrain_to_finetune_state_dict(state, self.encoder.state_dict())
        self.encoder.load_state_dict(state, strict=False)
        self.to(device)
        self.freeze()

    def forward(self, x):
        """[B, C, T, H, W] -> [B, N, D] token embeddings."""
        out = self.encoder.forward_features(x)
        if out.dim() == 2:
            out = out.unsqueeze(0)
        return out
