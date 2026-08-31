"""EchoPrime (MViT-v2-S) encoder adapter for EchoBench.

Requires:
    - torchvision (built-in, no extra install)
    - EchoPrime encoder checkpoint (not publicly available)

The checkpoint modifies the MViT-v2-S head to output 512-dim embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from echobench.adapters.base import BaseAdapter, IMAGENET_MEAN, IMAGENET_STD


# EchoPrime normalization (0-255 pixel space)
_EP_MEAN = torch.tensor([29.110628, 28.076836, 29.096405]).view(1, 3, 1, 1, 1)
_EP_STD = torch.tensor([47.989223, 46.456997, 47.20083]).view(1, 3, 1, 1, 1)
_INET_MEAN = IMAGENET_MEAN.unsqueeze(0)  # [1, 3, 1, 1, 1]
_INET_STD = IMAGENET_STD.unsqueeze(0)


class EchoPrimeAdapter(BaseAdapter):
    """Adapter for EchoPrime (MViT-v2-S) encoder.

    EchoPrime uses a modified MViT-v2-S with a 512-dim output head.
    Input is converted from ImageNet normalization to EchoPrime's
    0-255 pixel-space normalization internally.

    Args:
        checkpoint: path to echo_prime_encoder.pt
        device: torch device
        resolution: input resolution (resampled to 224 internally)
        frames: number of input frames (resampled to 16 internally)
    """

    def __init__(self, checkpoint, device="cuda", resolution=224, frames=16, **kwargs):
        super().__init__()
        import torchvision.models.video as video_models

        self.encoder = video_models.mvit_v2_s()
        self.encoder.head[-1] = nn.Linear(self.encoder.head[-1].in_features, 512)
        self.embed_dim = 512

        state_dict = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.encoder.load_state_dict(state_dict)
        self.to(device)
        self.freeze()

        self._device = device

    def forward(self, x):
        """[B, C, T, H, W] -> [B, 1, 512] pooled embedding."""
        B = x.shape[0]

        # Undo ImageNet norm -> [0, 1] -> [0, 255] -> EchoPrime norm
        inet_mean = _INET_MEAN.to(x.device, x.dtype)
        inet_std = _INET_STD.to(x.device, x.dtype)
        ep_mean = _EP_MEAN.to(x.device, x.dtype)
        ep_std = _EP_STD.to(x.device, x.dtype)

        x = x * inet_std + inet_mean  # undo ImageNet -> [0, 1]
        x = x * 255.0  # -> [0, 255]
        x = (x - ep_mean) / ep_std  # EchoPrime norm

        # Resample to 16 frames if needed
        T = x.shape[2]
        if T != 16:
            indices = torch.linspace(0, T - 1, 16).round().long()
            x = x[:, :, indices]

        # Resize to 224x224 if needed
        if x.shape[3] != 224 or x.shape[4] != 224:
            x = F.interpolate(
                x.reshape(-1, x.shape[1], x.shape[3], x.shape[4]),
                size=(224, 224), mode="bilinear", align_corners=False,
            ).reshape(B, x.shape[1], -1, 224, 224)

        out = self.encoder(x)  # [B, 512]
        return out.unsqueeze(1)  # [B, 1, 512]
