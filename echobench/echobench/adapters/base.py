"""Base adapter interface for EchoBench encoder models."""

from typing import Protocol, runtime_checkable

import torch
import torch.nn as nn


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)


@runtime_checkable
class EncoderAdapter(Protocol):
    """Protocol that all EchoBench encoder adapters must satisfy."""

    embed_dim: int

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode video clip to token embeddings.

        Args:
            x: [B, C, T, H, W] float32, ImageNet-normalized

        Returns:
            [B, N, D] token embeddings
        """
        ...


class BaseAdapter(nn.Module):
    """Convenience base class for building encoder adapters.

    Subclass this and implement ``_build_encoder`` and ``forward``.
    Provides common utilities for checkpoint loading and normalization.
    """

    embed_dim: int

    def normalize_clip(self, clip):
        """Apply ImageNet normalization to [C, T, H, W] or [B, C, T, H, W] clip."""
        if clip.dim() == 4:
            mean = IMAGENET_MEAN.squeeze(-1)  # [3, 1, 1]
            std = IMAGENET_STD.squeeze(-1)
        else:
            mean = IMAGENET_MEAN
            std = IMAGENET_STD
        return (clip - mean.to(clip.device)) / std.to(clip.device)

    def freeze(self):
        """Set eval mode and disable gradients."""
        self.eval()
        for p in self.parameters():
            p.requires_grad = False
        return self
