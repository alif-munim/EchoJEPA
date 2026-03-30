"""Lightweight segmentation decoders for EchoBench CAMUS evaluation.

Ported from evals/segmentation_frozen/eval.py.
"""

import torch.nn as nn
import torch.nn.functional as F


class LinearSegDecoder(nn.Module):
    """Minimal segmentation decoder: 1x1 conv on spatial token grid + upsample."""

    def __init__(self, embed_dim, num_classes=4, target_size=224):
        super().__init__()
        self.head = nn.Conv2d(embed_dim, num_classes, kernel_size=1)
        self.target_size = target_size

    def forward(self, x):
        # x: [B, H', W', D]
        x = x.permute(0, 3, 1, 2)  # [B, D, H', W']
        x = self.head(x)  # [B, C, H', W']
        x = F.interpolate(x, size=self.target_size, mode="bilinear", align_corners=False)
        return x  # [B, C, H, W]


class SmallConvSegDecoder(nn.Module):
    """4-stage transposed conv decoder: 14x14 -> 224x224 (16x upsample)."""

    def __init__(self, embed_dim, num_classes=4):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 256, 4, stride=2, padding=1),  # 14->28
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 28->56
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 56->112
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.ConvTranspose2d(64, num_classes, 4, stride=2, padding=1),  # 112->224
        )

    def forward(self, x):
        # x: [B, H', W', D]
        x = x.permute(0, 3, 1, 2)  # [B, D, H', W']
        return self.decoder(x)  # [B, C, H, W]
