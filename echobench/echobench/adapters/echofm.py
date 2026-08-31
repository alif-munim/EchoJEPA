"""EchoFM (ViT-L MAE) encoder adapter for EchoBench.

Requires:
    - EchoFM source repository cloned locally
    - Set ECHOFM_ROOT environment variable or pass echofm_root kwarg
    - EchoFM checkpoint file

EchoFM was trained on [0,1] pixel range, so ImageNet normalization
is undone internally before the encoder forward pass.
"""

import os
import sys

import torch
import torch.nn.functional as F

from echobench.adapters.base import BaseAdapter, IMAGENET_MEAN, IMAGENET_STD


class EchoFMAdapter(BaseAdapter):
    """Adapter for EchoFM (ViT-L MAE) encoder.

    EchoFM is a ViT-Large masked autoencoder pretrained on 290K echo clips.
    It expects [0,1] pixel range (not ImageNet-normalized), so this adapter
    undoes ImageNet normalization internally.

    Output: [B, 1568, 1024] spatiotemporal tokens (8 temporal x 14x14 spatial).

    Args:
        checkpoint: path to EchoFM checkpoint
        device: torch device
        resolution: input resolution (resampled to 224 internally)
        frames: number of input frames (resampled to 32 internally)
        echofm_root: path to EchoFM source directory.
            Falls back to ECHOFM_ROOT env var or ./EchoFM
    """

    def __init__(self, checkpoint, device="cuda", resolution=224, frames=16,
                 echofm_root=None, **kwargs):
        super().__init__()
        self.embed_dim = 1024
        self._target_frames = 32  # EchoFM expects 32 frames

        echofm_root = echofm_root or os.environ.get("ECHOFM_ROOT", "./EchoFM")
        if not os.path.isdir(echofm_root):
            raise FileNotFoundError(
                f"EchoFM source not found at {echofm_root}. "
                "Clone the EchoFM repository and set ECHOFM_ROOT."
            )

        self.encoder = self._load_with_isolation(echofm_root, checkpoint)

        # ImageNet stats for denormalization (as buffers)
        self.register_buffer("inet_mean", IMAGENET_MEAN.unsqueeze(0).unsqueeze(2))  # [1, 3, 1, 1, 1]
        self.register_buffer("inet_std", IMAGENET_STD.unsqueeze(0).unsqueeze(2))

        self.to(device)
        self.freeze()

    def _load_with_isolation(self, echofm_root, checkpoint):
        """Load EchoFM with namespace isolation."""
        vjepa_modules = {k: v for k, v in sys.modules.items() if k == "src" or k.startswith("src.")}
        original_path = list(sys.path)

        try:
            for k in vjepa_modules:
                sys.modules.pop(k, None)
            sys.path.insert(0, echofm_root)

            from EchoFM.models_mae import MaskedAutoencoderViT

            model = MaskedAutoencoderViT(
                img_size=224, patch_size=16, embed_dim=1024, depth=24, num_heads=16,
                num_frames=32, t_patch_size=4, sep_pos_embed=True, cls_embed=True,
            )

            ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
            state_dict = ckpt.get("model", ckpt)

            # Filter out decoder keys
            decoder_prefixes = ("decoder", "mask_token", "decoder_embed", "decoder_cls_token", "decoder_prj_cls_token")
            encoder_keys = {k: v for k, v in state_dict.items() if not any(k.startswith(p) for p in decoder_prefixes)}

            model.load_state_dict(encoder_keys, strict=False)
            return model
        finally:
            echofm_mods = {k: v for k, v in sys.modules.items()
                           if k == "EchoFM" or k.startswith("EchoFM.")}
            for k in echofm_mods:
                sys.modules.pop(k, None)
            sys.path = original_path
            sys.modules.update(vjepa_modules)

    def forward(self, x):
        """[B, C, T, H, W] -> [B, N, 1024] spatiotemporal tokens."""
        B = x.shape[0]

        # Undo ImageNet norm -> [0, 1]
        x = x * self.inet_std + self.inet_mean

        # Resample to 32 frames
        T = x.shape[2]
        if T != self._target_frames:
            indices = torch.linspace(0, T - 1, self._target_frames).round().long()
            x = x[:, :, indices]

        # Resize to 224x224 if needed
        if x.shape[3] != 224 or x.shape[4] != 224:
            x = x.reshape(B * self._target_frames, x.shape[1], x.shape[3], x.shape[4])
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
            x = x.reshape(B, x.shape[1] if x.dim() > 3 else 3, self._target_frames, 224, 224)

        out = self.encoder.forward_features(x)  # [B, N+1, D] with CLS

        # Strip CLS token if present
        if out.shape[1] > 1568:
            out = out[:, 1:]  # Remove CLS -> [B, 1568, 1024]

        return out
