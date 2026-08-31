"""PanEcho (ConvNeXt-Tiny + Transformer) encoder adapter for EchoBench.

Requires:
    - PanEcho source repository cloned locally
    - Set PANECHO_ROOT environment variable or pass panecho_root kwarg

PanEcho loads pretrained weights from its torch hub interface.
"""

import os
import sys

import torch
import torch.nn.functional as F

from echobench.adapters.base import BaseAdapter


class PanEchoAdapter(BaseAdapter):
    """Adapter for PanEcho encoder.

    PanEcho uses a ConvNeXt-Tiny backbone with a Transformer pooler.
    Weights are loaded from PanEcho's hub (pretrained=True).

    Args:
        checkpoint: unused (PanEcho loads from hub). Pass any string.
        device: torch device
        resolution: input resolution (upscaled to 224 internally)
        frames: number of input frames
        panecho_root: path to PanEcho source directory.
            Falls back to PANECHO_ROOT env var or ./PanEcho
    """

    def __init__(self, checkpoint=None, device="cuda", resolution=224, frames=16,
                 panecho_root=None, **kwargs):
        super().__init__()
        self.embed_dim = 768

        panecho_root = panecho_root or os.environ.get("PANECHO_ROOT", "./PanEcho")
        if not os.path.isdir(panecho_root):
            raise FileNotFoundError(
                f"PanEcho source not found at {panecho_root}. "
                "Clone it: git clone https://github.com/echonet/panecho.git PanEcho"
            )

        self.encoder = self._load_with_isolation(panecho_root, frames)
        self.to(device)
        self.freeze()

    def _load_with_isolation(self, panecho_root, frames):
        """Load PanEcho with namespace isolation to avoid src/ conflicts."""
        import importlib.util
        import ssl

        # SSL workaround for PanEcho weight downloads
        try:
            ssl._create_default_https_context = ssl._create_unverified_context
        except AttributeError:
            pass

        # Save vjepa2 modules
        vjepa_modules = {k: v for k, v in sys.modules.items() if k == "src" or k.startswith("src.")}
        original_path = list(sys.path)

        try:
            for k in vjepa_modules:
                sys.modules.pop(k, None)
            sys.path.insert(0, panecho_root)

            hubconf_path = os.path.join(panecho_root, "hubconf.py")
            spec = importlib.util.spec_from_file_location("hubconf", hubconf_path)
            hubconf = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(hubconf)

            model = hubconf.PanEcho(pretrained=True, clip_len=frames, backbone_only=True)
            return model
        finally:
            # Restore vjepa2 modules
            panecho_mods = {k: v for k, v in sys.modules.items() if k == "src" or k.startswith("src.")}
            for k in panecho_mods:
                sys.modules.pop(k, None)
            sys.path = original_path
            sys.modules.update(vjepa_modules)

    def forward(self, x):
        """[B, C, T, H, W] -> [B, 1, 768] pooled embedding."""
        if x.shape[3] != 224 or x.shape[4] != 224:
            B, C, T = x.shape[:3]
            x = x.reshape(B * T, C, x.shape[3], x.shape[4])
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
            x = x.reshape(B, C, T, 224, 224)

        out = self.encoder(x)  # [B, 768]
        if out.dim() == 2:
            out = out.unsqueeze(1)  # [B, 1, 768]
        return out
