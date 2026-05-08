"""Online V-JEPA clip encoder wrapper for EchoMV-JEPA Option A.

Loads a frozen V-JEPA ViT-L checkpoint once and exposes a single
``forward_tokens`` method that returns ``(N_clips, T_out, d_clip)`` token
embeddings. Optionally applies a spatial pool inside the wrapper to reduce
``T_out`` from 1568 (1x) to 392 (2x2 spatial pool) or fewer.

Intended to live in-process during training: called at every step to encode
K × B clips per GPU (typically 8 × 32 = 256 per step at batch_size 32, or
8 × 8 = 64 at the reduced batch size used by Option A's memory budget).

Reuses exactly the loading path that
``experiments/echoset_jepa/cache_cclip.py`` uses to produce the pooled c_clip
cache, with one difference: we do **not** apply ``.mean(dim=1)`` — the caller
receives the full token tensor and either keeps tokens or pools them itself.

Contract notes:
  - The encoder is always in ``eval()`` mode with ``requires_grad=False`` on
    all params (frozen — matches Stage-1's "frozen clip encoder" design).
  - The wrapper expects clips already on-device in the encoder's expected
    shape ``(B, 3, T_frames, H, W)`` (with B = total clip count after
    flattening study × clips-per-study) and in encoder-normalized color
    space.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml


class OnlineVJepaEncoder(nn.Module):
    """Frozen V-JEPA encoder returning per-clip tokens.

    Parameters
    ----------
    config_path : str
        Path to a YAML with ``experiment.data`` (frames_per_clip, resolution,
        frame_step, normalization) and ``model_kwargs`` (module_name,
        checkpoint, pretrain_kwargs, wrapper_kwargs) — same format the
        cache_cclip.py job uses.
    device : torch.device
        Where to place the encoder weights.
    token_spatial_pool : int, default 1
        Factor for a post-encoder 2D spatial pool. 1 = no pool (1568 tokens
        for ViT-L/16 on 224x224x16 input). 2 = 2x2 spatial pool (392 tokens).
        The pool is applied only along the spatial axes; temporal tubelets
        are preserved.
    spatial_hw : int, default 14
        Spatial grid side after ViT patchification (224/16=14 for ViT-L).
        Used to reshape N_tokens back to (T, H, W) before the spatial pool.
    temporal_tubelets : int, default 8
        Number of temporal tubelets (16 frames / tubelet_size=2 = 8).
    """

    def __init__(
        self,
        config_path: str,
        device: torch.device,
        *,
        token_spatial_pool: int = 1,
        spatial_hw: int = 14,
        temporal_tubelets: int = 8,
    ) -> None:
        super().__init__()
        if token_spatial_pool < 1 or spatial_hw % token_spatial_pool != 0:
            raise ValueError(f"token_spatial_pool={token_spatial_pool} must divide spatial_hw={spatial_hw}")
        self.token_spatial_pool = int(token_spatial_pool)
        self.spatial_hw = int(spatial_hw)
        self.temporal_tubelets = int(temporal_tubelets)

        with open(config_path) as f:
            config = yaml.safe_load(f)
        self._data_cfg: Dict[str, Any] = config["experiment"]["data"]
        self._model_kwargs: Dict[str, Any] = config["model_kwargs"]

        # Lazy import to keep this file importable in environments without the
        # heavyweight encoder modules installed (e.g. pure-CPU unit tests that
        # stub the encoder with a mock).
        from evals.video_classification_frozen.models import init_module

        self._encoder = init_module(
            module_name=self._model_kwargs["module_name"],
            frames_per_clip=self._data_cfg.get("frames_per_clip", 16),
            resolution=self._data_cfg.get("resolution", 224),
            checkpoint=self._model_kwargs.get("checkpoint"),
            model_kwargs=self._model_kwargs.get("pretrain_kwargs", {}),
            wrapper_kwargs=self._model_kwargs.get("wrapper_kwargs", {}),
            device=device,
        )
        self._encoder.eval()
        for p in self._encoder.parameters():
            p.requires_grad_(False)

        # d_clip is read from the first forward — avoid depending on model_kwargs
        # internals. Cached after the first forward_tokens call.
        self._d_clip: Optional[int] = None

    @property
    def d_clip(self) -> int:
        if self._d_clip is None:
            raise RuntimeError("d_clip is set on the first forward_tokens() call")
        return self._d_clip

    @property
    def tokens_per_clip(self) -> int:
        """T_out after the optional spatial pool."""
        hw = self.spatial_hw // self.token_spatial_pool
        return self.temporal_tubelets * hw * hw

    @torch.no_grad()
    def forward_tokens(
        self,
        clips: torch.Tensor,  # (N, 3, T_frames, H, W)
        clip_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode a flat batch of clips and return per-clip tokens.

        Parameters
        ----------
        clips : (N, 3, T_frames, H, W) float
            The raw clip tensor. N = total clips in the minibatch
            (study_batch_size * K).
        clip_indices : optional long tensor
            Some V-JEPA wrappers (e.g. multi-clip aggregators) use this for
            position bookkeeping. Pass ``None`` for the single-clip mode
            cache_cclip.py uses.

        Returns
        -------
        (N, T_out, d_clip) float
        """
        # Encoder wrappers used by cache_cclip.py expect a nested list of
        # clips and clip indices. We wrap our single-segment batch the same
        # way: one segment per sample.
        n = clips.shape[0]
        if clip_indices is None:
            clip_indices = torch.zeros(n, dtype=torch.long, device=clips.device)
        # Match cache_cclip's list-of-lists convention: [[segment_0_tensor], ...]
        outputs = self._encoder([[clips]], [clip_indices])
        # outputs is a list of tensors (one per segment); with one segment,
        # outputs[0] has shape (N, N_tokens, d_clip).
        toks = outputs[0] if isinstance(outputs, (list, tuple)) else outputs

        if self._d_clip is None:
            self._d_clip = int(toks.shape[-1])

        if self.token_spatial_pool == 1:
            return toks

        # Reshape to (N, T_tub, H, W, d) and apply 2D pool.
        H = W = self.spatial_hw
        T_tub = self.temporal_tubelets
        expected = T_tub * H * W
        if toks.shape[1] != expected:
            raise RuntimeError(f"token count {toks.shape[1]} != T_tub*H*W = {T_tub}*{H}*{W}={expected}")
        x = toks.reshape(n, T_tub, H, W, -1)
        # move d to channel for pool2d: (N*T_tub, d, H, W)
        x = x.permute(0, 1, 4, 2, 3).reshape(n * T_tub, -1, H, W)
        x = F.avg_pool2d(x, kernel_size=self.token_spatial_pool, stride=self.token_spatial_pool)
        H2 = H // self.token_spatial_pool
        # back to (N, T_tub, H2, W2, d) then flatten to (N, T_out, d)
        x = x.reshape(n, T_tub, -1, H2, H2).permute(0, 1, 3, 4, 2)
        return x.reshape(n, T_tub * H2 * H2, -1).contiguous()

    def forward(self, *args, **kwargs):  # type: ignore[override]
        raise RuntimeError("Call forward_tokens(clips) explicitly.")


__all__ = ["OnlineVJepaEncoder"]
