"""Mask generators for forward prediction and anomaly detection.

Two masking strategies:
1. RandomBlockMask: Random spatio-temporal blocks (reuses training-time masking).
   Used for anomaly detection — measures general prediction error.
2. FutureFrameMask: First half = context, second half = target.
   Used for temporal forward prediction — measures future-frame prediction error.
"""

import torch


class RandomBlockMask:
    """Generate random block masks matching training-time masking.

    Creates masks_enc (context) and masks_pred (target) as index tensors
    into the flattened [T, H, W] token grid.

    For anomaly detection, we average prediction error across K random masks
    per clip to get a stable anomaly score.
    """

    def __init__(
        self,
        num_frames=16,
        img_size=224,
        patch_size=16,
        tubelet_size=2,
        spatial_scale=(0.15, 0.15),
        temporal_scale=(1.0, 1.0),
        aspect_ratio=(0.75, 1.5),
        num_blocks=8,
    ):
        self.T = num_frames // tubelet_size
        self.H = img_size // patch_size
        self.W = img_size // patch_size
        self.num_tokens = self.T * self.H * self.W
        self.spatial_scale = spatial_scale
        self.temporal_scale = temporal_scale
        self.aspect_ratio = aspect_ratio
        self.num_blocks = num_blocks

    def _sample_block_size(self, rng):
        """Sample a block size (t, h, w) from scale ranges."""
        t_scale = self.temporal_scale[0] + rng.random() * (self.temporal_scale[1] - self.temporal_scale[0])
        t = max(1, int(self.T * t_scale))

        s_scale = self.spatial_scale[0] + rng.random() * (self.spatial_scale[1] - self.spatial_scale[0])
        spatial_keep = int(self.H * self.W * s_scale)

        ar = self.aspect_ratio[0] + rng.random() * (self.aspect_ratio[1] - self.aspect_ratio[0])
        import math

        h = min(int(round(math.sqrt(spatial_keep * ar))), self.H)
        w = min(int(round(math.sqrt(spatial_keep / ar))), self.W)
        return t, h, w

    def __call__(self, batch_size, seed=None):
        """Generate masks for a batch.

        Returns:
            masks_enc: [B, N_enc] — indices of context (visible) tokens
            masks_pred: [B, N_pred] — indices of target (masked) tokens
        """
        import numpy as np

        rng = np.random.RandomState(seed)

        all_enc, all_pred = [], []
        for _ in range(batch_size):
            mask = torch.ones(self.T, self.H, self.W, dtype=torch.int32)
            t, h, w = self._sample_block_size(rng)

            for _ in range(self.num_blocks):
                top = rng.randint(0, self.H - h + 1)
                left = rng.randint(0, self.W - w + 1)
                start = rng.randint(0, self.T - t + 1)
                mask[start : start + t, top : top + h, left : left + w] = 0

            mask_flat = mask.flatten()
            enc_idx = torch.nonzero(mask_flat).squeeze(-1)
            pred_idx = torch.nonzero(mask_flat == 0).squeeze(-1)

            # Ensure non-empty
            if len(enc_idx) == 0 or len(pred_idx) == 0:
                # Fallback: random 50/50 split
                perm = torch.randperm(self.num_tokens)
                half = self.num_tokens // 2
                enc_idx = perm[:half].sort().values
                pred_idx = perm[half:].sort().values

            all_enc.append(enc_idx)
            all_pred.append(pred_idx)

        # Pad to same length within batch
        masks_enc = _pad_and_stack(all_enc)
        masks_pred = _pad_and_stack(all_pred)
        return masks_enc, masks_pred


class FutureFrameMask:
    """Temporal split: first half context, second half target.

    For forward prediction evaluation. Splits along temporal axis:
    - Context: all tokens from frames [0, T//2)
    - Target: all tokens from frames [T//2, T)

    Can also do variable splits via context_ratio.
    """

    def __init__(
        self,
        num_frames=16,
        img_size=224,
        patch_size=16,
        tubelet_size=2,
        context_ratio=0.5,
    ):
        self.T = num_frames // tubelet_size
        self.H = img_size // patch_size
        self.W = img_size // patch_size
        self.num_tokens = self.T * self.H * self.W
        self.context_ratio = context_ratio
        self.spatial_tokens = self.H * self.W

    def __call__(self, batch_size):
        """Generate temporal split masks for a batch.

        Returns:
            masks_enc: [B, N_enc] — indices of context tokens (first half frames)
            masks_pred: [B, N_pred] — indices of target tokens (second half frames)
        """
        ctx_frames = max(1, int(self.T * self.context_ratio))
        pred_frames = self.T - ctx_frames

        # Context: tokens from first ctx_frames temporal positions
        # Token layout is [t, h, w] flattened: token_idx = t * H * W + h * W + w
        ctx_indices = []
        for t in range(ctx_frames):
            for hw in range(self.spatial_tokens):
                ctx_indices.append(t * self.spatial_tokens + hw)

        pred_indices = []
        for t in range(ctx_frames, self.T):
            for hw in range(self.spatial_tokens):
                pred_indices.append(t * self.spatial_tokens + hw)

        masks_enc = torch.tensor(ctx_indices, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
        masks_pred = torch.tensor(pred_indices, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
        return masks_enc, masks_pred

    def per_frame_indices(self):
        """Return list of index tensors, one per predicted frame.

        Useful for computing per-frame prediction error.
        """
        ctx_frames = max(1, int(self.T * self.context_ratio))
        frame_indices = []
        for t in range(ctx_frames, self.T):
            idx = []
            for hw in range(self.spatial_tokens):
                idx.append(t * self.spatial_tokens + hw)
            frame_indices.append(torch.tensor(idx, dtype=torch.long))
        return frame_indices


def _pad_and_stack(index_list):
    """Pad variable-length index tensors to the minimum length and stack."""
    min_len = min(len(t) for t in index_list)
    padded = torch.stack([t[:min_len] for t in index_list])
    return padded
