"""Layer-wise cosine diagnostic: hooks capture block outputs; cosine
between two copies of the same encoder is 1.0 at every layer."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from src.models.echomv_jepa.layerwise_drift import compute_layerwise_cosine


class _Block(nn.Module):
    def __init__(self, D: int):
        super().__init__()
        self.lin = nn.Linear(D, D)

    def forward(self, x, **kwargs):
        return x + self.lin(x)


class _MiniViT(nn.Module):
    def __init__(self, depth: int = 4, D: int = 16, num_tokens: int = 8):
        super().__init__()
        self.embed_dim = D
        self.num_tokens = num_tokens
        self.patch = nn.Linear(3, D)  # dummy patch proj
        self.blocks = nn.ModuleList([_Block(D) for _ in range(depth)])
        self.norm = nn.LayerNorm(D)

    def forward(self, x, **kwargs):
        # x is (N, 3, T, H, W); we pool spatial-temporal to (N, num_tokens, 3), then embed.
        pool = x.flatten(2).mean(-1)  # (N, 3)
        t = pool.unsqueeze(1).expand(-1, self.num_tokens, -1).contiguous()
        h = self.patch(t)
        for blk in self.blocks:
            h = blk(h)
        return self.norm(h)


class _Wrapper(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x_list, masks=None):
        out = []
        for x in x_list:
            out.append(self.backbone(x))
        return out


def test_cosine_is_one_on_identical_copies():
    torch.manual_seed(0)
    vit = _MiniViT(depth=4, D=16)
    online = _Wrapper(vit)
    anchor = _Wrapper(copy.deepcopy(vit))
    clips = torch.randn(3, 3, 2, 4, 4)
    out = compute_layerwise_cosine(online, anchor, clips, block_indices=[0, 1, 2, 3])
    for k in ("block_0", "block_1", "block_2", "block_3", "top_block"):
        assert k in out
        assert out[k] > 0.999, f"{k}: {out[k]}"


def test_cosine_drops_when_online_perturbed():
    torch.manual_seed(1)
    vit = _MiniViT(depth=4, D=16)
    online_vit = copy.deepcopy(vit)
    # Perturb the last block of the online encoder.
    with torch.no_grad():
        for p in online_vit.blocks[3].parameters():
            p.add_(torch.randn_like(p) * 0.5)
    online = _Wrapper(online_vit)
    anchor = _Wrapper(vit)
    clips = torch.randn(3, 3, 2, 4, 4)
    out = compute_layerwise_cosine(online, anchor, clips, block_indices=[0, 1, 2, 3])
    # Early blocks should still match closely.
    assert out["block_0"] > 0.99
    # Last block should have drifted.
    assert out["block_3"] < 0.99


def test_empty_clips_returns_nan():
    torch.manual_seed(2)
    vit = _MiniViT(depth=2, D=16)
    online = _Wrapper(vit)
    anchor = _Wrapper(copy.deepcopy(vit))
    clips = torch.empty(0, 3, 2, 4, 4)
    out = compute_layerwise_cosine(online, anchor, clips, block_indices=[0, 1])
    import math

    assert all(math.isnan(v) for v in out.values())
