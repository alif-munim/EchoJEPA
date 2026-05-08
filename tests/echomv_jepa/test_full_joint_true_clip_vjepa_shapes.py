"""Shape + finiteness contract for the true V-JEPA clip loss."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.echomv_jepa.full_joint_losses import clip_vjepa_true_loss


class ToyEncoder(nn.Module):
    """Mirrors MultiSeqWrapper(VisionTransformer).forward(x, masks=None)."""

    def __init__(self, D: int = 32, n_tokens: int = 64):
        super().__init__()
        self.proj = nn.Linear(D, D)
        self.n_tokens = n_tokens
        self.D = D

    def forward(self, clips_l, masks_l=None):
        out = []
        for clips in clips_l:
            pooled = clips.flatten(2).mean(-1)
            if pooled.size(-1) != self.D:
                reps = (self.D + pooled.size(-1) - 1) // pooled.size(-1)
                pooled = pooled.repeat(1, reps)[:, : self.D]
            base = pooled.unsqueeze(1).expand(-1, self.n_tokens, -1).contiguous()
            x = self.proj(base)
            if masks_l is None:
                out.append(x)
            else:
                per_fpc = []
                for mlist in masks_l:
                    m = mlist[0]
                    idx = m.to(clips.device)
                    gathered = x.gather(1, idx.unsqueeze(-1).expand(-1, -1, self.D))
                    per_fpc.append(gathered)
                return per_fpc
        return out


class ToyPredictor(nn.Module):
    """Mirrors PredictorMultiSeqWrapper(VisionTransformerPredictor)."""

    def __init__(self, D: int = 32):
        super().__init__()
        self.net = nn.Linear(D, D)
        self.D = D

    def forward(self, x_list, masks_x_list, masks_y_list, delta_phi=None):
        out = []
        for xi, mxi, myi in zip(x_list, masks_x_list, masks_y_list):
            row = []
            B = xi.size(0) if hasattr(xi, "size") else xi[0].size(0)
            ctx = xi.mean(dim=1, keepdim=True) if isinstance(xi, torch.Tensor) else xi[0].mean(dim=1, keepdim=True)
            for m_x, m_y in zip(mxi if isinstance(mxi, list) else [mxi], myi if isinstance(myi, list) else [myi]):
                N_tgt = m_y.size(1)
                t = ctx.expand(B, N_tgt, self.D).contiguous()
                row.append(self.net(t))
            out.append(row)
        return out


def _make_batch(N=2, T=4, H=16, W=16, n_tokens=64, n_ctx=40, n_tgt=20):
    clips = torch.randn(N, 3, T, H, W)
    enc = torch.stack([torch.randperm(n_tokens)[:n_ctx] for _ in range(N)])
    pred = torch.stack([torch.randperm(n_tokens)[:n_tgt] for _ in range(N)])
    return clips, enc, pred


def test_true_clip_vjepa_loss_is_finite_and_positive():
    torch.manual_seed(0)
    D = 32
    encoder = ToyEncoder(D)
    target = ToyEncoder(D)
    predictor = ToyPredictor(D)
    clips, m_enc, m_pred = _make_batch()
    loss = clip_vjepa_true_loss(clips, encoder, target, predictor, m_enc, m_pred)
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_true_clip_vjepa_loss_passes_wrapper_shapes_through_correctly():
    """The loss helper must accept (N, 3, T, H, W) clips and a single pair
    of mask tensors and return a scalar."""
    torch.manual_seed(1)
    D = 32
    encoder = ToyEncoder(D)
    target = ToyEncoder(D)
    predictor = ToyPredictor(D)
    clips, m_enc, m_pred = _make_batch(N=4)
    loss = clip_vjepa_true_loss(clips, encoder, target, predictor, m_enc, m_pred)
    assert loss.dim() == 0
