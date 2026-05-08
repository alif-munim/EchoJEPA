"""End-to-end shape + finiteness tests for forward_mcc_jepa.

Uses toy stand-in encoder / predictor modules so the test runs in <1 s.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from app.vjepa_multiview.mcc_jepa_forward import forward_mcc_jepa
from app.vjepa_multiview.train import PairBatch
from src.models.mcc_jepa import CrossClipAdapter


class ToyEncoder(nn.Module):
    """Stand-in for the V-JEPA encoder.

    Returns a list-over-fpc of ``[B, N_vis, D]`` tensors. When ``masks`` is
    None, returns the full token grid of ``num_tokens`` tokens; otherwise
    returns only the kept (visible) tokens.
    """

    def __init__(self, d: int, num_tokens: int):
        super().__init__()
        self.d = d
        self.num_tokens = num_tokens
        self.proj = nn.Linear(d, d)

    def forward(self, clip_list, masks=None):
        out = []
        for clip in clip_list:
            # Pool the pixels into d channels so gradients flow through proj.
            pooled = clip.flatten(2).mean(dim=-1)  # [B, C]
            if pooled.size(-1) != self.d:
                # pad or repeat channels up to d
                reps = (self.d + pooled.size(-1) - 1) // pooled.size(-1)
                pooled = pooled.repeat(1, reps)[:, : self.d]
            base = pooled.unsqueeze(1).expand(-1, self.num_tokens, -1).contiguous()
            x = self.proj(base)
            if masks is None:
                out.append(x)
            else:
                # masks is list-over-fpc of list-over-mask-generator of [B, N_ctx]
                # For the toy encoder treat the first entry as the visible set.
                out_per_fpc = []
                for mlist in masks:
                    if isinstance(mlist, list):
                        m = mlist[0]
                    else:
                        m = mlist
                    idx = m.to(clip.device)
                    gathered = x.gather(1, idx.unsqueeze(-1).expand(-1, -1, self.d))
                    out_per_fpc.append(gathered)
                return out_per_fpc
        return out


class ToyPredictor(nn.Module):
    """Stand-in for VisionTransformerPredictor.

    Always returns a list-of-list matching the (fpc, mask_generator) shape
    of ``masks_y``, with the predictor output dim equal to the encoder's
    embed dim. Batch dimension is B * len(masks_x[fpc]).
    """

    def __init__(self, d: int):
        super().__init__()
        self.d = d
        self.net = nn.Linear(d, d)

    def forward(self, x_list, masks_x_list, masks_y_list, delta_phi=None):
        out: list[list[torch.Tensor]] = []
        for xi, mxi, myi in zip(x_list, masks_x_list, masks_y_list):
            row = []
            B = xi.size(0)
            # Pool context tokens into a per-sample feature; broadcast to the
            # target length so gradients flow from target output into xi.
            ctx = xi.mean(dim=1, keepdim=True)  # [B, 1, d]
            for m_x, m_y in zip(mxi if isinstance(mxi, list) else [mxi], myi if isinstance(myi, list) else [myi]):
                N_tgt = m_y.size(1)
                t = ctx.expand(B, N_tgt, self.d).contiguous()
                row.append(self.net(t))
            out.append(row)
        return out


def _make_pair(B=2, N_total=32, N_ctx=20, N_tgt=12, D=32):
    device = torch.device("cpu")
    clip_a = [torch.randn(B, 3, 4, 32, 32, device=device)]
    clip_b = [torch.randn(B, 3, 4, 32, 32, device=device)]
    masks_enc = [[torch.arange(N_ctx).unsqueeze(0).expand(B, -1).contiguous()]]
    masks_pred = [[torch.arange(N_ctx, N_ctx + N_tgt).unsqueeze(0).expand(B, -1).contiguous()]]
    return PairBatch(clip_a=clip_a, clip_b=clip_b, masks_enc=masks_enc, masks_pred=masks_pred, phase_metadata=[])


def test_target_anchored_forward_shapes_and_finite_loss():
    torch.manual_seed(0)
    D, N_total, N_ctx, N_tgt, B = 32, 32, 20, 12, 2
    encoder = ToyEncoder(D, N_total)
    target_encoder = ToyEncoder(D, N_total)
    predictor = ToyPredictor(D)
    adapter = CrossClipAdapter(embed_dim=D, num_heads=4)
    pair = _make_pair(B, N_total, N_ctx, N_tgt, D)

    out = forward_mcc_jepa(
        pair,
        encoder,
        target_encoder,
        predictor,
        adapter,
        mode="target_anchored",
        lambda_mcc=0.2,
        lambda_vjepa=1.0,
    )
    assert torch.isfinite(out["total_loss"])
    assert torch.isfinite(out["loss_mcc"])
    assert torch.isfinite(out["loss_vjepa_self"])
    # At gamma=0 the MCC and vanilla-self losses should be equal.
    assert torch.allclose(out["loss_mcc"], out["loss_vjepa_self"])
    assert float(out["pred_delta_from_A"]) < 1e-5


def test_teacher_path_has_no_grad():
    torch.manual_seed(0)
    D = 32
    encoder = ToyEncoder(D, 32)
    target_encoder = ToyEncoder(D, 32)
    predictor = ToyPredictor(D)
    adapter = CrossClipAdapter(embed_dim=D, num_heads=4)
    pair = _make_pair()

    out = forward_mcc_jepa(pair, encoder, target_encoder, predictor, adapter, mode="target_anchored")
    out["total_loss"].backward()
    # Teacher params have no grad
    for p in target_encoder.parameters():
        assert p.grad is None
    # Student + adapter have grad
    any_grad = any((p.grad is not None and p.grad.abs().sum() > 0) for p in encoder.parameters())
    assert any_grad
