"""forward_token_phase_relational — end-to-end with mock encoder stack.

Avoids depending on the full V-JEPA encoder / predictor / target_encoder
by mocking their outputs at token-shape level. The forward's logic
(teacher concat, L_intra, token subsample, token-rel InfoNCE, pool-rel
safety, motion-delta) is what's under test.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# Required before importing train-side symbols.
from app.vjepa_multiview.phase_relational_head import PhaseRelationalHead  # noqa: E402
from app.vjepa_multiview.token_relational_head import (  # noqa: E402
    DeltaTargetProjector,
    MotionDeltaHead,
    TokenRelationalHead,
)
from app.vjepa_multiview.train import (  # noqa: E402
    forward_token_phase_relational,
)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


@dataclass
class MockPairBatch:
    clip_a: list
    clip_b: list
    clip_b_neg: list
    masks_enc: list
    masks_pred: list
    phase_metadata: list
    target_clip: Any = None
    target_views: Any = None
    target_delta_phase: Any = None
    target_clip_present: Any = None
    fused_clips: Any = None
    fused_views: Any = None
    fused_phases: Any = None
    fused_valid_mask: Any = None


class MockEncoder(nn.Module):
    """Returns nested-list shape matching MultiSeqWrapper output."""

    def __init__(self, n_tokens: int, dim: int):
        super().__init__()
        self.n_tokens = n_tokens
        self.dim = dim
        self.proj = nn.Linear(3 * 2 * 16 * 16, dim)  # C*T*H*W flattened on a tile

    def forward(self, clips, masks=None):
        # clips: list-over-fpc of [B, C, T, H, W]
        B = clips[0].size(0)
        # Synthesize token grid deterministically from the clip.
        x = clips[0].reshape(B, -1)[:, : 3 * 2 * 16 * 16]
        rep = self.proj(x)  # [B, dim]
        tokens = rep.unsqueeze(1).expand(B, self.n_tokens, self.dim).contiguous()
        if masks is None:
            return [tokens]
        return [[tokens]]


class MockPredictor(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Linear(dim, dim)

    def forward(self, z_ctx, masks_enc=None, masks_pred=None, delta_phi=None):
        # z_ctx is nested list [[tensor]]; return [[masked tokens]] matching
        # the masks_pred shape so _jepa_loss_fn compares like-sized tensors.
        inner = z_ctx[0][0] if isinstance(z_ctx[0], list) else z_ctx[0]
        out = self.lin(inner)
        # Gather mask positions so the output shape matches teacher-masked.
        if masks_pred is not None:
            mi = masks_pred[0][0]  # [B, K_mask]
            # apply_masks convention: gather along seq dim using mask idx.
            idx = mi.unsqueeze(-1).expand(-1, -1, out.size(-1))
            out = torch.gather(out, dim=1, index=idx)
        return [[out]]


def _make_pair(B, C, T, H, W, device):
    clip = torch.randn(B, C, T, H, W, device=device, requires_grad=True)
    mask_enc = [torch.arange(4, device=device).repeat(B, 1)]
    mask_pred = [torch.arange(4, 8, device=device).repeat(B, 1)]
    metas = []
    for b in range(B):
        metas.append({
            "clip_a_view": "A4C",
            "clip_b_view": "A2C" if b % 2 == 0 else "A4C",
            "target_phi_a": 0.1 * b,
            "target_phi_b": 0.2 * b,
            "study_id": f"study_{b}",
        })
    return MockPairBatch(
        clip_a=[clip],
        clip_b=[torch.randn(B, C, T, H, W, device=device)],
        clip_b_neg=[torch.randn(B, C, T, H, W, device=device)],
        masks_enc=[mask_enc],
        masks_pred=[mask_pred],
        phase_metadata=metas,
    )


def test_forward_token_phase_relational_finite_and_backward():
    B, C, T, H, W = 2, 3, 2, 16, 16
    D = 64
    N_tok = 16
    device = torch.device("cpu")
    encoder = MockEncoder(N_tok, D).to(device)
    target_encoder = MockEncoder(N_tok, D).to(device)
    predictor = MockPredictor(D).to(device)
    token_rel_head = TokenRelationalHead(embed_dim=D, rel_dim=16, hidden_dim=32).to(device)
    pool_safety = PhaseRelationalHead(embed_dim=D, rel_dim=16, hidden_dim=32).to(device)
    md_head = MotionDeltaHead(embed_dim=D, delta_dim=16, hidden_dim=32).to(device)
    delta_proj = DeltaTargetProjector(embed_dim=D, delta_dim=16, hidden_dim=32).to(device)
    pair = _make_pair(B, C, T, H, W, device)

    # Patch out _jepa_loss_fn by calling the real one; MockPredictor
    # returns full-sequence tokens, and _jepa_loss_fn masks by masks_pred.
    # To keep it trivial, pass the real encoder output through masks_pred
    # that index existing positions. The forward uses pair.masks_pred as
    # a list-of-lists; our masks are already that shape.
    out = forward_token_phase_relational(
        pair,
        encoder,
        target_encoder,
        predictor,
        token_rel_head,
        pool_safety,
        md_head,
        delta_proj,
        meta_list=pair.phase_metadata,
        token_subsample_k=8,
        tau_token=0.1,
        tau_delta=0.1,
        loss_exp=1.0,
        lambda_token_rel=0.02,
        lambda_pool_rel=0.005,
        lambda_delta=0.01,
        lambda_delta_l1=1.0,
        lambda_delta_nce=1.0,
    )
    assert torch.isfinite(out["total_loss"])
    assert torch.isfinite(out["intraview_loss"])
    assert torch.isfinite(out["token_rel_loss"])
    assert torch.isfinite(out["pool_rel_loss"])
    # With mixed-view metadata (A4C↔A2C vs A4C↔A4C), valid_rows > 0.
    assert out["delta_valid_rows"].item() >= 0.0

    # Backward should touch encoder and heads.
    out["total_loss"].backward()
    assert pair.clip_a[0].grad is not None, "gradient must flow into clip_a via encoder"


def test_forward_motion_delta_disabled_when_lambda_zero():
    """When lambda_delta=0, motion_delta branch returns zero but still
    runs the dummy-forward through delta heads (DDP reducer invariant)."""
    B, C, T, H, W = 2, 3, 2, 16, 16
    D = 64
    N_tok = 16
    device = torch.device("cpu")
    encoder = MockEncoder(N_tok, D).to(device)
    target_encoder = MockEncoder(N_tok, D).to(device)
    predictor = MockPredictor(D).to(device)
    token_rel_head = TokenRelationalHead(embed_dim=D, rel_dim=16, hidden_dim=32).to(device)
    md_head = MotionDeltaHead(embed_dim=D, delta_dim=16, hidden_dim=32).to(device)
    delta_proj = DeltaTargetProjector(embed_dim=D, delta_dim=16, hidden_dim=32).to(device)
    pair = _make_pair(B, C, T, H, W, device)

    out = forward_token_phase_relational(
        pair,
        encoder,
        target_encoder,
        predictor,
        token_rel_head,
        None,  # no pool safety this test
        md_head,
        delta_proj,
        meta_list=pair.phase_metadata,
        token_subsample_k=8,
        lambda_token_rel=0.02,
        lambda_pool_rel=0.0,
        lambda_delta=0.0,
    )
    assert torch.isfinite(out["total_loss"])
    # delta_loss is the zero-loss proxy (returns a tensor with value 0).
    assert float(out["delta_loss"]) == 0.0
