"""TokenRelationalHead shape + conditioning tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from app.vjepa_multiview.token_relational_head import (  # noqa: E402
    DeltaTargetProjector,
    MotionDeltaHead,
    TokenRelationalHead,
    subsample_tokens,
)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def test_token_rel_head_shapes():
    B, N, D = 4, 128, 1024
    head = TokenRelationalHead(embed_dim=D, rel_dim=256)
    z = torch.randn(B, N, D)
    src = torch.zeros(B, dtype=torch.long)
    tgt = torch.ones(B, dtype=torch.long)
    dphi = torch.rand(B)
    q = head.query(z, src, tgt, dphi)
    y = head.target(z)
    assert q.shape == (B, N, 256)
    assert y.shape == (B, N, 256)


def test_token_rel_head_view_phase_conditioning_changes_output():
    B, N, D = 2, 8, 64
    head = TokenRelationalHead(embed_dim=D, rel_dim=32)
    z = torch.randn(B, N, D)
    src_a = torch.zeros(B, dtype=torch.long)
    src_b = torch.ones(B, dtype=torch.long) * 4  # PLAX
    tgt_a = torch.ones(B, dtype=torch.long) * 2  # A4C
    tgt_b = torch.ones(B, dtype=torch.long) * 6  # PSAX-MV
    dphi1 = torch.zeros(B)
    dphi2 = torch.ones(B) * 0.5
    qA = head.query(z, src_a, tgt_a, dphi1)
    qB = head.query(z, src_b, tgt_a, dphi1)
    qC = head.query(z, src_a, tgt_b, dphi1)
    qD = head.query(z, src_a, tgt_a, dphi2)
    assert not torch.allclose(qA, qB), "src_view should change query"
    assert not torch.allclose(qA, qC), "tgt_view should change query"
    assert not torch.allclose(qA, qD), "delta_phase should change query"


def test_token_rel_head_unified_forward_touches_all_params():
    """DDP reducer invariant — unified forward exercises every param."""
    B, N, D = 2, 4, 32
    head = TokenRelationalHead(embed_dim=D, rel_dim=16)
    z = torch.randn(B, N, D)
    src = torch.zeros(B, dtype=torch.long)
    tgt = torch.ones(B, dtype=torch.long)
    dphi = torch.rand(B)
    y_pos = torch.randn(B, N, D)
    y_hard = torch.randn(B, N, D)
    q, y1, y2 = head(z, src, tgt, dphi, y_pos, y_hard)
    loss = q.sum() + y1.sum() + y2.sum()
    loss.backward()
    for n, p in head.named_parameters():
        assert p.grad is not None, f"{n} has no grad after unified forward"


def test_motion_delta_head_shapes():
    B, N, D = 3, 16, 1024
    head = MotionDeltaHead(embed_dim=D, delta_dim=128)
    z = torch.randn(B, N, D)
    src = torch.full((B,), 2, dtype=torch.long)
    dphi = torch.rand(B)
    q = head(z, src, dphi)
    assert q.shape == (B, N, 128)


def test_delta_target_projector_shapes():
    B, N, D = 3, 16, 1024
    proj = DeltaTargetProjector(embed_dim=D, delta_dim=128)
    delta = torch.randn(B, N, D)
    out = proj(delta)
    assert out.shape == (B, N, 128)


def test_subsample_tokens_k_less_than_n():
    B, N, D = 2, 100, 32
    tokens = torch.randn(B, N, D)
    sub, idx = subsample_tokens(tokens, 32)
    assert sub.shape == (B, 32, D)
    assert idx.shape == (32,)
    # The subsampling indices must be consistent across batch rows.
    expected = tokens.index_select(dim=1, index=idx)
    assert torch.allclose(sub, expected)


def test_subsample_tokens_k_ge_n_returns_full():
    B, N, D = 2, 16, 32
    tokens = torch.randn(B, N, D)
    sub, idx = subsample_tokens(tokens, 64)
    assert sub.shape == tokens.shape
    assert idx.shape == (N,)
