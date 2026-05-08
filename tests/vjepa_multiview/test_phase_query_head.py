"""Fix 2: PhaseQueryHead conditions on src/tgt view + Δφ.

The v1 implementation used q := z_phase directly, dropping all
conditioning. This test pins the v2 head's behaviour.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from app.vjepa_multiview.phase_query_head import PhaseQueryHead  # noqa: E402


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def _make_head(phase_dim=32, rel_dim=32):
    return PhaseQueryHead(
        phase_dim=phase_dim,
        rel_dim=rel_dim,
        hidden_dim=64,
        num_views=14,
        view_embedding_dim=8,
        n_phase_freqs=4,
    )


def test_query_shape():
    head = _make_head()
    B = 4
    z_phase = torch.randn(B, 32)
    src = torch.randint(0, 14, (B,))
    tgt = torch.randint(0, 14, (B,))
    dphi = torch.rand(B)
    q = head.query(z_phase, src, tgt, dphi)
    assert q.shape == (B, 32)
    assert torch.isfinite(q).all()


def test_target_shape():
    head = _make_head()
    B = 4
    z = torch.randn(B, 32)
    y = head.target(z)
    assert y.shape == (B, 32)


def test_query_depends_on_target_view():
    head = _make_head()
    B = 4
    z_phase = torch.zeros(B, 32)
    src = torch.full((B,), 2)
    tgt_a = torch.full((B,), 4)
    tgt_b = torch.full((B,), 0)
    dphi = torch.zeros(B)
    qa = head.query(z_phase, src, tgt_a, dphi)
    qb = head.query(z_phase, src, tgt_b, dphi)
    assert not torch.allclose(qa, qb, atol=1e-6), "query blind to tgt view"


def test_query_depends_on_source_view():
    head = _make_head()
    B = 4
    z_phase = torch.zeros(B, 32)
    tgt = torch.full((B,), 4)
    dphi = torch.zeros(B)
    src_a = torch.full((B,), 2)
    src_b = torch.full((B,), 0)
    qa = head.query(z_phase, src_a, tgt, dphi)
    qb = head.query(z_phase, src_b, tgt, dphi)
    assert not torch.allclose(qa, qb, atol=1e-6), "query blind to src view"


def test_query_depends_on_delta_phase():
    head = _make_head()
    B = 4
    z_phase = torch.zeros(B, 32)
    src = torch.full((B,), 2)
    tgt = torch.full((B,), 4)
    dphi_a = torch.zeros(B)
    dphi_b = torch.full((B,), 0.5)
    qa = head.query(z_phase, src, tgt, dphi_a)
    qb = head.query(z_phase, src, tgt, dphi_b)
    assert not torch.allclose(qa, qb, atol=1e-6), "query blind to Δφ"


def test_forward_returns_three_tensors_with_grad():
    head = _make_head()
    B = 4
    z_phase = torch.randn(B, 32, requires_grad=True)
    y_pos = torch.randn(B, 32)
    y_neg = torch.randn(B, 32)
    src = torch.randint(0, 14, (B,))
    tgt = torch.randint(0, 14, (B,))
    dphi = torch.rand(B)
    q_pre, y_pos_pre, y_hard_pre = head(z_phase, src, tgt, dphi, y_pos, y_neg)
    assert q_pre.shape == (B, 32)
    assert y_pos_pre.shape == (B, 32)
    assert y_hard_pre.shape == (B, 32)
    loss = q_pre.sum() + y_pos_pre.sum() + y_hard_pre.sum()
    loss.backward()
    # Every head parameter must see grad (single DDP-safe forward invariant).
    for name, p in head.named_parameters():
        assert p.grad is not None, f"no grad on {name}"
