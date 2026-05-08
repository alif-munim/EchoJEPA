"""Shape + view-ID dependence for ConditionalViewPredictor.

Covers both ``use_z_view=True`` (default, v2) and ``use_z_view=False``
(ablation / v1-legacy compatibility) paths.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from app.vjepa_multiview.view_predictor import ConditionalViewPredictor  # noqa: E402


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def _make_pred(shared_dim=32, phase_dim=32, view_dim=32, target_dim=32, use_z_view=True):
    return ConditionalViewPredictor(
        shared_dim=shared_dim,
        phase_dim=phase_dim,
        view_dim=view_dim,
        target_dim=target_dim,
        hidden_dim=64,
        num_views=14,
        view_embedding_dim=8,
        n_phase_freqs=4,
        use_z_view=use_z_view,
    )


def test_forward_shape_with_z_view():
    pred = _make_pred(use_z_view=True)
    B = 4
    z_shared = torch.randn(B, 32)
    z_phase = torch.randn(B, 32)
    z_view = torch.randn(B, 32)
    src = torch.randint(0, 14, (B,))
    tgt = torch.randint(0, 14, (B,))
    dphi = torch.rand(B)
    q = pred(z_shared, z_phase, src, tgt, dphi, z_view=z_view)
    assert q.shape == (B, 32)
    assert torch.isfinite(q).all()


def test_forward_shape_without_z_view():
    pred = _make_pred(use_z_view=False)
    B = 4
    z_shared = torch.randn(B, 32)
    z_phase = torch.randn(B, 32)
    src = torch.randint(0, 14, (B,))
    tgt = torch.randint(0, 14, (B,))
    dphi = torch.rand(B)
    q = pred(z_shared, z_phase, src, tgt, dphi)  # no z_view
    assert q.shape == (B, 32)


def test_use_z_view_true_requires_z_view_arg():
    pred = _make_pred(use_z_view=True)
    B = 2
    with pytest.raises(ValueError, match="z_view"):
        pred(
            torch.randn(B, 32),
            torch.randn(B, 32),
            torch.zeros(B, dtype=torch.long),
            torch.zeros(B, dtype=torch.long),
            torch.zeros(B),
        )


def test_output_depends_on_target_view_id():
    pred = _make_pred(use_z_view=True)
    B = 4
    z_shared = torch.zeros(B, 32)
    z_phase = torch.zeros(B, 32)
    z_view = torch.zeros(B, 32)
    src = torch.full((B,), 2)
    tgt1 = torch.full((B,), 4)
    tgt2 = torch.full((B,), 0)
    dphi = torch.zeros(B)
    q1 = pred(z_shared, z_phase, src, tgt1, dphi, z_view=z_view)
    q2 = pred(z_shared, z_phase, src, tgt2, dphi, z_view=z_view)
    assert not torch.allclose(q1, q2, atol=1e-6), "predictor is view-blind"


def test_output_depends_on_delta_phase():
    pred = _make_pred(use_z_view=True)
    B = 4
    z_shared = torch.zeros(B, 32)
    z_phase = torch.zeros(B, 32)
    z_view = torch.zeros(B, 32)
    src = torch.full((B,), 2)
    tgt = torch.full((B,), 4)
    dphi1 = torch.zeros(B)
    dphi2 = torch.full((B,), 0.5)
    q1 = pred(z_shared, z_phase, src, tgt, dphi1, z_view=z_view)
    q2 = pred(z_shared, z_phase, src, tgt, dphi2, z_view=z_view)
    assert not torch.allclose(q1, q2, atol=1e-6), "predictor ignored delta_phase"


def test_output_depends_on_z_view():
    """With slots + views zero-held constant, z_view alone must change
    the output — this is the invariant that lets L_pair_view actually
    teach z_view something."""
    pred = _make_pred(use_z_view=True)
    B = 4
    z_shared = torch.zeros(B, 32)
    z_phase = torch.zeros(B, 32)
    src = torch.full((B,), 2)
    tgt = torch.full((B,), 4)
    dphi = torch.zeros(B)
    z_view_a = torch.zeros(B, 32)
    z_view_b = torch.randn(B, 32)
    q1 = pred(z_shared, z_phase, src, tgt, dphi, z_view=z_view_a)
    q2 = pred(z_shared, z_phase, src, tgt, dphi, z_view=z_view_b)
    assert not torch.allclose(q1, q2, atol=1e-6), "predictor ignored z_view"


def test_all_params_receive_grad():
    pred = _make_pred(use_z_view=True)
    B = 4
    z_shared = torch.randn(B, 32, requires_grad=True)
    z_phase = torch.randn(B, 32, requires_grad=True)
    z_view = torch.randn(B, 32, requires_grad=True)
    src = torch.randint(0, 14, (B,))
    tgt = torch.randint(0, 14, (B,))
    dphi = torch.rand(B)
    q = pred(z_shared, z_phase, src, tgt, dphi, z_view=z_view)
    q.sum().backward()
    assert z_view.grad is not None, "z_view input received no grad"
    for name, p in pred.named_parameters():
        assert p.grad is not None, f"no grad on {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad on {name}"
