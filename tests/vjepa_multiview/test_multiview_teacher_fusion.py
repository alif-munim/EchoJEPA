"""Shape + detachment for MultiViewTeacherFusion."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from app.vjepa_multiview.mv_teacher_fusion import MultiViewTeacherFusion  # noqa: E402


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def _make_fusion(embed_dim=64, fused_dim=32):
    return MultiViewTeacherFusion(
        embed_dim=embed_dim,
        fused_dim=fused_dim,
        hidden_dim=128,
        num_views=14,
        view_embedding_dim=8,
        n_phase_freqs=4,
        num_heads=4,
    )


def _inputs(B, N, D):
    pooled = torch.randn(B, N, D)
    view_ids = torch.randint(0, 14, (B, N))
    phase = torch.rand(B, N)
    return pooled, view_ids, phase


@pytest.mark.parametrize("N", [2, 4])
def test_fusion_output_shape(N):
    fusion = _make_fusion()
    pooled, view_ids, phase = _inputs(3, N, 64)
    out = fusion(pooled, view_ids, phase)
    assert out.shape == (3, 32)
    assert torch.isfinite(out).all()


def test_no_grad_flows_into_detached_teacher_input():
    """If the caller detaches teacher pooled features (as it must), no
    gradient should flow back through `pooled_nv` into the teacher
    encoder's graph. We simulate this by making `pooled_nv` a leaf that
    is *not* requires_grad."""
    fusion = _make_fusion()
    pooled = torch.randn(2, 4, 64, requires_grad=False)
    view_ids = torch.randint(0, 14, (2, 4))
    phase = torch.rand(2, 4)
    out = fusion(pooled, view_ids, phase)
    out.sum().backward()
    # The detached teacher input has no grad.
    assert pooled.grad is None, "teacher pooled leaked grad"
    # But the fusion module's own params receive grad.
    grad_params = [n for n, p in fusion.named_parameters() if p.grad is not None]
    assert len(grad_params) > 0, "no fusion params received grad"


def test_fusion_params_all_touched():
    fusion = _make_fusion()
    pooled, view_ids, phase = _inputs(2, 4, 64)
    out = fusion(pooled, view_ids, phase)
    out.sum().backward()
    for name, p in fusion.named_parameters():
        assert p.grad is not None, f"no grad on {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad on {name}"


def test_key_padding_mask_changes_output():
    fusion = _make_fusion()
    pooled, view_ids, phase = _inputs(2, 4, 64)
    out_full = fusion(pooled, view_ids, phase)
    # Mask out the last clip of each batch.
    kpm = torch.zeros(2, 4, dtype=torch.bool)
    kpm[:, -1] = True
    out_masked = fusion(pooled, view_ids, phase, key_padding_mask=kpm)
    # Outputs should differ — masked clips no longer contribute.
    assert not torch.allclose(out_full, out_masked, atol=1e-4)


def test_view_id_permutation_changes_output():
    """Swapping view IDs on the same pooled tokens must change the fused
    target — otherwise view conditioning is inert."""
    fusion = _make_fusion()
    pooled, view_ids, phase = _inputs(2, 4, 64)
    out1 = fusion(pooled, view_ids, phase)
    view_ids_perm = view_ids.flip(dims=(1,))
    out2 = fusion(pooled, view_ids_perm, phase)
    assert not torch.allclose(out1, out2, atol=1e-4)
