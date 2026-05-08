"""Integration test for forward_privileged_multiview.

Runs the full loss computation on a synthetic batch with stubbed
encoder / target_encoder / predictor so we can assert:

  1. Every loss term is finite.
  2. Total loss is finite.
  3. On a step with `p_fused=0.0`, the student encoder, factorized head,
     view predictor, shared projector, and MV teacher fusion all
     receive grad (via the 0*dummy branch).
  4. On a step with `p_fused=1.0`, the same set receives grad via the
     real fused path.
  5. The teacher's `torch.no_grad` context blocks grad into the teacher
     encoder.

The encoder stubs return token grids of the expected shape. The JEPA
loss operates on mask-selected token slices; we use simple lists to
match the expected list-of-list shape.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from app.vjepa_multiview.factorized_head import FactorizedProjectionHead  # noqa: E402
from app.vjepa_multiview.mv_teacher_fusion import MultiViewTeacherFusion  # noqa: E402
from app.vjepa_multiview.shared_projector import SharedProjector  # noqa: E402
from app.vjepa_multiview.train import (  # noqa: E402
    PairBatch,
    _same_study_ntxent,
    forward_privileged_multiview_v1_legacy as forward_privileged_multiview,
)
from app.vjepa_multiview.view_predictor import ConditionalViewPredictor  # noqa: E402


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


# --- Stubs ----------------------------------------------------------------- #


class _StubEncoder(nn.Module):
    """Masked encoder: accepts a list-of-fpc of clip tensors and an
    optional list-of-fpc of masks. Returns a nested list (outer = fpc,
    inner = mask-generator) of [B_mask, N_tok, D] tensors. Without
    masks, returns the flat list-of-fpc form."""

    def __init__(self, n_tokens: int = 8, d: int = 16):
        super().__init__()
        self.n_tokens = n_tokens
        self.d = d
        self.proj = nn.Linear(3 * 16 * 4 * 4, n_tokens * d)  # will adapt

    def forward(self, clips_list, masks_enc=None):
        # clips_list: list-over-fpc of [B, C, T, H, W].
        out = []
        for clip in clips_list:
            B = clip.size(0)
            # Flatten to a per-sample latent, then reshape to [B, N, D].
            # Use a simple learned linear that adapts to the product.
            flat = clip.reshape(B, -1)
            if flat.size(-1) != self.proj.in_features:
                # Re-create projection on first call for this shape.
                self.proj = nn.Linear(flat.size(-1), self.n_tokens * self.d).to(flat.device, dtype=flat.dtype)
            latent = self.proj(flat).reshape(B, self.n_tokens, self.d)
            out.append(latent)
        if masks_enc is None:
            return out
        # With masks: emit nested list [fpc][mask_gen] of mask-selected.
        nested = []
        for fpc_idx, latent in enumerate(out):
            inner = []
            for m in masks_enc[fpc_idx]:
                # m: [B, N_ctx] — but encoder path expects context tokens
                # already produced here. For the stub we just return the
                # full latent (shape-correct for downstream pooling).
                inner.append(latent)
            nested.append(inner)
        return nested


class _StubPredictor(nn.Module):
    """Takes the nested student output and emits [list-over-fpc of
    list-over-mask-generator of [B*|m|, N_tgt, D] tensors]. For the
    stub we return the same nested list (the intraview JEPA loss just
    needs matching shapes against teacher h)."""

    def __init__(self):
        super().__init__()
        self.w = nn.Linear(16, 16)

    def forward(self, z_ctx, masks_enc, masks_pred, delta_phi=None):
        # z_ctx: list[list[Tensor]]. Apply the trainable linear so the
        # predictor's params receive grad.
        out = []
        for fpc in z_ctx:
            out.append([self.w(t) for t in fpc])
        return out


class _StubTargetEncoder(nn.Module):
    """Teacher encoder: called under no_grad, returns list-over-fpc
    of [B, N, D]. Its params should receive NO grad during this forward."""

    def __init__(self, n_tokens: int = 8, d: int = 16):
        super().__init__()
        self.proj = nn.Linear(3 * 16 * 4 * 4, n_tokens * d)
        self.n_tokens = n_tokens
        self.d = d

    def forward(self, clips_list):
        out = []
        for clip in clips_list:
            B = clip.size(0)
            flat = clip.reshape(B, -1)
            if flat.size(-1) != self.proj.in_features:
                self.proj = nn.Linear(flat.size(-1), self.n_tokens * self.d).to(flat.device, dtype=flat.dtype)
            out.append(self.proj(flat).reshape(B, self.n_tokens, self.d))
        return out


# --- Helpers --------------------------------------------------------------- #


def _make_pair_batch(B: int = 4, C: int = 3, T: int = 16, H: int = 4, W: int = 4, n_tgt: int = 8):
    """Build a PairBatch with stubbed masks."""
    device = torch.device("cpu")
    clip_a = [torch.randn(B, C, T, H, W, device=device)]
    clip_b = [torch.randn(B, C, T, H, W, device=device)]
    clip_b_neg = [torch.randn(B, C, T, H, W, device=device)]
    # Masks: context mask and pred mask, each [B, n_tgt] of token indices.
    masks_enc = [[torch.arange(n_tgt, device=device).unsqueeze(0).expand(B, -1)]]
    masks_pred = [[torch.arange(n_tgt, device=device).unsqueeze(0).expand(B, -1)]]
    meta_list = [
        {
            "clip_a_view": "A4C",
            "clip_b_view": ["PLAX", "A2C", "A5C", "A3C"][i % 4],
            "target_phi_a": 0.0,
            "target_phi_b": 0.25,
            "target_phi_b_neg": 0.75,
            "study_id": ["S0", "S1", "S0", "S2"][i],
            "sampling_mode": "uniform_phase",
        }
        for i in range(B)
    ]
    return (
        PairBatch(
            clip_a=clip_a,
            clip_b=clip_b,
            masks_enc=masks_enc,
            masks_pred=masks_pred,
            phase_metadata=meta_list,
            clip_b_neg=clip_b_neg,
        ),
        meta_list,
    )


def _make_modules(embed_dim: int = 16, shared_dim: int = 16, phase_dim: int = 16):
    encoder = _StubEncoder(n_tokens=8, d=embed_dim)
    target_encoder = _StubTargetEncoder(n_tokens=8, d=embed_dim)
    predictor = _StubPredictor()
    factorized = FactorizedProjectionHead(
        embed_dim=embed_dim,
        hidden_dim=32,
        shared_dim=shared_dim,
        phase_dim=phase_dim,
        view_dim=shared_dim,
    )
    view_pred = ConditionalViewPredictor(
        shared_dim=shared_dim,
        phase_dim=phase_dim,
        target_dim=shared_dim,
        hidden_dim=32,
        num_views=14,
        view_embedding_dim=8,
        n_phase_freqs=4,
        use_z_view=False,  # v1-legacy never passed z_view through predictor
    )
    fusion = MultiViewTeacherFusion(
        embed_dim=embed_dim,
        fused_dim=shared_dim,
        hidden_dim=32,
        num_views=14,
        view_embedding_dim=8,
        n_phase_freqs=4,
        num_heads=2,
    )
    shared_proj = SharedProjector(
        shared_dim=shared_dim,
        fused_dim=shared_dim,
        hidden_dim=32,
    )
    # Teacher params should never receive grad.
    for p in target_encoder.parameters():
        p.requires_grad = False
    return encoder, target_encoder, predictor, factorized, view_pred, fusion, shared_proj


# --- Tests ----------------------------------------------------------------- #


def test_forward_all_losses_finite():
    pair, meta = _make_pair_batch()
    enc, ten, pred, fh, vp, fusion, sp = _make_modules()
    torch.manual_seed(42)  # p_fused gate
    out = forward_privileged_multiview(
        pair,
        enc,
        ten,
        pred,
        fh,
        vp,
        fusion,
        sp,
        meta_list=meta,
        lambda_pair=0.25,
        lambda_fused=0.10,
        lambda_shared=0.05,
        lambda_phase=0.025,
        p_fused=0.25,
    )
    for k in (
        "intraview_loss",
        "pair_loss",
        "fused_loss",
        "shared_loss",
        "phase_rel_loss",
        "total_loss",
    ):
        v = out[k]
        assert torch.isfinite(v).all(), f"{k} not finite: {v}"


def test_backward_fused_path_active_grads_reach_all_modules():
    """With p_fused=1.0, the real fused path runs. Every trainable
    module (student encoder, predictor, factorized head, view predictor,
    MV fusion, shared projector) must receive grad."""
    pair, meta = _make_pair_batch()
    enc, ten, pred, fh, vp, fusion, sp = _make_modules()
    out = forward_privileged_multiview(
        pair,
        enc,
        ten,
        pred,
        fh,
        vp,
        fusion,
        sp,
        meta_list=meta,
        lambda_pair=0.25,
        lambda_fused=0.10,
        lambda_shared=0.05,
        lambda_phase=0.025,
        p_fused=1.0,
    )
    out["total_loss"].backward()
    assert out["fused_active"].item() == 1.0

    # Every module must have at least one parameter with a non-None,
    # non-zero grad.
    for name, module in [
        ("encoder", enc),
        ("predictor", pred),
        ("factorized_head", fh),
        ("view_predictor", vp),
        ("mv_teacher_fusion", fusion),
        ("shared_projector", sp),
    ]:
        any_grad = any((p.grad is not None and p.grad.abs().sum() > 0) for p in module.parameters() if p.requires_grad)
        assert any_grad, f"{name} received no grad on fused-active step"

    # Teacher must NOT have received grad.
    for p in ten.parameters():
        assert p.grad is None or p.grad.abs().sum() == 0, "teacher encoder received grad (structural no_grad failed)"


def test_backward_fused_path_inactive_grads_still_cover_all_modules():
    """With p_fused=0.0, the real fused path is skipped. The `0*dummy`
    branch must still route grad through mv_teacher_fusion and
    shared_projector so DDP's reducer does not see unused parameters."""
    pair, meta = _make_pair_batch()
    enc, ten, pred, fh, vp, fusion, sp = _make_modules()
    out = forward_privileged_multiview(
        pair,
        enc,
        ten,
        pred,
        fh,
        vp,
        fusion,
        sp,
        meta_list=meta,
        lambda_pair=0.25,
        lambda_fused=0.10,
        lambda_shared=0.05,
        lambda_phase=0.025,
        p_fused=0.0,
    )
    out["total_loss"].backward()
    assert out["fused_active"].item() == 0.0

    # On the fused-inactive path, mv_teacher_fusion and shared_projector
    # must still receive a *non-None* grad (may be exactly zero because
    # of the 0.0 multiplier, but must not be None — that's what
    # DistributedDataParallel requires).
    for name, module in [("mv_teacher_fusion", fusion), ("shared_projector", sp)]:
        for pname, p in module.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"{name}.{pname} has None grad on fused-inactive step"


def test_z_view_head_has_no_grad_by_default():
    """Documents the plan invariant: z_view is produced but not
    consumed by any loss. The z_view head parameters therefore receive
    None grad — callers need find_unused_parameters=True on the
    factorized_head DDP wrap, OR freeze z_view, OR add a z_view loss.
    If this test ever starts failing, that invariant has shifted and
    the config default in the YAML should be re-evaluated."""
    pair, meta = _make_pair_batch()
    enc, ten, pred, fh, vp, fusion, sp = _make_modules()
    out = forward_privileged_multiview(
        pair,
        enc,
        ten,
        pred,
        fh,
        vp,
        fusion,
        sp,
        meta_list=meta,
        lambda_pair=0.25,
        lambda_fused=0.10,
        lambda_shared=0.05,
        lambda_phase=0.025,
        p_fused=1.0,
    )
    out["total_loss"].backward()
    for pname, p in fh.view_mlp.named_parameters():
        assert p.grad is None, f"z_view head {pname} received grad — plan invariant broke"


def test_same_study_ntxent_scalar_and_diag():
    z = torch.randn(6, 8)
    z = torch.nn.functional.normalize(z, dim=-1)
    # 3 studies: pair (0,1), (2,3), (4,5).
    sh = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    loss, diag = _same_study_ntxent(z, sh, tau=0.1)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert diag["static_ntxent_pos_rows"].item() == 6.0
    assert diag["static_ntxent_batch_rows"].item() == 6.0


def test_same_study_ntxent_no_positives_returns_zero():
    z = torch.randn(4, 8)
    z = torch.nn.functional.normalize(z, dim=-1)
    # All-distinct studies.
    sh = torch.arange(4, dtype=torch.long)
    loss, diag = _same_study_ntxent(z, sh, tau=0.1)
    assert loss.item() == 0.0
    assert diag["static_ntxent_pos_rows"].item() == 0.0
