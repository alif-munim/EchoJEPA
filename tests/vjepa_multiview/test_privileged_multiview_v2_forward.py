"""v3 integration test for forward_privileged_multiview.

Covers:
  - Fix 2 (conditional phase query),
  - Fix 3 (paired shared NT-Xent),
  - Fix 4 (dual-target pair, z_view consumed),
  - Fix 5 (EMA factorized_head target),
  - Fix 6 (caller-supplied fused_active),
  - v3 fused_target_mode='mean_shared' (no mv_teacher_fusion needed).

Uses stubbed encoder / predictor / target_encoder so we can run
everything on CPU with a tiny batch.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from app.vjepa_multiview.factorized_head import FactorizedProjectionHead  # noqa: E402
from app.vjepa_multiview.phase_query_head import PhaseQueryHead  # noqa: E402
from app.vjepa_multiview.shared_projector import SharedProjector  # noqa: E402
from app.vjepa_multiview.train import (  # noqa: E402
    PairBatch,
    forward_privileged_multiview,
)
from app.vjepa_multiview.view_predictor import ConditionalViewPredictor  # noqa: E402


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


# --- Stubs ----------------------------------------------------------------- #


class _StubEncoder(nn.Module):
    def __init__(self, n_tokens: int = 8, d: int = 16):
        super().__init__()
        self.n_tokens = n_tokens
        self.d = d
        self.proj = nn.Linear(3 * 16 * 4 * 4, n_tokens * d)

    def forward(self, clips_list, masks_enc=None):
        out = []
        for clip in clips_list:
            B = clip.size(0)
            flat = clip.reshape(B, -1)
            if flat.size(-1) != self.proj.in_features:
                self.proj = nn.Linear(flat.size(-1), self.n_tokens * self.d).to(flat.device, dtype=flat.dtype)
            latent = self.proj(flat).reshape(B, self.n_tokens, self.d)
            out.append(latent)
        if masks_enc is None:
            return out
        nested = []
        for fpc_idx, latent in enumerate(out):
            nested.append([latent for _ in masks_enc[fpc_idx]])
        return nested


class _StubPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Linear(16, 16)

    def forward(self, z_ctx, masks_enc, masks_pred, delta_phi=None):
        return [[self.w(t) for t in fpc] for fpc in z_ctx]


class _StubTargetEncoder(nn.Module):
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


def _make_pair_batch(B=4, C=3, T=16, H=4, W=4, n_tgt=8):
    clip_a = [torch.randn(B, C, T, H, W)]
    clip_b = [torch.randn(B, C, T, H, W)]
    clip_b_neg = [torch.randn(B, C, T, H, W)]
    masks_enc = [[torch.arange(n_tgt).unsqueeze(0).expand(B, -1)]]
    masks_pred = [[torch.arange(n_tgt).unsqueeze(0).expand(B, -1)]]
    meta_list = [
        {
            "clip_a_view": "A4C",
            "clip_b_view": ["PLAX", "A2C", "A5C", "A3C"][i % 4],
            "target_phi_a": 0.0,
            "target_phi_b": 0.25,
            "target_phi_b_neg": 0.75,
            "study_id": f"S{i}",
            "sampling_mode": "uniform_phase",
        }
        for i in range(B)
    ]
    pair = PairBatch(
        clip_a=clip_a,
        clip_b=clip_b,
        masks_enc=masks_enc,
        masks_pred=masks_pred,
        phase_metadata=meta_list,
        clip_b_neg=clip_b_neg,
    )
    return pair, meta_list


def _make_modules(embed_dim=16, shared_dim=16, phase_dim=16, view_dim=16):
    enc = _StubEncoder(n_tokens=8, d=embed_dim)
    ten = _StubTargetEncoder(n_tokens=8, d=embed_dim)
    pred = _StubPredictor()
    fh = FactorizedProjectionHead(
        embed_dim=embed_dim,
        hidden_dim=32,
        shared_dim=shared_dim,
        phase_dim=phase_dim,
        view_dim=view_dim,
    )
    vp = ConditionalViewPredictor(
        shared_dim=shared_dim,
        phase_dim=phase_dim,
        view_dim=view_dim,
        target_dim=view_dim,
        hidden_dim=32,
        num_views=14,
        view_embedding_dim=8,
        n_phase_freqs=4,
        use_z_view=True,
    )
    pair_sp = SharedProjector(
        shared_dim=shared_dim,
        fused_dim=shared_dim,
        hidden_dim=32,
    )
    fused_sp = SharedProjector(
        shared_dim=shared_dim,
        fused_dim=shared_dim,
        hidden_dim=32,
    )
    pq = PhaseQueryHead(
        phase_dim=phase_dim,
        rel_dim=phase_dim,
        hidden_dim=32,
        num_views=14,
        view_embedding_dim=8,
        n_phase_freqs=4,
    )
    fh_ema = copy.deepcopy(fh)
    for p in ten.parameters():
        p.requires_grad = False
    for p in fh_ema.parameters():
        p.requires_grad = False
    return enc, ten, pred, fh, vp, pair_sp, fused_sp, pq, fh_ema


def _run(
    fused_active: bool = False,
    **lambdas,
):
    pair, meta = _make_pair_batch()
    enc, ten, pred, fh, vp, pair_sp, fused_sp, pq, fh_ema = _make_modules()
    out = forward_privileged_multiview(
        pair,
        enc,
        ten,
        pred,
        fh,
        vp,
        pair_sp,
        fused_sp,
        pq,
        fh_ema,
        meta_list=meta,
        fused_active=fused_active,
        fused_target_mode="mean_shared",
        lambda_pair_shared=lambdas.get("lambda_pair_shared", 0.05),
        lambda_pair_view=lambdas.get("lambda_pair_view", 0.10),
        lambda_view_nce=lambdas.get("lambda_view_nce", 0.025),
        lambda_shared=lambdas.get("lambda_shared", 0.05),
        lambda_phase=lambdas.get("lambda_phase", 0.0),
        lambda_fused=lambdas.get("lambda_fused", 0.05),
        lambda_local_motion=lambdas.get("lambda_local_motion", 0.0),
        # These v3/v4 tests predate the real target_clip loader; they
        # use the legacy clip_b reuse path and must explicitly opt into
        # the provisional fallback. The v5 target-clip tests (new file)
        # exercise the real path.
        allow_provisional_clip_b_fallback=True,
    )
    return out, (enc, ten, pred, fh, vp, pair_sp, fused_sp, pq, fh_ema)


# --- Tests ----------------------------------------------------------------- #


def test_v3_forward_all_losses_finite_fused_off():
    out, _ = _run(fused_active=False)
    for k in (
        "intraview_loss",
        "pair_shared_loss",
        "pair_view_loss",
        "shared_loss",
        "phase_rel_loss",
        "fused_loss",
        "total_loss",
    ):
        assert torch.isfinite(out[k]).all(), f"{k} not finite"
    assert out["fused_active"].item() == 0.0
    assert out["fused_loss"].item() == 0.0


def test_v3_forward_all_losses_finite_fused_on_mean_shared():
    out, _ = _run(fused_active=True)
    assert out["fused_active"].item() == 1.0
    assert torch.isfinite(out["fused_loss"])
    assert out["fused_loss"].item() > 0.0
    # Fresh fh_ema matches fh at init → target_norm should be small but > 0.
    assert out["fused_shared_target_norm"].item() > 0.0
    assert out["fused_valid_views_mean"].item() == pytest.approx(2.0)
    assert torch.isfinite(out["total_loss"])


def test_v3_z_view_receives_grad_via_pair_view_loss():
    out, mods = _run(
        fused_active=False,
        lambda_pair_shared=0.0,
        lambda_pair_view=1.0,
        lambda_shared=0.0,
        lambda_phase=0.0,
        lambda_fused=0.0,
    )
    out["total_loss"].backward()
    _, _, _, fh, _, _, _, _, _ = mods
    for pname, p in fh.view_mlp.named_parameters():
        assert p.grad is not None, f"z_view head {pname} got no grad"
        assert p.grad.abs().sum() > 0, f"z_view head {pname} got zero grad"


def test_v3_ema_factorized_head_never_receives_grad():
    for fa in (False, True):
        out, mods = _run(fused_active=fa)
        out["total_loss"].backward()
        _, _, _, _, _, _, _, _, fh_ema = mods
        for pname, p in fh_ema.named_parameters():
            assert p.grad is None, f"fh_ema {pname} received grad (fa={fa})"


def test_v3_teacher_encoder_no_grad():
    out, mods = _run(fused_active=True)
    out["total_loss"].backward()
    _, ten, _, _, _, _, _, _, _ = mods
    for p in ten.parameters():
        assert p.grad is None or p.grad.abs().sum() == 0


def test_v3_phase_query_head_receives_grad():
    out, mods = _run(
        fused_active=False,
        lambda_pair_shared=0.0,
        lambda_pair_view=0.0,
        lambda_shared=0.0,
        lambda_phase=1.0,
        lambda_fused=0.0,
    )
    out["total_loss"].backward()
    _, _, _, _, _, _, _, pq, _ = mods
    for pname, p in pq.named_parameters():
        assert p.grad is not None, f"pq {pname} got no grad"
        assert p.grad.abs().sum() > 0, f"pq {pname} got zero grad"


def test_v3_paired_shared_loss_grad_path():
    out, mods = _run(
        fused_active=False,
        lambda_pair_shared=0.0,
        lambda_pair_view=0.0,
        lambda_shared=1.0,
        lambda_phase=0.0,
        lambda_fused=0.0,
    )
    out["total_loss"].backward()
    _, _, _, fh, _, _, _, _, _ = mods
    shared_mlp_grads = [p.grad.abs().sum().item() for p in fh.shared_mlp.parameters()]
    assert sum(shared_mlp_grads) > 0, "shared_mlp got no grad from L_shared"


def test_v3_pair_shared_projector_receives_grad():
    """pair_shared_projector must receive grad when lambda_pair_shared > 0."""
    out, mods = _run(
        fused_active=False,
        lambda_pair_shared=1.0,
        lambda_pair_view=0.0,
        lambda_shared=0.0,
        lambda_phase=0.0,
        lambda_fused=0.0,
    )
    out["total_loss"].backward()
    _, _, _, _, _, pair_sp, _, _, _ = mods
    grad_sum = sum(p.grad.abs().sum().item() for p in pair_sp.parameters())
    assert grad_sum > 0, "pair_shared_projector got no grad"


def test_v3_fused_shared_projector_receives_grad_when_active():
    """fused_shared_projector receives grad only when fused_active=True and
    lambda_fused > 0."""
    out, mods = _run(
        fused_active=True,
        lambda_pair_shared=0.0,
        lambda_pair_view=0.0,
        lambda_shared=0.0,
        lambda_phase=0.0,
        lambda_fused=1.0,
    )
    out["total_loss"].backward()
    _, _, _, _, _, _, fused_sp, _, _ = mods
    grad_sum = sum(p.grad.abs().sum().item() for p in fused_sp.parameters())
    assert grad_sum > 0, "fused_shared_projector got no grad on active step"


def test_v3_fused_shared_projector_no_grad_when_inactive():
    out, mods = _run(
        fused_active=False,
        lambda_pair_shared=0.0,
        lambda_pair_view=0.0,
        lambda_shared=0.0,
        lambda_phase=0.0,
        lambda_fused=1.0,
    )
    out["total_loss"].backward()
    _, _, _, _, _, _, fused_sp, _, _ = mods
    grad_sum = sum(p.grad.abs().sum().item() if p.grad is not None else 0.0 for p in fused_sp.parameters())
    assert grad_sum == 0.0, "fused_shared_projector got grad on inactive step"


def test_v3_intra_only_parity():
    out, _ = _run(
        fused_active=False,
        lambda_pair_shared=0.0,
        lambda_pair_view=0.0,
        lambda_view_nce=0.0,
        lambda_shared=0.0,
        lambda_phase=0.0,
        lambda_fused=0.0,
        lambda_local_motion=0.0,
    )
    assert torch.allclose(out["total_loss"], out["intraview_loss"])


def test_v3_mean_shared_no_mv_teacher_fusion_required():
    """mean_shared mode must not require mv_teacher_fusion or its EMA."""
    pair, meta = _make_pair_batch()
    enc, ten, pred, fh, vp, pair_sp, fused_sp, pq, fh_ema = _make_modules()
    out = forward_privileged_multiview(
        pair,
        enc,
        ten,
        pred,
        fh,
        vp,
        pair_sp,
        fused_sp,
        pq,
        fh_ema,
        meta_list=meta,
        fused_active=True,
        fused_target_mode="mean_shared",
        mv_teacher_fusion=None,
        mv_teacher_fusion_ema=None,
        lambda_pair_shared=0.25,
        lambda_pair_view=0.05,
        lambda_shared=0.05,
        lambda_phase=0.0,
        lambda_fused=0.10,
        allow_provisional_clip_b_fallback=True,
    )
    assert torch.isfinite(out["total_loss"])
    assert torch.isfinite(out["fused_loss"])


def test_v3_attention_ema_requires_fusion_ema():
    pair, meta = _make_pair_batch()
    enc, ten, pred, fh, vp, pair_sp, fused_sp, pq, fh_ema = _make_modules()
    with pytest.raises(ValueError, match="mv_teacher_fusion_ema"):
        forward_privileged_multiview(
            pair,
            enc,
            ten,
            pred,
            fh,
            vp,
            pair_sp,
            fused_sp,
            pq,
            fh_ema,
            meta_list=meta,
            fused_active=True,
            fused_target_mode="attention_ema",
            mv_teacher_fusion=None,
            mv_teacher_fusion_ema=None,
            lambda_fused=1.0,
            allow_provisional_clip_b_fallback=True,
        )


def test_v3_unknown_fused_target_mode_rejected():
    pair, meta = _make_pair_batch()
    enc, ten, pred, fh, vp, pair_sp, fused_sp, pq, fh_ema = _make_modules()
    with pytest.raises(ValueError, match="fused_target_mode"):
        forward_privileged_multiview(
            pair,
            enc,
            ten,
            pred,
            fh,
            vp,
            pair_sp,
            fused_sp,
            pq,
            fh_ema,
            meta_list=meta,
            fused_active=True,
            fused_target_mode="bogus",
            lambda_fused=1.0,
            allow_provisional_clip_b_fallback=True,
        )


def test_v3_new_diagnostics_present():
    out, _ = _run(fused_active=True)
    for k in (
        "fused_valid_views_mean",
        "fused_valid_views_min",
        "fused_shared_target_norm",
        "fused_shared_q_norm",
        "fused_shared_cos_q_target",
        "diag_pair_shared_cos_q_target",
        "diag_pair_view_cos_q_target",
        "diag_z_shared_var",
        "diag_z_phase_var",
        "diag_z_view_var",
    ):
        assert k in out, f"missing diagnostic: {k}"
        assert torch.isfinite(out[k]).all(), f"{k} not finite"


# --- v4 tests ------------------------------------------------------------- #


def test_v4_view_nce_loss_emitted_and_finite():
    out, _ = _run(fused_active=False)
    assert "view_nce_loss" in out
    assert torch.isfinite(out["view_nce_loss"])
    assert out["view_nce_loss"].item() > 0.0


def test_v4_view_nce_top1_diagnostics_present():
    out, _ = _run(fused_active=False)
    for k in ("view_nce_top1", "view_nce_pos_sim_mean", "view_nce_neg_sim_mean"):
        assert k in out, f"missing v4 view_nce diagnostic: {k}"
        assert torch.isfinite(out[k]).all()


def test_v4_per_target_view_diagnostics_present():
    """diag_pair_view_cos_by_view and diag_view_nce_top1_by_view should
    be dicts keyed by view-name strings (subset of PLAX/A5C/A3C/A2C).
    With 4 rows cycling through those four views, at least three
    should appear."""
    out, _ = _run(fused_active=False)
    pv_by = out.get("diag_pair_view_cos_by_view", {})
    nce_by = out.get("diag_view_nce_top1_by_view", {})
    assert isinstance(pv_by, dict)
    assert isinstance(nce_by, dict)
    # The test batch uses clip_b_view in {PLAX, A2C, A5C, A3C}.
    expected = {"PLAX", "A2C", "A5C", "A3C"}
    assert set(pv_by.keys()) <= expected
    assert len(pv_by) >= 3, f"expected ≥3 target-view buckets, got {pv_by.keys()}"


def test_v4_pair_view_receives_grad_when_view_nce_alone():
    """view_nce ALONE (no SmoothL1 pair_view) must still pull grad into
    the view_predictor via q_pair_view."""
    out, mods = _run(
        fused_active=False,
        lambda_pair_shared=0.0,
        lambda_pair_view=0.0,
        lambda_view_nce=1.0,
        lambda_shared=0.0,
        lambda_phase=0.0,
        lambda_fused=0.0,
    )
    out["total_loss"].backward()
    _, _, _, _, vp, _, _, _, _ = mods
    grad_sum = sum(p.grad.abs().sum().item() for p in vp.parameters() if p.grad is not None)
    assert grad_sum > 0, "view_predictor got no grad from view_nce"


def test_v4_factorized_head_z_view_receives_grad_from_view_nce():
    """view_nce passes gradient back through q_pair_view, which consumes
    z_view — the z_view slot of factorized_head should receive grad."""
    out, mods = _run(
        fused_active=False,
        lambda_pair_shared=0.0,
        lambda_pair_view=0.0,
        lambda_view_nce=1.0,
        lambda_shared=0.0,
        lambda_phase=0.0,
        lambda_fused=0.0,
    )
    out["total_loss"].backward()
    _, _, _, fh, _, _, _, _, _ = mods
    grad_sum = sum(p.grad.abs().sum().item() for p in fh.view_mlp.parameters() if p.grad is not None)
    assert grad_sum > 0, "z_view head got no grad from view_nce"


def test_v4_local_motion_raises_when_enabled():
    """Scaffolding guard: enabling lambda_local_motion > 0 must raise
    until the sampler extension ships."""
    pair, meta = _make_pair_batch()
    enc, ten, pred, fh, vp, pair_sp, fused_sp, pq, fh_ema = _make_modules()
    with pytest.raises(NotImplementedError, match="local-motion"):
        forward_privileged_multiview(
            pair,
            enc,
            ten,
            pred,
            fh,
            vp,
            pair_sp,
            fused_sp,
            pq,
            fh_ema,
            meta_list=meta,
            fused_active=False,
            fused_target_mode="mean_shared",
            lambda_pair_shared=0.0,
            lambda_pair_view=0.0,
            lambda_view_nce=0.0,
            lambda_shared=0.0,
            lambda_phase=0.0,
            lambda_fused=0.0,
            lambda_local_motion=0.1,
        )


def test_v4_local_motion_loss_zero_by_default():
    out, _ = _run(fused_active=False)
    assert "local_motion_loss" in out
    assert out["local_motion_loss"].item() == 0.0
