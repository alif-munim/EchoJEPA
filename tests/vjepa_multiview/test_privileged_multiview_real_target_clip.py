"""v5 scientific-path smoke test: forward consumes real target_clip
+ fused_clips, not the v3/v4 provisional clip_b reuse.

Distinct from ``test_privileged_multiview_v2_forward.py`` which uses
``allow_provisional_clip_b_fallback=True`` throughout. Here we build a
synthetic batch where the target_clip differs from clip_b (different
tensor, different target_view from clip_b_view), and verify:

  1. Forward consumes pair.target_clip (not clip_b) for the pair loss.
  2. target_view_ids come from pair.target_views, not meta clip_b_view.
  3. view_nce per-target-view diagnostics are non-empty.
  4. pair_view + view_nce gradients reach z_view / view_mlp.
  5. Fused active path consumes pair.fused_clips and respects
     pair.fused_valid_mask; fused_valid_views_mean reflects the mask.
  6. Fail-loud guard trips when allow_provisional_clip_b_fallback=False
     and pair.target_clip is None.
  7. Fail-loud fused guard trips when fused_valid_mask.sum(dim=1).mean()
     is < 2 (insufficient fused pool).
  8. used_clip_b_fallback diagnostic is 0 when target_clip is real.
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
from app.vjepa_multiview.phase_relational_head import VIEW_ID_MAP  # noqa: E402
from app.vjepa_multiview.shared_projector import SharedProjector  # noqa: E402
from app.vjepa_multiview.train import (  # noqa: E402
    PairBatch,
    forward_privileged_multiview,
)
from app.vjepa_multiview.view_predictor import ConditionalViewPredictor  # noqa: E402


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


# --- Minimal stubs (mirror test_privileged_multiview_v2_forward) ---------- #


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
                self.proj = nn.Linear(flat.size(-1), self.n_tokens * self.d).to(
                    flat.device,
                    dtype=flat.dtype,
                )
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
                self.proj = nn.Linear(flat.size(-1), self.n_tokens * self.d).to(
                    flat.device,
                    dtype=flat.dtype,
                )
            out.append(self.proj(flat).reshape(B, self.n_tokens, self.d))
        return out


def _make_modules(embed_dim=16):
    enc = _StubEncoder(n_tokens=8, d=embed_dim)
    ten = _StubTargetEncoder(n_tokens=8, d=embed_dim)
    pred = _StubPredictor()
    fh = FactorizedProjectionHead(
        embed_dim=embed_dim,
        hidden_dim=32,
        shared_dim=16,
        phase_dim=16,
        view_dim=16,
    )
    vp = ConditionalViewPredictor(
        shared_dim=16,
        phase_dim=16,
        view_dim=16,
        target_dim=16,
        hidden_dim=32,
        num_views=14,
        view_embedding_dim=8,
        n_phase_freqs=4,
        use_z_view=True,
    )
    pair_sp = SharedProjector(shared_dim=16, fused_dim=16, hidden_dim=32)
    fused_sp = SharedProjector(shared_dim=16, fused_dim=16, hidden_dim=32)
    pq = PhaseQueryHead(
        phase_dim=16,
        rel_dim=16,
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


def _make_real_target_batch(B: int = 4, n_fused: int = 3):
    """Build a PairBatch with DISTINCT target_clip tensor and fused pool.

    target_views cycle through {A2C, A5C, PLAX, A3C} so the per-view
    diagnostics see multiple buckets. fused_valid_mask marks all fused
    slots valid in row 0 and the last one invalid in row 1.
    """
    C, T, H, W = 3, 16, 4, 4
    n_tgt = 8

    def _r(B):
        return torch.randn(B, C, T, H, W)

    clip_a = [_r(B)]
    clip_b = [_r(B)]
    clip_b_neg = [_r(B)]
    target_clip = [_r(B)]  # deliberately different from clip_b
    fused_video = torch.randn(B, n_fused, C, T, H, W)
    fused_valid_mask = torch.ones(B, n_fused, dtype=torch.bool)
    if B >= 2:
        fused_valid_mask[1, -1] = False  # one slot invalid for row 1

    masks_enc = [[torch.arange(n_tgt).unsqueeze(0).expand(B, -1)]]
    masks_pred = [[torch.arange(n_tgt).unsqueeze(0).expand(B, -1)]]
    target_views_list = [["A2C", "A5C", "PLAX", "A3C"][i % 4] for i in range(B)]
    target_delta_phase = torch.tensor([0.1, 0.2, 0.3, 0.4][:B], dtype=torch.float32)
    target_clip_present = torch.ones(B, dtype=torch.bool)
    fused_views = [[target_views_list[i]] + ["A5C"] * (n_fused - 1) for i in range(B)]
    fused_phases = torch.tensor(
        [[0.1 * (j + 1) for j in range(n_fused)] for _ in range(B)],
        dtype=torch.float32,
    )
    meta_list = [
        {
            "clip_a_view": "A4C",
            "clip_b_view": "A4C",  # intentionally NOT the target_view
            "target_phi_a": 0.0,
            "target_phi_b": 0.25,
            "target_phi_b_neg": 0.75,
            "target_clip_view": target_views_list[i],
            "target_delta_phase": float(target_delta_phase[i]),
            "target_clip_present": 1,
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
        target_clip=target_clip,
        target_views=target_views_list,
        target_delta_phase=target_delta_phase,
        target_clip_present=target_clip_present,
        fused_clips=[fused_video],
        fused_views=fused_views,
        fused_phases=fused_phases,
        fused_valid_mask=fused_valid_mask,
    )
    return pair, meta_list


def _run_real(
    pair,
    meta,
    fused_active: bool = False,
    **lambdas,
):
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
        lambda_fused=lambdas.get("lambda_fused", 0.0),
        lambda_local_motion=0.0,
        # NOT allowing provisional fallback — this test verifies the
        # scientific path works end to end with real target_clip.
        allow_provisional_clip_b_fallback=False,
    )
    return out, (enc, ten, pred, fh, vp, pair_sp, fused_sp, pq, fh_ema)


# --- Tests ---------------------------------------------------------------- #


def test_real_target_clip_forward_runs_without_fallback():
    pair, meta = _make_real_target_batch()
    out, _ = _run_real(pair, meta, fused_active=False)
    assert torch.isfinite(out["total_loss"])
    # Scientific invariant: no clip_b fallback used.
    assert out["used_clip_b_fallback"].item() == 0.0
    # pct_target_clip_present must be 1 when sampler delivers real targets.
    assert out["pct_target_clip_present"].item() == pytest.approx(1.0)


def test_forward_consumes_target_clip_not_clip_b():
    """Confirm the teacher encoder is fed pair.target_clip (not clip_b).

    Strategy: construct two batches that share clip_a + clip_b + clip_b_neg
    but differ in target_clip. pair_shared_loss depends on
    t_tgt_shared = factorized_head(pool(E_target(target_clip))), so it
    should differ between the two batches. If the forward were still
    consuming clip_b, the loss would be identical.
    """
    pair_A, meta = _make_real_target_batch()
    # Copy batch but replace target_clip tensor.
    pair_B = PairBatch(
        clip_a=pair_A.clip_a,
        clip_b=pair_A.clip_b,
        masks_enc=pair_A.masks_enc,
        masks_pred=pair_A.masks_pred,
        phase_metadata=pair_A.phase_metadata,
        clip_b_neg=pair_A.clip_b_neg,
        target_clip=[torch.randn_like(pair_A.target_clip[0])],
        target_views=pair_A.target_views,
        target_delta_phase=pair_A.target_delta_phase,
        target_clip_present=pair_A.target_clip_present,
        fused_clips=pair_A.fused_clips,
        fused_views=pair_A.fused_views,
        fused_phases=pair_A.fused_phases,
        fused_valid_mask=pair_A.fused_valid_mask,
    )
    # Same stubs for fair comparison.
    torch.manual_seed(123)
    out_a, _ = _run_real(pair_A, meta, fused_active=False)
    torch.manual_seed(123)
    out_b, _ = _run_real(pair_B, meta, fused_active=False)
    assert not torch.allclose(
        out_a["pair_shared_loss"], out_b["pair_shared_loss"]
    ), "pair_shared_loss identical → forward is not consuming target_clip"


def test_tgt_view_ids_come_from_target_views_not_meta():
    """Forward must use pair.target_views for tgt_view_ids, not
    meta.clip_b_view. Verify by checking the per-target-view
    diagnostic sees target_views (PLAX/A5C/A3C/A2C) NOT "A4C"
    (clip_b_view in the meta list)."""
    pair, meta = _make_real_target_batch(B=4)
    out, _ = _run_real(pair, meta, fused_active=False)
    # Batch has rows with target_views {A2C, A5C, PLAX, A3C}.
    tw_cos = out.get("diag_pair_view_cos_by_view", {}) or {}
    # Expected keys are target_views, NOT "A4C".
    assert set(tw_cos.keys()) <= {"A2C", "A5C", "PLAX", "A3C"}
    assert "A4C" not in tw_cos


def test_view_nce_per_target_view_buckets_populated():
    pair, meta = _make_real_target_batch(B=4)
    out, _ = _run_real(pair, meta, fused_active=False)
    nce_by_view = out.get("diag_view_nce_top1_by_view", {}) or {}
    assert isinstance(nce_by_view, dict)
    assert len(nce_by_view) >= 3, f"expected diagnostics for ≥3 target views, got {list(nce_by_view.keys())}"


def test_pair_view_and_view_nce_grads_reach_z_view():
    pair, meta = _make_real_target_batch()
    out, mods = _run_real(
        pair,
        meta,
        fused_active=False,
        lambda_pair_shared=0.0,
        lambda_pair_view=1.0,
        lambda_view_nce=1.0,
        lambda_shared=0.0,
        lambda_phase=0.0,
        lambda_fused=0.0,
    )
    out["total_loss"].backward()
    _, _, _, fh, _, _, _, _, _ = mods
    grad_sum = sum(p.grad.abs().sum().item() for p in fh.view_mlp.parameters() if p.grad is not None)
    assert grad_sum > 0, "z_view head received no grad from pair_view + view_nce"


def test_fused_active_consumes_fused_clips_tensor():
    """When fused_active=True and pair.fused_clips is provided,
    fused_valid_views_mean should reflect the actual mask (one invalid
    slot in row 1 out of n_fused=3 → mean = (3 + 2 + 3 + 3) / 4 = 2.75)."""
    pair, meta = _make_real_target_batch(B=4, n_fused=3)
    out, _ = _run_real(pair, meta, fused_active=True, lambda_fused=0.05)
    mean_valid = out["fused_valid_views_mean"].item()
    # Rows 0, 2, 3 have all 3 slots valid; row 1 has 2. Mean = 11/4 = 2.75.
    assert mean_valid == pytest.approx(2.75)
    assert torch.isfinite(out["fused_loss"])
    assert out["fused_active"].item() == 1.0


def test_fail_loud_when_target_clip_missing():
    """With pair_view/view_nce active and pair.target_clip=None, the
    forward must raise unless allow_provisional_clip_b_fallback=True."""
    pair, meta = _make_real_target_batch()
    # Strip the target_clip from the batch.
    pair_no_target = PairBatch(
        clip_a=pair.clip_a,
        clip_b=pair.clip_b,
        masks_enc=pair.masks_enc,
        masks_pred=pair.masks_pred,
        phase_metadata=pair.phase_metadata,
        clip_b_neg=pair.clip_b_neg,
        target_clip=None,
        target_views=None,
        target_delta_phase=None,
    )
    with pytest.raises(ValueError, match="target_clip is None"):
        _run_real(pair_no_target, meta, fused_active=False)


def test_fused_guard_insufficient_valid_views():
    """When fused_valid_mask.sum(dim=1).mean() < 2, forward must raise."""
    pair, meta = _make_real_target_batch(B=4, n_fused=3)
    # Knock out fused slots so mean valid < 2.
    bad_mask = torch.zeros(4, 3, dtype=torch.bool)
    bad_mask[:, 0] = True  # only slot 0 valid for every row
    pair_bad = PairBatch(
        clip_a=pair.clip_a,
        clip_b=pair.clip_b,
        masks_enc=pair.masks_enc,
        masks_pred=pair.masks_pred,
        phase_metadata=pair.phase_metadata,
        clip_b_neg=pair.clip_b_neg,
        target_clip=pair.target_clip,
        target_views=pair.target_views,
        target_delta_phase=pair.target_delta_phase,
        target_clip_present=pair.target_clip_present,
        fused_clips=pair.fused_clips,
        fused_views=pair.fused_views,
        fused_phases=pair.fused_phases,
        fused_valid_mask=bad_mask,
    )
    with pytest.raises(ValueError, match="mean valid views"):
        _run_real(pair_bad, meta, fused_active=True, lambda_fused=0.05)


def test_used_clip_b_fallback_is_zero_on_real_path():
    pair, meta = _make_real_target_batch()
    out, _ = _run_real(pair, meta, fused_active=False)
    assert out["used_clip_b_fallback"].item() == 0.0
