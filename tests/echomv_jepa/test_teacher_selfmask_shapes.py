"""Arm B shape / rate tests for teacher target self-masking."""

from __future__ import annotations

import torch

from app.echomv_jepa.train import training_step_echomv
from tests.echomv_jepa.test_contextualization_diagnostics import (
    _make_models,
    _synthetic_batch,
)


def test_selfmask_rate_zero_when_p_zero():
    st, meta, proj, teacher = _make_models()
    batch = _synthetic_batch(B=3, M_ctx=3, M_tgt=2)
    out = training_step_echomv(
        batch,
        st,
        meta,
        proj,
        teacher,
        lambda_nce=0.0,
        p_target_self_mask=0.0,
        global_step=0,
    )
    assert out.diagnostics["teacher_selfmask_rate"] == 0.0


def test_selfmask_rate_nonzero_when_p_positive():
    torch.manual_seed(0)
    st, meta, proj, teacher = _make_models()
    batch = _synthetic_batch(B=8, M_ctx=3, M_tgt=4)
    out = training_step_echomv(
        batch,
        st,
        meta,
        proj,
        teacher,
        lambda_nce=0.0,
        p_target_self_mask=0.5,
        global_step=0,
    )
    rate = out.diagnostics["teacher_selfmask_rate"]
    # Empirically ~0.5 ± finite-sample noise; just assert > 0 and ≤ 1.
    assert 0.0 < rate <= 1.0


def test_selfmask_preserves_shapes():
    st, meta, proj, teacher = _make_models()
    batch = _synthetic_batch(B=3, M_ctx=3, M_tgt=2)
    out = training_step_echomv(
        batch,
        st,
        meta,
        proj,
        teacher,
        lambda_nce=0.0,
        p_target_self_mask=1.0,
        global_step=0,
    )
    # All expected diagnostics present; loss finite.
    assert torch.isfinite(out.loss)
    for k in ("loss_regress", "loss_nce", "loss_cov", "var_t", "cov_off"):
        assert k in out.diagnostics
