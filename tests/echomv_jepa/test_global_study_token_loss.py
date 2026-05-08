"""Arm C — global study-token JEPA: shape and loss tests."""

from __future__ import annotations

import torch

from app.echomv_jepa.train import training_step_echomv_global
from tests.echomv_jepa.test_contextualization_diagnostics import (
    _make_models,
    _synthetic_batch,
)


def test_global_step_returns_finite_loss():
    st, meta, proj, teacher = _make_models()
    batch = _synthetic_batch(B=4, M_ctx=4, M_tgt=0)  # Arm C ignores per-elem masking
    # Expand full_* to the 4 context elements (Arm C uses full_elements directly).
    # _synthetic_batch already emits full_elements = concat(ctx, tgt); with M_tgt=0,
    # full = ctx. That matches Arm C's contract.
    out = training_step_echomv_global(
        batch,
        st,
        meta,
        proj,
        teacher,
        lambda_nce=0.01,
        lambda_cov=0.001,
        diag_extra_every_n_steps=1,
        global_step=0,
    )
    assert torch.isfinite(out.loss)
    # h_study predictions are (B, d_proj); matched_rank_top1 etc. computed.
    for k in ("loss_regress", "loss_nce", "loss_cov", "var_t", "cov_off"):
        assert k in out.diagnostics


def test_global_step_predicts_nonconstant_h_study():
    """With random student weights, different corrupted inputs should produce
    different h_study outputs. Confirms the [STUDY] readout is not stuck at
    a constant value."""
    torch.manual_seed(0)
    st, meta, proj, teacher = _make_models()
    batch = _synthetic_batch(B=6, M_ctx=5, M_tgt=0)
    out = training_step_echomv_global(
        batch,
        st,
        meta,
        proj,
        teacher,
        lambda_nce=0.0,
        lambda_cov=0.0,
        diag_extra_every_n_steps=1,
        global_step=0,
    )
    # var_t is the mean per-dim std over B. For a truly constant readout this
    # would be 0; for random init we expect > 0.
    assert out.diagnostics["var_t"] > 0.0


def test_global_step_study_context_delta_emitted():
    st, meta, proj, teacher = _make_models()
    batch = _synthetic_batch(B=4, M_ctx=4, M_tgt=0)
    out = training_step_echomv_global(
        batch,
        st,
        meta,
        proj,
        teacher,
        lambda_nce=0.0,
        lambda_cov=0.0,
        diag_extra_every_n_steps=1,
        global_step=0,
    )
    import math

    assert not math.isnan(out.diagnostics["study_context_delta"])
    assert not math.isnan(out.diagnostics["metadata_only_study_gap"])
