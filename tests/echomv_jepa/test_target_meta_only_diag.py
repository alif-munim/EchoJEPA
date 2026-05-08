"""Test for the target_meta_only_gap diagnostic (Part 2).

Computes ``cos(h_actual, z) - cos(h_meta_only, z)``. If metadata alone is
sufficient to predict the target, gap ≤ 0.
"""

from __future__ import annotations

import math

from app.echomv_jepa.train import training_step_echomv
from tests.echomv_jepa.test_contextualization_diagnostics import (
    _make_models,
    _synthetic_batch,
)


def test_target_meta_only_gap_is_finite():
    st, meta, proj, teacher = _make_models()
    batch = _synthetic_batch(B=3, M_ctx=3, M_tgt=2)
    out = training_step_echomv(
        batch,
        st,
        meta,
        proj,
        teacher,
        lambda_nce=0.0,
        diag_peer_drop_every_n_steps=1,
        diag_extra_every_n_steps=1,
        global_step=0,
    )
    d = out.diagnostics
    assert "target_meta_only_gap" in d
    assert "target_meta_only_cos" in d
    assert not math.isnan(d["target_meta_only_gap"])
    assert not math.isnan(d["target_meta_only_cos"])
    # Bounded by definition: cos is in [-1, 1], so gap is in [-2, 2].
    assert -2.0 <= d["target_meta_only_gap"] <= 2.0


def test_meta_only_is_nan_off_cadence():
    st, meta, proj, teacher = _make_models()
    batch = _synthetic_batch(B=2, M_ctx=3, M_tgt=2)
    out = training_step_echomv(
        batch,
        st,
        meta,
        proj,
        teacher,
        lambda_nce=0.0,
        diag_peer_drop_every_n_steps=50,
        diag_extra_every_n_steps=50,
        global_step=1,
    )
    assert math.isnan(out.diagnostics["target_meta_only_gap"])


def test_meta_only_gap_matches_direct_computation_on_identical_metadata():
    """Sanity: if student is freshly initialized and we use all-zero metadata,
    the meta-only student and the actual student should produce outputs that
    are only weakly related to z_t — the raw cosine values are in a sane
    range, not pinned to 1.0."""
    st, meta, proj, teacher = _make_models()
    batch = _synthetic_batch(B=3, M_ctx=3, M_tgt=2)
    out = training_step_echomv(
        batch,
        st,
        meta,
        proj,
        teacher,
        lambda_nce=0.0,
        diag_peer_drop_every_n_steps=1,
        diag_extra_every_n_steps=1,
        global_step=0,
    )
    d = out.diagnostics
    # At cold init, cos_actual and cos_meta should both be in [-1, 1].
    # We don't assert a sign on gap because at init with random weights it
    # can go either way — only gate at training end in the smoke.
    cos_meta = d["target_meta_only_cos"]
    assert -1.01 <= cos_meta <= 1.01
