"""Unit tests for matched_rank_metrics + its integration as a diagnostic."""

from __future__ import annotations

import math

import torch

from app.echomv_jepa.train import training_step_echomv
from src.models.echomv_jepa import matched_rank_metrics
from tests.echomv_jepa.test_contextualization_diagnostics import (
    _make_models,
    _synthetic_batch,
)


def test_matched_rank_perfect_alignment_top1_equals_one():
    """If h == z row-aligned and all negatives are valid, positive always
    scores highest → top1 = 1.0."""
    torch.manual_seed(0)
    N, D = 8, 16
    h = torch.randn(N, D)
    z = h.clone()  # perfect alignment
    neg_mask = torch.ones(N, N, dtype=torch.bool)
    out = matched_rank_metrics(h, z, neg_mask)
    assert out["matched_rank_top1"] == 1.0
    assert out["matched_rank_top5"] == 1.0
    # The positive should beat the hardest negative by a non-trivial margin.
    assert out["pos_minus_hardneg_gap_mean"] > 0.0


def test_matched_rank_random_alignment_near_chance():
    """Random h and z have expected top1 ~ 1/N."""
    torch.manual_seed(1)
    N, D = 32, 16
    h = torch.randn(N, D)
    z = torch.randn(N, D)
    neg_mask = torch.ones(N, N, dtype=torch.bool)
    out = matched_rank_metrics(h, z, neg_mask)
    # Chance top1 = 1/32 ≈ 0.031. Allow a loose bound (rare draws).
    assert out["matched_rank_top1"] < 0.4


def test_matched_rank_nan_when_n_lt_2():
    h = torch.randn(1, 8)
    z = torch.randn(1, 8)
    neg_mask = torch.ones(1, 1, dtype=torch.bool)
    out = matched_rank_metrics(h, z, neg_mask)
    assert math.isnan(out["matched_rank_top1"])


def test_matched_rank_metrics_integrated_into_step():
    st, meta, proj, teacher = _make_models()
    batch = _synthetic_batch(B=4, M_ctx=3, M_tgt=2)
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
    for k in ("matched_rank_top1", "matched_rank_top5", "pos_minus_hardneg_gap"):
        assert k in d
        assert not math.isnan(d[k])
        if "gap" not in k:
            assert 0.0 <= d[k] <= 1.0
