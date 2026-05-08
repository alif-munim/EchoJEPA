"""Quality proxy tests (v0.1)."""

from __future__ import annotations

import math

import pytest

from experiments.echoset_jepa.quality_proxy import (
    QUALITY_PROXY_VERSION,
    quality_buckets_from_train_tertiles,
    quality_score_row,
    quality_scores,
)


def test_quality_score_in_unit_interval():
    # High-confidence, sensible clip
    s = quality_score_row(view_confidence=0.9, video_duration_s=2.5, fps_video=30.0,
                          n_video_frames=64, ecg_quality_tier="high")
    assert 0.0 <= s <= 1.0
    assert s > 0.8

    # All-missing → neutral 0.5
    s2 = quality_score_row(None, None, None, None, None)
    assert 0.4 <= s2 <= 0.6


def test_quality_score_nan_inputs_safe():
    s = quality_score_row(float("nan"), float("nan"), float("nan"), None, "reject")
    assert 0.0 <= s <= 1.0


def test_quality_score_rejects_degenerate_clips():
    bad = quality_score_row(view_confidence=0.1, video_duration_s=0.0, fps_video=0.0,
                            n_video_frames=0, ecg_quality_tier="reject")
    good = quality_score_row(view_confidence=0.95, video_duration_s=2.5, fps_video=30.0,
                             n_video_frames=64, ecg_quality_tier="high")
    assert bad < good
    assert bad < 0.3


def test_quality_score_version_is_tracked():
    assert QUALITY_PROXY_VERSION.startswith("v")


def test_quality_scores_vectorized_matches_scalar():
    vc = [0.5, 0.9, 0.2]
    d = [1.5, 3.0, 0.3]
    fps = [30.0, 45.0, 10.0]
    nf = [30, 60, 5]
    tiers = ["high", "high", "low"]
    batch = quality_scores(vc, d, fps, nf, tiers)
    scalars = [quality_score_row(*a) for a in zip(vc, d, fps, nf, tiers)]
    for a, b in zip(batch, scalars):
        assert a == pytest.approx(b)


# ---- bucket from train-only tertiles --------------------------------------

def test_bucket_uses_train_tertiles_only():
    # Train rows are all low; val/test rows are all high. Bucket boundaries
    # should be set on the low side, so the val/test rows fall into 'high'.
    all_scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95,
                  0.9, 0.9, 0.9]
    train_mask = [True] * 10 + [False] * 3
    buckets = quality_buckets_from_train_tertiles(all_scores, train_mask)
    # Train rows split into 3 buckets
    train_buckets = buckets[:10]
    assert "low" in train_buckets
    assert "med" in train_buckets
    assert "high" in train_buckets
    # Val/test rows (0.9) should be 'high' since tertiles were set on train
    assert all(b == "high" for b in buckets[10:])


def test_bucket_fallback_when_few_train():
    scores = [0.5, 0.5]
    train_mask = [True, False]
    # Only 1 train row → not enough to compute tertiles robustly
    buckets = quality_buckets_from_train_tertiles(scores, train_mask)
    assert buckets == ["unknown", "unknown"]


def test_bucket_size_matches():
    scores = [0.1, 0.5, 0.9] * 10
    mask = [True] * 30
    buckets = quality_buckets_from_train_tertiles(scores, mask)
    assert len(buckets) == len(scores)
    # Should have roughly equal counts of low/med/high
    from collections import Counter
    c = Counter(buckets)
    assert c["low"] >= 5 and c["med"] >= 5 and c["high"] >= 5


def test_bucket_mask_length_mismatch_raises():
    with pytest.raises(ValueError):
        quality_buckets_from_train_tertiles([0.1, 0.2], [True])


# ---- no target-side leakage guard -----------------------------------------

def test_no_target_quality_by_default():
    """The Stage-2 target-mask slot must not consume quality tokens by default.

    The enforcement is in src/models/meta_embeddings.py (tested in
    test_target_metadata_no_quality_leak.py). This test just re-asserts that
    the quality proxy itself is context-only by construction: it emits a
    score, not a target feature.
    """
    # No target-side injection point exists in quality_proxy.py; the only
    # consumers are dedup tie-break, context-side meta token, and element
    # aggregation weight. Assert the module does not expose any target_* API.
    from experiments.echoset_jepa import quality_proxy as qp
    public = [n for n in dir(qp) if not n.startswith("_")]
    assert not any("target" in n.lower() for n in public), (
        f"quality_proxy must not expose any target_* API; found: {public}"
    )
