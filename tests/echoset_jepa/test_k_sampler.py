"""K-matched sampler tests (plan §3.6)."""

from __future__ import annotations

import random

from experiments.echoset_jepa.sample_K import sample_view_stratified


def _mkrow(clip_id, view_family, quality):
    return {
        "clip_id": clip_id,
        "view_family": view_family,
        "modality": "b_mode",
        "quality_score": quality,
    }


def test_sampler_respects_K():
    rows = [
        _mkrow(f"c{i}", "apical", 0.5)
        for i in range(12)
    ]
    picked = sample_view_stratified(rows, K=8, rng=random.Random(0))
    assert len(picked) == 8


def test_sampler_stratifies_across_view_families():
    rows = (
        [_mkrow(f"a{i}", "apical", 0.5) for i in range(10)]
        + [_mkrow(f"p{i}", "parasternal_long", 0.5) for i in range(5)]
        + [_mkrow(f"s{i}", "parasternal_short", 0.5) for i in range(3)]
    )
    picked = sample_view_stratified(rows, K=8, rng=random.Random(0))
    fams = {r["view_family"] for r in picked}
    # At K=8 with three families available, we should have all three represented.
    assert len(fams) == 3


def test_sampler_is_seed_deterministic():
    rows = [_mkrow(f"c{i}", "apical" if i % 2 else "parasternal_long", 0.5) for i in range(20)]
    a = sample_view_stratified(rows, K=8, rng=random.Random(42))
    b = sample_view_stratified(rows, K=8, rng=random.Random(42))
    assert [r["clip_id"] for r in a] == [r["clip_id"] for r in b]


def test_sampler_handles_study_with_fewer_than_K_clips():
    rows = [_mkrow(f"c{i}", "apical", 0.5) for i in range(3)]
    picked = sample_view_stratified(rows, K=8, rng=random.Random(0))
    assert len(picked) == 3
    assert len({r["clip_id"] for r in picked}) == 3


def test_sampler_prefers_higher_quality_within_family():
    rows = [
        _mkrow("low1", "apical", 0.1),
        _mkrow("high1", "apical", 0.95),
        _mkrow("mid1", "apical", 0.5),
    ]
    picked = sample_view_stratified(rows, K=2, rng=random.Random(0))
    ids = [r["clip_id"] for r in picked]
    assert "high1" in ids
