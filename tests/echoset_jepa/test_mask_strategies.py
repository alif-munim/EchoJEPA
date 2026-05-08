"""Mask strategy tests (plan §3.5)."""

from __future__ import annotations

import random
from collections import Counter

from src.datasets.echoset_jepa_dataset import DEFAULT_MASK_STRATEGY_WEIGHTS, pick_mask_indices


def _mixed_keys():
    return [
        ("apical", "b_mode", "systolic"),
        ("apical", "b_mode", "diastolic"),
        ("parasternal_long", "b_mode", "full_cycle"),
        ("parasternal_short", "b_mode", "full_cycle"),
        ("apical", "color_doppler", "not_applicable"),
        ("doppler_spectral", "cw_doppler", "not_applicable"),
    ]


def test_mask_invariants():
    keys = _mixed_keys()
    rng = random.Random(0)
    for _ in range(200):
        ctx, tgt, strat = pick_mask_indices(len(keys), keys, rng=rng)
        assert len(ctx) >= 1, strat
        assert len(tgt) >= 1, strat
        assert len(tgt) <= len(keys) - 1, strat
        assert set(ctx).isdisjoint(tgt)
        assert set(ctx) | set(tgt) == set(range(len(keys)))


def test_stratified_strategies_respect_filter():
    keys = _mixed_keys()
    rng = random.Random(0)
    # apical_holdout: should mask all apical, nothing else
    ctx, tgt, strat = pick_mask_indices(
        len(keys), keys, strategy_weights={"apical_holdout": 1.0}, rng=rng
    )
    assert strat == "apical_holdout"
    masked_keys = {keys[i] for i in tgt}
    assert all(k[0] == "apical" for k in masked_keys)


def test_whole_view_family_masks_entire_family():
    keys = _mixed_keys()
    rng = random.Random(0)
    ctx, tgt, strat = pick_mask_indices(
        len(keys), keys, strategy_weights={"whole_view_family": 1.0}, rng=rng
    )
    assert strat == "whole_view_family"
    # All targets should share the same view_family
    families = {keys[i][0] for i in tgt}
    assert len(families) == 1


def test_stratified_disabled_when_filter_is_empty():
    # No apical elements → apical_holdout should fall back to random_element
    keys = [
        ("parasternal_long", "b_mode", "full_cycle"),
        ("parasternal_short", "b_mode", "full_cycle"),
    ]
    rng = random.Random(0)
    ctx, tgt, strat = pick_mask_indices(
        len(keys), keys, strategy_weights={"apical_holdout": 1.0}, rng=rng
    )
    assert strat == "random_element"


def test_default_mixture_sums_to_one():
    assert abs(sum(DEFAULT_MASK_STRATEGY_WEIGHTS.values()) - 1.0) < 1e-6
