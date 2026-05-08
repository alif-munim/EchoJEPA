"""Element grouping tests (plan §3.1, §3.6)."""

from __future__ import annotations

import numpy as np
import pytest

from src.datasets.echoset_jepa_dataset import group_into_elements


def test_group_merges_same_key_clips():
    keys = [
        ("apical", "b_mode", "systolic"),
        ("apical", "b_mode", "systolic"),   # merged with above
        ("parasternal_long", "b_mode", "full_cycle"),
    ]
    vecs = np.array([[1.0, 0.0], [3.0, 4.0], [0.0, 1.0]])
    qs = np.array([0.5, 0.7, 0.9])
    elem_keys, elem_vecs, elem_q = group_into_elements(keys, vecs, qs, element_agg="mean")
    assert len(elem_keys) == 2
    # Apical merged into mean of [1,0] and [3,4] = [2,2]
    apical_idx = elem_keys.index(("apical", "b_mode", "systolic"))
    assert np.allclose(elem_vecs[apical_idx], [2.0, 2.0])
    assert elem_q[apical_idx] == pytest.approx(0.6)


def test_group_quality_weighted_reduces_to_mean_with_equal_quality():
    keys = [("apical", "b_mode", "systolic"), ("apical", "b_mode", "systolic")]
    vecs = np.array([[1.0, 0.0], [0.0, 1.0]])
    qs = np.array([0.5, 0.5])
    _, elem_vecs_qw, _ = group_into_elements(keys, vecs, qs, element_agg="quality_weighted")
    _, elem_vecs_mean, _ = group_into_elements(keys, vecs, qs, element_agg="mean")
    assert np.allclose(elem_vecs_qw, elem_vecs_mean)


def test_group_quality_weighted_favors_higher_quality():
    keys = [("apical", "b_mode", "systolic"), ("apical", "b_mode", "systolic")]
    vecs = np.array([[1.0, 0.0], [0.0, 1.0]])
    qs = np.array([0.1, 0.9])
    _, elem_vecs, _ = group_into_elements(keys, vecs, qs, element_agg="quality_weighted", tau_quality=0.5)
    # Higher-quality clip (vec [0,1], q=0.9) should dominate → result closer to [0,1]
    assert elem_vecs[0][1] > elem_vecs[0][0]


def test_group_quality_bucket_not_in_key():
    """Quality must not split clips with otherwise-equal keys (plan §3.1)."""
    keys = [
        ("apical", "b_mode", "systolic"),
        ("apical", "b_mode", "systolic"),
    ]
    vecs = np.ones((2, 4))
    qs = np.array([0.05, 0.95])   # low + high quality
    elem_keys, _, _ = group_into_elements(keys, vecs, qs, element_agg="mean")
    assert len(elem_keys) == 1, "quality bucket leaked into grouping key"
