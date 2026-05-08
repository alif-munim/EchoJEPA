"""Dedup tests — metadata-only mode (no c_clip)."""

from __future__ import annotations

import pytest

from experiments.echoset_jepa.dedup import DedupConfig, find_near_dup_clusters


def _cfg(require_cosine: bool = False) -> DedupConfig:
    return DedupConfig(require_cosine=require_cosine)


def test_metadata_only_catches_obvious_dups():
    # 3 identical A4C b_mode clips + 1 distinct PLAX clip
    clip_ids = ["c0", "c1", "c2", "c3"]
    n_dup, is_dup_of = find_near_dup_clusters(
        clip_ids=clip_ids,
        view_labels=["A4C", "A4C", "A4C", "PLAX"],
        view_confs=[0.9, 0.9, 0.9, 0.9],
        modalities=["b_mode"] * 4,
        n_frames=[60, 60, 60, 60],
        durations_s=[2.0, 2.0, 2.0, 2.0],
        quality_scores=[0.7, 0.8, 0.6, 0.9],   # c1 is cluster rep (highest quality)
        c_clip=None,
        cfg=_cfg(),
    )
    assert n_dup == [2, 2, 2, 0]
    # All non-rep duplicates point at c1
    assert is_dup_of[0] == "c1"
    assert is_dup_of[1] is None     # rep
    assert is_dup_of[2] == "c1"
    assert is_dup_of[3] is None


def test_metadata_only_respects_view_mismatch():
    clip_ids = ["a", "b"]
    n_dup, is_dup_of = find_near_dup_clusters(
        clip_ids=clip_ids,
        view_labels=["A4C", "PLAX"],
        view_confs=[0.9, 0.9],
        modalities=["b_mode"] * 2,
        n_frames=[60, 60],
        durations_s=[2.0, 2.0],
        quality_scores=[0.7, 0.8],
        c_clip=None,
        cfg=_cfg(),
    )
    assert n_dup == [0, 0]
    assert is_dup_of == [None, None]


def test_metadata_only_respects_modality_mismatch():
    clip_ids = ["a", "b"]
    n_dup, is_dup_of = find_near_dup_clusters(
        clip_ids=clip_ids,
        view_labels=["A4C", "A4C"],
        view_confs=[0.9, 0.9],
        modalities=["b_mode", "color_doppler"],
        n_frames=[60, 60],
        durations_s=[2.0, 2.0],
        quality_scores=[0.7, 0.8],
        c_clip=None,
        cfg=_cfg(),
    )
    assert n_dup == [0, 0]


def test_metadata_only_respects_frame_gap():
    clip_ids = ["a", "b"]
    n_dup, _ = find_near_dup_clusters(
        clip_ids=clip_ids,
        view_labels=["A4C", "A4C"],
        view_confs=[0.9, 0.9],
        modalities=["b_mode"] * 2,
        n_frames=[30, 60],      # 30-frame gap > max_frame_diff=3
        durations_s=[2.0, 2.0],
        quality_scores=[0.7, 0.8],
        c_clip=None,
        cfg=_cfg(),
    )
    assert n_dup == [0, 0]


def test_metadata_only_respects_low_view_conf():
    # One of the two clips has view_conf below threshold → cannot pair
    n_dup, _ = find_near_dup_clusters(
        clip_ids=["a", "b"],
        view_labels=["A4C", "A4C"],
        view_confs=[0.9, 0.5],
        modalities=["b_mode"] * 2,
        n_frames=[60, 60],
        durations_s=[2.0, 2.0],
        quality_scores=[0.7, 0.8],
        c_clip=None,
        cfg=_cfg(),
    )
    assert n_dup == [0, 0]


def test_require_cosine_without_c_clip_raises():
    with pytest.raises(ValueError):
        find_near_dup_clusters(
            clip_ids=["a"],
            view_labels=["A4C"],
            view_confs=[0.9],
            modalities=["b_mode"],
            n_frames=[60],
            durations_s=[2.0],
            quality_scores=[0.7],
            c_clip=None,
            cfg=_cfg(require_cosine=True),
        )
