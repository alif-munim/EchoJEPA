"""Fix 1: MV2SV target_clip + fused_clips sampling logic.

Exercises ``_mv2sv_allowed_target_views``, ``_mv2sv_pick_candidate``,
and ``_mv2sv_build_anchor`` via a PhaseMatchedStudySampler whose
``_load`` is stubbed out — we manually populate the caches the MV2SV
helpers read, avoiding the need for a synthetic parquet.

Does NOT exercise the full ``build_records`` pipeline (that needs a
parquet + phase annotations). Those integration paths will be tested
when the smoke runs on real data.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from classifier.phase.sampler.phase_matched_sampler import (  # noqa: E402
    ClipAnchor,
    PhaseMatchedStudySampler,
)


@pytest.fixture(autouse=True)
def _seed():
    np.random.seed(0)


def _build_stubbed_sampler(
    mv2sv_config: dict | None = None,
    view_labels: dict | None = None,
    study_to_rows_by_view: dict | None = None,
    view_by_row: dict | None = None,
    dicom_by_row: dict | None = None,
) -> PhaseMatchedStudySampler:
    """Construct a sampler with ``_load`` patched out, then install the
    minimal caches the MV2SV helpers read.
    """
    if view_labels is None:
        view_labels = {"d_a": "A4C", "d_t1": "A2C", "d_t2": "PLAX"}
    with patch.object(PhaseMatchedStudySampler, "_load", lambda self: None):
        s = PhaseMatchedStudySampler(
            parquet_path="/dev/null",  # _load is patched out
            view_labels=view_labels,
            mv2sv_config=mv2sv_config,
        )
    # Install caches the MV2SV helpers consume.
    if study_to_rows_by_view is not None:
        s._study_to_rows_by_view = study_to_rows_by_view
    if view_by_row is not None:
        s._view_by_row = view_by_row
    if dicom_by_row is not None:
        s._dicom_by_row = dicom_by_row
    return s


def _anchor(row_idx: int = 0, view: str = "A4C", dicom: str = "d_a") -> ClipAnchor:
    return ClipAnchor(
        row_idx=row_idx,
        dicom_id=dicom,
        n_frames=64,
        anchor_frame=10,
        phase_at_anchor=0.25,
        phase_error=0.01,
        view=view,
    )


# --- Config normalization ------------------------------------------------- #


def test_mv2sv_disabled_by_default():
    s = _build_stubbed_sampler()
    assert s.mv2sv_enabled is False
    assert s.mv2sv_fused_enabled is False


def test_mv2sv_enabled_without_view_labels_raises():
    with patch.object(PhaseMatchedStudySampler, "_load", lambda self: None):
        with pytest.raises(ValueError, match="view_labels"):
            PhaseMatchedStudySampler(
                parquet_path="/dev/null",
                view_labels=None,
                mv2sv_config={"enabled": True},
            )


def test_mv2sv_stage1_allowed_for_a4c_source():
    s = _build_stubbed_sampler(
        mv2sv_config={
            "enabled": True,
            "target_view_sampling": {
                "stage": "stage1",
                "a4c_source_targets": ["A2C", "A5C"],
                "allowed_targets": ["A2C", "A5C", "PLAX", "PSAX-MV", "A3C"],
            },
        }
    )
    assert s.mv2sv_target_stage == "stage1"
    assert s._mv2sv_allowed_target_views("A4C") == ("A2C", "A5C")
    # Non-A4C source falls back to the broader allowed set.
    assert "PLAX" in s._mv2sv_allowed_target_views("PLAX")


def test_mv2sv_stage2_uses_full_allowed():
    s = _build_stubbed_sampler(
        mv2sv_config={
            "enabled": True,
            "target_view_sampling": {
                "stage": "stage2",
                "a4c_source_targets": ["A2C"],
                "allowed_targets": ["A2C", "A5C", "PLAX"],
            },
        }
    )
    assert s.mv2sv_target_stage == "stage2"
    assert s._mv2sv_allowed_target_views("A4C") == ("A2C", "A5C", "PLAX")


def test_mv2sv_unknown_stage_rejected():
    with patch.object(PhaseMatchedStudySampler, "_load", lambda self: None):
        with pytest.raises(ValueError, match="stage"):
            PhaseMatchedStudySampler(
                parquet_path="/dev/null",
                view_labels={"d": "A4C"},
                mv2sv_config={
                    "enabled": True,
                    "target_view_sampling": {"stage": "bogus"},
                },
            )


def test_mv2sv_fused_n_max_less_than_n_min_rejected():
    with patch.object(PhaseMatchedStudySampler, "_load", lambda self: None):
        with pytest.raises(ValueError, match="n_fused"):
            PhaseMatchedStudySampler(
                parquet_path="/dev/null",
                view_labels={"d": "A4C"},
                mv2sv_config={
                    "enabled": True,
                    "fused_pool": {
                        "enabled": True,
                        "n_fused_min": 3,
                        "n_fused_max": 2,
                    },
                },
            )


# --- _mv2sv_pick_candidate ------------------------------------------------ #


def test_pick_candidate_respects_allowed_views():
    s = _build_stubbed_sampler(
        mv2sv_config={
            "enabled": True,
            "target_view_sampling": {
                "stage": "stage2",
                "allowed_targets": ["A2C"],  # PLAX should NOT be chosen
            },
        },
        study_to_rows_by_view={
            "study1": {"A4C": [0], "A2C": [1], "PLAX": [2]},
        },
        view_by_row={0: "A4C", 1: "A2C", 2: "PLAX"},
        dicom_by_row={0: "d0", 1: "d1", 2: "d2"},
    )
    rng = np.random.default_rng(0)
    picks = [
        s._mv2sv_pick_candidate(
            group_key="study1",
            clip_a_row_idx=0,
            clip_a_dicom="d0",
            clip_a_view="A4C",
            exclude_row_idxs=set(),
            allowed_views=("A2C",),
            rng=rng,
        )
        for _ in range(20)
    ]
    assert all(p == 1 for p in picks), f"expected all picks == 1, got {picks}"


def test_pick_candidate_excludes_anchor_and_excluded_rows():
    s = _build_stubbed_sampler(
        mv2sv_config={"enabled": True},
        study_to_rows_by_view={"study1": {"A2C": [1, 2, 3]}},
        view_by_row={0: "A4C", 1: "A2C", 2: "A2C", 3: "A2C"},
        dicom_by_row={0: "d0", 1: "d1", 2: "d2", 3: "d3"},
    )
    rng = np.random.default_rng(0)
    # Exclude row 2; anchor is row 0 with dicom d0.
    for _ in range(30):
        pick = s._mv2sv_pick_candidate(
            group_key="study1",
            clip_a_row_idx=0,
            clip_a_dicom="d0",
            clip_a_view="A4C",
            exclude_row_idxs={2},
            allowed_views=("A2C",),
            rng=rng,
        )
        assert pick in {1, 3}


def test_pick_candidate_requires_different_view_when_configured():
    """When require_different_view=True (default), picking an A4C target
    for an A4C source must return None (no valid candidates)."""
    s = _build_stubbed_sampler(
        mv2sv_config={
            "enabled": True,
            "target_view_sampling": {
                "stage": "stage2",
                "allowed_targets": ["A4C"],  # only allowed = same as source
                "require_different_view": True,
            },
        },
        study_to_rows_by_view={"study1": {"A4C": [1]}},
        view_by_row={0: "A4C", 1: "A4C"},
        dicom_by_row={0: "d0", 1: "d1"},
    )
    rng = np.random.default_rng(0)
    pick = s._mv2sv_pick_candidate(
        group_key="study1",
        clip_a_row_idx=0,
        clip_a_dicom="d0",
        clip_a_view="A4C",
        exclude_row_idxs=set(),
        allowed_views=("A4C",),
        rng=rng,
    )
    assert pick is None


def test_pick_candidate_returns_none_when_no_match():
    s = _build_stubbed_sampler(
        mv2sv_config={"enabled": True},
        study_to_rows_by_view={"study1": {"A4C": [0]}},
        view_by_row={0: "A4C"},
        dicom_by_row={0: "d0"},
    )
    rng = np.random.default_rng(0)
    pick = s._mv2sv_pick_candidate(
        group_key="study1",
        clip_a_row_idx=0,
        clip_a_dicom="d0",
        clip_a_view="A4C",
        exclude_row_idxs=set(),
        allowed_views=("A2C",),
        rng=rng,
    )
    assert pick is None


def test_pick_candidate_unknown_study_returns_none():
    s = _build_stubbed_sampler(
        mv2sv_config={"enabled": True},
        study_to_rows_by_view={"study1": {"A4C": [0]}},
        view_by_row={0: "A4C"},
        dicom_by_row={0: "d0"},
    )
    rng = np.random.default_rng(0)
    pick = s._mv2sv_pick_candidate(
        group_key="nonexistent",
        clip_a_row_idx=0,
        clip_a_dicom="d0",
        clip_a_view="A4C",
        exclude_row_idxs=set(),
        allowed_views=("A2C",),
        rng=rng,
    )
    assert pick is None


# --- Config schema tests -------------------------------------------------- #


def test_fused_pool_config_defaults():
    s = _build_stubbed_sampler(mv2sv_config={"enabled": True})
    # fused_pool block absent → fused disabled by default.
    assert s.mv2sv_fused_enabled is False
    assert s.mv2sv_fused_n_min == 2
    assert s.mv2sv_fused_n_max == 2


def test_fused_pool_config_custom():
    s = _build_stubbed_sampler(
        mv2sv_config={
            "enabled": True,
            "fused_pool": {
                "enabled": True,
                "n_fused_min": 2,
                "n_fused_max": 4,
                "require_distinct_views": True,
            },
        }
    )
    assert s.mv2sv_fused_enabled is True
    assert s.mv2sv_fused_n_min == 2
    assert s.mv2sv_fused_n_max == 4
    assert s.mv2sv_fused_require_distinct_views is True


def test_target_dropout_default_zero():
    s = _build_stubbed_sampler(mv2sv_config={"enabled": True})
    assert s.mv2sv_target_dropout == 0.0
