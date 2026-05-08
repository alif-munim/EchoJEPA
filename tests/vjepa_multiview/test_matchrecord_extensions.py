"""MatchRecord privileged_multiview extensions must not break default use.

Existing consumers (phase_relational, phase_curriculum, pair-only) rely
on the dataclass being constructable without the new fields. This test
pins that behaviour and confirms the new fields round-trip.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from classifier.phase.sampler.phase_matched_sampler import (  # noqa: E402
    ClipAnchor,
    MatchRecord,
)


def _anchor(dicom: str = "a.dcm") -> ClipAnchor:
    return ClipAnchor(
        row_idx=0,
        dicom_id=dicom,
        n_frames=64,
        anchor_frame=10,
        phase_at_anchor=0.25,
        phase_error=0.01,
        view="A4C",
    )


def _base_kwargs() -> dict:
    return dict(
        study_id="S1",
        subject_id="P1",
        acquisition_datetime_a="2026-01-01",
        acquisition_datetime_b="2026-01-01",
        clip_a=_anchor("a.dcm"),
        clip_b=_anchor("b.dcm"),
        target_phi_a=0.0,
        target_phi_b=0.5,
        circular_phase_diff=0.5,
        sampling_mode="phase_matched",
        frame_step=1,
        frames_per_clip=16,
        source_span_frames=16,
        source_span_seconds_a=0.5,
        source_span_seconds_b=0.5,
        source_span_cycles_a=0.5,
        source_span_cycles_b=0.5,
    )


def test_default_construction_backward_compatible():
    """Constructing a MatchRecord with only the pre-existing kwargs must
    still succeed — the new fields all carry defaults."""
    rec = MatchRecord(**_base_kwargs())
    assert rec.target_clip is None
    assert rec.target_view is None
    assert rec.target_delta_phase is None
    assert rec.fused_clips == ()
    assert rec.fused_views == ()
    assert rec.fused_phases == ()
    # Pre-existing hardneg fields still default too.
    assert rec.clip_b_neg_phase is None
    assert rec.hard_neg_available is False


def test_pairwise_target_roundtrip():
    rec = MatchRecord(
        **_base_kwargs(),
        target_clip=_anchor("t.dcm"),
        target_view="PLAX",
        target_delta_phase=0.25,
    )
    assert rec.target_clip is not None
    assert rec.target_clip.dicom_id == "t.dcm"
    assert rec.target_view == "PLAX"
    assert rec.target_delta_phase == pytest.approx(0.25)


def test_fused_clips_roundtrip():
    rec = MatchRecord(
        **_base_kwargs(),
        fused_clips=(_anchor("f1.dcm"), _anchor("f2.dcm")),
        fused_views=("A2C", "PLAX"),
        fused_phases=(0.1, 0.7),
    )
    assert len(rec.fused_clips) == 2
    assert rec.fused_views == ("A2C", "PLAX")
    assert rec.fused_phases == (0.1, 0.7)


def test_hashable_frozen():
    """MatchRecord is frozen=True; new tuple fields must stay hashable.
    (Old list-based field would have broken this. Using tuple preserves
    hashability and immutability.)"""
    rec = MatchRecord(
        **_base_kwargs(),
        fused_clips=(_anchor("f1.dcm"),),
        fused_views=("A2C",),
        fused_phases=(0.3,),
    )
    # frozen=True dataclass has __hash__; must not raise.
    h = hash(rec)
    assert isinstance(h, int)
