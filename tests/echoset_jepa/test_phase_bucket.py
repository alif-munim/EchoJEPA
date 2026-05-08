"""Tests for per-clip phase_bucket derivation."""

from __future__ import annotations

import json

from experiments.echoset_jepa.phase_bucket import derive_clip_phase_bucket


def _json(vals):
    return json.dumps(vals)


def test_full_cycle_when_multiple_rpeaks():
    phases = _json([0.0, 0.2, 0.4, 0.6, 0.8, 0.0, 0.2, 0.4])
    mask = _json([1] * 8)
    # 2 R-peaks listed → rule 2 fires
    rpks = _json([0, 5])
    assert derive_clip_phase_bucket(phases, mask, rpks) == "full_cycle"


def test_full_cycle_when_phase_span_large():
    phases = _json([0.05, 0.15, 0.45, 0.75, 0.95])
    mask = _json([1] * 5)
    rpks = _json([])   # no R-peaks; fall through to span rule
    assert derive_clip_phase_bucket(phases, mask, rpks) == "full_cycle"


def test_systolic_when_dominant():
    # All phases < 0.35 → dominant systolic
    phases = _json([0.05, 0.10, 0.15, 0.20, 0.25])
    mask = _json([1] * 5)
    rpks = _json([])
    assert derive_clip_phase_bucket(phases, mask, rpks) == "systolic"


def test_diastolic_when_dominant():
    phases = _json([0.40, 0.50, 0.60, 0.70, 0.80])
    mask = _json([1] * 5)
    rpks = _json([])
    assert derive_clip_phase_bucket(phases, mask, rpks) == "diastolic"


def test_unknown_when_too_few_confident():
    # Only 1 confident frame out of 10 (< 30% threshold)
    phases = _json([None, None, None, None, None, 0.5, None, None, None, None])
    mask = _json([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
    rpks = _json([])
    assert derive_clip_phase_bucket(phases, mask, rpks) == "unknown"


def test_handles_nones_in_phases():
    # Frames outside confident windows are None in phase_json
    phases_list = [None, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, None]
    phases = json.dumps(phases_list)
    mask = json.dumps([0, 1, 1, 1, 1, 1, 1, 1, 0])
    rpks = json.dumps([])
    # Span is 0.7 - 0.1 = 0.6, below default 0.8 threshold → should be
    # full_cycle (mixed, rule 5) not systolic or diastolic
    out = derive_clip_phase_bucket(phases, mask, rpks)
    assert out in {"full_cycle"}


def test_handles_raw_list_inputs():
    # Skip JSON decoding when caller passes a list directly
    phases = [0.1, 0.5, 0.9]
    mask = [1, 1, 1]
    rpks = [0, 2]
    assert derive_clip_phase_bucket(phases, mask, rpks) == "full_cycle"


def test_empty_json_is_unknown():
    assert derive_clip_phase_bucket(None) == "unknown"
    assert derive_clip_phase_bucket("") == "unknown"
    assert derive_clip_phase_bucket("[]") == "unknown"
