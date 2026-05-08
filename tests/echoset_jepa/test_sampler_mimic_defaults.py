"""Sampler smoke tests tuned for the MIMIC 2-modality cohort.

MIMIC has only {b_mode, color_doppler}. The K=8 default policy (bmode=6,
color=2, spectral=0) should:
  - select at most 6 B-mode clips
  - select at most 2 color clips
  - never error on studies that lack one of the two modalities
  - remain seed-deterministic
"""

from __future__ import annotations

import random

from experiments.echoset_jepa.sample_K import sample_mixed_modality


def _mk(clip_id: str, view: str, modality: str, q: float = 0.5) -> dict:
    return {
        "clip_id": clip_id,
        "study_id": "s_test",
        "view_family": view,
        "modality": modality,
        "phase_bucket": "full_cycle",
        "quality_score": q,
    }


def test_mimic_defaults_respect_budgets():
    rows = (
        [_mk(f"b{i}", "apical", "b_mode", 0.8) for i in range(10)]
        + [_mk(f"p{i}", "parasternal_long", "b_mode", 0.8) for i in range(5)]
        + [_mk(f"c{i}", "apical", "color_doppler", 0.8) for i in range(8)]
    )
    picked = sample_mixed_modality(rows, K=8, rng=random.Random(0))
    assert len(picked) == 8
    mods = [r["modality"] for r in picked]
    assert mods.count("b_mode") == 6
    assert mods.count("color_doppler") == 2


def test_mimic_defaults_handle_bmode_only_study():
    # Color absent — 2 color slots reallocate to B-mode via fill step.
    rows = [_mk(f"b{i}", "apical", "b_mode", 0.5) for i in range(15)]
    picked = sample_mixed_modality(rows, K=8, rng=random.Random(0))
    assert len(picked) == 8
    assert all(r["modality"] == "b_mode" for r in picked)


def test_mimic_defaults_handle_color_only_study():
    # B-mode absent — 6 B-mode slots reallocate across color + fill.
    rows = [_mk(f"c{i}", "apical", "color_doppler", 0.5) for i in range(15)]
    picked = sample_mixed_modality(rows, K=8, rng=random.Random(0))
    assert len(picked) == 8
    assert all(r["modality"] == "color_doppler" for r in picked)


def test_mimic_defaults_ignore_absent_spectral_budget():
    # spectral_budget defaults to 0; even if spectral clips existed, they
    # should only be taken via the final fill step (quality-sorted).
    rows = (
        [_mk(f"b{i}", "apical", "b_mode", 0.9) for i in range(10)]
        + [_mk(f"s{i}", "apical", "cw_doppler", 0.5) for i in range(3)]
    )
    picked = sample_mixed_modality(rows, K=8, rng=random.Random(0))
    # With spectral_budget=0, all 8 should be B-mode (higher quality wins fill).
    assert sum(1 for r in picked if r["modality"] == "cw_doppler") == 0


def test_mimic_defaults_seed_deterministic():
    rows = (
        [_mk(f"b{i}", "apical", "b_mode", 0.5) for i in range(10)]
        + [_mk(f"c{i}", "parasternal_long", "color_doppler", 0.5) for i in range(5)]
    )
    a = sample_mixed_modality(rows, K=8, rng=random.Random(42))
    b = sample_mixed_modality(rows, K=8, rng=random.Random(42))
    assert [r["clip_id"] for r in a] == [r["clip_id"] for r in b]
