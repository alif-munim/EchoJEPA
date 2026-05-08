"""Taxonomy tests — view_family / modality / measurement_site / phase_bucket."""

from __future__ import annotations

import pytest

from experiments.echoset_jepa.taxonomy import (
    MODALITIES,
    VIEW_FAMILIES,
    infer_measurement_site,
    is_excluded_view,
    normalize_modality,
    normalize_phase_bucket,
    normalize_view_family,
)


# ---- view_family ----------------------------------------------------------

@pytest.mark.parametrize(
    "label,expected",
    [
        ("A4C", "apical"),
        ("a3c", "apical"),
        ("PLAX", "parasternal_long"),
        ("PSAX-MV", "parasternal_short"),
        ("PSAX-AV", "parasternal_short"),
        ("SUBCOSTAL", "subcostal"),
        ("SSN", "suprasternal"),
        ("RV-FOCUSED", "rv_focused"),
        ("UNKNOWN_GARBAGE_LABEL", "unknown"),
        ("", "unknown"),
        (None, "unknown"),
    ],
)
def test_normalize_view_family_basic(label, expected):
    assert normalize_view_family(label) == expected


def test_doppler_not_in_view_family():
    # The v2 plan put these in view_family; v3 says they're modalities.
    # Downstream code should route them to 'unknown' unless a separate
    # anatomic view label exists.
    assert normalize_view_family("DOPPLER_SPECTRAL") == "unknown"
    assert normalize_view_family("MMODE") == "unknown"
    assert normalize_view_family("TDI") == "unknown"


def test_view_family_low_confidence_becomes_unknown():
    assert normalize_view_family("A4C", view_conf=0.3) == "unknown"
    assert normalize_view_family("A4C", view_conf=0.5) == "apical"   # mid-range: use label
    assert normalize_view_family("A4C", view_conf=0.9) == "apical"


def test_excluded_views_detected():
    assert is_excluded_view("Exclude")
    assert is_excluded_view("EXCLUDE")
    assert is_excluded_view("TEE")
    assert not is_excluded_view("A4C")
    assert not is_excluded_view(None)


def test_view_families_vocab_complete():
    """Every canonical view_family value exists in VIEW_FAMILIES."""
    assert "apical" in VIEW_FAMILIES
    assert "rv_focused" in VIEW_FAMILIES
    assert "unknown" in VIEW_FAMILIES
    # doppler_spectral / m_mode / tdi must NOT be in view_families
    assert "doppler_spectral" not in VIEW_FAMILIES
    assert "m_mode" not in VIEW_FAMILIES
    assert "tdi" not in VIEW_FAMILIES


# ---- modality -------------------------------------------------------------

def test_normalize_modality_from_color_flag():
    assert normalize_modality(color_flag="Yes") == "color_doppler"
    assert normalize_modality(color_flag="No") == "b_mode"
    assert normalize_modality(color_flag="yes") == "color_doppler"


def test_normalize_modality_filename_hints():
    assert normalize_modality(filename="foo_mmode_bar.mp4") == "m_mode"
    assert normalize_modality(filename="foo_tdi.mp4") == "tdi"
    assert normalize_modality(filename="foo_cwdoppler.mp4") == "cw_doppler"
    assert normalize_modality(filename="foo_pwdoppler.mp4") == "pw_doppler"


def test_normalize_modality_raw_override():
    assert normalize_modality(raw_modality="cw_doppler", color_flag="No") == "cw_doppler"


def test_normalize_modality_default_b_mode():
    assert normalize_modality() == "b_mode"
    assert normalize_modality(color_flag="unknown_value") == "b_mode"


def test_modalities_vocab_includes_all():
    for m in ("b_mode", "color_doppler", "pw_doppler", "cw_doppler", "m_mode", "tdi", "contrast", "unknown"):
        assert m in MODALITIES


# ---- measurement_site -----------------------------------------------------

def test_measurement_site_b_mode_is_none():
    assert infer_measurement_site(modality="b_mode") == "none"


def test_measurement_site_cw_doppler_default_unknown():
    assert infer_measurement_site(modality="cw_doppler") == "unknown"


def test_measurement_site_filename_cues():
    assert infer_measurement_site(modality="cw_doppler", filename="foo_tr_peak.mp4") == "TR"
    assert infer_measurement_site(modality="cw_doppler", filename="foo_lvot.mp4") == "LVOT"
    assert infer_measurement_site(modality="pw_doppler", filename="foo_mv_inflow.mp4") == "MV_inflow"


# ---- phase_bucket ---------------------------------------------------------

def test_phase_bucket_doppler_is_not_applicable():
    assert normalize_phase_bucket("systolic", modality="cw_doppler") == "not_applicable"
    assert normalize_phase_bucket("whatever", modality="pw_doppler") == "not_applicable"
    assert normalize_phase_bucket(None, modality="m_mode") == "not_applicable"


def test_phase_bucket_bmode_uses_label():
    assert normalize_phase_bucket("systolic", "b_mode") == "systolic"
    assert normalize_phase_bucket("DIASTOLIC", "b_mode") == "diastolic"
    assert normalize_phase_bucket("full_cycle", "b_mode") == "full_cycle"


def test_phase_bucket_bmode_missing_is_unknown():
    assert normalize_phase_bucket(None, "b_mode") == "unknown"
    assert normalize_phase_bucket("garbage", "b_mode") == "unknown"


def test_phase_bucket_color_follows_bmode_rules():
    # Color Doppler may have phase labels if classifier assigns them.
    assert normalize_phase_bucket("systolic", "color_doppler") == "systolic"
    assert normalize_phase_bucket(None, "color_doppler") == "unknown"
