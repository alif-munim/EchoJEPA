"""Element manifest build tests (plan §3.1, PR-N1)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from experiments.echoset_jepa.element_grouping import build_element_manifest


def _fake_clip_manifest(tmp_path: Path) -> Path:
    rows = []
    # Study A: 3 A4C B-mode systolic + 1 PLAX B-mode + 1 A4C color
    for i in range(3):
        rows.append(dict(
            patient_id="p1", study_id="sA", clip_id=f"cA{i}",
            view_family="apical", modality="b_mode", phase_bucket="systolic",
            measurement_site="none", quality_score=0.7,
        ))
    rows.append(dict(
        patient_id="p1", study_id="sA", clip_id="cA3",
        view_family="parasternal_long", modality="b_mode", phase_bucket="full_cycle",
        measurement_site="none", quality_score=0.8,
    ))
    rows.append(dict(
        patient_id="p1", study_id="sA", clip_id="cA4",
        view_family="apical", modality="color_doppler", phase_bucket="unknown",
        measurement_site="unknown", quality_score=0.6,
    ))
    # Study B: 1 clip
    rows.append(dict(
        patient_id="p2", study_id="sB", clip_id="cB0",
        view_family="parasternal_short", modality="b_mode", phase_bucket="full_cycle",
        measurement_site="none", quality_score=0.5,
    ))
    df = pd.DataFrame(rows)
    # Fill in the schema the loader expects
    for col in ["s3_uri", "view_label", "view_conf", "phase_label", "phase_conf",
                "quality_proxy_version", "frame_rate_hz", "clip_duration_s", "n_frames",
                "pixel_spacing_cm_per_px", "acquisition_ts", "site_id", "vendor",
                "cached_cclip_s3", "n_duplicates", "is_duplicate_of", "dicom_series_uid"]:
        if col not in df.columns:
            df[col] = ""
    p = tmp_path / "clip_manifest.parquet"
    df.to_parquet(p, index=False)
    return p


def test_element_manifest_groups_by_3tuple_key(tmp_path):
    clip_manifest = _fake_clip_manifest(tmp_path)
    out = tmp_path / "element_manifest.parquet"
    build_element_manifest(str(clip_manifest), str(out))
    em = pd.read_parquet(out)

    # Study A should have 3 elements: (apical, b_mode, systolic),
    # (parasternal_long, b_mode, full_cycle), (apical, color_doppler, unknown)
    sA = em[em.study_id == "sA"]
    assert len(sA) == 3, sA[["view_family","modality","phase_bucket"]].to_string()

    # The apical b_mode systolic element should have n_clips_in_element == 3
    row = sA[(sA.view_family=="apical") & (sA.modality=="b_mode") & (sA.phase_bucket=="systolic")].iloc[0]
    assert row["n_clips_in_element"] == 3
    assert sorted(row["clip_ids"]) == ["cA0","cA1","cA2"]


def test_element_manifest_quality_not_in_key(tmp_path):
    """Two apical-b_mode-systolic clips of different quality → ONE element."""
    rows = [
        dict(patient_id="p1", study_id="sZ", clip_id="c0",
             view_family="apical", modality="b_mode", phase_bucket="systolic",
             measurement_site="none", quality_score=0.2),
        dict(patient_id="p1", study_id="sZ", clip_id="c1",
             view_family="apical", modality="b_mode", phase_bucket="systolic",
             measurement_site="none", quality_score=0.95),
    ]
    df = pd.DataFrame(rows)
    p = tmp_path / "cm.parquet"
    df.to_parquet(p, index=False)
    out = tmp_path / "em.parquet"
    build_element_manifest(str(p), str(out))
    em = pd.read_parquet(out)
    assert len(em) == 1, "quality bucket leaked into element key"
    assert em.iloc[0]["n_clips_in_element"] == 2


def test_element_manifest_measurement_site_aggregated(tmp_path):
    rows = [
        dict(patient_id="p1", study_id="sX", clip_id="c0",
             view_family="apical", modality="cw_doppler", phase_bucket="not_applicable",
             measurement_site="TR", quality_score=0.7),
        dict(patient_id="p1", study_id="sX", clip_id="c1",
             view_family="apical", modality="cw_doppler", phase_bucket="not_applicable",
             measurement_site="LVOT", quality_score=0.7),
    ]
    df = pd.DataFrame(rows)
    p = tmp_path / "cm.parquet"
    df.to_parquet(p, index=False)
    out = tmp_path / "em.parquet"
    build_element_manifest(str(p), str(out))
    em = pd.read_parquet(out)
    # One element by the 3-tuple key, but both TR and LVOT sites preserved
    assert len(em) == 1
    assert set(em.iloc[0]["measurement_sites"]) == {"TR", "LVOT"}


def test_element_manifest_cap_max_M(tmp_path):
    rows = []
    # Make 80 distinct elements in one study
    views = ["apical","parasternal_long","parasternal_short","subcostal","suprasternal"]
    mods = ["b_mode","color_doppler","cw_doppler","pw_doppler"]
    phases = ["systolic","diastolic","full_cycle","not_applicable"]
    seen = set()
    for v in views:
        for m in mods:
            for p in phases:
                if (v,m,p) in seen:
                    continue
                seen.add((v,m,p))
                rows.append(dict(patient_id="p1", study_id="sY", clip_id=f"c_{v}_{m}_{p}",
                                 view_family=v, modality=m, phase_bucket=p,
                                 measurement_site="none", quality_score=0.6))
    df = pd.DataFrame(rows)
    assert len(df) >= 30
    p_path = tmp_path / "cm.parquet"
    df.to_parquet(p_path, index=False)
    out = tmp_path / "em.parquet"
    build_element_manifest(str(p_path), str(out), max_M=8)
    em = pd.read_parquet(out)
    assert len(em) <= 8
    # Diversity-preserving: cap prioritizes unique (view_family, modality) pairs.
    # We should see multiple distinct pairs, and multiple modalities.
    pairs = set(zip(em.view_family, em.modality))
    assert len(pairs) == len(em), "cap failed to diversify across (view, modality) pairs"
    assert em.modality.nunique() >= 3, f"expected >=3 modalities, got {em.modality.nunique()}"
