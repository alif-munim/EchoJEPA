"""Coverage audit tests — synthetic manifests + gate behaviour."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from experiments.echoset_jepa.view_modality_coverage_audit import run_audit


def _synth_manifests(tmp_path: Path, n_studies: int = 30):
    """Build tiny synthetic manifests that mirror the MIMIC schema."""
    rows = []
    for sid in range(n_studies):
        patient = f"p{sid // 3}"
        study = f"s{sid}"
        # Every study gets 4 apical b_mode + 2 plax b_mode + 2 color
        for i in range(4):
            rows.append(dict(patient_id=patient, study_id=study, clip_id=f"{study}_a{i}",
                             view_family="apical", modality="b_mode",
                             phase_bucket="full_cycle", measurement_site="none",
                             view_label="A4C", view_conf=0.9,
                             n_frames=60, clip_duration_s=2.0,
                             quality_score=0.7, split="train"))
        for i in range(2):
            rows.append(dict(patient_id=patient, study_id=study, clip_id=f"{study}_p{i}",
                             view_family="parasternal_long", modality="b_mode",
                             phase_bucket="full_cycle", measurement_site="none",
                             view_label="PLAX", view_conf=0.9,
                             n_frames=60, clip_duration_s=2.0,
                             quality_score=0.7, split="train"))
        for i in range(2):
            rows.append(dict(patient_id=patient, study_id=study, clip_id=f"{study}_c{i}",
                             view_family="apical", modality="color_doppler",
                             phase_bucket="unknown", measurement_site="unknown",
                             view_label="A4C", view_conf=0.9,
                             n_frames=60, clip_duration_s=2.0,
                             quality_score=0.7, split="train"))
    clip_df = pd.DataFrame(rows)
    clip_path = tmp_path / "clip.parquet"
    clip_df.to_parquet(clip_path, index=False)

    # Elements: for each study, group by (vf, mod, phase)
    elements = (clip_df.groupby(["study_id", "view_family", "modality", "phase_bucket"])
                       .size().reset_index(name="n_clips_in_element"))
    elements["patient_id"] = elements["study_id"].map(
        clip_df.set_index("study_id")["patient_id"].to_dict()
    )
    elem_path = tmp_path / "elem.parquet"
    elements.to_parquet(elem_path, index=False)

    # K=8 sample: pick 4 b_mode apical + 2 b_mode plax + 2 color
    k_rows = []
    for sid in range(n_studies):
        study = f"s{sid}"
        for i in range(4):
            k_rows.append(dict(study_id=study, clip_id=f"{study}_a{i}",
                               view_family="apical", modality="b_mode",
                               phase_bucket="full_cycle", slot=i))
        for i in range(2):
            k_rows.append(dict(study_id=study, clip_id=f"{study}_p{i}",
                               view_family="parasternal_long", modality="b_mode",
                               phase_bucket="full_cycle", slot=4+i))
        for i in range(2):
            k_rows.append(dict(study_id=study, clip_id=f"{study}_c{i}",
                               view_family="apical", modality="color_doppler",
                               phase_bucket="unknown", slot=6+i))
    k_df = pd.DataFrame(k_rows)
    k_path = tmp_path / "k.parquet"
    k_df.to_parquet(k_path, index=False)
    return clip_path, elem_path, k_path


def test_audit_emits_expected_files(tmp_path):
    clip, elem, k = _synth_manifests(tmp_path)
    out = tmp_path / "report"
    summary = run_audit(str(clip), str(elem), str(k), str(out))
    assert (out / "coverage_audit.json").exists()
    assert (out / "coverage_audit.md").exists()
    assert summary["overall"]["n_studies"] == 30
    assert summary["overall"]["n_clips"] == 30 * 8


def test_audit_gate_passes_on_diverse_synth(tmp_path):
    clip, elem, k = _synth_manifests(tmp_path)
    summary = run_audit(str(clip), str(elem), str(k), str(tmp_path / "r"))
    # Every study has apical + parasternal_long + color → should pass
    assert summary["gate_passed"] is True
    assert summary["gates"]["frac_studies_ge2_view_families_in_K8"] == 1.0
    assert summary["gates"]["color_retention_in_K8"] == 1.0


def test_audit_gate_fails_on_single_view_cohort(tmp_path):
    # Build a cohort where every study has ONLY apical b_mode clips
    rows = []
    for sid in range(20):
        for i in range(8):
            rows.append(dict(patient_id=f"p{sid}", study_id=f"s{sid}",
                             clip_id=f"s{sid}_{i}", view_family="apical",
                             modality="b_mode", phase_bucket="full_cycle",
                             view_label="A4C", view_conf=0.9,
                             n_frames=60, clip_duration_s=2.0,
                             measurement_site="none", quality_score=0.7, split="train"))
    clip_df = pd.DataFrame(rows)
    clip_path = tmp_path / "clip.parquet"
    clip_df.to_parquet(clip_path, index=False)

    elements = (clip_df.groupby(["study_id", "view_family", "modality", "phase_bucket"])
                       .size().reset_index(name="n_clips_in_element"))
    elements["patient_id"] = elements["study_id"].map(
        clip_df.set_index("study_id")["patient_id"].to_dict()
    )
    elem_path = tmp_path / "elem.parquet"
    elements.to_parquet(elem_path, index=False)
    k_path = tmp_path / "k.parquet"
    clip_df.assign(slot=clip_df.groupby("study_id").cumcount()).to_parquet(k_path, index=False)

    summary = run_audit(str(clip_path), str(elem_path), str(k_path), str(tmp_path / "r"))
    # Single view family everywhere → frac_ge2 = 0 → gate fails
    assert summary["gate_passed"] is False
    assert summary["gates"]["frac_studies_ge2_view_families_in_K8"] == 0.0
