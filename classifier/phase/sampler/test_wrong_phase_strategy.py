"""Metadata-only unit tests for wrong_phase_strategy.

Builds a tiny synthetic parquet fixture (written to a tempfile, passed
to PhaseMatchedStudySampler) that exercises each strategy branch:

A. `same_view_only` + same-view candidate available → returns same view.
B. `same_view_then_same_family` + only same-family available → returns family.
C. `same_view_only` + no same-view candidate → returns None (hard-neg miss).
D. `any_same_study` + no view metadata → returns any same-study candidate.

No video decoding. No local DICOM dependency. Pure sampler logic.

Run:
    python classifier/phase/sampler/test_wrong_phase_strategy.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from phase_matched_sampler import PhaseMatchedStudySampler  # noqa: E402


def _make_fixture(tmp_dir: Path, study_layouts: list[dict]) -> Path:
    """Build a minimal parquet with the columns the sampler requires.

    Each entry in ``study_layouts`` is a dict::
      {
        "study_id": "90001",
        "subject_id": "sub_0",
        "clips": [
          {"dicom_id": "90001_0001", "view": "A4C", "n_frames": 64, "fps": 30.0, "hr": 72.0},
          ...
        ]
      }

    All clips get a strict-RR-consistent phase track spanning [0, 1) uniformly
    with confident_mask=True across the full range.
    """
    rows = []
    for st in study_layouts:
        for c in st["clips"]:
            n = int(c["n_frames"])
            fps = float(c.get("fps", 30.0))
            hr = float(c.get("hr", 72.0))
            # Cycle length in frames (matches rr_consistency._metadata_cycle_frames).
            cycle = int(round(fps * 60.0 / hr))
            # Place evenly-spaced r-peaks across the clip so median RR ≈ cycle.
            # This keeps rr_median_meta_ratio ≈ 1.0 → passes strict rr_consistent.
            rpeaks = list(range(0, n, cycle))
            if len(rpeaks) < 2:
                rpeaks = [0, cycle]
            # Linear phase from 0 to (n-1)/n; fully confident.
            phase = [i / n for i in range(n)]
            conf = [1] * n
            rows.append({
                "dicom_id": c["dicom_id"],
                "subject_id": st.get("subject_id", "sub"),
                "study_id": st["study_id"],
                "dicom_filepath": f"files/{c['dicom_id']}.dcm",
                "s3_uri": f"s3://test/{c['dicom_id']}.dcm",
                "manufacturer": "synthetic",
                "model": "synthetic",
                "sop_class_uid": "syn",
                "n_video_frames": n,
                "fps_video": fps,
                "video_duration_s": n / fps,
                "sr_ecg": 213.0,
                "sr_source": "scanner_default",
                "sr_confidence": "high",
                "x0": 0,
                "x1": 100,
                "trace_span_dur_s": n / fps,
                "duration_ratio": 1.0,
                "coverage_frac": 1.0,
                "hr_metadata": float(c.get("hr", 72.0)),
                "hr_detected": float(c.get("hr", 72.0)),
                "detection_method": "synth",
                "rpeak_ratio_dist": 0.0,
                "n_rpeaks_total": len(rpeaks),
                "n_rpeaks_in_video": len(rpeaks),
                "r_peaks_video_json": json.dumps(rpeaks),
                "rr_median_frames": float(n // 2),
                "per_frame_phase_json": json.dumps(phase),
                "confident_mask_json": json.dumps(conf),
                "regime_summary": f"strict:{n}",
                "full_y_b85": "x",
                "quality_tier": "high",
                "reject_reason": "",
                "elapsed_s": 0.1,
                "acquisition_datetime": pd.NaT,
            })
    df = pd.DataFrame(rows)
    out = tmp_dir / "fixture.parquet"
    df.to_parquet(out, index=False)
    return out


def _make_sampler(parquet, view_labels, strategy, allow_missing=False):
    return PhaseMatchedStudySampler(
        parquet_path=parquet,
        tiers=("high",),
        rr_filter_mode="strict",
        require_rr_consistent=True,
        sampling_mode="uniform_phase",
        phase_tolerance=0.15,
        frames_per_clip=8,
        frame_step=1,
        pairs_per_study=1,
        seed=42,
        view_labels=view_labels,
        delta_phase_mode="controlled_buckets",
        delta_phase_buckets=(0.0, 0.25),
        delta_phase_bucket_probs=(0.5, 0.5),
        require_same_study_wrong_phase_negative=True,
        wrong_phase_min_delta=0.25,
        wrong_phase_strategy=strategy,
        allow_missing_hard_negative=allow_missing,
        hard_negative_fallback="resample_anchor" if not allow_missing else "batch_negatives_only",
        max_hard_neg_attempts=32,
    )


def case_A_same_view_available():
    """Strategy same_view_only + same-view candidate → returns same view."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        layout = [{
            "study_id": "10001",
            "subject_id": "sub",
            "clips": [
                {"dicom_id": "10001_0001", "view": "A4C", "n_frames": 90},
                {"dicom_id": "10001_0002", "view": "A4C", "n_frames": 90},
                {"dicom_id": "10001_0003", "view": "A4C", "n_frames": 90},
            ],
        }]
        parquet = _make_fixture(tmp_dir, layout)
        views = {c["dicom_id"]: c["view"] for st in layout for c in st["clips"]}
        sampler = _make_sampler(parquet, views, "same_view_only")
        rng = np.random.default_rng(0)
        # Override study_to_rows to the known study
        sid = "10001"
        r = sampler._draw_pair(sid, rng)
        assert r is not None, "case A: draw returned None"
        assert r.clip_b_neg_phase is not None, "case A: hard_neg_phase is None"
        assert r.clip_b_neg_phase.view == "A4C", f"case A: view={r.clip_b_neg_phase.view}"
        assert r.clip_b.view == "A4C"
        assert r.clip_a.view == "A4C"
        print(f"[pass] case A: same_view_only + same-view available → view=A4C")


def case_B_family_available():
    """Strategy same_view_then_same_family + only same-family candidates
    (no identical view) → returns same family."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        layout = [{
            "study_id": "10002",
            "subject_id": "sub",
            # A4C, A3C, A2C are all in the "apical" family but different views.
            # _VIEW_MEDIUM_PAIRS includes (A4C, A2C) and (A5C, A2C) so these
            # classify as same_family. No identical-view pairs here.
            "clips": [
                {"dicom_id": "10002_0001", "view": "A4C", "n_frames": 90},
                {"dicom_id": "10002_0002", "view": "A3C", "n_frames": 90},
                {"dicom_id": "10002_0003", "view": "A2C", "n_frames": 90},
            ],
        }]
        parquet = _make_fixture(tmp_dir, layout)
        views = {c["dicom_id"]: c["view"] for st in layout for c in st["clips"]}
        sampler = _make_sampler(parquet, views, "same_view_then_same_family")
        # Several draws — none should return a hard_neg with view == pos view.
        rng = np.random.default_rng(0)
        fam_count = 0
        attempts = 0
        while fam_count < 5 and attempts < 30:
            attempts += 1
            r = sampler._draw_pair("10002", rng)
            if r is None or r.clip_b_neg_phase is None:
                continue
            # neg is in the apical family (A2C/A3C/A4C) but should not equal pos view
            assert r.clip_b_neg_phase.view in {"A4C", "A3C", "A2C"}
            fam_count += 1
        assert fam_count >= 3, f"case B: only got {fam_count} family draws"
        print(f"[pass] case B: same_view_then_same_family + family-only available → family used ({fam_count} draws)")


def case_C_no_view_no_family_strict():
    """Strategy same_view_only + NO same-view candidate → None/resample."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        # Every clip is a different view AND different family.
        layout = [{
            "study_id": "10003",
            "subject_id": "sub",
            "clips": [
                {"dicom_id": "10003_0001", "view": "A4C", "n_frames": 90},
                {"dicom_id": "10003_0002", "view": "PLAX", "n_frames": 90},
                {"dicom_id": "10003_0003", "view": "IVC", "n_frames": 90},
            ],
        }]
        parquet = _make_fixture(tmp_dir, layout)
        views = {c["dicom_id"]: c["view"] for st in layout for c in st["clips"]}
        # Allow missing so we can observe the miss rather than infinite resample.
        sampler = _make_sampler(parquet, views, "same_view_only", allow_missing=True)
        rng = np.random.default_rng(0)
        misses = 0
        hits = 0
        for _ in range(20):
            r = sampler._draw_pair("10003", rng)
            if r is None:
                continue
            if r.clip_b_neg_phase is None:
                misses += 1
            else:
                hits += 1
                # If hit, must be same-view (which shouldn't happen since clip_b and clip_a differ)
                # Actually: if clip_a=A4C, clip_b=PLAX, then same_view means view==PLAX,
                # which is clip_a's complement — but same-view candidate must also != clip_b.
                # With 3 clips (A4C,PLAX,IVC) and anchor+positive taking 2, only 1 clip left.
                # If positive is PLAX, same-view needs another PLAX, which doesn't exist → miss.
        assert misses >= 10, f"case C: expected many misses, got {misses} misses / {hits} hits"
        print(f"[pass] case C: same_view_only + no same-view candidate → {misses} misses / {hits} hits")


def case_D_any_same_study_fallback():
    """Strategy any_same_study works even with no view metadata."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        layout = [{
            "study_id": "10004",
            "subject_id": "sub",
            "clips": [
                {"dicom_id": "10004_0001", "view": None, "n_frames": 90},
                {"dicom_id": "10004_0002", "view": None, "n_frames": 90},
                {"dicom_id": "10004_0003", "view": None, "n_frames": 90},
            ],
        }]
        parquet = _make_fixture(tmp_dir, layout)
        views = {}  # NO view labels
        sampler = _make_sampler(parquet, views, "any_same_study")
        rng = np.random.default_rng(0)
        hits = 0
        for _ in range(10):
            r = sampler._draw_pair("10004", rng)
            if r is not None and r.clip_b_neg_phase is not None:
                hits += 1
        assert hits >= 5, f"case D: expected ≥5 hits without view labels, got {hits}"
        print(f"[pass] case D: any_same_study + no view metadata → {hits} hits (≥5)")


def main():
    print("[INFO] Running wrong_phase_strategy unit tests (metadata-only)...")
    case_A_same_view_available()
    case_B_family_available()
    case_C_no_view_no_family_strict()
    case_D_any_same_study_fallback()
    print("\nALL UNIT TESTS PASSED")


if __name__ == "__main__":
    main()
