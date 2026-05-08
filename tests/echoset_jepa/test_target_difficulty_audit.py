"""Target-difficulty audit tests — synthetic c_clip cache + gate behaviour."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.echoset_jepa.target_difficulty_audit import (
    _cos,
    _element_vectors,
    _sample_mask,
    run_audit,
)


def _make_synth_cache(tmp_path: Path, n_studies: int = 10, d: int = 16):
    """Build a synthetic c_clip cache + manifests.

    Study layout: each study has 4 apical + 2 parasternal_long + 2 color clips.
    Apical and parasternal clips are distinct random vectors (different mean),
    so within-study same-view clips are close and across-study same-view are
    closer to each other than to across-view.
    """
    rng = np.random.RandomState(0)
    cache = tmp_path / "cclip"
    rows = []
    for sid_i in range(n_studies):
        sid = f"s{sid_i}"
        # Unique study signature
        study_sig = rng.randn(d) * 0.3
        for i, (view, mod, phase, count) in enumerate([
            ("apical", "b_mode", "full_cycle", 4),
            ("parasternal_long", "b_mode", "full_cycle", 2),
            ("apical", "color_doppler", "unknown", 2),
        ]):
            view_sig = np.zeros(d)
            view_sig[i % d] = 2.0   # distinct per-view direction
            for k in range(count):
                clip_id = f"{sid}_{view}_{mod}_{k}"
                vec = study_sig + view_sig + rng.randn(d) * 0.1
                path = cache / sid / f"{clip_id}.npy"
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, vec.astype(np.float32))
                rows.append(dict(
                    patient_id=f"p{sid_i // 2}",
                    study_id=sid, clip_id=clip_id,
                    view_family=view, modality=mod, phase_bucket=phase,
                    split="val" if sid_i < n_studies // 2 else "train",
                ))
    clip_df = pd.DataFrame(rows)
    elements = (clip_df.groupby(["study_id", "view_family", "modality", "phase_bucket"])
                       .size().reset_index(name="n_clips_in_element"))
    clip_path = tmp_path / "clip.parquet"
    elem_path = tmp_path / "elem.parquet"
    clip_df.to_parquet(clip_path, index=False)
    elements.to_parquet(elem_path, index=False)
    return clip_path, elem_path, cache


def test_element_vectors_groups_by_3tuple(tmp_path):
    clip_path, _, cache = _make_synth_cache(tmp_path, n_studies=4)
    df = pd.read_parquet(clip_path)
    rows = df[df.study_id == "s0"].to_dict("records")
    keys, mat, clip_ids = _element_vectors(rows, str(cache))
    # s0 has 3 distinct element keys
    assert len(keys) == 3
    assert mat.shape[0] == 3
    for cid_list in clip_ids:
        assert len(cid_list) >= 1


def test_sample_mask_invariants():
    import random
    rng = random.Random(0)
    for M in [2, 3, 5, 10]:
        ctx, tgt = _sample_mask(M, rng)
        assert len(ctx) >= 1
        assert len(tgt) >= 1
        assert len(ctx) + len(tgt) == M
        assert set(ctx).isdisjoint(tgt)


def test_cos_range():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert _cos(a, a) == pytest.approx(1.0)
    assert _cos(a, b) == pytest.approx(0.0, abs=1e-6)
    assert _cos(a, -a) == pytest.approx(-1.0)


def test_run_audit_emits_report(tmp_path):
    clip, elem, cache = _make_synth_cache(tmp_path, n_studies=8)
    out = tmp_path / "report"
    summary = run_audit(
        clip_manifest_path=str(clip),
        element_manifest_path=str(elem),
        cache_prefix=str(cache),
        out_dir=str(out),
        audit_split="val",
        proto_split="train",
        num_studies=100,
        seed=0,
    )
    assert (out / "target_difficulty.json").exists()
    assert (out / "target_difficulty.md").exists()
    assert (out / "target_difficulty_per_row.csv").exists()
    assert summary["n_rows"] > 0
    # Gate should pass on well-constructed synthetic data where same-study
    # different-view clips share a small study signature but targets are
    # view-distinct.
    assert "gate_passed" in summary


def test_run_audit_detects_easy_targets(tmp_path):
    """If we build a cache where every target IS literally a context element
    (zero-variance clips), nearest_context cosine should be ~1.0 and the
    gate should FAIL."""
    rng = np.random.RandomState(42)
    cache = tmp_path / "cclip"
    rows = []
    for sid_i in range(20):
        sid = f"s{sid_i}"
        # Every clip is the same vector (duplicate target + context)
        clip_vec = rng.randn(8).astype(np.float32)
        for i, (view, mod) in enumerate([("apical", "b_mode"), ("parasternal_long", "b_mode"),
                                         ("parasternal_short", "b_mode"), ("subcostal", "b_mode")]):
            cid = f"{sid}_{view}"
            path = cache / sid / f"{cid}.npy"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, clip_vec)
            rows.append(dict(
                patient_id=f"p{sid_i}", study_id=sid, clip_id=cid,
                view_family=view, modality=mod, phase_bucket="full_cycle",
                split="val" if sid_i < 10 else "train",
            ))
    clip_df = pd.DataFrame(rows)
    elements = (clip_df.groupby(["study_id", "view_family", "modality", "phase_bucket"])
                       .size().reset_index(name="n_clips_in_element"))
    clip_df.to_parquet(tmp_path / "clip.parquet", index=False)
    elements.to_parquet(tmp_path / "elem.parquet", index=False)

    summary = run_audit(
        clip_manifest_path=str(tmp_path / "clip.parquet"),
        element_manifest_path=str(tmp_path / "elem.parquet"),
        cache_prefix=str(cache),
        out_dir=str(tmp_path / "r"),
        num_studies=100,
        seed=0,
    )
    # Every target is identical to its context → frac_gt_0p9 should be 1.0
    assert summary["cos_nearest_context"]["frac_gt_0p9"] > 0.9
    assert summary["gate_passed"] is False
