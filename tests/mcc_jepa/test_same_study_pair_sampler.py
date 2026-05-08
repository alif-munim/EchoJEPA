"""Target-anchored MCC-JEPA pair sampler returns same-study pairs."""

from __future__ import annotations

from src.datasets.mcc_pair_dataset import build_pair_manifest, sampler_diagnostics


def test_sampler_returns_same_study_pairs(synth_clip_manifest):
    pair_df = build_pair_manifest(synth_clip_manifest, seed=1)
    # one pair per study
    assert len(pair_df) == synth_clip_manifest["study_id"].nunique()

    # every pair within the same study when not shuffled
    for _, row in pair_df.iterrows():
        a_study = synth_clip_manifest.loc[synth_clip_manifest["path"] == row["path_a"], "study_id"].iloc[0]
        b_study = synth_clip_manifest.loc[synth_clip_manifest["path"] == row["path_b"], "study_id"].iloc[0]
        assert a_study == b_study == row["study_id"]


def test_distinct_clip_rate_above_threshold(synth_clip_manifest):
    pair_df = build_pair_manifest(synth_clip_manifest, seed=11)
    diag = sampler_diagnostics(pair_df)
    # 6 multi-clip studies + 1 single-clip → ≥ 6/7 distinct ≈ 0.857
    assert diag["pair_distinct_clip_rate"] >= 6 / 7 - 1e-6


def test_mixture_diagnostics_contain_buckets(synth_clip_manifest):
    pair_df = build_pair_manifest(synth_clip_manifest, seed=3)
    diag = sampler_diagnostics(pair_df)
    # Every MVP bucket key is present even if 0.
    for k in (
        "same_view_fraction",
        "same_broad_family_fraction",
        "cross_view_fraction",
        "cross_modality_fraction",
    ):
        assert k in diag
        assert 0.0 <= diag[k] <= 1.0
