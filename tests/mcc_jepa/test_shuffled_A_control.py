"""Shuffled-A control: build_pair_manifest with shuffle_source=True draws
clip_A from a different study. Used for the anti-hallucination diagnostic."""

from __future__ import annotations

import pandas as pd

from src.datasets.mcc_pair_dataset import build_pair_manifest, sampler_diagnostics


def _df():
    rows = []
    for s in range(3):
        for i in range(4):
            rows.append(dict(study_id=f"s{s}", path=f"s{s}_c{i}.mp4", view="A4C", modality="bmode"))
    return pd.DataFrame(rows)


def test_shuffled_source_draws_from_different_study():
    df = _df()
    pair_df = build_pair_manifest(df, seed=5, shuffle_source=True)
    assert len(pair_df) == df["study_id"].nunique()
    # every row marked shuffled
    assert pair_df["shuffled_source"].all()
    # every clip_A comes from a different study than clip_B's study
    for _, row in pair_df.iterrows():
        a_study = df.loc[df["path"] == row["path_a"], "study_id"].iloc[0]
        assert a_study != row["study_id"]


def test_non_shuffled_rate_is_one_when_not_shuffled():
    df = _df()
    pair_df = build_pair_manifest(df, seed=5, shuffle_source=False)
    diag = sampler_diagnostics(pair_df)
    assert diag["pair_same_study_rate"] == 1.0


def test_shuffled_rate_is_zero_when_shuffled():
    df = _df()
    pair_df = build_pair_manifest(df, seed=5, shuffle_source=True)
    diag = sampler_diagnostics(pair_df)
    assert diag["pair_same_study_rate"] == 0.0
