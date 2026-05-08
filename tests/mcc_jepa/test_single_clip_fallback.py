"""Studies with only one eligible clip fall back to (clip, clip)."""

from __future__ import annotations

import pandas as pd

from src.datasets.mcc_pair_dataset import build_pair_manifest


def test_single_clip_study_falls_back_to_self_pair():
    df = pd.DataFrame(
        [
            dict(study_id="A", path="a1.mp4", view="A4C", modality="bmode"),
            dict(study_id="A", path="a2.mp4", view="A2C", modality="bmode"),
            dict(study_id="B", path="b1.mp4", view="A4C", modality="bmode"),
        ]
    )
    pair_df = build_pair_manifest(df, seed=0)
    solo = pair_df[pair_df["study_id"] == "B"].iloc[0]
    assert solo["path_a"] == solo["path_b"] == "b1.mp4"
    assert solo["fallback"]
    assert solo["bucket"] == "fallback_single_clip"

    multi = pair_df[pair_df["study_id"] == "A"].iloc[0]
    assert multi["path_a"] != multi["path_b"]
