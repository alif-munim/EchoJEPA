"""Shared fixtures for EchoMV-JEPA tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.meta_embeddings import MetaEmbeddings


@pytest.fixture
def synth_cache(tmp_path):
    """Small synthetic study manifest + cached c_clip vectors on local disk."""
    cache = tmp_path / "cclip"
    rng = np.random.RandomState(0)
    rows = []
    for sid_i in range(6):
        sid = f"s{sid_i}"
        for i, (view, mod) in enumerate(
            [
                ("apical", "b_mode"),
                ("parasternal_long", "b_mode"),
                ("apical", "color_doppler"),
                ("parasternal_short", "b_mode"),
            ]
        ):
            cid = f"{sid}_c{i}"
            vec = rng.randn(32).astype(np.float32)
            path = cache / sid / f"{cid}.npy"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, vec)
            rows.append(
                dict(
                    study_id=sid,
                    clip_id=cid,
                    view_family=view,
                    modality=mod,
                    phase_bucket="full_cycle",
                    measurement_site="none",
                    quality_score=0.7,
                    cached_cclip_s3="",
                )
            )
    k_path = tmp_path / "k.parquet"
    pd.DataFrame(rows).to_parquet(k_path, index=False)
    return k_path, cache


@pytest.fixture
def meta32():
    return MetaEmbeddings(d_model=32)
