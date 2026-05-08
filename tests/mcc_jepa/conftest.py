"""Shared fixtures for MCC-JEPA tests."""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def synth_clip_manifest():
    """Flat 6-study manifest with 8 clips per study covering view families
    and modalities, plus one single-clip study for fallback coverage."""
    views = ["A4C", "A2C", "A3C", "A5C", "PLAX", "PSAX-MV", "PSAX-AV", "A4C"]
    mods = ["bmode"] * 7 + ["color"]
    rows = []
    for s in range(6):
        for i in range(8):
            rows.append(
                dict(
                    study_id=f"s{s}",
                    path=f"s{s}_c{i}.mp4",
                    view=views[i],
                    modality=mods[i],
                )
            )
    rows.append(dict(study_id="solo", path="solo_only.mp4", view="A4C", modality="bmode"))
    return pd.DataFrame(rows)
