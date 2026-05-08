"""The full_joint trainer must read the pixel dataset, not the cached c_clip.

This test asserts the right imports exist and checks that the config
schema for full_joint does not reference ``cache_local_prefix`` /
``cache_s3_prefix`` (those belong to the pooled path).
"""

from __future__ import annotations

from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]


def test_train_full_joint_imports_pixel_dataset():
    text = (REPO / "app" / "echomv_jepa" / "train_full_joint.py").read_text()
    assert "from src.datasets.echomv_jepa_pixel_dataset import" in text
    assert "EchoMVJEPAPixelDataset" in text
    # Must NOT depend on the cached-element dataset.
    assert "from src.datasets.echomv_jepa_dataset" not in text


def test_full_joint_smoke_config_does_not_reference_cache_prefix():
    cfg_path = REPO / "configs" / "train" / "echomv_jepa" / "full_joint_global_study_smoke.yaml"
    if not cfg_path.exists():
        # Config written later in implementation; skip if not present yet.
        return
    cfg = yaml.safe_load(cfg_path.read_text())
    exp = cfg.get("experiment", {}) or {}
    clip = exp.get("clip_encoder", {}) or {}
    assert clip.get("source") == "online_trainable"
    assert "cache_local_prefix" not in clip
    assert "cache_s3_prefix" not in clip
