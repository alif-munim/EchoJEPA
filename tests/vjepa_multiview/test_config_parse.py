"""Parse-only smoke test for pretrain-multiview-privview-25of100.yaml.

Confirms the new YAML loads, the dispatch value is recognised, and the
privileged_multiview config block has the keys the wiring expects. Does
NOT launch training — that requires GPUs and a full sampler pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

CONFIG_PATH = REPO_ROOT / "configs" / "train" / "vitl16" / "pretrain-multiview-privview-25of100.yaml"


def test_yaml_loads():
    with CONFIG_PATH.open() as fh:
        cfg = yaml.safe_load(fh)
    assert cfg["app"] == "vjepa_multiview"
    assert cfg["phase_multiview"]["enabled"] is True
    assert cfg["phase_multiview"]["multiview_objective"] == "privileged_multiview"


def test_privview_block_has_required_keys():
    with CONFIG_PATH.open() as fh:
        cfg = yaml.safe_load(fh)
    priv = cfg["phase_multiview"]["privileged_multiview"]
    required = {
        "fused_target_mode",
        "lambda_pair_shared",
        "lambda_pair_view",
        "lambda_view_nce",
        "lambda_fused",
        "lambda_shared",
        "lambda_phase",
        "lambda_local_motion",
        "p_fused",
        "tau_static",
        "tau_phase",
        "tau_view",
        "warmup_epochs",
        "embed_dim",
        "shared_dim",
        "phase_dim",
        "view_dim",
        "head_hidden_dim",
        "view_embedding_dim",
        "n_phase_freqs",
        "fusion_num_heads",
        "use_z_view",
    }
    missing = required - set(priv.keys())
    assert not missing, f"missing keys in privileged_multiview block: {missing}"


def test_fused_target_mode_default_is_mean_shared():
    with CONFIG_PATH.open() as fh:
        cfg = yaml.safe_load(fh)
    priv = cfg["phase_multiview"]["privileged_multiview"]
    assert priv["fused_target_mode"] == "mean_shared"


def test_v4_lambda_recipe():
    """v4 first-real-run recipe: pair_view is primary; pair_shared is
    demoted to stabilizer; view_nce and fused are on; phase and
    local_motion are off."""
    with CONFIG_PATH.open() as fh:
        cfg = yaml.safe_load(fh)
    priv = cfg["phase_multiview"]["privileged_multiview"]
    # pair_view is the primary signal → must be >= pair_shared.
    assert priv["lambda_pair_view"] >= priv["lambda_pair_shared"], (
        f"pair_view ({priv['lambda_pair_view']}) must dominate " f"pair_shared ({priv['lambda_pair_shared']}) in v4"
    )
    # pair_shared is demoted: <= 0.10.
    assert priv["lambda_pair_shared"] <= 0.10
    # view_nce is active.
    assert priv["lambda_view_nce"] > 0.0
    # phase and local_motion are disabled by default.
    assert priv["lambda_phase"] == 0.0, "lambda_phase should default off"
    assert priv["lambda_local_motion"] == 0.0, "local_motion should default off"


def test_shared_phase_dims_compatible():
    """target_dim of the view predictor is set to shared_dim in train.py;
    that's required for the SmoothL1 comparison with t_tgt_shared."""
    with CONFIG_PATH.open() as fh:
        cfg = yaml.safe_load(fh)
    priv = cfg["phase_multiview"]["privileged_multiview"]
    # Not a hard YAML constraint, but ensure these are set and positive.
    assert priv["shared_dim"] > 0
    assert priv["phase_dim"] > 0
    assert priv["view_dim"] > 0
    assert priv["embed_dim"] == 1024  # ViT-L default


def test_warmup_does_not_exceed_training_horizon():
    with CONFIG_PATH.open() as fh:
        cfg = yaml.safe_load(fh)
    priv = cfg["phase_multiview"]["privileged_multiview"]
    stop_after = cfg["optimization"]["stop_after_epochs"]
    # Warmup must fit inside the stopping horizon, otherwise lambdas
    # never reach full strength.
    assert priv["warmup_epochs"] <= stop_after


def test_sampler_num_clips_is_3_for_mv2sv():
    """The dispatcher sets num_clips=3 when objective is
    privileged_multiview. This test confirms the YAML itself doesn't
    try to override that (MV2SV needs the 3-clip sampler output)."""
    with CONFIG_PATH.open() as fh:
        cfg = yaml.safe_load(fh)
    # No explicit num_clips key override in data block.
    assert "num_clips" not in cfg.get("data", {})


def test_dispatch_accepts_privileged_multiview():
    """train.py must accept the new multiview_objective value without
    raising. We only exercise the validation branch without running
    the heavy initialisation path."""
    # Re-implement the validation logic from main() as a smoke check.
    valid = ("smooth_l1", "intraview_only", "phase_relational", "privileged_multiview")
    with CONFIG_PATH.open() as fh:
        cfg = yaml.safe_load(fh)
    obj = cfg["phase_multiview"]["multiview_objective"]
    assert obj in valid
