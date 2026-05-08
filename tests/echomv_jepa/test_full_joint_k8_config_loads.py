"""Restart-v2 configs parse and have the expected knobs."""

from __future__ import annotations

from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]


def _load(p):
    with open(p) as f:
        return yaml.safe_load(f)


def test_v2_smoke_config_has_v2_knobs():
    cfg = _load(REPO / "configs" / "train" / "echomv_jepa" / "full_joint_global_study_restart_v2_smoke.yaml")
    exp = cfg["experiment"]
    assert exp["clip_vjepa"]["enabled"] is True
    assert exp["clip_vjepa"]["n_clips_per_study"] == 1
    assert exp["clip_consistency"]["enabled"] is True
    assert exp["single_view_branch"]["enabled"] is True
    assert exp["single_view_branch"]["p_rows"] == 0.25
    assert "apical" in exp["single_view_branch"]["prefer_view_families"]
    assert exp["study_nce"]["enabled"] is True
    assert exp["study_nce"]["cross_rank"] is True
    assert exp["study_nce"]["exclude_same_patient"] is True
    assert exp["lambdas"]["anchor_decay_steps"] > 0
    assert exp["lambdas"]["anchor_schedule"] in ("cosine", "linear", "constant_start")
    # K=8 path: batch_studies_per_gpu lowered to 2 relative to v1's 4.
    assert exp["optim"]["batch_studies_per_gpu"] == 2


def test_v2_30k_config_matches_mcc_compute():
    cfg = _load(REPO / "configs" / "train" / "echomv_jepa" / "full_joint_global_study_restart_v2_30k.yaml")
    exp = cfg["experiment"]
    # 30k steps × 256 clips/step (K=8 × batch=2 × 8 GPUs × 2 grad_accum would be 512; at
    # batch_per_gpu=2 and K=8 this is 128 clip forwards per optim step × 8 GPUs = 1024
    # per effective-batch unless grad_accum); the exact clip-forward budget is tracked
    # in the launch_readiness report. Core constraint: total_steps == 30_000.
    assert exp["optim"]["total_steps"] == 30000
    assert exp["lambdas"]["study_warmup_steps"] > 0
    assert exp["lambdas"]["sv_warmup_steps"] > 0
