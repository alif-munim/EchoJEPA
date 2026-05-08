"""Checkpoint round-trip: save new heads, reload into fresh instances.

Does not test target_encoder load through vit_encoder_multiclip (that
adapter is exercised by the existing factorized probe adapter tests);
this test confirms the new head state_dict schemas round-trip cleanly.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from app.vjepa_multiview.token_relational_head import (  # noqa: E402
    DeltaTargetProjector,
    MotionDeltaHead,
    TokenRelationalHead,
)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def test_roundtrip_token_rel_head():
    head1 = TokenRelationalHead(embed_dim=64, rel_dim=16, hidden_dim=32)
    head2 = TokenRelationalHead(embed_dim=64, rel_dim=16, hidden_dim=32)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "ckpt.pt"
        torch.save({"token_rel_head": head1.state_dict()}, p)
        ck = torch.load(p, map_location="cpu", weights_only=False)
        msg = head2.load_state_dict(ck["token_rel_head"], strict=True)
        assert not msg.missing_keys, f"missing keys: {msg.missing_keys}"
        assert not msg.unexpected_keys, f"unexpected keys: {msg.unexpected_keys}"


def test_roundtrip_motion_delta_head():
    h1 = MotionDeltaHead(embed_dim=64, delta_dim=16, hidden_dim=32)
    h2 = MotionDeltaHead(embed_dim=64, delta_dim=16, hidden_dim=32)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "ckpt.pt"
        torch.save({"motion_delta_head": h1.state_dict()}, p)
        ck = torch.load(p, map_location="cpu", weights_only=False)
        h2.load_state_dict(ck["motion_delta_head"], strict=True)


def test_roundtrip_delta_target_projector():
    p1 = DeltaTargetProjector(embed_dim=64, delta_dim=16, hidden_dim=32)
    p2 = DeltaTargetProjector(embed_dim=64, delta_dim=16, hidden_dim=32)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "ckpt.pt"
        torch.save({"delta_target_projector": p1.state_dict()}, p)
        ck = torch.load(p, map_location="cpu", weights_only=False)
        p2.load_state_dict(ck["delta_target_projector"], strict=True)


def test_loading_missing_keys_is_nonfatal_with_strict_false():
    """Probe loaders use strict=False; a vanilla encoder checkpoint must
    not crash when we try to load missing head keys with strict=False."""
    head = TokenRelationalHead(embed_dim=64, rel_dim=16, hidden_dim=32)
    empty_state = {}
    msg = head.load_state_dict(empty_state, strict=False)
    # All keys missing — expected. No unexpected keys.
    assert len(msg.missing_keys) > 0
    assert not msg.unexpected_keys
