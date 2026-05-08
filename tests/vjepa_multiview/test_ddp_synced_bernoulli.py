"""Fix 6: DDP-synced per-step Bernoulli gate is identical across ranks.

Since the draw is seeded by ``(global_step, seed_salt)`` with a fresh
CPU generator on every call, two simulated ranks calling
``ddp_synced_bernoulli`` with the same arguments must produce the
same bool — regardless of the per-rank RNG state at the time of the
call.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from app.vjepa_multiview.train import ddp_synced_bernoulli  # noqa: E402


def test_gate_deterministic_same_step():
    for step in range(20):
        r1 = ddp_synced_bernoulli(p=0.5, global_step=step)
        r2 = ddp_synced_bernoulli(p=0.5, global_step=step)
        assert r1 == r2, f"step {step}: same call disagreed ({r1} vs {r2})"


def test_gate_differs_between_steps_for_mid_p():
    """With p=0.5 and deterministic seeding, the draws should NOT all
    be identical across steps — if they were, the Bernoulli is
    degenerate."""
    seen = set()
    for step in range(50):
        seen.add(ddp_synced_bernoulli(p=0.5, global_step=step))
    assert seen == {True, False}, f"gate is degenerate: {seen}"


def test_gate_rank_independence_under_simulated_drift():
    """Simulate two ranks whose local Python RNG states have drifted.
    Draws must still agree if seed args match."""
    for step in (1, 7, 42, 101):
        # Rank 0's local RNG
        torch.manual_seed(0xDEADBEEF)
        r0 = ddp_synced_bernoulli(p=0.3, global_step=step)
        # Rank 1's local RNG (different state)
        torch.manual_seed(0xA5A5A5A5)
        r1 = ddp_synced_bernoulli(p=0.3, global_step=step)
        assert r0 == r1, f"step {step}: rank drift produced different gate"


def test_p_zero_always_false():
    for step in range(100):
        assert ddp_synced_bernoulli(p=0.0, global_step=step) is False


def test_p_one_always_true():
    for step in range(100):
        assert ddp_synced_bernoulli(p=1.0, global_step=step) is True


def test_seed_salt_changes_draw():
    salt_a_trues = sum(ddp_synced_bernoulli(p=0.5, global_step=s, seed_salt=0x1111) for s in range(200))
    salt_b_trues = sum(ddp_synced_bernoulli(p=0.5, global_step=s, seed_salt=0x2222) for s in range(200))
    # The two seed salts should yield different draw sequences; the
    # fraction of Trues for p=0.5 should both be near 100 but the
    # per-step comparison should not be identical.
    diffs = sum(
        1
        for s in range(200)
        if ddp_synced_bernoulli(p=0.5, global_step=s, seed_salt=0x1111)
        != ddp_synced_bernoulli(p=0.5, global_step=s, seed_salt=0x2222)
    )
    assert diffs > 0, "seed_salt had no effect"
