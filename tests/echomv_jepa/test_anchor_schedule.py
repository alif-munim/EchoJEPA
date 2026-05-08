"""LossDecay (anchor schedule) — cosine + linear + constant_start."""

from __future__ import annotations

from src.models.echomv_jepa.full_joint_losses import LossDecay


def test_cosine_starts_at_start_weight():
    d = LossDecay(start_weight=0.05, final_weight=0.005, decay_steps=10000, schedule="cosine")
    assert abs(d.value_at(0) - 0.05) < 1e-9


def test_cosine_ends_at_final_weight():
    d = LossDecay(start_weight=0.05, final_weight=0.005, decay_steps=10000, schedule="cosine")
    assert abs(d.value_at(10000) - 0.005) < 1e-9
    assert abs(d.value_at(20000) - 0.005) < 1e-9  # saturates after decay_steps


def test_cosine_halfway_is_midpoint_by_value():
    d = LossDecay(start_weight=0.05, final_weight=0.005, decay_steps=10000, schedule="cosine")
    # cos(π/2)=0 → value = final + 0.5*(start-final)*(1+0) = final + 0.5*(start-final) = midpoint
    expected = 0.005 + 0.5 * (0.05 - 0.005)
    assert abs(d.value_at(5000) - expected) < 1e-9


def test_cosine_monotone_nonincreasing():
    d = LossDecay(start_weight=0.1, final_weight=0.01, decay_steps=1000, schedule="cosine")
    prev = d.value_at(0)
    for s in range(1, 1001, 50):
        curr = d.value_at(s)
        assert curr <= prev + 1e-9, f"step={s}: {curr} > {prev}"
        prev = curr


def test_linear_schedule():
    d = LossDecay(start_weight=0.05, final_weight=0.005, decay_steps=100, schedule="linear")
    # Halfway should be exactly the midpoint.
    assert abs(d.value_at(50) - (0.05 + 0.5 * (0.005 - 0.05))) < 1e-9
    assert abs(d.value_at(0) - 0.05) < 1e-9
    assert abs(d.value_at(100) - 0.005) < 1e-9


def test_constant_start_schedule():
    d = LossDecay(start_weight=0.05, final_weight=0.005, decay_steps=100, schedule="constant_start")
    for s in (0, 50, 100, 1000):
        assert d.value_at(s) == 0.05


def test_zero_decay_steps_returns_start():
    d = LossDecay(start_weight=0.05, final_weight=0.005, decay_steps=0, schedule="cosine")
    assert d.value_at(0) == 0.05
    assert d.value_at(1000) == 0.05
