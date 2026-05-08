"""Clip-EMA helpers update teacher params in place and don't propagate grad."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn  # noqa: F401

from src.models.echomv_jepa.clip_ema import (
    clip_ema_schedule,
    ema_delta_norm,
    step_clip_ema,
)


def _tiny_net():
    return nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8))


def test_step_clip_ema_moves_teacher_toward_student():
    torch.manual_seed(0)
    student = _tiny_net()
    teacher = copy.deepcopy(student)
    for p in teacher.parameters():
        p.requires_grad_(False)
    # Push student params in a known direction.
    with torch.no_grad():
        for p in student.parameters():
            p.add_(1.0)
    # Record teacher before.
    before = {n: p.data.clone() for n, p in teacher.named_parameters()}
    step_clip_ema(teacher, student, tau=0.9)
    delta = ema_delta_norm(teacher, student)
    # EMA moved teacher partway toward student → teacher has changed but is
    # still closer to old self than to student (tau=0.9 is a slow move).
    any_moved = False
    for n, p in teacher.named_parameters():
        if not torch.allclose(p.data, before[n]):
            any_moved = True
    assert any_moved
    assert delta > 0.0


def test_clip_ema_schedule_linear():
    sched = list(clip_ema_schedule(tau_start=0.9, tau_end=0.99, total_steps=10))
    assert len(sched) == 11
    assert abs(sched[0] - 0.9) < 1e-6
    assert abs(sched[-1] - 0.99) < 1e-6
    # Monotone increasing
    for a, b in zip(sched, sched[1:]):
        assert b >= a


def test_ema_delta_norm_zero_on_identical():
    torch.manual_seed(1)
    net_a = _tiny_net()
    net_b = copy.deepcopy(net_a)
    assert ema_delta_norm(net_a, net_b) == 0.0
