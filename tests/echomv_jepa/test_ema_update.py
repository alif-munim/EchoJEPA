"""EMA update correctness for StudyTransformerEMA."""

from __future__ import annotations

import copy

import torch

from src.models.echomv_jepa import StudyTransformerEMA, ema_update_
from src.models.study_transformer import StudyTransformer, StudyTransformerConfig


def _make_student(d_clip=16, d_model=16):
    return StudyTransformer(
        StudyTransformerConfig(
            d_clip=d_clip,
            d_model=d_model,
            n_layers=1,
            n_heads=2,
            ffn_mult=2,
            dropout_ffn=0.0,
            dropout_attn=0.0,
            max_M=4,
        )
    )


def test_ema_update_math():
    s = _make_student()
    t = copy.deepcopy(s)
    for p in t.parameters():
        p.requires_grad_(False)

    # Snapshot originals
    s_orig = [p.detach().clone() for p in s.parameters()]
    t_orig = [p.detach().clone() for p in t.parameters()]

    # Change the student by a known delta
    with torch.no_grad():
        for p in s.parameters():
            p.add_(torch.ones_like(p))

    ema_update_(t, s, tau=0.5)
    # teacher <- 0.5 * t_orig + 0.5 * (s_orig + 1)
    for p_t, po_t, po_s in zip(t.parameters(), t_orig, s_orig):
        expected = 0.5 * po_t + 0.5 * (po_s + 1.0)
        assert torch.allclose(p_t.detach(), expected, atol=1e-6)


def test_ema_tau_one_is_noop():
    s = _make_student()
    teacher = StudyTransformerEMA(s)
    before = [p.detach().clone() for p in teacher.teacher.parameters()]
    teacher.update_teacher(s, tau=1.0)
    for p, b in zip(teacher.teacher.parameters(), before):
        assert torch.allclose(p.detach(), b, atol=1e-7)


def test_ema_tau_zero_copies_student():
    s = _make_student()
    teacher = StudyTransformerEMA(s)
    # Move the student away from its initialization
    with torch.no_grad():
        for p in s.parameters():
            p.add_(torch.randn_like(p) * 0.5)
    teacher.update_teacher(s, tau=0.0)
    for pt, ps in zip(teacher.teacher.parameters(), s.parameters()):
        assert torch.allclose(pt.detach(), ps.detach(), atol=1e-7)


def test_teacher_parameters_require_grad_false_after_update():
    s = _make_student()
    teacher = StudyTransformerEMA(s)
    teacher.update_teacher(s, tau=0.5)
    assert all(not p.requires_grad for p in teacher.teacher.parameters())
