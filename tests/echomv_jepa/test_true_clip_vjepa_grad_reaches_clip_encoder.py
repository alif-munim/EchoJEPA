"""Gradients from the true V-JEPA loss must reach f_theta (student)
AND the predictor. Otherwise the clip encoder is not getting
self-supervision."""

from __future__ import annotations

import torch

from src.models.echomv_jepa.full_joint_losses import clip_vjepa_true_loss
from tests.echomv_jepa.test_full_joint_true_clip_vjepa_shapes import ToyEncoder, ToyPredictor, _make_batch


def test_student_encoder_has_nonzero_grad():
    torch.manual_seed(0)
    D = 32
    encoder = ToyEncoder(D)
    target = ToyEncoder(D)
    predictor = ToyPredictor(D)
    for p in target.parameters():
        p.requires_grad_(False)
    clips, m_enc, m_pred = _make_batch()
    loss = clip_vjepa_true_loss(clips, encoder, target, predictor, m_enc, m_pred)
    loss.backward()
    any_nonzero = any((p.grad is not None and p.grad.abs().sum().item() > 0.0) for p in encoder.parameters())
    assert any_nonzero, "student encoder received no gradient — broken forward"


def test_predictor_has_nonzero_grad():
    torch.manual_seed(1)
    D = 32
    encoder = ToyEncoder(D)
    target = ToyEncoder(D)
    predictor = ToyPredictor(D)
    for p in target.parameters():
        p.requires_grad_(False)
    clips, m_enc, m_pred = _make_batch()
    loss = clip_vjepa_true_loss(clips, encoder, target, predictor, m_enc, m_pred)
    loss.backward()
    any_nonzero = any((p.grad is not None and p.grad.abs().sum().item() > 0.0) for p in predictor.parameters())
    assert any_nonzero, "predictor received no gradient — broken forward"
