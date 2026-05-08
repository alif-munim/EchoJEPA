"""Teacher encoder must not receive gradients through the V-JEPA loss."""

from __future__ import annotations

import torch

from src.models.echomv_jepa.full_joint_losses import clip_vjepa_true_loss
from tests.echomv_jepa.test_full_joint_true_clip_vjepa_shapes import ToyEncoder, ToyPredictor, _make_batch


def test_teacher_encoder_has_no_grad_after_backward():
    torch.manual_seed(0)
    D = 32
    encoder = ToyEncoder(D)
    target = ToyEncoder(D)
    predictor = ToyPredictor(D)
    # Simulate the real contract: target encoder has requires_grad=False.
    for p in target.parameters():
        p.requires_grad_(False)
    clips, m_enc, m_pred = _make_batch()
    loss = clip_vjepa_true_loss(clips, encoder, target, predictor, m_enc, m_pred)
    loss.backward()
    # Every teacher param stays with grad=None.
    for p in target.parameters():
        assert p.grad is None, "teacher param has a grad — breaks EMA semantics"
