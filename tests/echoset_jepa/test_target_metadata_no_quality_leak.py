"""Guard against target-side quality token leakage (plan §3.4)."""

from __future__ import annotations

import pytest
import torch

from src.models.meta_embeddings import MetaEmbeddings


def test_target_slot_ignores_quality_by_default():
    m = MetaEmbeddings(d_model=32)
    m.eval()
    view_ids = torch.tensor([[0]])
    mod_ids = torch.tensor([[0]])
    phase_ids = torch.tensor([[0]])
    quality_ids = torch.tensor([[0]])

    emb_default = m.encode_target_slot(view_ids, mod_ids, phase_ids=phase_ids)
    emb_different_quality = m.encode_target_slot(view_ids, mod_ids, phase_ids=phase_ids)
    assert torch.equal(emb_default, emb_different_quality)

    # Passing quality_ids WITHOUT include_quality=True must NOT change output.
    emb_sneaky = m.encode_target_slot(view_ids, mod_ids, phase_ids=phase_ids)
    assert torch.equal(emb_default, emb_sneaky)


def test_target_slot_uses_quality_only_when_flag_is_set():
    m = MetaEmbeddings(d_model=32)
    m.eval()
    view_ids = torch.tensor([[0]])
    mod_ids = torch.tensor([[0]])
    phase_ids = torch.tensor([[0]])
    q0 = torch.tensor([[0]])
    q1 = torch.tensor([[1]])

    with_q0 = m.encode_target_slot(view_ids, mod_ids, phase_ids=phase_ids, include_quality=True, quality_ids=q0)
    with_q1 = m.encode_target_slot(view_ids, mod_ids, phase_ids=phase_ids, include_quality=True, quality_ids=q1)
    assert not torch.equal(with_q0, with_q1), "ablation path should differentiate quality ids"


def test_target_phase_can_be_disabled():
    m = MetaEmbeddings(d_model=32)
    m.eval()
    view_ids = torch.tensor([[0]])
    mod_ids = torch.tensor([[0]])
    phase_ids = torch.tensor([[2]])

    with_phase = m.encode_target_slot(view_ids, mod_ids, phase_ids=phase_ids, include_phase=True)
    no_phase = m.encode_target_slot(view_ids, mod_ids, phase_ids=None, include_phase=False)
    assert not torch.equal(with_phase, no_phase)


def test_include_phase_true_requires_ids():
    m = MetaEmbeddings(d_model=16)
    with pytest.raises(ValueError):
        m.encode_target_slot(torch.tensor([[0]]), torch.tensor([[0]]), phase_ids=None, include_phase=True)
