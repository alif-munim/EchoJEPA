"""Per-modality projector (Stage-1m) routing and EMA behavior."""

from __future__ import annotations

import torch

from src.models.echomv_jepa import ModalityProjectorPair


def test_num_modalities_one_acts_like_shared_projector():
    torch.manual_seed(0)
    mp = ModalityProjectorPair(num_modalities=1, d_model=8, d_hidden=16, d_proj=4)
    x = torch.randn(5, 8)
    ids = torch.zeros(5, dtype=torch.long)
    out_s = mp.student_forward(x, ids)
    out_t = mp.teacher_forward(x, ids)
    assert out_s.shape == (5, 4)
    assert out_t.shape == (5, 4)
    # At initialization, student and teacher are exact deep copies → identical
    assert torch.allclose(out_s, out_t, atol=1e-6)


def test_routing_matches_direct_pair_call():
    torch.manual_seed(1)
    mp = ModalityProjectorPair(num_modalities=3, d_model=8, d_hidden=16, d_proj=4)
    x = torch.randn(6, 8)
    ids = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.long)
    out = mp.student_forward(x, ids)
    # Check each row matches the direct forward of its pair
    for i, m in enumerate(ids.tolist()):
        expected = mp.pairs[m].student_forward(x[i : i + 1])
        assert torch.allclose(out[i : i + 1], expected, atol=1e-6), f"row {i} modality={m}"


def test_oob_modality_falls_back_to_pair_zero():
    torch.manual_seed(2)
    mp = ModalityProjectorPair(num_modalities=3, d_model=8, d_hidden=16, d_proj=4)
    x = torch.randn(3, 8)
    ids = torch.tensor([-1, 99, 2], dtype=torch.long)  # -1 and 99 are OOB
    out = mp.student_forward(x, ids)
    # Rows 0 and 1 should equal pair_0's output; row 2 should equal pair_2's.
    exp0 = mp.pairs[0].student_forward(x[0:1])
    exp1 = mp.pairs[0].student_forward(x[1:2])
    exp2 = mp.pairs[2].student_forward(x[2:3])
    assert torch.allclose(out[0:1], exp0, atol=1e-6)
    assert torch.allclose(out[1:2], exp1, atol=1e-6)
    assert torch.allclose(out[2:3], exp2, atol=1e-6)


def test_update_teacher_advances_all_pairs():
    torch.manual_seed(3)
    mp = ModalityProjectorPair(num_modalities=4, d_model=4, d_hidden=8, d_proj=2)
    # Perturb students
    with torch.no_grad():
        for p in mp.pairs:
            for param in p.student.parameters():
                param.add_(torch.ones_like(param))
    # Snapshot teacher state
    before = [[param.detach().clone() for param in pair.teacher.parameters()] for pair in mp.pairs]
    mp.update_teacher(tau=0.0)  # replace teacher with student
    for i, pair in enumerate(mp.pairs):
        for pt, bs in zip(pair.teacher.parameters(), before[i]):
            assert not torch.allclose(pt.detach(), bs, atol=1e-6), f"pair {i} teacher did not update"


def test_empty_modality_id_set_produces_empty_output_slice():
    torch.manual_seed(4)
    mp = ModalityProjectorPair(num_modalities=3, d_model=4, d_hidden=8, d_proj=2)
    x = torch.zeros(0, 4)
    ids = torch.zeros(0, dtype=torch.long)
    out = mp.student_forward(x, ids)
    assert out.shape == (0, 2)
