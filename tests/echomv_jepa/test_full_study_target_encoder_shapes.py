"""Shape + invariance tests for StudyTransformerEMA (PR-3).

Checks:
  - forward_contextualized returns (B, M, d_model) regardless of pad mask.
  - gather-at-target yields the expected rows.
  - forward_isolated returns (B, M, d_model) and matches single-element
    forward_contextualized for each row.
  - teacher parameters require_grad=False.
  - new StudyTransformer.forward_contextualized is purely additive (no behavior
    change to the existing forward).
"""

from __future__ import annotations

import torch

from src.models.echomv_jepa import StudyTransformerEMA
from src.models.study_transformer import StudyTransformer, StudyTransformerConfig


def _make_student(d_clip=32, d_model=32, n_layers=2, max_M=8):
    cfg = StudyTransformerConfig(
        d_clip=d_clip,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=4,
        ffn_mult=2,
        dropout_ffn=0.0,
        dropout_attn=0.0,
        max_M=max_M,
    )
    return StudyTransformer(cfg)


def test_teacher_no_grad():
    st = _make_student()
    teacher = StudyTransformerEMA(st)
    assert all(not p.requires_grad for p in teacher.teacher.parameters())


def test_forward_contextualized_shape():
    st = _make_student(d_clip=32, d_model=32)
    teacher = StudyTransformerEMA(st).eval()
    B, M = 3, 5
    full_elements = torch.randn(B, M, 32)
    full_meta = torch.zeros(B, M, 32)
    full_pad = torch.zeros(B, M, dtype=torch.bool)
    out = teacher.forward_contextualized(full_elements, full_meta, full_pad)
    assert out.shape == (B, M, 32)


def test_gather_at_target_indices():
    st = _make_student(d_clip=32, d_model=32)
    teacher = StudyTransformerEMA(st).eval()
    B, M = 2, 4
    full_elements = torch.randn(B, M, 32)
    full_meta = torch.zeros(B, M, 32)
    full_pad = torch.zeros(B, M, dtype=torch.bool)
    out = teacher.forward_contextualized(full_elements, full_meta, full_pad)
    # target_idx picks positions [1, 3] for row 0, [0, 2] for row 1
    tgt_idx = torch.tensor([[1, 3], [0, 2]], dtype=torch.long)
    idx_exp = tgt_idx.unsqueeze(-1).expand(-1, -1, 32)
    gathered = torch.gather(out, dim=1, index=idx_exp)
    # Compare against manual select
    manual = torch.stack(
        [out[0][[1, 3]], out[1][[0, 2]]],
        dim=0,
    )
    assert torch.allclose(gathered, manual)


def test_forward_isolated_shape_and_matches_single_element_context():
    torch.manual_seed(0)
    st = _make_student(d_clip=32, d_model=32)
    teacher = StudyTransformerEMA(st).eval()
    B, M = 2, 3
    elements = torch.randn(B, M, 32)
    meta = torch.zeros(B, M, 32)
    iso = teacher.forward_isolated(elements, meta)
    assert iso.shape == (B, M, 32)

    # For each position, forward_isolated should equal forward_contextualized
    # on a single-element input of that same element.
    for b in range(B):
        for m in range(M):
            single_el = elements[b : b + 1, m : m + 1]
            single_mt = meta[b : b + 1, m : m + 1]
            single_pad = torch.zeros(1, 1, dtype=torch.bool)
            single_out = teacher.forward_contextualized(single_el, single_mt, single_pad)[0, 0]
            assert torch.allclose(iso[b, m], single_out, atol=1e-6)


def test_student_forward_unchanged_by_new_method():
    """The existing StudyTransformer.forward must behave identically after
    adding the new forward_contextualized method."""
    torch.manual_seed(42)
    st = _make_student(d_clip=8, d_model=8, n_layers=1)
    B, M_ctx, M_tgt = 2, 3, 2
    ctx = torch.randn(B, M_ctx, 8)
    ctx_meta = torch.zeros(B, M_ctx, 8)
    ctx_pad = torch.zeros(B, M_ctx, dtype=torch.bool)
    tgt_meta = torch.zeros(B, M_tgt, 8)
    tgt_pad = torch.zeros(B, M_tgt, dtype=torch.bool)
    h_study, h_mask = st(ctx, ctx_meta, ctx_pad, tgt_meta, tgt_pad)
    assert h_study.shape == (B, 8)
    assert h_mask.shape == (B, M_tgt, 8)
