"""Permutation-invariance tests for the EchoMV-JEPA teacher and student.

The study transformer uses no positional encoding across elements. Two
consequences must hold:

1. Teacher ``forward_contextualized(elements, meta, pad)`` is *equivariant*
   under a shared permutation of rows along the M axis. Permuting the input
   by ``perm`` produces an output whose rows are the same permutation of the
   original rows.

2. Student ``forward`` has an *invariant* read-out at the ``[STUDY]`` token
   position: shuffling context rows (and their meta / pad rows identically)
   does not change ``h_study``. Target-mask outputs at masked positions are
   equivariant under a shared permutation of the target block.

These are the guardrails against accidentally introducing any positional
dependence (e.g. via an added sinusoidal embedding in a refactor).
"""

from __future__ import annotations

import torch

from src.models.echomv_jepa import StudyTransformerEMA
from src.models.study_transformer import StudyTransformer, StudyTransformerConfig


def _make_student(d_clip=16, d_model=16, n_layers=2):
    return StudyTransformer(
        StudyTransformerConfig(
            d_clip=d_clip,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=4,
            ffn_mult=2,
            dropout_ffn=0.0,  # deterministic
            dropout_attn=0.0,
            max_M=16,
        )
    )


def _eval(mod):
    mod.eval()
    return mod


def test_teacher_forward_contextualized_is_equivariant_to_shared_permutation():
    torch.manual_seed(0)
    st = _eval(_make_student())
    teacher = StudyTransformerEMA(st).eval()
    B, M, d_clip, d_model = 2, 5, 16, 16
    elements = torch.randn(B, M, d_clip)
    meta = torch.randn(B, M, d_model)
    pad = torch.zeros(B, M, dtype=torch.bool)

    out_orig = teacher.forward_contextualized(elements, meta, pad)

    # Permute the M axis identically across elements / meta / pad
    perm = torch.tensor([3, 0, 4, 2, 1], dtype=torch.long)
    elements_p = elements[:, perm, :]
    meta_p = meta[:, perm, :]
    pad_p = pad[:, perm]

    out_perm = teacher.forward_contextualized(elements_p, meta_p, pad_p)

    # out_perm[:, k, :] should equal out_orig[:, perm[k], :]
    expected = out_orig[:, perm, :]
    assert torch.allclose(out_perm, expected, atol=1e-5), "teacher output not equivariant under element permutation"


def test_teacher_output_differs_under_non_structural_perturbation():
    """Sanity check that the equivariance test is non-trivial: without the
    matching permutation on meta/pad, outputs must differ."""
    torch.manual_seed(1)
    st = _eval(_make_student())
    teacher = StudyTransformerEMA(st).eval()
    B, M, d_clip, d_model = 2, 5, 16, 16
    elements = torch.randn(B, M, d_clip)
    meta = torch.randn(B, M, d_model)
    pad = torch.zeros(B, M, dtype=torch.bool)

    out_orig = teacher.forward_contextualized(elements, meta, pad)

    perm = torch.tensor([3, 0, 4, 2, 1], dtype=torch.long)
    elements_p = elements[:, perm, :]  # permute elements
    out_mismatched = teacher.forward_contextualized(elements_p, meta, pad)  # NOT meta

    # Should NOT equal the trivially-permuted original
    assert not torch.allclose(out_mismatched, out_orig[:, perm, :], atol=1e-3)


def test_student_h_study_invariant_under_ctx_permutation():
    torch.manual_seed(2)
    st = _eval(_make_student())
    B, M_ctx, M_tgt, d_clip, d_model = 2, 4, 2, 16, 16
    ctx = torch.randn(B, M_ctx, d_clip)
    ctx_meta = torch.randn(B, M_ctx, d_model)
    ctx_pad = torch.zeros(B, M_ctx, dtype=torch.bool)
    tgt_meta = torch.randn(B, M_tgt, d_model)
    tgt_pad = torch.zeros(B, M_tgt, dtype=torch.bool)

    h_study_orig, _ = st(ctx, ctx_meta, ctx_pad, tgt_meta, tgt_pad)

    perm = torch.tensor([2, 0, 3, 1], dtype=torch.long)
    h_study_perm, _ = st(
        ctx[:, perm, :],
        ctx_meta[:, perm, :],
        ctx_pad[:, perm],
        tgt_meta,
        tgt_pad,
    )

    assert torch.allclose(
        h_study_orig, h_study_perm, atol=1e-5
    ), "[STUDY] token readout not invariant under ctx permutation"


def test_student_h_mask_equivariant_under_tgt_permutation():
    torch.manual_seed(3)
    st = _eval(_make_student())
    B, M_ctx, M_tgt, d_clip, d_model = 2, 3, 4, 16, 16
    ctx = torch.randn(B, M_ctx, d_clip)
    ctx_meta = torch.randn(B, M_ctx, d_model)
    ctx_pad = torch.zeros(B, M_ctx, dtype=torch.bool)
    tgt_meta = torch.randn(B, M_tgt, d_model)
    tgt_pad = torch.zeros(B, M_tgt, dtype=torch.bool)

    _, h_mask_orig = st(ctx, ctx_meta, ctx_pad, tgt_meta, tgt_pad)

    perm = torch.tensor([3, 0, 2, 1], dtype=torch.long)
    _, h_mask_perm = st(
        ctx,
        ctx_meta,
        ctx_pad,
        tgt_meta[:, perm, :],
        tgt_pad[:, perm],
    )

    expected = h_mask_orig[:, perm, :]
    assert torch.allclose(h_mask_perm, expected, atol=1e-5), "h_mask not equivariant under target permutation"


def test_teacher_respects_pad_mask():
    """Padded positions must not change the outputs of unpadded positions."""
    torch.manual_seed(4)
    st = _eval(_make_student())
    teacher = StudyTransformerEMA(st).eval()
    B, M, d_clip, d_model = 1, 6, 16, 16
    elements = torch.randn(B, M, d_clip)
    meta = torch.randn(B, M, d_model)
    pad = torch.zeros(B, M, dtype=torch.bool)
    pad[0, 4:] = True  # mark last two as padded

    out_with_pad = teacher.forward_contextualized(elements, meta, pad)

    # Corrupt the padded positions' inputs and re-run. Outputs at unpadded
    # positions must be identical.
    elements2 = elements.clone()
    elements2[0, 4:] = torch.randn_like(elements2[0, 4:]) * 100.0
    meta2 = meta.clone()
    meta2[0, 4:] = torch.randn_like(meta2[0, 4:]) * 100.0

    out_corrupted = teacher.forward_contextualized(elements2, meta2, pad)
    assert torch.allclose(
        out_with_pad[:, :4, :], out_corrupted[:, :4, :], atol=1e-5
    ), "unpadded outputs changed when padded inputs were perturbed"
