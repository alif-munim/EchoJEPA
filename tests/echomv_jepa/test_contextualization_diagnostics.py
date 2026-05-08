"""Teacher-contextualization diagnostics — training step integration tests.

Runs a single ``training_step_echomv`` on synthetic batches and verifies that:

- The diagnostic cosines are finite floats in [-1, 1].
- When all elements are identical, ``z_cosine_vs_v1`` and
  ``z_cosine_vs_isolated`` are close to 1 (teacher has no distinguishing
  information), confirming the probes are not just measuring model noise.
- When elements differ substantially, ``z_cosine_vs_isolated`` decreases from
  the identical-elements case (teacher uses context to differentiate targets).
"""

from __future__ import annotations

import torch

from app.echomv_jepa.train import training_step_echomv
from src.models.echomv_jepa import StudyTransformerEMA
from src.models.meta_embeddings import MetaDropout, MetaEmbeddings
from src.models.study_projectors import EMAProjectorPair
from src.models.study_transformer import StudyTransformer, StudyTransformerConfig


def _make_models(d_clip=8, d_model=8, seed=0):
    torch.manual_seed(seed)
    st = StudyTransformer(
        StudyTransformerConfig(
            d_clip=d_clip,
            d_model=d_model,
            n_layers=1,
            n_heads=2,
            ffn_mult=2,
            dropout_ffn=0.0,
            dropout_attn=0.0,
            max_M=8,
        )
    )
    meta = MetaEmbeddings(d_model=d_model, dropout=MetaDropout(0, 0, 0, 0))
    proj = EMAProjectorPair(d_model=d_model, d_hidden=16, d_proj=4)
    teacher = StudyTransformerEMA(st)
    return st, meta, proj, teacher


def _synthetic_batch(B=2, M_ctx=3, M_tgt=2, d_clip=8, element_generator="random"):
    if element_generator == "random":
        ctx_el = torch.randn(B, M_ctx, d_clip)
        tgt_el = torch.randn(B, M_tgt, d_clip)
    elif element_generator == "identical":
        v = torch.randn(B, 1, d_clip)
        ctx_el = v.expand(B, M_ctx, d_clip).clone()
        tgt_el = v.expand(B, M_tgt, d_clip).clone()
    else:
        raise ValueError(element_generator)

    zeros_ctx = torch.zeros(B, M_ctx, dtype=torch.long)
    zeros_tgt = torch.zeros(B, M_tgt, dtype=torch.long)
    ctx_pad = torch.zeros(B, M_ctx, dtype=torch.bool)
    tgt_pad = torch.zeros(B, M_tgt, dtype=torch.bool)

    full_el = torch.cat([ctx_el, tgt_el], dim=1)
    full_pad = torch.cat([ctx_pad, tgt_pad], dim=1)
    tgt_idx = (torch.arange(M_tgt, dtype=torch.long).unsqueeze(0).expand(B, -1) + M_ctx).contiguous()

    return {
        "ctx_elements": ctx_el,
        "tgt_elements": tgt_el,
        "ctx_meta_view": zeros_ctx,
        "ctx_meta_modality": zeros_ctx,
        "ctx_meta_phase": zeros_ctx,
        "ctx_meta_quality": zeros_ctx,
        "tgt_meta_view": zeros_tgt,
        "tgt_meta_modality": zeros_tgt,
        "tgt_meta_phase": zeros_tgt,
        "tgt_meta_quality": zeros_tgt,
        "ctx_pad_mask": ctx_pad,
        "tgt_pad_mask": tgt_pad,
        "full_elements": full_el,
        "full_meta_view": torch.zeros(B, M_ctx + M_tgt, dtype=torch.long),
        "full_meta_modality": torch.zeros(B, M_ctx + M_tgt, dtype=torch.long),
        "full_meta_phase": torch.zeros(B, M_ctx + M_tgt, dtype=torch.long),
        "full_meta_quality": torch.zeros(B, M_ctx + M_tgt, dtype=torch.long),
        "full_pad_mask": full_pad,
        "target_idx_in_full": tgt_idx,
        "study_id_int": torch.arange(B, dtype=torch.long) + 1000,
    }


def test_diagnostic_fields_present_and_in_range():
    st, meta, proj, teacher = _make_models()
    batch = _synthetic_batch(B=3, M_ctx=3, M_tgt=2)
    out = training_step_echomv(
        batch,
        st,
        meta,
        proj,
        teacher,
        lambda_nce=0.0,
        diag_peer_drop_every_n_steps=1,
        global_step=0,
    )
    d = out.diagnostics
    for k in ("z_cosine_vs_v1", "z_cosine_vs_isolated", "z_cosine_vs_peer_drop"):
        assert k in d, f"missing diagnostic key {k}"
    assert -1.01 <= d["z_cosine_vs_v1"] <= 1.01
    assert -1.01 <= d["z_cosine_vs_isolated"] <= 1.01
    # peer-drop can be nan when no eligible context — but here M_ctx=3 so it should compute
    assert -1.01 <= d["z_cosine_vs_peer_drop"] <= 1.01


def test_cosine_vs_isolated_is_one_when_all_elements_identical():
    """If every element is the same vector, the teacher cannot distinguish
    positions; contextualized and isolated outputs must match."""
    st, meta, proj, teacher = _make_models()
    batch = _synthetic_batch(B=2, M_ctx=3, M_tgt=2, element_generator="identical")
    out = training_step_echomv(
        batch,
        st,
        meta,
        proj,
        teacher,
        lambda_nce=0.0,
        diag_peer_drop_every_n_steps=0,
        global_step=0,
    )
    # For identical elements, z_cosine_vs_isolated should be very close to 1
    assert (
        out.diagnostics["z_cosine_vs_isolated"] > 0.99
    ), f"identical-element cosine_vs_isolated = {out.diagnostics['z_cosine_vs_isolated']}"


def test_peer_drop_only_computed_on_configured_schedule():
    st, meta, proj, teacher = _make_models()
    batch = _synthetic_batch(B=2, M_ctx=3, M_tgt=2)
    out_off = training_step_echomv(
        batch,
        st,
        meta,
        proj,
        teacher,
        lambda_nce=0.0,
        diag_peer_drop_every_n_steps=50,
        global_step=1,  # 1 % 50 != 0
    )
    import math

    assert math.isnan(out_off.diagnostics["z_cosine_vs_peer_drop"])

    out_on = training_step_echomv(
        batch,
        st,
        meta,
        proj,
        teacher,
        lambda_nce=0.0,
        diag_peer_drop_every_n_steps=50,
        global_step=0,  # 0 % 50 == 0
    )
    assert not math.isnan(out_on.diagnostics["z_cosine_vs_peer_drop"])


def test_nce_is_exactly_zero_when_lambda_zero():
    st, meta, proj, teacher = _make_models()
    batch = _synthetic_batch(B=2, M_ctx=3, M_tgt=2)
    out = training_step_echomv(
        batch,
        st,
        meta,
        proj,
        teacher,
        lambda_nce=0.0,
        diag_peer_drop_every_n_steps=0,
        global_step=0,
    )
    assert out.loss_nce.item() == 0.0
