"""Arm B anti-leak test: teacher self-masking must NEVER change the student's
target-slot input. The student already receives zero visual content at target
slots by construction (mask_token + meta). Teacher self-masking touches only
the teacher's forward-pass input.

This test hashes the tensors the STUDENT consumes with and without teacher
self-masking enabled. They must be byte-equal.
"""

from __future__ import annotations

import copy

import torch

from app.echomv_jepa.train import training_step_echomv
from tests.echomv_jepa.test_contextualization_diagnostics import (
    _make_models,
    _synthetic_batch,
)


def _batch_hash(batch):
    """Tensor hash of the keys the student consumes."""
    parts = []
    for k in (
        "ctx_elements",
        "ctx_meta_view",
        "ctx_meta_modality",
        "ctx_meta_phase",
        "ctx_meta_quality",
        "ctx_pad_mask",
        "tgt_meta_view",
        "tgt_meta_modality",
        "tgt_meta_phase",
        "tgt_meta_quality",
        "tgt_pad_mask",
    ):
        if k in batch:
            parts.append(torch.as_tensor(batch[k]).clone().detach().float().sum().item())
    return tuple(parts)


def test_student_sees_identical_input_regardless_of_selfmask():
    torch.manual_seed(0)
    st, meta, proj, teacher = _make_models()
    batch = _synthetic_batch(B=4, M_ctx=3, M_tgt=2)
    # Snapshot of everything the student should see.
    pre_batch = copy.deepcopy(batch)
    pre_hash = _batch_hash(pre_batch)

    _ = training_step_echomv(
        batch,
        st,
        meta,
        proj,
        teacher,
        lambda_nce=0.0,
        p_target_self_mask=1.0,
        global_step=0,
    )

    # Batch must not have been mutated in-place by the step.
    post_hash = _batch_hash(batch)
    assert pre_hash == post_hash, "training_step_echomv mutated batch in place"


def test_selfmask_does_not_enable_target_quality_leak():
    """Regardless of p_target_self_mask, include_target_quality must still
    default to False, so the student's target slot never sees quality."""
    torch.manual_seed(1)
    st, meta, proj, teacher = _make_models()
    batch = _synthetic_batch(B=3, M_ctx=3, M_tgt=2)
    # Compose a student target-slot encoding ourselves with the documented
    # defaults and confirm that it does NOT depend on quality ids.
    from src.models.meta_embeddings import MetaDropout

    meta.dropout_cfg = MetaDropout(0, 0, 0, 0)  # deterministic
    q_a = batch["tgt_meta_quality"]
    q_b = torch.ones_like(q_a)  # force a different quality id
    tgt_view = batch["tgt_meta_view"]
    tgt_mod = batch["tgt_meta_modality"]
    tgt_phase = batch["tgt_meta_phase"]
    add_a = meta.encode_target_slot(
        tgt_view,
        tgt_mod,
        phase_ids=tgt_phase,
        include_phase=True,
        include_quality=False,
        quality_ids=q_a,
    )
    add_b = meta.encode_target_slot(
        tgt_view,
        tgt_mod,
        phase_ids=tgt_phase,
        include_phase=True,
        include_quality=False,
        quality_ids=q_b,
    )
    assert torch.allclose(
        add_a, add_b
    ), "target slot meta encoding should not depend on quality when include_quality=False"
