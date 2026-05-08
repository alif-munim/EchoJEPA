"""Arm B positive test: at p_target_self_mask = 1.0, the teacher's target
embedding z_t should differ from the unmasked baseline (p=0.0) on most rows.
This confirms the knob actually reaches the teacher.
"""

from __future__ import annotations

import torch  # noqa: F401  (imported for side effects in the reused helpers)

from app.echomv_jepa.train import training_step_echomv
from tests.echomv_jepa.test_contextualization_diagnostics import (
    _make_models,
    _synthetic_batch,
)


def test_teacher_target_differs_under_selfmask():
    torch.manual_seed(0)
    st, meta, proj, teacher = _make_models()
    batch = _synthetic_batch(B=4, M_ctx=4, M_tgt=3)

    # Diagnostic z_cosine_vs_v1 already captures how far the teacher's output
    # sits from the pre-context linear target. Under self-masking the teacher's
    # output should additionally shift compared to no-selfmask — we observe
    # this as a change in the logged z_cosine_vs_isolated or z_cosine_vs_v1.
    out_nomask = training_step_echomv(
        batch,
        st,
        meta,
        proj,
        teacher,
        lambda_nce=0.0,
        p_target_self_mask=0.0,
        diag_peer_drop_every_n_steps=1,
        diag_extra_every_n_steps=1,
        global_step=0,
    )
    out_mask = training_step_echomv(
        batch,
        st,
        meta,
        proj,
        teacher,
        lambda_nce=0.0,
        p_target_self_mask=1.0,
        diag_peer_drop_every_n_steps=1,
        diag_extra_every_n_steps=1,
        global_step=0,
    )

    # The two runs share the same weights but feed different teacher input.
    # At least one of the teacher-based diagnostics should differ.
    d_nomask = out_nomask.diagnostics
    d_mask = out_mask.diagnostics
    assert d_mask["teacher_selfmask_rate"] > 0.0
    assert d_nomask["teacher_selfmask_rate"] == 0.0
    # z_cosine_vs_v1: when target visual content is zeroed on the teacher,
    # the teacher's output should no longer track v1 = proj(clip_in(tgt)) as
    # closely. Expect a detectable difference.
    assert abs(d_mask["z_cosine_vs_v1"] - d_nomask["z_cosine_vs_v1"]) > 1e-3, (
        f"expected z_v1 to shift under selfmask, got "
        f"nomask={d_nomask['z_cosine_vs_v1']:.4f} mask={d_mask['z_cosine_vs_v1']:.4f}"
    )
