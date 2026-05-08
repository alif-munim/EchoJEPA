"""End-to-end smoke test for EchoMV-JEPA Stage-1 / 1b / 1m.

Runs several steps of the real training step (forward + backward + optimizer
+ EMA updates for both the projector and the teacher study transformer) on
synthetic batches. Does not launch the full ``main`` loop — that is blocked
on a real manifest + cached c_clip. This test catches:

- Teacher EMA is actually advancing (parameters change from the initial copy).
- Projector EMA is actually advancing.
- Loss is finite for many consecutive steps.
- Student parameters receive gradient (non-None ``.grad``).
- Teacher parameters never receive gradient.
- Stage-1m (``ModalityProjectorPair``) runs and all per-modality student
  projectors advance.
"""

from __future__ import annotations

import torch

from app.echomv_jepa.train import training_step_echomv
from src.models.echomv_jepa import ModalityProjectorPair, StudyTransformerEMA
from src.models.meta_embeddings import MetaDropout, MetaEmbeddings
from src.models.study_projectors import EMAProjectorPair, cosine_schedule
from src.models.study_transformer import StudyTransformer, StudyTransformerConfig


def _synthetic_batch(B=3, M_ctx=3, M_tgt=2, d_clip=8, n_modalities=3, seed=0):
    torch.manual_seed(seed)
    ctx_el = torch.randn(B, M_ctx, d_clip)
    tgt_el = torch.randn(B, M_tgt, d_clip)
    # Modality ids: cycle across 0, 1, 2 so Stage-1m has per-pair traffic
    ctx_mod = torch.tensor([[(i + j) % n_modalities for j in range(M_ctx)] for i in range(B)], dtype=torch.long)
    tgt_mod = torch.tensor([[(i + j) % n_modalities for j in range(M_tgt)] for i in range(B)], dtype=torch.long)
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
        "ctx_meta_modality": ctx_mod,
        "ctx_meta_phase": zeros_ctx,
        "ctx_meta_quality": zeros_ctx,
        "tgt_meta_view": zeros_tgt,
        "tgt_meta_modality": tgt_mod,
        "tgt_meta_phase": zeros_tgt,
        "tgt_meta_quality": zeros_tgt,
        "ctx_pad_mask": ctx_pad,
        "tgt_pad_mask": tgt_pad,
        "full_elements": full_el,
        "full_meta_view": torch.zeros(B, M_ctx + M_tgt, dtype=torch.long),
        "full_meta_modality": torch.cat([ctx_mod, tgt_mod], dim=1),
        "full_meta_phase": torch.zeros(B, M_ctx + M_tgt, dtype=torch.long),
        "full_meta_quality": torch.zeros(B, M_ctx + M_tgt, dtype=torch.long),
        "full_pad_mask": full_pad,
        "target_idx_in_full": tgt_idx,
        "study_id_int": torch.arange(B, dtype=torch.long) + 1000,
    }


def _build_models(d_clip=8, d_model=8, num_modalities=1):
    torch.manual_seed(42)
    st = StudyTransformer(
        StudyTransformerConfig(
            d_clip=d_clip,
            d_model=d_model,
            n_layers=2,
            n_heads=2,
            ffn_mult=2,
            dropout_ffn=0.0,
            dropout_attn=0.0,
            max_M=16,
        )
    )
    meta = MetaEmbeddings(d_model=d_model, dropout=MetaDropout(0.1, 0.1, 0.1, 0.1))
    if num_modalities <= 1:
        proj = EMAProjectorPair(d_model=d_model, d_hidden=16, d_proj=4)
    else:
        proj = ModalityProjectorPair(num_modalities=num_modalities, d_model=d_model, d_hidden=16, d_proj=4)
    teacher_st = StudyTransformerEMA(st)
    return st, meta, proj, teacher_st


def _student_params(proj):
    if isinstance(proj, ModalityProjectorPair):
        out = []
        for p in proj.pairs:
            out += list(p.student.parameters())
        return out
    return list(proj.student.parameters())


def _run_steps(st, meta, proj, teacher_st, *, n_steps=10, lambda_nce=0.0, num_modalities=1):
    params = list(st.parameters()) + list(meta.parameters()) + _student_params(proj)
    opt = torch.optim.AdamW(params, lr=1e-3)

    teacher_snapshot = [p.detach().clone() for p in teacher_st.teacher.parameters()]
    proj_teacher_snapshot = (
        [p.detach().clone() for pair in proj.pairs for p in pair.teacher.parameters()]
        if isinstance(proj, ModalityProjectorPair)
        else [p.detach().clone() for p in proj.teacher.parameters()]
    )
    student_st_snapshot = [p.detach().clone() for p in st.parameters()]
    student_proj_snapshot = [p.detach().clone() for p in _student_params(proj)]

    losses = []
    total = max(n_steps, 1)
    for step in range(n_steps):
        batch = _synthetic_batch(n_modalities=num_modalities if num_modalities > 1 else 3, seed=step)
        out = training_step_echomv(
            batch,
            st,
            meta,
            proj,
            teacher_st,
            lambda_nce=lambda_nce,
            diag_peer_drop_every_n_steps=5,
            global_step=step,
        )
        opt.zero_grad(set_to_none=True)
        out.loss.backward()
        opt.step()
        tau = cosine_schedule(step, total, 0.9, 0.99)
        # EMA updates
        if isinstance(proj, ModalityProjectorPair):
            proj.update_teacher(tau)
        else:
            proj.update_teacher(tau)
        teacher_st.update_teacher(st, tau)

        assert torch.isfinite(out.loss), f"step {step}: non-finite loss {out.loss.item()}"
        losses.append(out.loss.item())

    return {
        "losses": losses,
        "teacher_snapshot": teacher_snapshot,
        "proj_teacher_snapshot": proj_teacher_snapshot,
        "student_st_snapshot": student_st_snapshot,
        "student_proj_snapshot": student_proj_snapshot,
    }


def test_stage1_end_to_end_runs_and_emas_advance():
    st, meta, proj, teacher_st = _build_models(num_modalities=1)
    r = _run_steps(st, meta, proj, teacher_st, n_steps=10, lambda_nce=0.0)

    # Teacher study transformer advanced
    advanced = False
    for p, snap in zip(teacher_st.teacher.parameters(), r["teacher_snapshot"]):
        if not torch.allclose(p.detach(), snap, atol=1e-6):
            advanced = True
            break
    assert advanced, "teacher_st did not advance after 10 EMA updates"

    # Projector teacher advanced
    advanced = False
    for p, snap in zip(proj.teacher.parameters(), r["proj_teacher_snapshot"]):
        if not torch.allclose(p.detach(), snap, atol=1e-6):
            advanced = True
            break
    assert advanced, "projector teacher did not advance after 10 EMA updates"

    # Student ST advanced (optimizer applied gradients)
    advanced = False
    for p, snap in zip(st.parameters(), r["student_st_snapshot"]):
        if not torch.allclose(p.detach(), snap, atol=1e-6):
            advanced = True
            break
    assert advanced, "student StudyTransformer did not advance"

    # Student projector advanced
    advanced = False
    for p, snap in zip(_student_params(proj), r["student_proj_snapshot"]):
        if not torch.allclose(p.detach(), snap, atol=1e-6):
            advanced = True
            break
    assert advanced, "student projector did not advance"


def test_teacher_parameters_never_require_grad_during_training():
    st, meta, proj, teacher_st = _build_models(num_modalities=1)
    _run_steps(st, meta, proj, teacher_st, n_steps=3)
    for p in teacher_st.teacher.parameters():
        assert not p.requires_grad, "teacher_st parameter acquired requires_grad=True"


def test_stage1b_tiny_nce_runs_e2e():
    """Stage-1b uses lambda_nce > 0. Confirm gradients still flow and no NaNs."""
    st, meta, proj, teacher_st = _build_models(num_modalities=1)
    r = _run_steps(st, meta, proj, teacher_st, n_steps=5, lambda_nce=0.005)
    assert all(loss_v == loss_v for loss_v in r["losses"]), f"NaN losses: {r['losses']}"  # x==x catches NaN


def test_stage1m_modality_projector_all_pairs_advance():
    """Stage-1m: every per-modality student pair that sees traffic must advance."""
    n_mod = 3
    st, meta, proj, teacher_st = _build_models(num_modalities=n_mod)
    # Snapshot each pair's student params
    snapshots = [[p.detach().clone() for p in pair.student.parameters()] for pair in proj.pairs]
    _run_steps(st, meta, proj, teacher_st, n_steps=10, num_modalities=n_mod)

    for idx, pair in enumerate(proj.pairs):
        advanced = False
        for p, snap in zip(pair.student.parameters(), snapshots[idx]):
            if not torch.allclose(p.detach(), snap, atol=1e-6):
                advanced = True
                break
        # All three pairs see traffic because _synthetic_batch cycles ids 0..n_mod-1
        assert advanced, f"Stage-1m pair {idx} student did not advance"


def test_stage1m_teacher_pairs_advance():
    n_mod = 3
    st, meta, proj, teacher_st = _build_models(num_modalities=n_mod)
    snapshots = [[p.detach().clone() for p in pair.teacher.parameters()] for pair in proj.pairs]
    _run_steps(st, meta, proj, teacher_st, n_steps=10, num_modalities=n_mod)

    for idx, pair in enumerate(proj.pairs):
        advanced = False
        for p, snap in zip(pair.teacher.parameters(), snapshots[idx]):
            if not torch.allclose(p.detach(), snap, atol=1e-6):
                advanced = True
                break
        assert advanced, f"Stage-1m pair {idx} teacher EMA did not advance"


def test_loss_is_finite_over_many_steps():
    st, meta, proj, teacher_st = _build_models(num_modalities=1)
    r = _run_steps(st, meta, proj, teacher_st, n_steps=30, lambda_nce=0.005)
    assert all(loss_v < 10.0 for loss_v in r["losses"]), f"loss exploded: {r['losses']}"
    assert all(loss_v > -1.0 for loss_v in r["losses"]), f"loss went very negative: {r['losses']}"


def test_frozen_teacher_stays_frozen_when_tau_one():
    """Sanity: if EMA schedule is always tau=1.0, the teacher does NOT move.

    This validates the ablation_no_ema pattern (PR-5)."""
    st, meta, proj, teacher_st = _build_models(num_modalities=1)
    params = list(st.parameters()) + list(meta.parameters()) + list(proj.student.parameters())
    opt = torch.optim.AdamW(params, lr=1e-3)
    before = [p.detach().clone() for p in teacher_st.teacher.parameters()]
    for step in range(5):
        batch = _synthetic_batch(seed=step)
        out = training_step_echomv(
            batch,
            st,
            meta,
            proj,
            teacher_st,
            lambda_nce=0.0,
            global_step=step,
        )
        opt.zero_grad(set_to_none=True)
        out.loss.backward()
        opt.step()
        teacher_st.update_teacher(st, tau=1.0)  # no-op EMA
    for p, b in zip(teacher_st.teacher.parameters(), before):
        assert torch.allclose(p.detach(), b, atol=1e-7), "teacher moved despite tau=1.0"
