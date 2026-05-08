"""Cross-rank study NCE: scalar loss + finite + reasonable diagnostics."""

from __future__ import annotations

import torch

from src.models.echomv_jepa.cross_rank_study_nce import cross_rank_study_nce_loss


def _setup(B_local: int = 4, N_global: int = 16, D: int = 8, seed: int = 0):
    torch.manual_seed(seed)
    # h_student_local corresponds to local_offset .. local_offset + B_local
    local_offset = 0
    h = torch.randn(B_local, D, requires_grad=True)
    z = torch.randn(N_global, D)
    study_id = torch.arange(N_global, dtype=torch.long)
    patient_id = torch.arange(N_global, dtype=torch.long)
    view_bucket = torch.zeros(N_global, dtype=torch.long)  # all same bucket
    mod_bucket = torch.zeros(N_global, dtype=torch.long)
    clip_bucket = torch.zeros(N_global, dtype=torch.long)
    return h, z, local_offset, study_id, patient_id, view_bucket, mod_bucket, clip_bucket


def test_scalar_loss_and_finite():
    h, z, offset, sid, pid, vb, mb, cb = _setup()
    loss, diag = cross_rank_study_nce_loss(h, z, offset, sid, pid, vb, mb, cb)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_diagnostics_schema():
    h, z, offset, sid, pid, vb, mb, cb = _setup()
    _loss, diag = cross_rank_study_nce_loss(h, z, offset, sid, pid, vb, mb, cb)
    for k in (
        "study_nce_pool_size",
        "study_nce_valid_neg_mean",
        "study_nce_valid_neg_min",
        "study_nce_fallback_fraction",
        "study_matched_rank_top1_global",
        "study_matched_rank_top5_global",
        "pos_minus_hardneg_gap_global",
    ):
        assert k in diag
        assert isinstance(diag[k], float)


def test_loss_backward_reaches_h():
    h, z, offset, sid, pid, vb, mb, cb = _setup()
    loss, _diag = cross_rank_study_nce_loss(h, z, offset, sid, pid, vb, mb, cb)
    loss.backward()
    assert h.grad is not None
    assert h.grad.abs().sum().item() > 0.0
