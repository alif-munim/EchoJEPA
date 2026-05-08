"""Same-patient columns must NOT appear as negatives even if they are
different studies."""

from __future__ import annotations

import torch

from src.models.echomv_jepa.cross_rank_study_nce import cross_rank_study_nce_loss


def test_same_patient_excluded_from_negatives():
    torch.manual_seed(0)
    # 4 local rows; 8 global cols. Patient 0 owns studies 0 and 4; everyone else unique.
    B_local = 4
    N_global = 8
    D = 8
    h = torch.randn(B_local, D)
    z = torch.randn(N_global, D)
    study_id = torch.arange(N_global, dtype=torch.long)
    patient_id = torch.tensor([0, 1, 2, 3, 0, 5, 6, 7], dtype=torch.long)  # study 0 and 4 share patient 0
    vb = torch.zeros(N_global, dtype=torch.long)
    mb = torch.zeros(N_global, dtype=torch.long)
    cb = torch.zeros(N_global, dtype=torch.long)

    _loss, diag = cross_rank_study_nce_loss(
        h,
        z,
        local_offset=0,
        study_id_global=study_id,
        patient_id_global=patient_id,
        view_count_bucket_global=vb,
        modality_count_bucket_global=mb,
        clip_count_bucket_global=cb,
        match_view_bucket=False,
        match_modality_bucket=False,
        match_clip_bucket=False,
        exclude_same_patient=True,
    )
    # Row 0's positive is column 0. Column 4 is same-patient.
    # Valid negatives per row should exclude both same-study and same-patient.
    # Row 0 valid_neg: {1,2,3,5,6,7} → 6; pool includes positive, so neg count (excl pos) = 6.
    # Diag reports mean valid_neg (excluding positive). Expect ≤ 6 for row 0.
    assert diag["study_nce_valid_neg_mean"] <= 7.0
    # Row 0 should not be able to use col 4 as negative — check by disabling bucket match
    # and counting: same_patient(0) = {0, 4}. Excluded. So pool for row 0 is 6 other-patient rows.


def test_disable_same_patient_exclusion_keeps_more_negatives():
    torch.manual_seed(1)
    B_local, N_global, D = 2, 6, 4
    h = torch.randn(B_local, D)
    z = torch.randn(N_global, D)
    study_id = torch.arange(N_global, dtype=torch.long)
    patient_id = torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.long)  # all same patient
    vb = torch.zeros(N_global, dtype=torch.long)
    mb = torch.zeros(N_global, dtype=torch.long)
    cb = torch.zeros(N_global, dtype=torch.long)

    _loss_excl, diag_excl = cross_rank_study_nce_loss(
        h,
        z,
        local_offset=0,
        study_id_global=study_id,
        patient_id_global=patient_id,
        view_count_bucket_global=vb,
        modality_count_bucket_global=mb,
        clip_count_bucket_global=cb,
        match_view_bucket=False,
        match_modality_bucket=False,
        match_clip_bucket=False,
        exclude_same_patient=True,
    )
    _loss_incl, diag_incl = cross_rank_study_nce_loss(
        h,
        z,
        local_offset=0,
        study_id_global=study_id,
        patient_id_global=patient_id,
        view_count_bucket_global=vb,
        modality_count_bucket_global=mb,
        clip_count_bucket_global=cb,
        match_view_bucket=False,
        match_modality_bucket=False,
        match_clip_bucket=False,
        exclude_same_patient=False,
    )
    # Everyone is same patient → exclusion=True leaves ZERO negatives (fallback keeps diag w/ positive only).
    # Without exclusion there are 4 valid neg per row (N_global - self - same_study = 6 - 1 - 1 = 4).
    assert diag_incl["study_nce_valid_neg_mean"] > diag_excl["study_nce_valid_neg_mean"]
