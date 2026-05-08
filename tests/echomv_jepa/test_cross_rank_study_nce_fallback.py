"""When preferred-bucket negatives are too few, fall back to the full
valid pool and log the fallback fraction."""

from __future__ import annotations

import torch

from src.models.echomv_jepa.cross_rank_study_nce import cross_rank_study_nce_loss


def test_fallback_triggers_when_buckets_disagree():
    """Make every row have zero matched-bucket candidates (bucket per row is unique),
    forcing every row into fallback."""
    torch.manual_seed(0)
    B_local, N_global, D = 4, 8, 8
    h = torch.randn(B_local, D)
    z = torch.randn(N_global, D)
    study_id = torch.arange(N_global, dtype=torch.long)
    patient_id = torch.arange(N_global, dtype=torch.long)
    # Every study has a UNIQUE view bucket → no preferred negatives exist for any row.
    vb = torch.arange(N_global, dtype=torch.long)
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
        match_view_bucket=True,
        match_modality_bucket=False,
        match_clip_bucket=False,
        min_negatives=1,
        exclude_same_patient=False,
    )
    # All 4 rows fall back.
    assert diag["study_nce_fallback_fraction"] == 1.0


def test_no_fallback_when_buckets_agree():
    """All studies in the same bucket → no fallback needed."""
    torch.manual_seed(1)
    B_local, N_global, D = 4, 8, 8
    h = torch.randn(B_local, D)
    z = torch.randn(N_global, D)
    study_id = torch.arange(N_global, dtype=torch.long)
    patient_id = torch.arange(N_global, dtype=torch.long)
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
        match_view_bucket=True,
        match_modality_bucket=True,
        match_clip_bucket=True,
        min_negatives=1,
        exclude_same_patient=False,
    )
    assert diag["study_nce_fallback_fraction"] == 0.0
