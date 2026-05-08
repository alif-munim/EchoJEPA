"""Single-view branch must fire for a deterministic fraction of rows
every step, and work correctly when the per-rank population differs
(DDP safety: no Bernoulli draw required across ranks)."""

from __future__ import annotations

import torch

from src.models.echomv_jepa.single_view_branch import sample_single_view_rows
from src.models.meta_embeddings import VIEW_FAMILY_VOCAB

APICAL_ID = VIEW_FAMILY_VOCAB.index("apical")
PLAX_ID = VIEW_FAMILY_VOCAB.index("parasternal_long")
PSAX_ID = VIEW_FAMILY_VOCAB.index("parasternal_short")


def _three_studies():
    # 3 studies × 4 elements: all unpadded, mix of views.
    pad = torch.zeros(3, 4, dtype=torch.bool)
    view = torch.tensor(
        [
            [APICAL_ID, APICAL_ID, PLAX_ID, PSAX_ID],
            [PLAX_ID, PLAX_ID, APICAL_ID, PSAX_ID],
            [PSAX_ID, PSAX_ID, PLAX_ID, APICAL_ID],
        ],
        dtype=torch.long,
    )
    return pad, view


def test_sv_fires_for_p_rows_fraction():
    pad, view = _three_studies()
    gen = torch.Generator().manual_seed(7)
    sv_pad, sv_row, stats = sample_single_view_rows(pad, view, p_rows=1.0, generator=gen)
    # All 3 rows should have a valid SV context.
    assert sv_row.sum().item() == 3
    assert stats["sv_valid_fraction"] == 1.0


def test_sv_fires_on_batch_of_one_row_nonzero_prob():
    """Single-row batch + p_rows=0.5 should fire on at least 0 rows and
    up to 1 row; p_rows=1.0 must fire on the only row."""
    pad = torch.zeros(1, 4, dtype=torch.bool)
    view = torch.tensor([[APICAL_ID, PLAX_ID, PSAX_ID, APICAL_ID]], dtype=torch.long)
    gen = torch.Generator().manual_seed(1)
    _sv_pad, sv_row, stats = sample_single_view_rows(pad, view, p_rows=1.0, generator=gen)
    assert sv_row.item() is True
    assert stats["sv_num_rows"] == 1


def test_zero_p_rows_returns_empty_valid_mask():
    pad, view = _three_studies()
    sv_pad, sv_row, stats = sample_single_view_rows(pad, view, p_rows=0.0)
    assert sv_row.sum().item() == 0
    assert sv_pad.all().item()  # every element padded
    assert stats["sv_num_rows"] == 0


def test_handles_all_padded_row_gracefully():
    pad = torch.tensor([[False, False], [True, True]], dtype=torch.bool)
    view = torch.tensor([[APICAL_ID, PLAX_ID], [APICAL_ID, PLAX_ID]], dtype=torch.long)
    gen = torch.Generator().manual_seed(2)
    _sv_pad, sv_row, stats = sample_single_view_rows(pad, view, p_rows=1.0, generator=gen)
    # Row 1 is all-padded; can't be SV. Row 0 should succeed.
    assert sv_row[0].item() is True
    assert sv_row[1].item() is False
    assert stats["sv_valid_fraction"] == 0.5
