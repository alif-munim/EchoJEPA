"""The SV branch must prefer the 'apical' family (our A4C proxy) when
both apical and other families are available."""

from __future__ import annotations

import torch

from src.models.echomv_jepa.single_view_branch import sample_single_view_rows
from src.models.meta_embeddings import VIEW_FAMILY_VOCAB

APICAL_ID = VIEW_FAMILY_VOCAB.index("apical")
PLAX_ID = VIEW_FAMILY_VOCAB.index("parasternal_long")
PSAX_ID = VIEW_FAMILY_VOCAB.index("parasternal_short")


def test_apical_chosen_when_present():
    # 4 studies; apical is present in every row. Should always pick apical.
    pad = torch.zeros(4, 3, dtype=torch.bool)
    view = torch.tensor(
        [
            [APICAL_ID, PLAX_ID, PSAX_ID],
            [PLAX_ID, APICAL_ID, PSAX_ID],
            [PSAX_ID, PLAX_ID, APICAL_ID],
            [APICAL_ID, APICAL_ID, PLAX_ID],
        ],
        dtype=torch.long,
    )
    gen = torch.Generator().manual_seed(0)
    sv_pad, sv_row, stats = sample_single_view_rows(pad, view, p_rows=1.0, generator=gen)
    # Every SV row should have only apical elements unpadded.
    for b in range(4):
        if sv_row[b]:
            kept = ~sv_pad[b]
            kept_views = view[b, kept]
            assert (kept_views == APICAL_ID).all(), f"row {b} kept non-apical: {kept_views.tolist()}"
    assert stats["a4c_sv_count"] == stats["sv_num_rows"]


def test_fallback_to_plax_when_no_apical():
    pad = torch.zeros(2, 3, dtype=torch.bool)
    view = torch.tensor(
        [
            [PLAX_ID, PLAX_ID, PSAX_ID],  # no apical; prefer PLAX
            [PSAX_ID, PSAX_ID, PSAX_ID],  # no apical, no PLAX; take PSAX
        ],
        dtype=torch.long,
    )
    gen = torch.Generator().manual_seed(5)
    sv_pad, sv_row, stats = sample_single_view_rows(pad, view, p_rows=1.0, generator=gen)
    assert sv_row.all().item()
    # Row 0: should keep PLAX.
    kept_0 = view[0, ~sv_pad[0]]
    assert (kept_0 == PLAX_ID).all()
    # Row 1: should keep PSAX.
    kept_1 = view[1, ~sv_pad[1]]
    assert (kept_1 == PSAX_ID).all()
    assert stats["a4c_sv_count"] == 0
    assert stats["sv_family_counts"].get("parasternal_long", 0) == 1
    assert stats["sv_family_counts"].get("parasternal_short", 0) == 1
