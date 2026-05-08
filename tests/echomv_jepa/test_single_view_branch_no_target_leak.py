"""The SV branch emits a *context* pad mask only. The teacher (full
study) must be unaffected. Test this at the pad-mask level: the
returned sv_pad_mask is strictly no less padded than the original
pad mask, and the teacher path (full_pad_mask) is not mutated."""

from __future__ import annotations

import torch

from src.models.echomv_jepa.single_view_branch import sample_single_view_rows
from src.models.meta_embeddings import VIEW_FAMILY_VOCAB

APICAL_ID = VIEW_FAMILY_VOCAB.index("apical")
PLAX_ID = VIEW_FAMILY_VOCAB.index("parasternal_long")


def test_sv_pad_is_superset_of_full_pad():
    pad = torch.tensor(
        [
            [False, False, False, False],
            [False, False, True, True],  # last 2 already padded
            [False, False, False, False],
        ],
        dtype=torch.bool,
    )
    view = torch.tensor(
        [
            [APICAL_ID, APICAL_ID, PLAX_ID, PLAX_ID],
            [PLAX_ID, APICAL_ID, APICAL_ID, PLAX_ID],
            [APICAL_ID, PLAX_ID, APICAL_ID, PLAX_ID],
        ],
        dtype=torch.long,
    )
    gen = torch.Generator().manual_seed(9)
    sv_pad, sv_row, _stats = sample_single_view_rows(pad, view, p_rows=1.0, generator=gen)
    # sv_pad can only grow relative to pad; wherever pad is True, sv_pad must also be True.
    assert (pad & ~sv_pad).sum().item() == 0, "SV mask unpadded a position that was padded — leak"


def test_full_pad_unchanged_after_call():
    """The input pad mask must not be mutated in place."""
    pad = torch.zeros(3, 4, dtype=torch.bool)
    view = torch.full((3, 4), APICAL_ID, dtype=torch.long)
    pad_before = pad.clone()
    sample_single_view_rows(pad, view, p_rows=1.0)
    assert torch.equal(pad, pad_before)
