"""K_actual diagnostic: computed from the pad mask, not the config K."""

from __future__ import annotations

import torch


def test_k_actual_counts_unpadded():
    pad = torch.tensor(
        [
            [False, False, False, False, True, True, True, True],  # K=4
            [False, False, False, False, False, False, False, False],  # K=8
            [False, False, False, True, True, True, True, True],  # K=3
        ],
        dtype=torch.bool,
    )
    k_actual = (~pad).float().sum(dim=1)
    assert k_actual.tolist() == [4.0, 8.0, 3.0]
    assert float(k_actual.mean().item()) == 5.0


def test_a4c_present_fraction_computed_per_row():
    from src.models.meta_embeddings import VIEW_FAMILY_VOCAB

    apical_id = VIEW_FAMILY_VOCAB.index("apical")
    plax_id = VIEW_FAMILY_VOCAB.index("parasternal_long")
    meta_view = torch.tensor(
        [
            [apical_id, apical_id, plax_id, plax_id],  # has apical
            [plax_id, plax_id, plax_id, plax_id],  # no apical
            [apical_id, plax_id, plax_id, plax_id],  # has apical
        ],
        dtype=torch.long,
    )
    pad = torch.zeros(3, 4, dtype=torch.bool)
    a4c_present_mask = (meta_view == apical_id) & ~pad
    a4c_per_row = a4c_present_mask.any(dim=1)
    fraction = float(a4c_per_row.float().mean().item())
    assert abs(fraction - 2 / 3) < 1e-6
