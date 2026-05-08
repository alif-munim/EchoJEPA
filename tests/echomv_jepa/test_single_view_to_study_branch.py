"""Single-view-to-study branch subset logic and loss behavior."""

from __future__ import annotations

import torch

from app.echomv_jepa.train_full_joint import _single_view_subset


def test_single_view_subset_drops_other_views():
    # 2 rows, 5 elements each; views 0, 0, 1, 1, 2 (unpadded)
    full_pad = torch.zeros(2, 5, dtype=torch.bool)
    full_view = torch.tensor(
        [
            [0, 0, 1, 1, 2],
            [0, 1, 2, 2, 2],
        ],
        dtype=torch.long,
    )
    torch.manual_seed(0)
    new_pad = _single_view_subset(full_pad, full_view)
    # Every row must have at least one unpadded position.
    assert ((~new_pad).sum(dim=1) >= 1).all()
    # Unpadded positions in new_pad must all share the same view per row.
    for b in range(full_view.shape[0]):
        keep = ~new_pad[b]
        vs = full_view[b, keep]
        if vs.numel() > 0:
            assert (vs == vs[0]).all()


def test_single_view_subset_handles_all_padded():
    full_pad = torch.ones(1, 3, dtype=torch.bool)
    full_view = torch.tensor([[0, 1, 2]], dtype=torch.long)
    new_pad = _single_view_subset(full_pad, full_view)
    # Can't subset — all still padded.
    assert new_pad.all()
