"""Sampler for the true-V-JEPA clip subset."""

from __future__ import annotations

import torch

from src.models.echomv_jepa.clip_subset_sampler import sample_clip_subset


def _fake_clips(B: int = 3, M: int = 4, C: int = 3, T: int = 2, H: int = 4, W: int = 4) -> torch.Tensor:
    # Marker so we can check which element was picked.
    x = torch.zeros(B, M, C, T, H, W)
    for b in range(B):
        for m in range(M):
            x[b, m] = float(b * 10 + m)
    return x


def test_first_valid_policy_deterministic():
    clips = _fake_clips()
    pad = torch.tensor(
        [
            [False, False, False, False],
            [True, False, False, False],
            [True, True, False, False],
        ],
        dtype=torch.bool,
    )
    sel, idx, valid = sample_clip_subset(clips, pad, n_per_study=1, policy="first_valid")
    # First valid elements per row: 0, 1, 2.
    assert idx[:, 1].tolist() == [0, 1, 2]
    assert valid.all().item()
    # Marker values confirm the selection came from the right clip.
    assert sel[0, 0, 0, 0, 0].item() == 0.0  # row 0, elem 0
    assert sel[1, 0, 0, 0, 0].item() == 11.0  # row 1, elem 1
    assert sel[2, 0, 0, 0, 0].item() == 22.0  # row 2, elem 2


def test_all_padded_row_is_marked_invalid():
    clips = _fake_clips()
    pad = torch.tensor(
        [
            [False, False, False, False],
            [True, True, True, True],  # all padded
            [False, False, False, False],
        ],
        dtype=torch.bool,
    )
    sel, idx, valid = sample_clip_subset(clips, pad, n_per_study=1, policy="first_valid")
    assert valid.tolist() == [True, False, True]
    # Sentinel element index for the invalid row is 0.
    assert idx[1, 1].item() == 0


def test_random_valid_picks_from_unpadded_only():
    torch.manual_seed(0)
    clips = _fake_clips()
    pad = torch.tensor(
        [
            [False, True, False, True],  # valid = {0, 2}
            [False, False, False, False],  # all valid
            [True, True, False, False],  # valid = {2, 3}
        ],
        dtype=torch.bool,
    )
    for seed in range(10):
        g = torch.Generator().manual_seed(seed)
        _sel, idx, _valid = sample_clip_subset(clips, pad, n_per_study=1, policy="random_valid", generator=g)
        assert idx[0, 1].item() in (0, 2)
        # row 1 any of {0, 1, 2, 3}
        assert idx[1, 1].item() in (0, 1, 2, 3)
        assert idx[2, 1].item() in (2, 3)


def test_n_per_study_greater_than_one():
    torch.manual_seed(0)
    clips = _fake_clips()
    pad = torch.zeros(3, 4, dtype=torch.bool)
    sel, idx, valid = sample_clip_subset(clips, pad, n_per_study=3, policy="first_valid")
    # 3 studies × 3 clips each → 9 selections.
    assert sel.shape[0] == 9
    assert idx.shape == (9, 2)
    assert valid.all().item()
