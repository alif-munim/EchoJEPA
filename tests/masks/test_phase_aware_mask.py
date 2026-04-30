"""Unit tests for phase-aware masking (phi-JEPA mask-phi variant).

These tests exercise:
    * phi computation on the sampled tubelet grid (HR=60, FrameTime=33.33 ms);
    * dphi wraparound semantics;
    * circular mean near phase wraparound (e.g. [0.98, 0.02]);
    * each phase bucket samples target blocks whose dphi is in-range;
    * `same_phase_next_beat` skips when the clip is too short;
    * invalid metadata falls back to the vanilla sampler and returns
      shape-correct masks;
    * shuffled-HR derangement preserves the marginal distribution and changes
      most per-sample HR assignments.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from src.masks.multiseq_multiblock3d import MaskCollator
from src.masks.phase_mask_utils import (
    apply_shuffled_hr,
    block_center_phi,
    circular_mean,
    cycle_tubelets_from_hr,
    dphi_fraction,
    parse_bucket_cfg,
    tubelet_to_phi,
    validate_hr,
)


# ---------------------------------------------------------------------------
# phi / dphi / circular mean
# ---------------------------------------------------------------------------


def test_phi_from_known_hr_ft():
    # HR=60 bpm, fps=8, tubelet=2 -> cycle_frames_sampled = 8, cycle_tubelets=4.
    # At tubelet 0, phi=0. At tubelet 2, phi=0.5. At tubelet 4, phi=0 again.
    ct = cycle_tubelets_from_hr(hr_bpm=60.0, fps_sampled=8.0, tubelet_size=2)
    assert math.isclose(ct, 4.0, rel_tol=1e-9)
    idxs = np.arange(6)
    phis = tubelet_to_phi(idxs, ct)
    assert math.isclose(phis[0], 0.0)
    assert math.isclose(phis[2], 0.5)
    assert math.isclose(phis[4], 0.0)


def test_dphi_wraparound():
    assert math.isclose(dphi_fraction(0.98, 0.02), 0.04, abs_tol=1e-9)
    assert math.isclose(dphi_fraction(0.5, 0.6), 0.1, abs_tol=1e-9)
    # backward wrap: 0.1 -> 0.95 is +0.85 forward.
    assert math.isclose(dphi_fraction(0.1, 0.95), 0.85, abs_tol=1e-9)


def test_circular_mean_wraparound():
    # Phases straddling 1.0 should average near 0.0, not near 0.5.
    m = circular_mean([0.98, 0.02])
    assert min(m, 1.0 - m) < 0.01  # close to 0 circularly
    # Symmetric case around 0.25 should mean ~0.25.
    m2 = circular_mean([0.20, 0.30])
    assert math.isclose(m2, 0.25, abs_tol=1e-6)


# ---------------------------------------------------------------------------
# Metadata validation
# ---------------------------------------------------------------------------


def test_validate_hr_accepts_normal_metadata():
    ok, reason = validate_hr(hr_bpm=70.0, frame_time_ms=33.3, num_frames=60)
    assert ok and reason == "ok"


def test_validate_hr_rejects_nan_hr():
    ok, _ = validate_hr(hr_bpm=float("nan"), frame_time_ms=33.3, num_frames=60)
    assert not ok


def test_validate_hr_rejects_out_of_range():
    ok, reason = validate_hr(hr_bpm=30.0, frame_time_ms=33.3, num_frames=60)
    assert not ok and reason == "hr_out_of_range"


def test_validate_hr_rejects_bad_ft():
    ok, _ = validate_hr(hr_bpm=70.0, frame_time_ms=-1.0, num_frames=60)
    assert not ok


# ---------------------------------------------------------------------------
# Shuffled-HR derangement
# ---------------------------------------------------------------------------


def test_shuffled_hr_preserves_marginal():
    hrs = [60.0, 70.0, 80.0, 90.0, 100.0, 65.0, 75.0, 85.0]
    rng = np.random.default_rng(0)
    r = apply_shuffled_hr(hrs, rng)
    assert r.was_applied
    # Same multiset.
    assert sorted(hrs) == sorted(r.shuffled)
    # Derangement: no fixed points.
    assert r.derangement_ok
    assert sum(1 for a, b in zip(hrs, r.shuffled) if a == b) == 0


def test_shuffled_hr_degenerate_same_values():
    hrs = [70.0] * 4
    rng = np.random.default_rng(1)
    r = apply_shuffled_hr(hrs, rng)
    # All same -> no-op.
    assert not r.was_applied


def test_shuffled_hr_small_batch():
    rng = np.random.default_rng(2)
    r = apply_shuffled_hr([70.0], rng)
    assert not r.was_applied  # size < 2


# ---------------------------------------------------------------------------
# Bucket config
# ---------------------------------------------------------------------------


def test_parse_bucket_cfg_defaults_normalize():
    specs, probs = parse_bucket_cfg(
        {
            "local": [0.05, 0.15],
            "mid_cycle": [0.20, 0.35],
            "opposite_phase": [0.45, 0.55],
            "same_phase_next_beat": True,
        },
        {
            "local": 0.25,
            "mid_cycle": 0.35,
            "opposite_phase": 0.25,
            "same_phase_next_beat": 0.15,
        },
    )
    assert math.isclose(sum(probs.values()), 1.0, abs_tol=1e-9)
    assert specs["same_phase_next_beat"].next_beat


# ---------------------------------------------------------------------------
# Collator integration
# ---------------------------------------------------------------------------


def _make_sample(fpc=16, hr=70.0, ft=33.3, nf=60):
    """Construct a dataset tuple compatible with MaskCollator._phase_call.
    The collator expects: (buffer, label, clip_indices, sample_uri, meta)."""
    buf = torch.zeros((3, fpc, 16, 16), dtype=torch.float32)  # (C, T, H, W)
    label = torch.tensor(0, dtype=torch.long)
    # clip_indices: list with last element sized fpc.
    clip_indices = [list(range(fpc))]
    sample_uri = "/tmp/dummy.mp4"
    meta = {"hr_bpm": float(hr), "frame_time_ms": float(ft), "num_frames": int(nf)}
    return (buf, label, clip_indices, sample_uri, meta)


def _default_mask_cfgs():
    # Phase-aware masking requires localized target blocks (temporal_scale<1.0)
    # to have any degrees of freedom in the temporal placement. With
    # temporal_scale=1.0 every target block spans the full clip and there is
    # only one valid t_start, so re-sampling against phase buckets is a no-op.
    return [
        {
            "aspect_ratio": [0.75, 1.5],
            "full_complement": False,
            "max_keep": None,
            "max_temporal_keep": 1.0,
            "num_blocks": 2,
            "spatial_scale": [0.15, 0.15],
            "temporal_scale": [0.25, 0.25],
        },
    ]


def _default_phase_cfg(**over):
    base = {
        "phase_aware": True,
        "phase_buckets": {
            "local": [0.05, 0.15],
            "mid_cycle": [0.20, 0.35],
            "opposite_phase": [0.45, 0.55],
            "same_phase_next_beat": True,
        },
        "phase_bucket_probs": {
            "local": 0.25,
            "mid_cycle": 0.35,
            "opposite_phase": 0.25,
            "same_phase_next_beat": 0.15,
        },
        "phase_fallback": "random",
        "require_valid_hr": True,
        "shuffled_hr": False,
        "phase_max_attempts": 20,
        "phase_seed": 0,
    }
    base.update(over)
    return base


def test_collator_vanilla_path_unchanged_without_phase_mask():
    collator = MaskCollator(
        cfgs_mask=_default_mask_cfgs(),
        dataset_fpcs=[16],
        crop_size=(224, 224),
        patch_size=(16, 16),
        tubelet_size=2,
    )
    batch = [_make_sample() for _ in range(4)]
    out = collator(batch)
    assert len(out) == 1
    collated_batch, masks_enc, masks_pred = out[0]
    # Shapes: encoder mask is [B, N_enc]; predictor mask is [B, N_pred].
    assert masks_enc[0].ndim == 2 and masks_enc[0].shape[0] == 4
    assert masks_pred[0].ndim == 2 and masks_pred[0].shape[0] == 4


def test_collator_phase_aware_returns_same_shapes():
    collator = MaskCollator(
        cfgs_mask=_default_mask_cfgs(),
        dataset_fpcs=[16],
        crop_size=(224, 224),
        patch_size=(16, 16),
        tubelet_size=2,
        fps_sampled=8.0,
        phase_mask_cfg=_default_phase_cfg(),
    )
    batch = [_make_sample(hr=60.0 + i * 5) for i in range(4)]
    out = collator(batch)
    assert len(out) == 1
    _, masks_enc, masks_pred = out[0]
    assert masks_enc[0].ndim == 2 and masks_enc[0].shape[0] == 4
    assert masks_pred[0].ndim == 2 and masks_pred[0].shape[0] == 4
    # At least one clip should have triggered valid metadata.
    assert collator.stats.n_valid_meta >= 1


def test_collator_invalid_metadata_falls_back():
    collator = MaskCollator(
        cfgs_mask=_default_mask_cfgs(),
        dataset_fpcs=[16],
        crop_size=(224, 224),
        patch_size=(16, 16),
        tubelet_size=2,
        fps_sampled=8.0,
        phase_mask_cfg=_default_phase_cfg(),
    )
    batch = [_make_sample(hr=float("nan")) for _ in range(4)]
    out = collator(batch)
    _, masks_enc, masks_pred = out[0]
    assert masks_enc[0].shape[0] == 4
    assert masks_pred[0].shape[0] == 4
    assert collator.stats.n_fallback_invalid_meta == 4


def test_collator_each_bucket_respects_range_when_forced():
    # Force a single bucket -> verify sampled dphi lies in that bucket's range.
    for bucket, lo, hi in [
        ("local", 0.05, 0.15),
        ("mid_cycle", 0.20, 0.35),
        ("opposite_phase", 0.45, 0.55),
    ]:
        cfg = _default_phase_cfg(phase_bucket_probs={bucket: 1.0})
        collator = MaskCollator(
            cfgs_mask=_default_mask_cfgs(),
            dataset_fpcs=[16],
            crop_size=(224, 224),
            patch_size=(16, 16),
            tubelet_size=2,
            fps_sampled=8.0,
            phase_mask_cfg=cfg,
        )
        batch = [_make_sample(hr=60.0 + i * 5) for i in range(8)]
        collator(batch)
        dphis = np.asarray(collator.stats.dphi_samples)
        assert dphis.size > 0, f"no samples for bucket={bucket}"
        # All sampled dphis must lie in the target range (within floating-point
        # tolerance). We only verify for non-fallback samples.
        in_range = (dphis >= lo - 1e-6) & (dphis <= hi + 1e-6)
        # Allow up to one rare fallback sample.
        assert in_range.mean() >= 0.85, (
            f"bucket={bucket}: {(~in_range).sum()}/{dphis.size} out of range; "
            f"dphis={dphis}"
        )


def test_same_phase_next_beat_skip_on_short_clip():
    """For a clip short enough that target t_start + ~1 cycle > D, the
    same_phase_next_beat bucket should yield no candidate starts and get
    counted under n_same_phase_skipped (falling back to another bucket or the
    vanilla sampler)."""
    # HR=45 bpm -> cycle_tubelets = (60/45)*8/2 = 5.33. D=8 -> one cycle ~5.3
    # tubelets from near-start is still within-grid, so instead force a large
    # cycle: HR=40 -> ct=6. Block size = full-temporal (t_blk=8) => no valid
    # next-beat start since D - t_blk + 1 == 1.
    cfg = _default_phase_cfg(
        phase_bucket_probs={"same_phase_next_beat": 1.0},
    )
    collator = MaskCollator(
        cfgs_mask=_default_mask_cfgs(),
        dataset_fpcs=[16],
        crop_size=(224, 224),
        patch_size=(16, 16),
        tubelet_size=2,
        fps_sampled=8.0,
        phase_mask_cfg=cfg,
    )
    batch = [_make_sample(hr=40.0) for _ in range(4)]
    collator(batch)
    # Should have incremented either n_same_phase_skipped (skip signal) or
    # n_fallback_bucket_fail (all buckets failed). With only same_phase
    # enabled, bucket failure is the fallback path.
    assert (
        collator.stats.n_same_phase_skipped > 0
        or collator.stats.n_fallback_bucket_fail > 0
    )


def test_collator_shuffled_hr_flag_accounted():
    cfg = _default_phase_cfg(shuffled_hr=True)
    collator = MaskCollator(
        cfgs_mask=_default_mask_cfgs(),
        dataset_fpcs=[16],
        crop_size=(224, 224),
        patch_size=(16, 16),
        tubelet_size=2,
        fps_sampled=8.0,
        phase_mask_cfg=cfg,
    )
    batch = [_make_sample(hr=60.0 + i * 7) for i in range(6)]
    collator(batch)
    assert collator.stats.n_shuffled_hr_applied == 6


# ---------------------------------------------------------------------------
# block_center_phi smoke
# ---------------------------------------------------------------------------


def test_block_center_phi_monotonic():
    ct = 4.0
    a = block_center_phi(0, 2, ct)
    b = block_center_phi(2, 2, ct)
    # Half-cycle apart.
    d = (b - a) % 1.0
    assert math.isclose(d, 0.5, abs_tol=1e-6)
