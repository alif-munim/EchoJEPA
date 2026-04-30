"""Exercise _compute_clip_indices against the spec's anchor cases.

Verifies anchor-centered indexing for the PhaseMatchedStudySampler before
training is launched. Uses no real video data — this is a pure-math check
on the index-selection function.

Expected spec cases (printed and asserted):
  n_frames=100, fpc=16, frame_step=1, anchor=50  -> center ~42..57
  n_frames=100, fpc=16, frame_step=2, anchor=50  -> center ~35,37,...,65
  anchor=2   -> clamp to start
  anchor=98  -> clamp to end
  anchor=None -> legacy strided-from-0 behavior
  very short video (n_frames=10) -> padded, indices <= 9
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Add the repo root so `src.datasets.video_group_dataset` imports.
VJEPA_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(VJEPA_ROOT))

# Stub decord before the dataset imports (CPU-only dev box).
try:
    import decord  # noqa: F401
except ImportError:
    import types

    mod = types.ModuleType("decord")
    mod.VideoReader = object
    mod.cpu = lambda: None
    sys.modules["decord"] = mod

from src.datasets.video_group_dataset import _compute_clip_indices  # noqa: E402


def _line(s):
    print("-" * 72)
    print(s)


def _pp(name, inds, meta):
    print(f"  {name}: indices={inds.tolist()}")
    print(f"           meta={meta}")


def check_anchor_math() -> None:
    _line("CASE 1  n_frames=100, fpc=16, frame_step=1, anchor=50")
    inds, meta = _compute_clip_indices(100, 16, frame_step=1, anchor_frame=50)
    _pp("fs=1 anchor=50", inds, meta)
    # Spec expects "roughly 42..57". With Python's banker's rounding
    # round(50 - 7.5) -> 42, so indices are 42..57. anchor_pos == 8
    # (distance 0 between indices[8]=50 and anchor=50).
    assert inds[0] == 42 and inds[-1] == 57, f"expected 42..57, got {inds[0]}..{inds[-1]}"
    assert meta["anchor_pos"] == 8, f"anchor_pos expected 8, got {meta['anchor_pos']}"
    assert meta["was_clamped"] is False
    assert meta["source_span_frames"] == 16
    print("  OK: anchor 50 at position 8 of 16 (near-center), no clamp.")

    _line("CASE 2  n_frames=100, fpc=16, frame_step=2, anchor=50")
    inds, meta = _compute_clip_indices(100, 16, frame_step=2, anchor_frame=50)
    _pp("fs=2 anchor=50", inds, meta)
    # Expected: source_span = 31, start = round(50 - 15) = 35, indices = 35,37,...,65.
    assert inds[0] == 35 and inds[-1] == 65, f"expected 35..65 stride 2, got {inds[0]}..{inds[-1]}"
    assert (np.diff(inds) == 2).all()
    assert meta["anchor_pos"] == 7, f"anchor_pos expected 7 (closest to 50 at index 49), got {meta['anchor_pos']}"
    assert meta["source_span_frames"] == 31
    assert meta["was_clamped"] is False
    print("  OK: stride 2, span 31, centered on 50 -> 35..65.")

    _line("CASE 3  anchor=2 clamps to start (n_frames=100, fpc=16, fs=1)")
    inds, meta = _compute_clip_indices(100, 16, frame_step=1, anchor_frame=2)
    _pp("fs=1 anchor=2", inds, meta)
    assert inds[0] == 0 and inds[-1] == 15, f"expected 0..15, got {inds[0]}..{inds[-1]}"
    assert meta["was_clamped"] is True
    assert meta["anchor_pos"] == 2
    print("  OK: clamped to [0..15]; anchor_frame=2 at index 2.")

    _line("CASE 4  anchor=98 clamps to end (n_frames=100, fpc=16, fs=1)")
    inds, meta = _compute_clip_indices(100, 16, frame_step=1, anchor_frame=98)
    _pp("fs=1 anchor=98", inds, meta)
    # Expected: end frame is n_frames-1 = 99; source_span = 16 -> start = 84
    assert inds[0] == 84 and inds[-1] == 99, f"expected 84..99, got {inds[0]}..{inds[-1]}"
    assert meta["was_clamped"] is True
    # anchor 98 lands at index 14 of indices 84..99.
    assert meta["anchor_pos"] == 14
    print("  OK: clamped to [84..99]; anchor_frame=98 at index 14.")

    _line("CASE 5  anchor=None returns legacy strided-from-0 (single clip)")
    inds, meta = _compute_clip_indices(100, 16, frame_step=1, anchor_frame=None)
    _pp("fs=1 anchor=None", inds, meta)
    assert inds[0] == 0 and inds[-1] == 15
    assert meta["anchor_pos"] is None
    assert meta["anchor_frame"] is None
    print("  OK: legacy behavior unchanged.")

    _line("CASE 6  anchor=None, frame_step=2, legacy strided-from-0")
    inds, meta = _compute_clip_indices(100, 16, frame_step=2, anchor_frame=None)
    _pp("fs=2 anchor=None", inds, meta)
    # Legacy strided-from-0: start=0, stride 2 -> 0,2,4,...,30.
    assert inds[0] == 0 and inds[-1] == 30
    assert (np.diff(inds) == 2).all()
    print("  OK: legacy strided (fs=2) starts at 0.")

    _line("CASE 7  short video (n=10, fpc=16, fs=1, anchor=5) -> padded")
    inds, meta = _compute_clip_indices(10, 16, frame_step=1, anchor_frame=5)
    _pp("n=10 anchor=5", inds, meta)
    assert inds.max() <= 9, f"indices must stay in [0,9], got {inds}"
    assert meta["padded"] is True
    print("  OK: short video padded with last valid index.")

    _line("CASE 8  K=2 anchor-centered single-anchor at V=300, fpc=16, fs=1")
    # Phase-matched sampler uses K=1; K>1 with a single int anchor is
    # redundant. We still want the helper to be sane when called with K>1
    # (each clip gets the same window).
    inds1, meta1 = _compute_clip_indices(300, 16, frame_step=1, clip_idx=0, num_clips=2, anchor_frame=150)
    inds2, meta2 = _compute_clip_indices(300, 16, frame_step=1, clip_idx=1, num_clips=2, anchor_frame=150)
    _pp("K=2 clip0", inds1, meta1)
    _pp("K=2 clip1", inds2, meta2)
    # For anchor mode we intentionally center every clip on the anchor;
    # callers who want distinct windows per clip should use K=1 and issue
    # one call per clip (which is what the phase-matched sampler does).
    assert inds1[0] == inds2[0]
    print("  OK: K=2 anchor mode yields identical windows (as intended).")

    _line("CASE 9  K=2 legacy strided (anchor=None) spans separate windows")
    inds1, _ = _compute_clip_indices(300, 16, frame_step=1, clip_idx=0, num_clips=2, anchor_frame=None)
    inds2, _ = _compute_clip_indices(300, 16, frame_step=1, clip_idx=1, num_clips=2, anchor_frame=None)
    assert inds1[0] == 0 and inds1[-1] == 15
    assert inds2[0] == 16 and inds2[-1] == 31
    print(f"  OK: K=2 legacy clip0=[{inds1[0]}..{inds1[-1]}], clip1=[{inds2[0]}..{inds2[-1]}].")

    print("\nALL ANCHOR-MATH CASES PASS")


if __name__ == "__main__":
    check_anchor_math()
