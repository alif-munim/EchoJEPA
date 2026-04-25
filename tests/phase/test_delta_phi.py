"""Hand-verified test cases for the Δφ formula in phase_utils.compute_delta_phi.

Convention:
  target_t, ctx_t are tubelet indices (0..D-1) where D = num_frames / tubelet_size.
  fps_sampled is the dataloader's sampled fps (default 8).
  tubelet_size default 2.

At fps=8, tubelet=2:  seconds_per_tubelet = 2/8 = 0.25 s.

Case table:
  HR=60, target_t=4, ctx_t=0:
    1 s elapsed between ctx and tgt -> at 60 bpm that's 1/1 = 1.0 cycle.
  HR=60, target_t=2, ctx_t=0:
    0.5 s -> 0.5 cycle.
  HR=120, target_t=2, ctx_t=0:
    0.5 s -> 0.5 * 120/60 = 1.0 cycle.
  HR=60, target_t=0, ctx_t=4:
    -1.0 cycle (sign convention).
  HR=nan (irregular):
    nan (propagates).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

# Make the repo importable when running via `pytest tests/phase` from repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.datasets.phase_utils import compute_delta_phi  # noqa: E402


FPS = 8.0
TUBELET = 2


def _dphi(hr, tgt, ctx):
    target_t = torch.tensor([[float(tgt)]], dtype=torch.float32)  # [B=1, N=1]
    ctx_t = torch.tensor([float(ctx)], dtype=torch.float32)       # [B=1]
    hr_bpm = torch.tensor([float(hr)], dtype=torch.float32)        # [B=1]
    out = compute_delta_phi(target_t, ctx_t, hr_bpm, FPS, TUBELET)
    return out.item()


def test_full_cycle_at_hr60():
    # 1 second between frames at 60 bpm -> 1.0 cycle.
    assert math.isclose(_dphi(60, 4, 0), 1.0, rel_tol=1e-6)


def test_half_cycle_at_hr60():
    assert math.isclose(_dphi(60, 2, 0), 0.5, rel_tol=1e-6)


def test_hr_scaling():
    # 0.5 s at 120 bpm = 1.0 cycle.
    assert math.isclose(_dphi(120, 2, 0), 1.0, rel_tol=1e-6)


def test_sign_convention():
    # target before context -> negative delta_phi.
    assert math.isclose(_dphi(60, 0, 4), -1.0, rel_tol=1e-6)


def test_nan_hr_propagates():
    val = _dphi(float("nan"), 2, 0)
    assert math.isnan(val)


def test_zero_offset():
    assert math.isclose(_dphi(72, 3, 3), 0.0, rel_tol=1e-6, abs_tol=1e-7)


def test_batched_shape():
    # B=2, N=3 targets
    target_t = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.float32)
    ctx_t = torch.tensor([0.0, 1.0], dtype=torch.float32)
    hr_bpm = torch.tensor([60.0, 60.0], dtype=torch.float32)
    out = compute_delta_phi(target_t, ctx_t, hr_bpm, FPS, TUBELET)
    assert out.shape == (2, 3)
    # Row 0: (0,1,2) - 0 = (0, 0.25, 0.5 cycle) at HR=60
    assert torch.allclose(out[0], torch.tensor([0.0, 0.25, 0.5]), atol=1e-6)
    # Row 1: (1,2,3) - 1 = (0, 0.25, 0.5 cycle)
    assert torch.allclose(out[1], torch.tensor([0.0, 0.25, 0.5]), atol=1e-6)


def test_typical_16frame_clip_at_hr70():
    # A 16-frame clip at sampled fps=8 spans 2.0 s total.
    # tubelets 0..7 (D = 16/2 = 8).
    # ctx_t at tubelet 2, target at tubelet 6 -> 4 tubelets apart -> 1.0 s.
    # At HR=70: 1.0 * 70 / 60 = ~1.167 cycle.
    val = _dphi(70, 6, 2)
    expected = (6 - 2) * (TUBELET / FPS) * 70.0 / 60.0
    assert math.isclose(val, expected, rel_tol=1e-6)
    assert 1.15 < val < 1.18


if __name__ == "__main__":
    # Allow direct execution without pytest.
    import traceback

    tests = [
        test_full_cycle_at_hr60,
        test_half_cycle_at_hr60,
        test_hr_scaling,
        test_sign_convention,
        test_nan_hr_propagates,
        test_zero_offset,
        test_batched_shape,
        test_typical_16frame_clip_at_hr70,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception:
            failed += 1
            print(f"  FAIL  {t.__name__}")
            traceback.print_exc()
    if failed:
        sys.exit(1)
    print(f"\n{len(tests)} tests passed")
