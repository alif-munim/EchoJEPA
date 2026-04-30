"""Phase-aware masking helpers for phi-JEPA.

This module provides the primitives for *relative* cardiac-cycle target
selection. All quantities are phase fractions in [0, 1) modulo 1 cycle; no
ED/ES anchoring is assumed and the phase origin is arbitrary.

Important unit convention:
    * DICOM ``FrameTime`` describes the *native* acquisition timing.
    * The V-JEPA dataloader resamples every clip to a uniform ``fps`` before
      the encoder / predictor see it, and masking operates on that uniformly
      sampled tubelet grid.
    * Therefore phase-aware masking uses *sampled-sequence* phase:

          cycle_frames_sampled = (60.0 / hr_bpm) * fps_sampled
          cycle_tubelets       = cycle_frames_sampled / tubelet_size
          phi_at_tubelet_i     = (i / cycle_tubelets) % 1.0

      ``frame_time_ms`` is used only for metadata validation (must be finite,
      positive) -- it is not arithmetically part of the Δφ used here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from multiprocessing import Array, Value
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


HR_LO = 40.0
HR_HI = 180.0


def validate_hr(
    hr_bpm: Optional[float],
    frame_time_ms: Optional[float],
    num_frames: Optional[int],
    hr_lo: float = HR_LO,
    hr_hi: float = HR_HI,
) -> Tuple[bool, str]:
    """Return (ok, reason). ok=True means metadata can be used for phase-aware
    sampling. FrameTime is validated (finite, positive) but not required to be
    used arithmetically -- the encoder sees uniformly resampled frames.
    ``num_frames`` is validated to be a positive int.
    """
    if hr_bpm is None or not np.isfinite(float(hr_bpm)):
        return False, "hr_nan"
    hr_f = float(hr_bpm)
    if hr_f < hr_lo or hr_f > hr_hi:
        return False, "hr_out_of_range"
    if frame_time_ms is None or not np.isfinite(float(frame_time_ms)):
        return False, "ft_nan"
    if float(frame_time_ms) <= 0.0:
        return False, "ft_nonpositive"
    if num_frames is None:
        return False, "nf_missing"
    try:
        if int(num_frames) <= 0:
            return False, "nf_nonpositive"
    except (TypeError, ValueError):
        return False, "nf_bad"
    return True, "ok"


# ---------------------------------------------------------------------------
# Phase math (sampled-sequence)
# ---------------------------------------------------------------------------


def cycle_tubelets_from_hr(hr_bpm: float, fps_sampled: float, tubelet_size: int) -> float:
    """Length of one cardiac cycle measured in tubelets on the sampled grid."""
    cycle_frames_sampled = (60.0 / float(hr_bpm)) * float(fps_sampled)
    return cycle_frames_sampled / float(tubelet_size)


def tubelet_to_phi(tubelet_idx: np.ndarray, cycle_tubelets: float) -> np.ndarray:
    """Wrap tubelet index -> phase fraction in [0, 1)."""
    return np.mod(np.asarray(tubelet_idx, dtype=np.float64) / float(cycle_tubelets), 1.0)


def circular_mean(phis: Sequence[float]) -> float:
    """Circular mean of phases in [0, 1) -> phase in [0, 1).

    Uses the sin/cos vector average so that e.g. mean([0.98, 0.02]) ≈ 0.0,
    not 0.5.
    """
    arr = np.asarray(list(phis), dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    angles = 2.0 * math.pi * arr
    c = float(np.mean(np.cos(angles)))
    s = float(np.mean(np.sin(angles)))
    if c == 0.0 and s == 0.0:
        # Perfectly antipodal -- circular mean is undefined. Return nan.
        return float("nan")
    theta = math.atan2(s, c)
    return (theta / (2.0 * math.pi)) % 1.0


def dphi_fraction(phi_c: float, phi_y: float) -> float:
    """Relative phase displacement (phi_y - phi_c) wrapped to [0, 1)."""
    return (float(phi_y) - float(phi_c)) % 1.0


def circular_dist(phi_a: float, phi_b: float) -> float:
    """Shortest circular distance between two phases, in [0, 0.5]."""
    d = abs(float(phi_a) - float(phi_b)) % 1.0
    return min(d, 1.0 - d)


# ---------------------------------------------------------------------------
# Bucket definitions
# ---------------------------------------------------------------------------


@dataclass
class PhaseBucketSpec:
    """Per-bucket configuration."""

    name: str
    # dphi interval in [0, 1); only meaningful for ordinary buckets.
    lo: float = 0.0
    hi: float = 0.0
    # same_phase_next_beat: target center must be ~1 cycle away in TIME from
    # the context center, regardless of the dphi interval (which is ~0).
    next_beat: bool = False
    next_beat_tolerance: float = 0.10  # ±10% of cycle length on dphi magnitude
    next_beat_min_extra_tubelets: int = 0  # filled at runtime from cycle


def default_bucket_specs() -> Dict[str, PhaseBucketSpec]:
    return {
        "local": PhaseBucketSpec("local", 0.05, 0.15),
        "mid_cycle": PhaseBucketSpec("mid_cycle", 0.20, 0.35),
        "opposite_phase": PhaseBucketSpec("opposite_phase", 0.45, 0.55),
        "same_phase_next_beat": PhaseBucketSpec("same_phase_next_beat", next_beat=True),
    }


def parse_bucket_cfg(
    buckets_cfg: Optional[dict],
    bucket_probs: Optional[dict],
) -> Tuple[Dict[str, PhaseBucketSpec], Dict[str, float]]:
    """Merge user-provided bucket ranges + probabilities onto the defaults."""
    specs = default_bucket_specs()
    if buckets_cfg:
        for name, val in buckets_cfg.items():
            if name not in specs:
                specs[name] = PhaseBucketSpec(name)
            if name == "same_phase_next_beat":
                # Allow overriding via bool or dict.
                if isinstance(val, dict):
                    specs[name].next_beat = bool(val.get("enabled", True))
                    specs[name].next_beat_tolerance = float(
                        val.get("tolerance", specs[name].next_beat_tolerance)
                    )
                else:
                    specs[name].next_beat = bool(val)
            else:
                lo, hi = float(val[0]), float(val[1])
                if not (0.0 <= lo < hi <= 1.0):
                    raise ValueError(
                        f"phase bucket '{name}' must satisfy 0<=lo<hi<=1, got [{lo},{hi}]"
                    )
                specs[name].lo = lo
                specs[name].hi = hi

    probs = {name: 0.0 for name in specs}
    if bucket_probs:
        for name, p in bucket_probs.items():
            if name not in specs:
                raise ValueError(f"phase_bucket_probs has unknown bucket '{name}'")
            probs[name] = float(p)
    else:
        # Uniform over enabled (non-disabled) buckets.
        enabled = [
            n for n, s in specs.items() if (not s.next_beat) or s.next_beat
        ]
        for n in enabled:
            probs[n] = 1.0 / max(1, len(enabled))

    # Normalize over buckets with positive probability only.
    total = sum(max(0.0, p) for p in probs.values())
    if total <= 0:
        raise ValueError("phase_bucket_probs sums to 0; enable at least one bucket")
    probs = {n: max(0.0, p) / total for n, p in probs.items()}
    return specs, probs


def choose_bucket(probs: Dict[str, float], rng: np.random.Generator) -> str:
    names = list(probs.keys())
    p = np.asarray([probs[n] for n in names], dtype=np.float64)
    idx = int(rng.choice(len(names), p=p))
    return names[idx]


# ---------------------------------------------------------------------------
# Block-level phase helpers (on the sampled tubelet grid)
# ---------------------------------------------------------------------------


def block_center_phi(
    t_start: int,
    t_len: int,
    cycle_tubelets: float,
) -> float:
    """Circular mean of per-tubelet phases spanned by a block of length t_len
    starting at tubelet index t_start (inclusive).
    """
    if t_len <= 0:
        return float("nan")
    idxs = np.arange(t_start, t_start + t_len, dtype=np.float64)
    phis = tubelet_to_phi(idxs, cycle_tubelets)
    return circular_mean(phis.tolist())


def block_center_time_tubelets(t_start: int, t_len: int) -> float:
    """Midpoint of a block in tubelet units, ignoring phase wrap."""
    return float(t_start) + 0.5 * float(max(0, t_len - 1))


# ---------------------------------------------------------------------------
# Shuffled-HR (in-batch derangement)
# ---------------------------------------------------------------------------


@dataclass
class ShuffledHrResult:
    shuffled: List[float]
    was_applied: bool
    derangement_ok: bool
    n_fixed_points: int


def apply_shuffled_hr(
    hr_bpm: Sequence[float],
    rng: np.random.Generator,
    max_tries: int = 20,
) -> ShuffledHrResult:
    """Return a permutation of ``hr_bpm`` that is a derangement when possible.

    * If batch size < 2 or all HRs equal, returns the inputs unchanged with
      ``was_applied=False``.
    * NaN entries are carried through at their new position (the permutation
      shuffles index positions, not values).
    * If a derangement cannot be constructed in ``max_tries``, falls back to a
      random permutation and reports ``derangement_ok=False``.
    """
    arr = np.asarray(list(hr_bpm), dtype=np.float64)
    n = arr.size
    if n < 2:
        return ShuffledHrResult(list(arr), False, False, 0)

    # Degenerate: all same (or all nan) -> derangement is meaningless.
    finite = arr[np.isfinite(arr)]
    if finite.size == 0 or (finite.size > 0 and np.all(finite == finite[0])):
        return ShuffledHrResult(list(arr), False, False, 0)

    for _ in range(max_tries):
        perm = rng.permutation(n)
        fixed = np.sum(perm == np.arange(n))
        if fixed == 0:
            return ShuffledHrResult(list(arr[perm]), True, True, 0)

    # Fallback: accept a random permutation even if not a derangement.
    perm = rng.permutation(n)
    fixed = int(np.sum(perm == np.arange(n)))
    return ShuffledHrResult(list(arr[perm]), True, False, fixed)


# ---------------------------------------------------------------------------
# Lightweight stats accumulator (used by MaskCollator on rank 0)
# ---------------------------------------------------------------------------


class PhaseMaskStats:
    """Process-shared phase-mask diagnostics.

    The collator's ``__call__`` runs inside DataLoader worker subprocesses (one
    worker per ``num_workers``). Plain Python attributes live in the worker's
    own address space and never propagate back to the main process, so any
    counters based on ordinary ints/lists would stay at zero from the main
    process's perspective. This class stores every counter in
    ``multiprocessing.Value`` / ``multiprocessing.Array`` backed by shared
    memory, so worker updates are visible to the main process at end-of-epoch.

    Bucket names are registered up-front (at collator construction) so we can
    preallocate fixed-size shared-memory slots for each.

    dphi and cycle_tubelets samples are collected into capped shared ring
    buffers (default 16384 slots). When the buffer fills in one epoch, older
    samples are overwritten -- this is fine for summary stats.
    """

    def __init__(
        self,
        bucket_names: Sequence[str],
        dphi_buf_size: int = 16384,
        cycle_buf_size: int = 16384,
    ):
        # Stable bucket index.
        self._bucket_index = {n: i for i, n in enumerate(bucket_names)}
        n_buckets = len(bucket_names)

        # Scalar counters (int64).
        self._n_clips = Value("q", 0)
        self._n_valid_meta = Value("q", 0)
        self._n_fallback_invalid_meta = Value("q", 0)
        self._n_fallback_bucket_fail = Value("q", 0)
        self._n_shuffled_hr_applied = Value("q", 0)
        self._n_same_phase_skipped = Value("q", 0)

        # Bucket counts (int64 per bucket).
        self._bucket_counts = Array("q", [0] * max(1, n_buckets))
        self._bucket_fail_counts = Array("q", [0] * max(1, n_buckets))

        # Ring buffers for dphi and cycle_tubelets samples (float64).
        self._dphi_buf = Array("d", [0.0] * dphi_buf_size)
        self._dphi_idx = Value("q", 0)  # total writes (monotonic); mod size = slot
        self._cycle_buf = Array("d", [0.0] * cycle_buf_size)
        self._cycle_idx = Value("q", 0)
        self._dphi_cap = dphi_buf_size
        self._cycle_cap = cycle_buf_size

    # -- counter-like API (kept compatible with the old dataclass) ----------

    @property
    def n_clips(self) -> int:
        return self._n_clips.value

    @n_clips.setter
    def n_clips(self, v: int) -> None:
        with self._n_clips.get_lock():
            self._n_clips.value = int(v)

    def _inc(self, slot: "Value", delta: int = 1) -> None:
        with slot.get_lock():
            slot.value += int(delta)

    @property
    def n_valid_meta(self) -> int:
        return self._n_valid_meta.value

    @n_valid_meta.setter
    def n_valid_meta(self, v: int) -> None:
        with self._n_valid_meta.get_lock():
            self._n_valid_meta.value = int(v)

    @property
    def n_fallback_invalid_meta(self) -> int:
        return self._n_fallback_invalid_meta.value

    @n_fallback_invalid_meta.setter
    def n_fallback_invalid_meta(self, v: int) -> None:
        with self._n_fallback_invalid_meta.get_lock():
            self._n_fallback_invalid_meta.value = int(v)

    @property
    def n_fallback_bucket_fail(self) -> int:
        return self._n_fallback_bucket_fail.value

    @n_fallback_bucket_fail.setter
    def n_fallback_bucket_fail(self, v: int) -> None:
        with self._n_fallback_bucket_fail.get_lock():
            self._n_fallback_bucket_fail.value = int(v)

    @property
    def n_shuffled_hr_applied(self) -> int:
        return self._n_shuffled_hr_applied.value

    @n_shuffled_hr_applied.setter
    def n_shuffled_hr_applied(self, v: int) -> None:
        with self._n_shuffled_hr_applied.get_lock():
            self._n_shuffled_hr_applied.value = int(v)

    @property
    def n_same_phase_skipped(self) -> int:
        return self._n_same_phase_skipped.value

    @n_same_phase_skipped.setter
    def n_same_phase_skipped(self, v: int) -> None:
        with self._n_same_phase_skipped.get_lock():
            self._n_same_phase_skipped.value = int(v)

    # Support ``stats.n_clips += 1`` style from the collator code which
    # translates into __iadd__ on the property setter; Python will read, add,
    # write. That's racy across workers without the lock, so expose explicit
    # add methods and have callers use them instead.

    def add_clips(self, delta: int = 1) -> None:
        self._inc(self._n_clips, delta)

    def add_valid_meta(self, delta: int = 1) -> None:
        self._inc(self._n_valid_meta, delta)

    def add_fallback_invalid_meta(self, delta: int = 1) -> None:
        self._inc(self._n_fallback_invalid_meta, delta)

    def add_fallback_bucket_fail(self, delta: int = 1) -> None:
        self._inc(self._n_fallback_bucket_fail, delta)

    def add_shuffled_hr_applied(self, delta: int = 1) -> None:
        self._inc(self._n_shuffled_hr_applied, delta)

    def add_same_phase_skipped(self, delta: int = 1) -> None:
        self._inc(self._n_same_phase_skipped, delta)

    def inc_bucket(self, name: str) -> None:
        idx = self._bucket_index.get(name)
        if idx is None:
            return
        with self._bucket_counts.get_lock():
            self._bucket_counts[idx] += 1

    def inc_bucket_fail(self, name: str) -> None:
        idx = self._bucket_index.get(name)
        if idx is None:
            return
        with self._bucket_fail_counts.get_lock():
            self._bucket_fail_counts[idx] += 1

    # -- sample buffers ----------------------------------------------------

    def push_dphi(self, values: Iterable[float]) -> None:
        with self._dphi_idx.get_lock():
            i = self._dphi_idx.value
            # Lock both the counter and the buffer together to avoid torn
            # writes when two workers wrap around.
            with self._dphi_buf.get_lock():
                for v in values:
                    self._dphi_buf[i % self._dphi_cap] = float(v)
                    i += 1
                self._dphi_idx.value = i

    def push_cycle(self, values: Iterable[float]) -> None:
        with self._cycle_idx.get_lock():
            i = self._cycle_idx.value
            with self._cycle_buf.get_lock():
                for v in values:
                    self._cycle_buf[i % self._cycle_cap] = float(v)
                    i += 1
                self._cycle_idx.value = i

    # Back-compat shims for the old list-based API (used by tests that
    # inspect `stats.dphi_samples` directly).
    @property
    def dphi_samples(self) -> List[float]:
        with self._dphi_idx.get_lock():
            n = min(int(self._dphi_idx.value), self._dphi_cap)
            with self._dphi_buf.get_lock():
                return [self._dphi_buf[i] for i in range(n)]

    @property
    def cycle_tubelets_samples(self) -> List[float]:
        with self._cycle_idx.get_lock():
            n = min(int(self._cycle_idx.value), self._cycle_cap)
            with self._cycle_buf.get_lock():
                return [self._cycle_buf[i] for i in range(n)]

    @property
    def bucket_counts(self) -> Dict[str, int]:
        with self._bucket_counts.get_lock():
            return {
                name: int(self._bucket_counts[i])
                for name, i in self._bucket_index.items()
            }

    @property
    def bucket_fail_counts(self) -> Dict[str, int]:
        with self._bucket_fail_counts.get_lock():
            return {
                name: int(self._bucket_fail_counts[i])
                for name, i in self._bucket_index.items()
            }

    # -- summary / reset ---------------------------------------------------

    def summarize(self) -> Dict[str, float]:
        out: Dict[str, float] = {
            "n_clips": float(self.n_clips),
            "n_valid_meta": float(self.n_valid_meta),
            "n_fallback_invalid_meta": float(self.n_fallback_invalid_meta),
            "n_fallback_bucket_fail": float(self.n_fallback_bucket_fail),
            "n_shuffled_hr_applied": float(self.n_shuffled_hr_applied),
            "n_same_phase_skipped": float(self.n_same_phase_skipped),
        }
        for name, count in self.bucket_counts.items():
            out[f"bucket_{name}"] = float(count)
        for name, count in self.bucket_fail_counts.items():
            out[f"bucket_fail_{name}"] = float(count)
        dsamp = self.dphi_samples
        arr = np.asarray(dsamp, dtype=np.float64) if dsamp else np.empty(0)
        if arr.size:
            out["dphi_mean"] = float(arr.mean())
            out["dphi_std"] = float(arr.std())
            out["dphi_p05"] = float(np.quantile(arr, 0.05))
            out["dphi_p50"] = float(np.quantile(arr, 0.50))
            out["dphi_p95"] = float(np.quantile(arr, 0.95))
        csamp = self.cycle_tubelets_samples
        ct = np.asarray(csamp, dtype=np.float64) if csamp else np.empty(0)
        if ct.size:
            out["cycle_tubelets_mean"] = float(ct.mean())
            out["cycle_tubelets_std"] = float(ct.std())
            out["cycle_tubelets_p05"] = float(np.quantile(ct, 0.05))
            out["cycle_tubelets_p95"] = float(np.quantile(ct, 0.95))
        return out

    def reset(self) -> None:
        """Reset all shared-memory counters and ring-buffer indices. Called
        on the main process at end of each epoch. Workers will continue to
        write into the same (now-empty) buffers on the next epoch."""
        for slot in (
            self._n_clips,
            self._n_valid_meta,
            self._n_fallback_invalid_meta,
            self._n_fallback_bucket_fail,
            self._n_shuffled_hr_applied,
            self._n_same_phase_skipped,
            self._dphi_idx,
            self._cycle_idx,
        ):
            with slot.get_lock():
                slot.value = 0
        with self._bucket_counts.get_lock():
            for i in range(len(self._bucket_counts)):
                self._bucket_counts[i] = 0
        with self._bucket_fail_counts.get_lock():
            for i in range(len(self._bucket_fail_counts)):
                self._bucket_fail_counts[i] = 0
