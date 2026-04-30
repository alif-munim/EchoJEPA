"""RR-interval consistency utilities for phase_annotations parquet rows.

The quality-tier gate on the parquet uses ``rpeak_ratio_dist`` (symmetric
log-ratio of detected-beat-count vs expected count given metadata HR and
clip duration). That metric is count-vs-duration-consistent and accepts
clips where the detector marked every 2nd or 4th beat. For phase-matched
training those clips are toxic because all frames inside a "single RR"
get labeled linearly [0,1) while spanning multiple real cycles.

Two independent checks:

  median_vs_metadata:
      median(RR) / (60 * fps / HR_metadata) must be in [0.80, 1.25].

  max_min_rr_ratio:
      max(RR) / min(RR) <= 1.40 catches missed beats inside an otherwise-
      good clip (one interval becomes 2x the others).

The second check is *intentionally not applied* by the training-time
sampler: AFib and other legitimate arrhythmias produce beat-to-beat RR
variance above that threshold, and we want those patients in. Use it
only for visualization candidate selection.
"""

from __future__ import annotations

import json
from typing import Iterable

import numpy as np
import pandas as pd


def _parse_r_peaks(val) -> np.ndarray | None:
    """Accept JSON string, list, or numpy-like; return int64 1D array or None."""
    if val is None:
        return None
    try:
        if isinstance(val, (str, bytes)):
            obj = json.loads(val)
        else:
            obj = val
        arr = np.asarray(obj, dtype=np.int64).reshape(-1)
        return arr
    except Exception:
        return None


def _metadata_cycle_frames(hr_bpm, fps) -> float | None:
    try:
        hr = float(hr_bpm)
        f = float(fps)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(hr) or not np.isfinite(f) or hr <= 0 or f <= 0:
        return None
    return 60.0 * f / hr


def rr_stats(row) -> dict:
    """Return diagnostic stats for a parquet row. Missing fields produce
    None for the corresponding entries."""
    rp = _parse_r_peaks(getattr(row, "r_peaks_video_json", None))
    out = {
        "n_rpeaks_video": int(rp.size) if rp is not None else 0,
        "n_rr_intervals": 0,
        "median_rr_frames": None,
        "max_min_rr_ratio": None,
        "metadata_cycle_frames": _metadata_cycle_frames(
            getattr(row, "hr_metadata", None), getattr(row, "fps_video", None)
        ),
        "rr_median_meta_ratio": None,
    }
    if rp is None or rp.size < 2:
        return out
    rr = np.diff(rp)
    if (rr <= 0).any():
        # Non-monotonic R-peak list — treat as not-consistent.
        return out
    out["n_rr_intervals"] = int(rr.size)
    med = float(np.median(rr))
    out["median_rr_frames"] = med
    mn, mx = float(rr.min()), float(rr.max())
    out["max_min_rr_ratio"] = mx / mn if mn > 0 else float("inf")
    if out["metadata_cycle_frames"] is not None and out["metadata_cycle_frames"] > 0:
        out["rr_median_meta_ratio"] = med / out["metadata_cycle_frames"]
    return out


def rr_consistent(
    row,
    median_tol: tuple[float, float] = (0.80, 1.25),
    max_min_rr_ratio: float | None = 1.40,
) -> bool:
    """Reject every-Nth-beat / missed-beat detector failures.

    ``median_tol``: median(RR) / metadata_cycle_frames must fall in this
    range. Default [0.80, 1.25].

    ``max_min_rr_ratio``: if set, max(RR)/min(RR) must be <= this value.
    Default 1.40. **Set to None for training-time use** — AFib legitimately
    violates this bound and we want those patients in training.

    Returns False on any insufficient/missing input.
    """
    stats = rr_stats(row)
    if stats["n_rpeaks_video"] < 2 or stats["n_rr_intervals"] < 1:
        return False
    if stats["rr_median_meta_ratio"] is None:
        return False
    lo, hi = median_tol
    if not (lo <= stats["rr_median_meta_ratio"] <= hi):
        return False
    if max_min_rr_ratio is not None:
        if stats["max_min_rr_ratio"] is None:
            return False
        if stats["max_min_rr_ratio"] > max_min_rr_ratio:
            return False
    return True


def add_rr_consistency_columns(
    df: pd.DataFrame,
    median_tol: tuple[float, float] = (0.80, 1.25),
    max_min_rr_ratio: float | None = 1.40,
) -> pd.DataFrame:
    """Return a copy of ``df`` with per-row RR consistency columns added:

      rr_consistent            (bool — both layers applied)
      median_rr_frames         (float or nan)
      metadata_cycle_frames    (float or nan)
      rr_max_min_ratio         (float or nan)
      rr_median_meta_ratio     (float or nan)
      n_rpeaks_video           (int)
      n_rr_intervals           (int)

    ``max_min_rr_ratio=None`` disables the second layer (use for training).
    """
    need = {"r_peaks_video_json", "hr_metadata", "fps_video"}
    missing = need - set(df.columns)
    if missing:
        raise KeyError(f"add_rr_consistency_columns: missing columns {missing}")
    records = []
    for row in df.itertuples(index=False):
        s = rr_stats(row)
        s["rr_consistent"] = rr_consistent(row, median_tol=median_tol, max_min_rr_ratio=max_min_rr_ratio)
        records.append(s)
    stats_df = pd.DataFrame(records, index=df.index)
    stats_df = stats_df.rename(columns={"max_min_rr_ratio": "rr_max_min_ratio"})
    out = df.copy()
    for col in [
        "rr_consistent",
        "median_rr_frames",
        "metadata_cycle_frames",
        "rr_max_min_ratio",
        "rr_median_meta_ratio",
        "n_rpeaks_video",
        "n_rr_intervals",
    ]:
        out[col] = stats_df[col].values
    return out


__all__ = [
    "rr_stats",
    "rr_consistent",
    "add_rr_consistency_columns",
]
