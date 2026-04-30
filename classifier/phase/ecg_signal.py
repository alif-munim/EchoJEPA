#!/usr/bin/env python3
"""Convert cropped ECG strip images to 1D amplitude signals and detect R-waves.

Downstream of ``crop_waveform_frame.py``. The cropped strips contain the
burned-in ECG trace as green-family pixels on a near-black background; this
module:

1. Segments trace pixels using the same criterion as extraction (saturated
   and green-dominant).
2. Filters out spurious cyan/green blobs from UI icons via connected-component
   horizontal-extent thresholding.
3. Collapses the 2D mask to a 1D amplitude signal (median trace-y per column).
4. Detects R-waves with ``scipy.signal.find_peaks`` using adaptive prominence.

The caller is responsible for determining ``sampling_rate`` (from sweep speed
or HR-based calibration) — this module deliberately does not mix calibration
with detection.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from skimage.measure import label, regionprops


# ---------------------------------------------------------------------------
# 2D → 1D conversion
# ---------------------------------------------------------------------------

def trace_mask(strip: np.ndarray, sat_thresh: int = 60) -> np.ndarray:
    """Binary mask of ECG-trace pixels.

    Uses the same criterion as the extractor: saturated
    (``max_ch - min_ch > sat_thresh``) AND green-dominant
    (``G > R`` and ``G > B``). Keeping this consistent with
    ``crop_waveform_frame.py`` ensures the downstream 1D signal sees the
    same pixel set the extractor was tuned to preserve.
    """
    assert strip.ndim == 3 and strip.shape[-1] == 3, f"expected HxWx3, got {strip.shape}"
    s = strip.astype(np.int16)
    R, G, B = s[..., 0], s[..., 1], s[..., 2]
    sat = s.max(axis=-1) - s.min(axis=-1)
    return (sat > sat_thresh) & (G > R) & (G > B)


def filter_trace_blobs(mask: np.ndarray, min_width_frac: float = 0.30) -> np.ndarray:
    """Drop localized blobs; keep the set of components that makes up the trace.

    Uses 8-connectivity. The main ECG trace frequently breaks into several
    horizontal segments (large R-wave/Q-wave excursions momentarily leave
    columns empty), so a per-component horizontal-extent threshold is too
    aggressive in practice. Instead, select components greedily in order
    of horizontal extent and keep them until the union of their
    ``[min_col, max_col)`` ranges covers at least ``min_width_frac`` of
    the strip width. Spurious UI blobs (scale markers, icons, numerals)
    are always small in horizontal extent relative to trace segments and
    get excluded naturally.
    """
    if not mask.any():
        return mask
    H, W = mask.shape
    target_coverage = max(1, int(min_width_frac * W))

    labeled = label(mask, connectivity=2)
    components: list[tuple[int, int, int]] = []  # (min_col, max_col, label)
    for prop in regionprops(labeled):
        _, min_col, _, max_col = prop.bbox
        components.append((min_col, max_col, prop.label))
    if not components:
        return mask

    # Sort by horizontal extent descending; break ties by area (via label order).
    components.sort(key=lambda c: (c[1] - c[0]), reverse=True)

    keep = np.zeros_like(mask, dtype=bool)
    kept_ranges: list[tuple[int, int]] = []

    def _covered_width(ranges: list[tuple[int, int]]) -> int:
        if not ranges:
            return 0
        ranges = sorted(ranges)
        total = 0
        cur_lo, cur_hi = ranges[0]
        for lo, hi in ranges[1:]:
            if lo <= cur_hi:
                cur_hi = max(cur_hi, hi)
            else:
                total += cur_hi - cur_lo
                cur_lo, cur_hi = lo, hi
        total += cur_hi - cur_lo
        return total

    for min_col, max_col, lbl in components:
        if _covered_width(kept_ranges) >= target_coverage:
            break
        keep[labeled == lbl] = True
        kept_ranges.append((min_col, max_col))

    if _covered_width(kept_ranges) < target_coverage:
        # Even the union of all components doesn't clear the threshold —
        # fall back to keeping nothing. The caller will see an all-NaN
        # signal and flag it via gap_fraction.
        return np.zeros_like(mask, dtype=bool)

    return keep


def cropped_to_signal(
    strip: np.ndarray,
    sat_thresh: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a cropped ECG strip image to a 1D amplitude signal.

    Parameters
    ----------
    strip : (H, W, 3) uint8 RGB array
        Cropped strip from ``crop_waveform_frame.py``.
    sat_thresh : int
        Minimum ``max_ch - min_ch`` for a pixel to count as colored.

    Returns
    -------
    amplitude : (W,) float32
        1D amplitude signal. Higher = upward (toward smaller image-y).
        NaN at columns with no detected trace.
    valid_mask : (W,) bool
        True at columns where trace pixels were detected.
    """
    mask = trace_mask(strip, sat_thresh=sat_thresh)
    mask = filter_trace_blobs(mask)

    H, W = mask.shape
    trace_y = np.full(W, np.nan, dtype=np.float32)
    valid = np.zeros(W, dtype=bool)

    # Per-column median y over trace pixels. Median is more robust than
    # centroid against anti-aliasing at peak tips and tracks the peak
    # position even when the peak is flat-topped due to edge cropping.
    ys = np.arange(H)
    for x in range(W):
        col = mask[:, x]
        if col.any():
            trace_y[x] = float(np.median(ys[col]))
            valid[x] = True

    if not valid.any():
        return trace_y, valid  # all-NaN

    # Center on the signal median so "positive = upward deflection".
    baseline_y = float(np.nanmedian(trace_y))
    amplitude = (baseline_y - trace_y).astype(np.float32)
    return amplitude, valid


# ---------------------------------------------------------------------------
# R-wave detection
# ---------------------------------------------------------------------------

def _interpolate_short_nan_gaps(
    x: np.ndarray,
    max_gap_frac: float,
) -> tuple[np.ndarray, int, float]:
    """Linearly interpolate NaN runs shorter than ``max_gap_frac * len(x)``.

    Returns the filled signal, the number of gap regions that were
    interpolated, and the original fraction of the signal that was NaN.
    """
    x = x.astype(np.float32).copy()
    W = len(x)
    gap_fraction = float(np.isnan(x).sum()) / max(1, W)
    if W == 0 or not np.isnan(x).any():
        return x, 0, gap_fraction

    max_gap = max(1, int(max_gap_frac * W))
    n_interp = 0

    # Walk contiguous NaN runs.
    i = 0
    while i < W:
        if np.isnan(x[i]):
            j = i
            while j < W and np.isnan(x[j]):
                j += 1
            # x[i:j] is a NaN run. Need values on both sides to interpolate.
            has_left = i > 0 and not np.isnan(x[i - 1])
            has_right = j < W and not np.isnan(x[j])
            if has_left and has_right and (j - i) <= max_gap:
                lo, hi = x[i - 1], x[j]
                x[i:j] = np.linspace(lo, hi, j - i + 2)[1:-1]
                n_interp += 1
            i = j
        else:
            i += 1
    return x, n_interp, gap_fraction


def detect_rwaves(
    amplitude: np.ndarray,
    sampling_rate: float,
    min_distance_ms: float = 250.0,
    interpolate_short_gaps: bool = True,
    max_gap_frac: float = 0.05,
    nk_method: str = "neurokit",
) -> dict:
    """Detect R-waves in a 1D amplitude signal via neurokit2.

    Delegates to ``neurokit2.ecg_peaks`` with the specified method
    (default ``neurokit``), which combines QRS morphology and gradient
    steepness and is more resistant to T-wave false positives than a
    pure prominence-based peak detector. The minimum-distance constraint
    is enforced as a post-filter: peaks closer than ``min_distance_ms``
    (at the given sampling rate) are merged, keeping the earlier one.

    Neurokit's signature expects ~ECG-shaped input; we pass the
    gap-interpolated, baseline-centered amplitude directly. It's tolerant
    of modest noise and handles short records (our strips are typically
    4–5 s at 200 Hz, i.e. 800–1000 samples).
    """
    W = len(amplitude)
    sig = amplitude.astype(np.float32)

    if interpolate_short_gaps:
        sig, n_interp, gap_fraction = _interpolate_short_nan_gaps(sig, max_gap_frac)
    else:
        n_interp = 0
        gap_fraction = float(np.isnan(sig).sum()) / max(1, W)

    # Neurokit doesn't tolerate NaN; replace remaining NaN with 0 (baseline).
    detector_sig = np.where(np.isnan(sig), 0.0, sig).astype(np.float64)

    peaks: np.ndarray
    if detector_sig.size == 0 or np.allclose(detector_sig, 0.0):
        peaks = np.array([], dtype=np.int64)
    else:
        try:
            import neurokit2 as nk  # local import so module import stays cheap
            _, info = nk.ecg_peaks(detector_sig, sampling_rate=sampling_rate,
                                   method=nk_method)
            peaks = np.asarray(info["ECG_R_Peaks"], dtype=np.int64)
        except Exception:
            # Safety net if neurokit fails on pathological input.
            peaks = np.array([], dtype=np.int64)

    # Enforce refractory period: merge peaks closer than min_distance_ms.
    if peaks.size > 1:
        min_dist = max(1, int(round(min_distance_ms / 1000.0 * sampling_rate)))
        kept = [int(peaks[0])]
        for x in peaks[1:]:
            if int(x) - kept[-1] >= min_dist:
                kept.append(int(x))
        peaks = np.asarray(kept, dtype=np.int64)

    if len(peaks) >= 2:
        rr = np.diff(peaks).astype(np.float32)
        mean_rr_s = float(np.mean(rr)) / sampling_rate
        detected_hr_bpm = 60.0 / mean_rr_s if mean_rr_s > 0 else float("nan")
    else:
        rr = np.array([], dtype=np.float32)
        detected_hr_bpm = float("nan")

    if len(peaks) >= 3:
        rr_cv = float(np.std(rr) / np.mean(rr)) if np.mean(rr) > 0 else float("nan")
    else:
        rr_cv = float("nan")

    return {
        "rwave_positions": peaks.astype(np.int64),
        "rr_intervals_samples": rr,
        "detected_hr_bpm": detected_hr_bpm,
        "rr_cv": rr_cv,
        "n_rwaves": int(len(peaks)),
        "gaps_interpolated": int(n_interp),
        "gap_fraction": gap_fraction,
    }


# ---------------------------------------------------------------------------
# Diagnostic plot
# ---------------------------------------------------------------------------

def plot_signal_with_peaks(
    strip: np.ndarray,
    amplitude: np.ndarray,
    rwave_positions: np.ndarray,
    output_path: Path,
    title: str = "",
) -> None:
    """Two-panel diagnostic: strip image + R-wave lines on top, 1D signal +
    R-wave dots on bottom. Single primary debug tool."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_img, ax_sig) = plt.subplots(
        2, 1,
        figsize=(max(8, len(amplitude) / 100), 4),
        gridspec_kw={"height_ratios": [1, 2]},
    )

    ax_img.imshow(strip, aspect="auto")
    ax_img.set_xlim(0, strip.shape[1])
    for p in rwave_positions:
        ax_img.axvline(p, color="red", linewidth=0.8, alpha=0.8)
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    if title:
        ax_img.set_title(title)

    ax_sig.plot(amplitude, color="tab:blue", linewidth=1)
    if len(rwave_positions):
        ax_sig.plot(rwave_positions, amplitude[rwave_positions], "o",
                    color="red", markersize=5)
    ax_sig.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
    ax_sig.set_xlim(0, len(amplitude))
    ax_sig.set_xlabel("column (x)")
    ax_sig.set_ylabel("amplitude (baseline - y)")

    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
