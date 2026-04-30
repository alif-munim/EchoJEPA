#!/usr/bin/env python3
"""Robust R-peak detection with metadata-HR-supervised ensemble + fallbacks.

Public API:
  - ``robust_rpeaks(signal, sr, hr_metadata) -> (peaks, method, ratio_dist)``

Strategy (see design doc). Metadata HR is the supervisory signal:
  1. Run a small bank of black-box detectors (NeuroKit2, HR-primed Pan-Tompkins).
  2. Score each by ``|log(n_detected / expected)|`` — symmetric in over/under.
  3. Keep the best. If still off by more than ~35% (log-dist > 0.3), try
     a deterministic bandpass + refractory peak-picker, then the same picker
     on the polarity-flipped signal. Return whichever is closest.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


# ---------------------------------------------------------------------------
# Individual detectors
# ---------------------------------------------------------------------------

def neurokit_detector(signal: np.ndarray, sr: float) -> np.ndarray:
    """NeuroKit2 default (method='neurokit'). Treats signal as black-box input."""
    import neurokit2 as nk
    _, info = nk.ecg_peaks(signal, sampling_rate=sr, method="neurokit")
    return np.asarray(info["ECG_R_Peaks"], dtype=np.int64)


def pan_tompkins_with_hr_prior(
    signal: np.ndarray,
    sr: float,
    hr_metadata: float | None = None,
) -> np.ndarray:
    """Pan-Tompkins with metadata HR prior for refractory + initial threshold.

    - Refractory: ``max(200ms, 0.6 * expected_rr)`` — widens the rejection
      window when HR is slow so T-waves don't trigger a second detection.
    - Initial threshold: primed from the median amplitude of local maxima
      spaced ~expected_rr apart, instead of a fraction of running max.
    """
    sr = float(sr)
    nyq = sr / 2.0
    b, a = butter(2, [5.0 / nyq, 15.0 / nyq], btype="band")
    x = filtfilt(b, a, signal.astype(np.float64))
    dx = np.diff(x, prepend=x[0])
    sq = dx * dx
    win = max(1, int(0.150 * sr))
    mwi = np.convolve(sq, np.ones(win) / win, mode="same")

    if hr_metadata and hr_metadata > 0:
        expected_rr = 60.0 / float(hr_metadata) * sr
        refrac = max(int(0.200 * sr), int(0.6 * expected_rr))
    else:
        expected_rr = 0.0
        refrac = max(1, int(0.250 * sr))

    # Prime threshold from local maxima spaced ~expected_rr apart.
    if expected_rr > 0:
        spacing = max(1, int(0.5 * expected_rr))
        candidates, _ = find_peaks(mwi, distance=spacing)
        if len(candidates):
            prime = float(np.median(mwi[candidates]))
            thr = 0.5 * prime
        else:
            thr = 0.3 * float(np.max(mwi))
    else:
        thr = 0.3 * float(np.max(mwi))
    thr = max(thr, 1e-9)

    snap = max(1, int(0.050 * sr))
    peaks: list[int] = []
    i = 0
    n = len(mwi)
    while i < n:
        if mwi[i] > thr:
            j = min(n, i + refrac)
            local = int(np.argmax(mwi[i:j])) + i
            lo = max(0, local - snap)
            hi = min(n, local + snap + 1)
            snap_idx = int(np.argmax(signal[lo:hi])) + lo
            peaks.append(snap_idx)
            i = local + refrac
        else:
            i += 1
    return np.asarray(peaks, dtype=np.int64)


def fallback_rpeaks(
    signal: np.ndarray,
    sr: float,
    hr_metadata: float,
) -> np.ndarray:
    """Deterministic peak picker with metadata-derived refractory. No adaptive
    thresholding — pick the highest peaks subject to the spacing constraint,
    then snap back to the local maximum of |signal|."""
    expected_rr = max(1.0, 60.0 / max(hr_metadata, 1e-6) * sr)
    min_distance = max(1, int(0.4 * expected_rr))
    nyq = 0.5 * sr
    b, a = butter(2, [5.0 / nyq, 15.0 / nyq], btype="band")
    filtered = filtfilt(b, a, signal.astype(np.float64))
    rectified = np.abs(filtered)
    height_floor = float(np.percentile(rectified, 75))
    peaks, _ = find_peaks(rectified, distance=min_distance, height=height_floor)

    snap_window = max(1, int(0.05 * sr))
    out: list[int] = []
    for p in peaks:
        lo = max(0, int(p) - snap_window)
        hi = min(len(signal), int(p) + snap_window)
        if hi <= lo:
            out.append(int(p))
        else:
            local = lo + int(np.argmax(np.abs(signal[lo:hi])))
            out.append(local)
    return np.asarray(out, dtype=np.int64)


# ---------------------------------------------------------------------------
# Ensemble + robust wrapper
# ---------------------------------------------------------------------------

def _log_ratio_dist(n_peaks: int, expected: float) -> float:
    if expected <= 0:
        return float("inf")
    ratio = n_peaks / expected
    return abs(np.log(max(ratio, 1e-3)))


def ensemble_rpeaks(
    signal: np.ndarray,
    sr: float,
    hr_metadata: float,
    detectors: dict | None = None,
) -> tuple[np.ndarray, str, list[tuple]]:
    """Run each detector; return (best_peaks, best_name, [(name, peaks, ratio, err)])."""
    if detectors is None:
        detectors = {
            "neurokit2": lambda s, sr: neurokit_detector(s, sr),
            "pan_tompkins_hr_prior":
                lambda s, sr: pan_tompkins_with_hr_prior(s, sr, hr_metadata),
        }
    duration_s = len(signal) / sr
    expected = duration_s * hr_metadata / 60.0
    best_peaks = np.array([], dtype=np.int64)
    best_dist = float("inf")
    best_name = "none"
    results: list[tuple] = []
    for name, fn in detectors.items():
        try:
            peaks = fn(signal, sr)
        except Exception as e:
            results.append((name, None, None, str(e)[:120]))
            continue
        ratio = len(peaks) / expected if expected > 0 else 0
        d = _log_ratio_dist(len(peaks), expected)
        results.append((name, peaks, ratio, None))
        if d < best_dist:
            best_dist = d
            best_peaks = peaks
            best_name = name
    return best_peaks, best_name, results


def robust_rpeaks(
    signal: np.ndarray,
    sr: float,
    hr_metadata: float,
    confidence_threshold: float = 0.3,
) -> tuple[np.ndarray, str, float]:
    """Cascading detection. Returns (peaks, method_used, ratio_distance).

    If the ensemble's best log-ratio distance is <= ``confidence_threshold``
    (default 0.3, ~35% off expected), return it. Otherwise try
    ``fallback_rpeaks(signal)`` and ``fallback_rpeaks(-signal)`` and pick the
    overall minimum-distance candidate.
    """
    duration_s = len(signal) / sr
    expected = duration_s * hr_metadata / 60.0
    peaks, name, _ = ensemble_rpeaks(signal, sr, hr_metadata)
    dist = _log_ratio_dist(len(peaks), expected)
    if dist <= confidence_threshold:
        return peaks, name, dist

    candidates: list[tuple[np.ndarray, str, float]] = [(peaks, name, dist)]
    try:
        fb = fallback_rpeaks(signal, sr, hr_metadata)
        candidates.append((fb, "fallback", _log_ratio_dist(len(fb), expected)))
    except Exception:
        pass
    try:
        fbf = fallback_rpeaks(-signal, sr, hr_metadata)
        # Peaks are in index space; snap inverted-signal peaks back to positions
        # on the *original* amplitude so downstream uses consistent indices.
        candidates.append((fbf, "fallback_flipped",
                           _log_ratio_dist(len(fbf), expected)))
    except Exception:
        pass

    return min(candidates, key=lambda c: c[2])
