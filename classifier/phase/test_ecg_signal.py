"""Smoke tests for ecg_signal.cropped_to_signal / detect_rwaves."""

from __future__ import annotations

import numpy as np

from ecg_signal import cropped_to_signal, detect_rwaves, filter_trace_blobs


def test_synthetic_signal():
    """Verify R-wave detection on a synthetic clean signal."""
    sampling_rate = 250.0
    duration_s = 4.0
    hr_bpm = 75
    n_samples = int(duration_s * sampling_rate)

    rr_interval_s = 60.0 / hr_bpm
    expected_rwave_times = np.arange(0.5, duration_s, rr_interval_s)
    signal = np.zeros(n_samples, dtype=np.float32)
    for rwave_t in expected_rwave_times:
        rwave_idx = int(rwave_t * sampling_rate)
        for i in range(max(0, rwave_idx - 20), min(n_samples, rwave_idx + 20)):
            signal[i] += np.exp(-((i - rwave_idx) ** 2) / (2 * 5 ** 2))

    result = detect_rwaves(signal, sampling_rate=sampling_rate)

    assert abs(result["detected_hr_bpm"] - hr_bpm) < 2.0, \
        f"detected_hr_bpm={result['detected_hr_bpm']}, expected ~{hr_bpm}"
    assert abs(result["n_rwaves"] - len(expected_rwave_times)) <= 1, \
        f"n_rwaves={result['n_rwaves']}, expected ~{len(expected_rwave_times)}"
    assert result["rr_cv"] < 0.05, f"rr_cv={result['rr_cv']}, expected <0.05"


def test_blob_filtering():
    """Synthetic image: a full-width horizontal trace plus a localized blob.

    The blob must be removed, the trace must survive, and the resulting
    1D signal must be non-trivial across the full width of the trace.
    """
    H, W = 60, 400
    img = np.zeros((H, W, 3), dtype=np.uint8)

    # Main trace: a green horizontal line at y=30 spanning the full width.
    trace_row = 30
    img[trace_row - 1 : trace_row + 2, :, 1] = 200  # green
    img[trace_row - 1 : trace_row + 2, :, 0] = 50   # small R so G-dominant & saturated
    img[trace_row - 1 : trace_row + 2, :, 2] = 80

    # Spurious blob: a 15x15 green patch at x=300..315.
    img[5:20, 300:315, 1] = 220
    img[5:20, 300:315, 0] = 40
    img[5:20, 300:315, 2] = 60

    # Sanity: both the trace and the blob pass the trace-pixel test.
    from ecg_signal import trace_mask
    m_raw = trace_mask(img)
    assert m_raw.any()

    # Filter: expect the blob is dropped; the trace survives.
    m_filt = filter_trace_blobs(m_raw)
    # Column 307 is inside the blob but outside the trace vertically — in
    # the filtered mask, the only True pixels at column 307 should be the
    # trace rows (29, 30, 31). The blob rows (5–19) should be cleared.
    assert not m_filt[5:20, 307].any(), "blob should be removed"
    assert m_filt[trace_row, 307], "trace row at blob column should survive"

    # Full 1D signal end-to-end.
    amplitude, valid = cropped_to_signal(img)
    assert valid.all(), "flat trace should have signal at every column"
    # Signal should be ~0 everywhere (flat trace centered on its own median).
    assert np.nanmax(np.abs(amplitude)) <= 1.0


def test_empty_strip():
    """All-black strip: signal is all-NaN, detector returns no peaks."""
    img = np.zeros((40, 200, 3), dtype=np.uint8)
    amplitude, valid = cropped_to_signal(img)
    assert not valid.any()
    assert np.all(np.isnan(amplitude))

    result = detect_rwaves(amplitude, sampling_rate=100.0)
    assert result["n_rwaves"] == 0
    assert np.isnan(result["detected_hr_bpm"])
    assert np.isnan(result["rr_cv"])


def test_gap_interpolation():
    """A signal with a short NaN gap should be interpolated; a long gap shouldn't."""
    rng = np.random.default_rng(0)
    W = 1000
    sig = np.sin(np.linspace(0, 20 * np.pi, W)).astype(np.float32)

    # 2% gap (short) at middle
    sig_short = sig.copy()
    sig_short[100:120] = np.nan

    # 10% gap (long) at middle
    sig_long = sig.copy()
    sig_long[100:200] = np.nan

    r_short = detect_rwaves(sig_short, sampling_rate=100.0, max_gap_frac=0.05)
    r_long = detect_rwaves(sig_long, sampling_rate=100.0, max_gap_frac=0.05)

    assert r_short["gaps_interpolated"] == 1
    assert r_long["gaps_interpolated"] == 0
    assert r_long["gap_fraction"] > r_short["gap_fraction"]


if __name__ == "__main__":
    test_synthetic_signal()
    test_blob_filtering()
    test_empty_strip()
    test_gap_interpolation()
    print("All smoke tests passed.")
