#!/usr/bin/env python3
"""Compute per-sample Hilbert phase on ECG strips and compare across
HR-inconsistent within-study pairs.

For each selected pair:
  - compute bandpass-filtered Hilbert phase for both clips
  - plot phase(t) for A and B overlaid
  - plot phase_A - phase_B (circular difference) over time
  - report summary stats (mean circular offset, drift, etc.)

If phase trajectories are smooth and phi_A(t) - phi_B(t) is near constant,
per-frame phase is doing useful work. If noisy, the approach doesn't hold up.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.signal import butter, filtfilt, hilbert

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ecg_signal import cropped_to_signal

HERE = Path(__file__).resolve().parent
STRIP_DIR = HERE / "lastframe" / "waveform"
CALIB_CSV = HERE / "calibration_results.csv"
META_CSV = HERE / "dicom_metadata.csv"
OUT_DIR = HERE / "lastframe" / "hilbert_phase"


def load_calib() -> dict[str, float]:
    out = {}
    with CALIB_CSV.open() as f:
        for row in csv.DictReader(f):
            try:
                out[row["dicom_id"].replace(".dcm", "")] = float(row["sampling_rate_hz"])
            except (ValueError, KeyError):
                pass
    return out


def load_hr() -> dict[str, float]:
    out = {}
    with META_CSV.open() as f:
        for row in csv.DictReader(f):
            try:
                hr = float(row.get("heart_rate", "") or 0)
                if hr > 0:
                    out[row.get("dicom", "").replace(".dcm", "")] = hr
            except ValueError:
                pass
    return out


def compute_per_frame_phase(
    amplitude: np.ndarray,
    valid_mask: np.ndarray,
    sampling_rate_hz: float,
) -> np.ndarray:
    """Per-sample phase in [0, 1) via bandpass + Hilbert."""
    nyq = sampling_rate_hz / 2.0
    b, a = butter(4, [0.5 / nyq, 3.0 / nyq], btype="band")
    sig = np.where(valid_mask, amplitude, 0.0).astype(np.float64)
    filtered = filtfilt(b, a, sig)
    analytic = hilbert(filtered)
    phase = np.angle(analytic)                    # in [-pi, pi]
    phase01 = (phase + np.pi) / (2 * np.pi)       # in [0, 1)
    phase01[~valid_mask] = np.nan
    return phase01


def load_signal(clip: str) -> tuple[np.ndarray, np.ndarray] | None:
    p = STRIP_DIR / f"{clip}.png"
    if not p.exists():
        return None
    img = np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)
    amp, valid = cropped_to_signal(img)
    if valid.sum() < 100:
        return None
    return amp, valid


def circular_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Wrapped difference of two phase-in-[0,1) sequences, returned in [-0.5, 0.5]."""
    d = (a - b) % 1.0
    d[d > 0.5] -= 1.0
    return d


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sr_by = load_calib()
    hr_by = load_hr()

    # HR-inconsistent within-study pairs from the padded-run CSV.
    pairs = [
        ("97877557_0044", "97877557_0073"),
        ("96166542_0043", "96166542_0011"),
        ("99626741_0102", "99626741_0165"),
        ("96257816_0012", "96257816_0028"),
        ("90548321_0063", "90548321_0073"),
        ("90197916_0070", "90197916_0094"),
    ]

    summary_rows = []

    for ca, cb in pairs:
        la, lb = load_signal(ca), load_signal(cb)
        if la is None or lb is None:
            print(f"  skip {ca} <-> {cb}: missing strip or no trace")
            continue
        amp_a, valid_a = la
        amp_b, valid_b = lb
        sr_a = sr_by.get(ca, 213.0)
        sr_b = sr_by.get(cb, 213.0)
        hr_a = hr_by.get(ca, float("nan"))
        hr_b = hr_by.get(cb, float("nan"))

        # Sampling rate is effectively same-scanner within a study.
        sr = sr_a
        phase_a = compute_per_frame_phase(amp_a, valid_a, sr)
        phase_b = compute_per_frame_phase(amp_b, valid_b, sr)

        # Align lengths: truncate to the shorter for the difference plot.
        n = min(len(phase_a), len(phase_b))
        pa = phase_a[:n]
        pb = phase_b[:n]
        both_valid = ~(np.isnan(pa) | np.isnan(pb))
        if both_valid.sum() < 10:
            print(f"  skip {ca} <-> {cb}: too few joint valid samples")
            continue

        diff = circular_diff(pa, pb)
        diff_v = diff[both_valid]

        # Circular mean/std of diff in cycles.
        ang = 2 * np.pi * diff_v
        mean_cos = np.mean(np.cos(ang))
        mean_sin = np.mean(np.sin(ang))
        R = np.sqrt(mean_cos ** 2 + mean_sin ** 2)         # in [0,1]: resultant length
        circ_mean = np.arctan2(mean_sin, mean_cos) / (2 * np.pi)
        circ_std_cycles = np.sqrt(-2 * np.log(R)) / (2 * np.pi) if R > 1e-6 else float("nan")

        # Linear drift of unwrapped diff (detect HR mismatch between clips).
        unwrapped = np.unwrap(ang)  # same length as diff_v (valid-only)
        t = np.arange(n) / sr
        tv = t[both_valid]
        if len(tv) >= 2:
            slope_rad_per_s, _ = np.polyfit(tv, unwrapped, 1)
            drift_hz = slope_rad_per_s / (2 * np.pi)
        else:
            drift_hz = float("nan")

        # ---------------- plot ----------------
        fig, axes = plt.subplots(
            3, 1, figsize=(10, 7),
            gridspec_kw={"height_ratios": [1, 1, 1]}
        )
        # Panel 1: amplitudes
        axes[0].plot(t, amp_a[:n], color="tab:blue",
                     linewidth=0.7, alpha=0.8, label=f"A {ca} HR={hr_a:.0f}")
        axes[0].plot(t, amp_b[:n], color="tab:orange",
                     linewidth=0.7, alpha=0.8, label=f"B {cb} HR={hr_b:.0f}")
        axes[0].set_title(f"{ca} <-> {cb}  sr={sr:.0f}Hz")
        axes[0].set_ylabel("amp")
        axes[0].legend(loc="upper right", fontsize=8)

        # Panel 2: phase trajectories
        axes[1].plot(t, pa, color="tab:blue", linewidth=0.8, label="phase A")
        axes[1].plot(t, pb, color="tab:orange", linewidth=0.8, label="phase B")
        axes[1].set_ylabel("phase [0,1)")
        axes[1].set_ylim(-0.05, 1.05)
        axes[1].legend(loc="upper right", fontsize=8)

        # Panel 3: circular difference
        axes[2].plot(t, diff, color="tab:green", linewidth=0.8)
        axes[2].axhline(circ_mean, color="black", linestyle="--",
                        linewidth=0.7,
                        label=f"mean={circ_mean:+.3f}, R={R:.2f}, drift={drift_hz:.2f}Hz")
        axes[2].set_ylabel("phase A - B")
        axes[2].set_xlabel("time (s)")
        axes[2].set_ylim(-0.55, 0.55)
        axes[2].legend(loc="upper right", fontsize=8)

        fig.tight_layout()
        fig.savefig(OUT_DIR / f"pair_{ca}_{cb}.png", dpi=120)
        plt.close(fig)

        summary_rows.append({
            "clip_a": ca, "clip_b": cb,
            "hr_a": round(hr_a, 1), "hr_b": round(hr_b, 1),
            "circ_mean_cycles": round(float(circ_mean), 3),
            "R_length": round(float(R), 3),          # 1 = perfectly locked
            "circ_std_cycles": round(float(circ_std_cycles), 3),
            "drift_hz": round(float(drift_hz), 3),
            "n_valid": int(both_valid.sum()),
        })

        print(f"  {ca} <-> {cb}: mean_offset={circ_mean:+.3f}c  R={R:.2f}  "
              f"circ_std={circ_std_cycles:.3f}c  drift={drift_hz:+.3f}Hz  "
              f"(HR {hr_a:.0f} vs {hr_b:.0f})")

    if summary_rows:
        out_csv = OUT_DIR / "hilbert_pair_summary.csv"
        with out_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)
        print(f"\nWrote {out_csv}")
        print(f"Plots in {OUT_DIR}/")

        print()
        print("Interpretation guide:")
        print("  R_length near 1.0 = phase-locked (offset stable over time)")
        print("  R_length near 0.0 = random (phase not tracking cardiac cycle)")
        print("  circ_std_cycles <0.15 = tight alignment  |  >0.25 = loose/noisy")
        print("  |drift_hz| > 0.3    = phases diverging (HR mismatch between clips)")


if __name__ == "__main__":
    main()
