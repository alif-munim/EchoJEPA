#!/usr/bin/env python3
"""Dig into R-peak detection failures on process_waveform NPZs.

Runs the current detector (neurokit2) on every processed clip, compares
count-vs-expected (metadata HR × duration), and:
  - writes ``rpeak_failure_categories.txt`` with counts of each class
  - writes ``rpeak_failures.png`` (4x3 grid: 6 over-detecting + 6 under-detecting)
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
PROCESSED_DIR = HERE / "lastframe" / "waveform_processed"
CALIB_CSV = HERE / "calibration_results.csv"
META_CSV = HERE / "dicom_metadata.csv"


def load_calib() -> dict[str, float]:
    out: dict[str, float] = {}
    with CALIB_CSV.open() as f:
        for row in csv.DictReader(f):
            try:
                out[row["dicom_id"].replace(".dcm", "")] = float(row["sampling_rate_hz"])
            except (ValueError, KeyError):
                pass
    return out


def load_hr() -> dict[str, float]:
    out: dict[str, float] = {}
    with META_CSV.open() as f:
        for row in csv.DictReader(f):
            try:
                hr = float(row.get("heart_rate", "") or 0)
                if hr > 0:
                    k = (row.get("dicom", "") or
                         row.get("dicom_id", "")).replace(".dcm", "")
                    out[k] = hr
            except ValueError:
                pass
    return out


def neurokit_peaks(signal: np.ndarray, sr: float) -> np.ndarray:
    import neurokit2 as nk
    _, info = nk.ecg_peaks(signal, sampling_rate=sr, method="neurokit")
    return np.asarray(info["ECG_R_Peaks"], dtype=np.int64)


def load_signal(clip_stem: str) -> tuple[np.ndarray, np.ndarray] | None:
    p = PROCESSED_DIR / f"{clip_stem}.npz"
    if not p.exists():
        return None
    d = np.load(p)
    full_y = d["full_y"].astype(np.float64)
    span = d["trace_span_mask"].astype(bool)
    if span.sum() < 100:
        return None
    lo, hi = int(d["x0"]), int(d["x1"]) + 1
    seg = np.nan_to_num(full_y[lo:hi], nan=0.0)
    return seg, np.arange(lo, hi)


def classify_failure(
    signal: np.ndarray, peaks: np.ndarray,
    expected: float, sr: float, hr: float,
) -> str:
    """Heuristic failure categorization. Used for summary counts only."""
    n = len(peaks)
    ratio = n / expected if expected > 0 else 0
    duration_s = len(signal) / sr

    # Signal-quality flags first.
    abs_sig = np.abs(signal - np.median(signal))
    peak_amp = np.percentile(abs_sig, 99)
    ptp = np.ptp(signal)
    if ptp < 5:
        return "low_qrs_amplitude"

    # Saturation: if >5% of samples sit within 2% of min or max of range.
    top = signal.max()
    bot = signal.min()
    at_top = (signal > top - 0.02 * ptp).sum() / max(1, len(signal))
    at_bot = (signal < bot + 0.02 * ptp).sum() / max(1, len(signal))
    if at_top > 0.05 or at_bot > 0.05:
        return "saturation_clipping"

    # Polarity: QRS typically dominates with *one* sign. If |min| >> |max|, trace
    # may be effectively inverted relative to what the detector expects.
    med = float(np.median(signal))
    max_dev_up = signal.max() - med
    max_dev_dn = med - signal.min()
    inverted = max_dev_dn > 2 * max_dev_up and n / max(expected, 1) < 0.75

    # Baseline wander: dominant power in the 0-0.8 Hz band.
    from scipy.signal import welch
    if len(signal) >= max(32, int(sr)):
        freqs, psd = welch(signal, fs=sr, nperseg=min(len(signal), int(2 * sr)))
        low_mask = freqs < 0.8
        if psd[low_mask].sum() > 0.5 * psd.sum() and ratio < 0.75:
            return "baseline_wander"

    if duration_s < 1.0 or expected < 3:
        return "short_clip"

    if inverted:
        return "polarity_flip"

    if ratio > 1.25:
        return "t_wave_double"
    return "other_under" if ratio < 0.75 else "other"


def main() -> None:
    sr_by = load_calib()
    hr_by = load_hr()

    records: list[dict] = []
    for npz in sorted(PROCESSED_DIR.glob("*.npz")):
        stem = npz.stem
        sr = sr_by.get(stem)
        hr = hr_by.get(stem)
        if sr is None or hr is None or hr <= 0:
            continue
        res = load_signal(stem)
        if res is None:
            continue
        sig, cols = res
        duration_s = len(sig) / sr
        expected = duration_s * hr / 60.0
        if expected < 1:
            continue
        try:
            peaks = neurokit_peaks(sig, sr)
        except Exception:
            peaks = np.array([], dtype=np.int64)
        ratio = len(peaks) / expected
        records.append({
            "stem": stem, "sr": sr, "hr": hr,
            "duration_s": duration_s, "expected": expected,
            "n_peaks": int(len(peaks)), "ratio": ratio,
            "sig": sig, "peaks": peaks,
            "within_25": abs(len(peaks) - expected) / expected <= 0.25,
        })

    print(f"Analysed {len(records)} clips "
          f"(both calibration and HR available; processed NPZ present)")
    passing = [r for r in records if r["within_25"]]
    failing = [r for r in records if not r["within_25"]]
    print(f"  within ±25%: {len(passing)}/{len(records)} "
          f"({100 * len(passing) / max(1, len(records)):.0f}%)")
    print(f"  failing    : {len(failing)}")

    # Classify failures.
    from collections import Counter
    cats = Counter()
    for r in failing:
        r["category"] = classify_failure(
            r["sig"], r["peaks"], r["expected"], r["sr"], r["hr"]
        )
        cats[r["category"]] += 1
    print("\nFailure categories:")
    for c, n in cats.most_common():
        print(f"  {c:25s}  {n}")

    (HERE / "rpeak_failure_categories.txt").write_text(
        "R-peak failure categories (processed NPZs, neurokit2-only baseline)\n"
        f"Clips analysed: {len(records)}\n"
        f"Within ±25% (passing): {len(passing)}\n"
        f"Failing: {len(failing)}\n\n"
        + "\n".join(f"{c:25s}  {n}" for c, n in cats.most_common())
        + "\n"
    )

    # Pick 6 over-detecting (highest ratio) and 6 under-detecting (lowest ratio>0).
    over = sorted([r for r in failing if r["ratio"] > 1.25],
                  key=lambda r: -r["ratio"])[:6]
    under = sorted([r for r in failing if r["ratio"] < 0.75],
                   key=lambda r: r["ratio"])[:6]
    picks = over + under
    print(f"\nPlotting {len(over)} over + {len(under)} under = {len(picks)} cases")

    if not picks:
        print("No failure cases to plot.")
        return

    rows = 4
    cols = 3
    fig, axes = plt.subplots(rows, cols, figsize=(16, 12))
    axes = axes.flatten()
    for ax, r in zip(axes, picks):
        t = np.arange(len(r["sig"])) / r["sr"]
        ax.plot(t, r["sig"], color="tab:blue", linewidth=0.8)
        if len(r["peaks"]):
            ax.scatter(r["peaks"] / r["sr"], r["sig"][r["peaks"]],
                       color="red", s=20, zorder=3, label=f"n={len(r['peaks'])}")
        tag = "OVER" if r["ratio"] > 1.25 else "UNDER"
        cat = r.get("category", "?")
        ax.set_title(
            f"{tag}  {r['stem']}  ratio={r['ratio']:.2f}  "
            f"HR={r['hr']:.0f}  exp={r['expected']:.1f}\n"
            f"cat: {cat}",
            fontsize=9,
        )
        ax.set_xlabel("time (s)")
        ax.legend(fontsize=7, loc="upper right")
    for ax in axes[len(picks):]:
        ax.axis("off")
    fig.tight_layout()
    out_path = HERE / "rpeak_failures.png"
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
