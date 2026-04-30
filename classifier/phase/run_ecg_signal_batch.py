#!/usr/bin/env python3
"""Run cropped_to_signal + detect_rwaves across every strip in
phase/lastframe/waveform/ and dump diagnostic plots + a results CSV.

Uses a placeholder sampling_rate based on sweep speed assumption (25 mm/s
at typical echo display resolution → ~4 samples per mm × strip width).
For development-time spot checking we just take sampling_rate such that
the strip covers ~4s of ECG, matching typical echo cine length. This is
good enough for dev-time HR sanity checks; real downstream calibration is
out of scope for this module.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image

from ecg_signal import cropped_to_signal, detect_rwaves, plot_signal_with_peaks

HERE = Path(__file__).resolve().parent
STRIPS = HERE / "lastframe" / "waveform"
OUT_DIR = HERE / "lastframe" / "ecg_diagnostics"
RESULTS_CSV = HERE / "lastframe" / "ecg_results.csv"

# Assumed clip duration → sampling rate. Matches ballpark echo cine length
# (2–4 s) at common sweep speeds. Real calibration is the caller's job.
ASSUMED_DURATION_S = 3.0


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Optional: pair with DICOM-reported HR if available.
    md_path = HERE / "dicom_metadata.csv"
    displayed_hr: dict[str, str] = {}
    if md_path.exists():
        with md_path.open() as f:
            for row in csv.DictReader(f):
                displayed_hr[Path(row["dicom"]).stem] = row.get("heart_rate", "")

    rows = []
    strips = sorted(STRIPS.glob("*.png"))
    print(f"Processing {len(strips)} strips → {OUT_DIR}")
    for p in strips:
        img = np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)
        amplitude, valid = cropped_to_signal(img)
        sr = len(amplitude) / ASSUMED_DURATION_S
        res = detect_rwaves(amplitude, sampling_rate=sr)

        title = (
            f"{p.name}  |  W={len(amplitude)}  sr≈{sr:.0f}Hz  "
            f"n_R={res['n_rwaves']}  HR={res['detected_hr_bpm']:.0f}  "
            f"CV={res['rr_cv']:.2f}  gap={res['gap_fraction']:.1%}"
        )
        plot_signal_with_peaks(img, amplitude, res["rwave_positions"],
                               OUT_DIR / p.name, title=title)

        rows.append({
            "strip": p.name,
            "width": len(amplitude),
            "assumed_duration_s": ASSUMED_DURATION_S,
            "sampling_rate": round(sr, 2),
            "n_rwaves": res["n_rwaves"],
            "detected_hr_bpm": round(res["detected_hr_bpm"], 2)
                if not np.isnan(res["detected_hr_bpm"]) else "",
            "rr_cv": round(res["rr_cv"], 4) if not np.isnan(res["rr_cv"]) else "",
            "gaps_interpolated": res["gaps_interpolated"],
            "gap_fraction": round(res["gap_fraction"], 4),
            "displayed_hr": displayed_hr.get(p.stem, ""),
        })
        print(f"  {p.name}: n_R={res['n_rwaves']} "
              f"HR={res['detected_hr_bpm']:.0f} CV={res['rr_cv']:.2f} "
              f"gap={res['gap_fraction']:.1%}  (displayed={displayed_hr.get(p.stem, '?')})")

    with RESULTS_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {RESULTS_CSV}")
    print(f"Diagnostic plots in {OUT_DIR}/")


if __name__ == "__main__":
    main()
