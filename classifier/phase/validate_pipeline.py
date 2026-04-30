#!/usr/bin/env python3
"""End-to-end pipeline validation.

For each cropped strip, run cropped_to_signal + detect_rwaves at the
calibrated sampling rate, compare the detected HR to the DICOM-reported
HR, and classify each clip. Aggregate by calibration source and scanner.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from ecg_signal import cropped_to_signal, detect_rwaves

HERE = Path(__file__).resolve().parent


def _classify(
    n_rwaves: int,
    detected_hr: float,
    displayed_hr: float | None,
    rr_cv: float,
    sampling_rate: float | None,
) -> str:
    if sampling_rate is None or math.isnan(sampling_rate):
        return "no_calibration"
    if n_rwaves < 2 or math.isnan(detected_hr):
        return "no_detection"
    if displayed_hr is None or math.isnan(displayed_hr):
        # No ground truth — can't classify further. Treat as "no_detection"
        # bucket since we can't validate HR match. (Rare in MIMIC — 320/320
        # have HR populated in the current sample.)
        return "no_detection"
    err_pct = abs(detected_hr - displayed_hr) / displayed_hr * 100.0
    if err_pct <= 10.0 and (math.isnan(rr_cv) or rr_cv < 0.10):
        return "good"
    if err_pct <= 10.0:
        return "irregular"
    return "hr_mismatch"


def validate_pipeline(
    dicom_dir: Path,
    cropped_strip_dir: Path,
    calibration_results_csv: Path,
    metadata_csv: Path,
    output_csv: Path,
) -> dict:
    # Load calibration and metadata.
    calib: dict[str, dict] = {}
    with calibration_results_csv.open() as f:
        for row in csv.DictReader(f):
            calib[row["dicom_id"]] = row

    meta: dict[str, dict] = {}
    with metadata_csv.open() as f:
        for row in csv.DictReader(f):
            meta[row["dicom"]] = row

    strips = sorted(cropped_strip_dir.glob("*.png"))
    out_rows: list[dict] = []

    for p in strips:
        dicom_id = p.stem + ".dcm"
        calib_row = calib.get(dicom_id, {})
        meta_row = meta.get(dicom_id, {})

        sr_raw = calib_row.get("sampling_rate_hz", "")
        sr = float(sr_raw) if sr_raw not in ("", None) else float("nan")
        source = calib_row.get("source", "")
        confidence = calib_row.get("confidence", "")
        manu = meta_row.get("manufacturer", "")
        model = meta_row.get("model", "")
        displayed_hr_raw = meta_row.get("heart_rate", "")
        displayed_hr = float(displayed_hr_raw) if displayed_hr_raw not in ("", None) \
            else float("nan")

        if math.isnan(sr):
            n_rwaves = 0
            detected_hr = float("nan")
            rr_cv = float("nan")
            gap_fraction = float("nan")
        else:
            img = np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)
            amp, _valid = cropped_to_signal(img)
            res = detect_rwaves(amp, sampling_rate=sr)
            n_rwaves = res["n_rwaves"]
            detected_hr = res["detected_hr_bpm"]
            rr_cv = res["rr_cv"]
            gap_fraction = res["gap_fraction"]

        if not math.isnan(detected_hr) and not math.isnan(displayed_hr) \
                and displayed_hr > 0:
            hr_error_pct = abs(detected_hr - displayed_hr) / displayed_hr * 100.0
        else:
            hr_error_pct = float("nan")

        quality = _classify(n_rwaves, detected_hr, displayed_hr, rr_cv,
                            None if math.isnan(sr) else sr)

        out_rows.append({
            "dicom_id": dicom_id,
            "manufacturer": manu,
            "model": model,
            "sampling_rate_hz": "" if math.isnan(sr) else round(sr, 4),
            "calibration_source": source,
            "confidence": confidence,
            "n_rwaves": n_rwaves,
            "detected_hr": "" if math.isnan(detected_hr) else round(detected_hr, 2),
            "displayed_hr": "" if math.isnan(displayed_hr) else int(displayed_hr),
            "hr_error_pct": "" if math.isnan(hr_error_pct) else round(hr_error_pct, 2),
            "rr_cv": "" if math.isnan(rr_cv) else round(rr_cv, 4),
            "gap_fraction": "" if math.isnan(gap_fraction) else round(gap_fraction, 4),
            "quality": quality,
        })

    with output_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    # Aggregate.
    by_quality = Counter(r["quality"] for r in out_rows)
    err_good = [r["hr_error_pct"] for r in out_rows
                if r["quality"] in ("good", "irregular") and r["hr_error_pct"] != ""]
    median_err = float(np.median(err_good)) if err_good else float("nan")

    by_src: dict[str, Counter] = defaultdict(Counter)
    for r in out_rows:
        by_src[r["calibration_source"]][r["quality"]] += 1

    by_scanner: dict[str, Counter] = defaultdict(Counter)
    for r in out_rows:
        scanner = f"{r['manufacturer']} {r['model']}".strip()
        by_scanner[scanner][r["quality"]] += 1

    stats = {
        "n_total": len(out_rows),
        "n_good": by_quality["good"],
        "n_irregular": by_quality["irregular"],
        "n_hr_mismatch": by_quality["hr_mismatch"],
        "n_no_detection": by_quality["no_detection"],
        "n_no_calibration": by_quality["no_calibration"],
        "median_hr_error_pct": median_err,
        "breakdown_by_calibration_source": {k: dict(v) for k, v in by_src.items()},
        "breakdown_by_scanner": {k: dict(v) for k, v in by_scanner.items()},
    }
    return stats


def _print_stats(s: dict) -> None:
    n = s["n_total"]
    print(f"\n=== Pipeline validation over {n} strips ===\n")
    print(f"good:           {s['n_good']:>4d}  ({s['n_good']/n:.1%})")
    print(f"irregular:      {s['n_irregular']:>4d}  ({s['n_irregular']/n:.1%})")
    print(f"hr_mismatch:    {s['n_hr_mismatch']:>4d}  ({s['n_hr_mismatch']/n:.1%})")
    print(f"no_detection:   {s['n_no_detection']:>4d}  ({s['n_no_detection']/n:.1%})")
    print(f"no_calibration: {s['n_no_calibration']:>4d}  ({s['n_no_calibration']/n:.1%})")
    if not math.isnan(s["median_hr_error_pct"]):
        print(f"\nmedian HR error on good+irregular: {s['median_hr_error_pct']:.2f}%")

    print(f"\nBy calibration source:")
    for src, cnts in sorted(s["breakdown_by_calibration_source"].items()):
        total = sum(cnts.values())
        good = cnts.get("good", 0)
        print(f"  {src:18}  n={total:>3d}  good={good}  ({good/total:.0%})  {dict(cnts)}")

    print(f"\nBy scanner:")
    for scanner, cnts in sorted(s["breakdown_by_scanner"].items()):
        total = sum(cnts.values())
        good = cnts.get("good", 0)
        print(f"  {scanner:40}  n={total:>3d}  good={good}  ({good/total:.0%})  {dict(cnts)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dicom-dir", type=Path, default=HERE / "dicoms")
    ap.add_argument("--strip-dir", type=Path, default=HERE / "lastframe" / "waveform")
    ap.add_argument("--calibration-csv", type=Path,
                    default=HERE / "calibration_results.csv")
    ap.add_argument("--metadata-csv", type=Path,
                    default=HERE / "dicom_metadata.csv")
    ap.add_argument("-o", "--out", type=Path,
                    default=HERE / "pipeline_validation.csv")
    args = ap.parse_args()

    stats = validate_pipeline(
        args.dicom_dir, args.strip_dir, args.calibration_csv,
        args.metadata_csv, args.out,
    )
    _print_stats(stats)
    print(f"\nPer-clip results: {args.out}")


if __name__ == "__main__":
    main()
