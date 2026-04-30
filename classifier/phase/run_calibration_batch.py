#!/usr/bin/env python3
"""Calibrate sampling rate for every DICOM in phase/dicoms/ and emit a CSV.

Columns: dicom_id, sampling_rate_hz, source, confidence.
"""

import argparse
import csv
from pathlib import Path

from ecg_calibration import calibrate_sampling_rate, load_scanner_defaults

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dicom-dir", type=Path, default=HERE / "dicoms")
    ap.add_argument("-o", "--out", type=Path, default=HERE / "calibration_results.csv")
    args = ap.parse_args()

    defaults = load_scanner_defaults()
    dcms = sorted(args.dicom_dir.glob("*.dcm"))
    print(f"Calibrating {len(dcms)} DICOMs → {args.out}")

    rows = []
    for p in dcms:
        r = calibrate_sampling_rate(p, defaults)
        rows.append({
            "dicom_id": p.name,
            "sampling_rate_hz": (round(r["sampling_rate_hz"], 4)
                                 if r["sampling_rate_hz"] is not None else ""),
            "source": r["source"],
            "confidence": r["confidence"],
        })

    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dicom_id", "sampling_rate_hz",
                                          "source", "confidence"])
        w.writeheader()
        w.writerows(rows)

    from collections import Counter
    by_src = Counter(r["source"] for r in rows)
    by_conf = Counter(r["confidence"] for r in rows)
    print(f"by source:     {dict(by_src)}")
    print(f"by confidence: {dict(by_conf)}")


if __name__ == "__main__":
    main()
