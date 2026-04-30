#!/usr/bin/env python3
"""Compare detector R-wave positions against hand-labeled ground truth.

Expects a hand-label CSV with columns:
    dicom_id, rwave_columns
where `rwave_columns` is a ``|``-delimited list of pixel-x positions
(e.g. ``56|222|386|550``).

Matches each hand-labeled R-wave to the nearest detected R-wave within
``--tol-samples`` (default 3); computes precision, recall, F1. Also prints
per-clip mismatches so failure modes are visible.

To populate the hand-label CSV:
    1. Run `select_handlabel_set.py` to pick a stratified sample of 30–50
       clips across quality classes.
    2. Open each strip (or the diagnostic plot in lastframe/ecg_diagnostics/)
       in an image viewer and note R-wave x-positions.
    3. Write one row per clip into `hand_labels.csv`.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from ecg_signal import cropped_to_signal, detect_rwaves


def match_peaks(
    detected: np.ndarray, hand: np.ndarray, tol: int
) -> tuple[int, int, int]:
    """Greedy bipartite match by nearest-neighbor within tolerance.

    Returns (n_matched, n_fp, n_fn).
    """
    detected = np.sort(detected.astype(int))
    hand = np.sort(hand.astype(int))
    matched_d = np.zeros(len(detected), dtype=bool)
    matched_h = np.zeros(len(hand), dtype=bool)
    for i, h in enumerate(hand):
        # Nearest unused detection within tol.
        avail = np.where(~matched_d)[0]
        if len(avail) == 0:
            break
        dists = np.abs(detected[avail] - h)
        j = avail[np.argmin(dists)]
        if dists.min() <= tol:
            matched_d[j] = True
            matched_h[i] = True
    n_matched = int(matched_h.sum())
    n_fp = int((~matched_d).sum())
    n_fn = int((~matched_h).sum())
    return n_matched, n_fp, n_fn


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--labels-csv", type=Path, default=here / "hand_labels.csv")
    ap.add_argument("--strip-dir", type=Path, default=here / "lastframe" / "waveform")
    ap.add_argument("--calibration-csv", type=Path,
                    default=here / "calibration_results.csv")
    ap.add_argument("--tol-samples", type=int, default=3)
    args = ap.parse_args()

    if not args.labels_csv.exists():
        print(f"No hand-label CSV at {args.labels_csv}. "
              f"Create it with columns [dicom_id, rwave_columns] "
              f"(pipe-delimited x-positions) to run this comparison.")
        return

    calib = {r["dicom_id"]: r for r in csv.DictReader(open(args.calibration_csv))}
    tot_match = tot_fp = tot_fn = 0
    per_clip_stats: list[dict] = []

    for row in csv.DictReader(open(args.labels_csv)):
        dcm = row["dicom_id"]
        stem = dcm.replace(".dcm", "")
        strip = args.strip_dir / f"{stem}.png"
        if not strip.exists():
            print(f"  SKIP {dcm}: no strip at {strip}")
            continue
        sr_raw = calib.get(dcm, {}).get("sampling_rate_hz", "")
        if sr_raw in ("", None):
            print(f"  SKIP {dcm}: no calibration")
            continue
        sr = float(sr_raw)

        img = np.asarray(Image.open(strip).convert("RGB"))
        amp, _ = cropped_to_signal(img)
        res = detect_rwaves(amp, sampling_rate=sr)
        det = res["rwave_positions"]

        hand_str = row.get("rwave_columns", "")
        hand = np.array([int(x) for x in hand_str.split("|") if x.strip()])

        nm, nfp, nfn = match_peaks(det, hand, args.tol_samples)
        tot_match += nm
        tot_fp += nfp
        tot_fn += nfn
        per_clip_stats.append({
            "dicom_id": dcm,
            "n_detected": len(det),
            "n_hand": len(hand),
            "matched": nm,
            "fp": nfp,
            "fn": nfn,
            "detected_cols": "|".join(str(int(x)) for x in det),
            "hand_cols": hand_str,
        })

    precision = tot_match / (tot_match + tot_fp) if (tot_match + tot_fp) else float("nan")
    recall = tot_match / (tot_match + tot_fn) if (tot_match + tot_fn) else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else float("nan")

    print(f"\n=== Hand-label comparison (tol={args.tol_samples} samples) ===")
    print(f"Clips evaluated: {len(per_clip_stats)}")
    print(f"  matched={tot_match}  fp={tot_fp}  fn={tot_fn}")
    print(f"  precision={precision:.3f}  recall={recall:.3f}  F1={f1:.3f}")

    # Failure modes
    print(f"\nPer-clip (sorted by FP+FN desc):")
    for r in sorted(per_clip_stats, key=lambda x: -(x["fp"] + x["fn"]))[:20]:
        tag = "OK" if (r["fp"] + r["fn"]) == 0 else "!!"
        print(f"  {tag}  {r['dicom_id']:25} det={r['n_detected']} hand={r['n_hand']} "
              f"matched={r['matched']} fp={r['fp']} fn={r['fn']}")


if __name__ == "__main__":
    main()
