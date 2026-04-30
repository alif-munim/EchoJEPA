#!/usr/bin/env python3
"""Pick a stratified random sample of clips for hand-label validation.

Sampling plan: N clips total, stratified across quality classes so that
both successes and failure modes are covered. Writes a stub
``hand_labels.csv`` with empty ``rwave_columns`` ready for manual entry.

Default N=40 per the Phase 4 plan.
"""

import argparse
import csv
import random
from pathlib import Path


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--validation-csv", type=Path,
                    default=here / "pipeline_validation.csv")
    ap.add_argument("-o", "--out", type=Path, default=here / "hand_labels.csv")
    ap.add_argument("--n", type=int, default=40,
                    help="Total clips to sample (default 40)")
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.validation_csv)))
    # Target mix: 60% good, 20% irregular, 15% hr_mismatch, 5% no_detection.
    targets = {"good": int(args.n * 0.60),
               "irregular": int(args.n * 0.20),
               "hr_mismatch": int(args.n * 0.15),
               "no_detection": max(1, args.n - int(args.n * 0.60)
                                   - int(args.n * 0.20) - int(args.n * 0.15))}

    by_q: dict[str, list] = {k: [] for k in targets}
    for r in rows:
        if r["quality"] in by_q:
            by_q[r["quality"]].append(r)

    rng = random.Random(args.seed)
    picked: list[dict] = []
    for q, want in targets.items():
        avail = by_q[q]
        take = min(want, len(avail))
        picked.extend(rng.sample(avail, take))

    # Write stub — rwave_columns empty for manual fill-in.
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dicom_id", "quality", "detected_hr",
                                          "displayed_hr", "rr_cv",
                                          "rwave_columns"])
        w.writeheader()
        for r in picked:
            w.writerow({
                "dicom_id": r["dicom_id"],
                "quality": r["quality"],
                "detected_hr": r.get("detected_hr", ""),
                "displayed_hr": r.get("displayed_hr", ""),
                "rr_cv": r.get("rr_cv", ""),
                "rwave_columns": "",
            })

    from collections import Counter
    c = Counter(r["quality"] for r in picked)
    print(f"Wrote stub with {len(picked)} clips → {args.out}")
    print(f"  distribution: {dict(c)}")
    print(f"Fill in `rwave_columns` as pipe-delimited x-positions per clip, "
          f"then run `python hand_label_compare.py`.")


if __name__ == "__main__":
    main()
