#!/usr/bin/env python3
"""Diagnose why some within-study xcorr pairs are HR-inconsistent.

Checks two metadata-testable hypotheses:
  1. HR drift between clip A and clip B (we use only clip A's HR as the
     reference for the consistent_with_hr gate).
  2. Scanner / model / frame-time mismatch between same-study clips.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
XCORR_CSV = HERE / "xcorr_test_results.csv"
META_CSV = HERE / "dicom_metadata.csv"


def _f(x) -> float | None:
    try:
        v = float(x)
        return v if not np.isnan(v) else None
    except (TypeError, ValueError):
        return None


def main() -> None:
    meta: dict[str, dict] = {}
    with META_CSV.open() as f:
        for row in csv.DictReader(f):
            key = row.get("dicom", "").replace(".dcm", "")
            meta[key] = row

    def field(clip: str, name: str) -> str:
        return (meta.get(clip, {}) or {}).get(name, "")

    rows = list(csv.DictReader(XCORR_CSV.open()))
    within = [r for r in rows if r["kind"] == "within_study"]

    per_pair = []
    for r in within:
        a, b = r["clip_a"], r["clip_b"]
        hr_a = _f(field(a, "heart_rate"))
        hr_b = _f(field(b, "heart_rate"))
        ft_a = _f(field(a, "frame_time_ms"))
        ft_b = _f(field(b, "frame_time_ms"))
        model_a = field(a, "model")
        model_b = field(b, "model")
        manu_a = field(a, "manufacturer")
        manu_b = field(b, "manufacturer")
        nframes_a = _f(field(a, "n_frames"))
        nframes_b = _f(field(b, "n_frames"))

        per_pair.append({
            "pair": f"{a} <-> {b}",
            "consistent": r["consistent_with_hr"] in ("True", "true", True),
            "peak_corr": _f(r["peak_correlation"]),
            "hr_a": hr_a,
            "hr_b": hr_b,
            "hr_delta": abs(hr_a - hr_b) if (hr_a and hr_b) else None,
            "ft_a": ft_a, "ft_b": ft_b,
            "ft_match": (ft_a == ft_b) if (ft_a and ft_b) else None,
            "model_match": model_a == model_b and model_a != "",
            "manu_match": manu_a == manu_b and manu_a != "",
            "nframes_a": nframes_a, "nframes_b": nframes_b,
            "nframes_delta": abs(nframes_a - nframes_b)
                             if (nframes_a and nframes_b) else None,
            "model": f"{model_a}|{model_b}",
            "lag_mod_cycle": _f(r["peak_lag_mod_cycle"]),
            "cycle_period_s": _f(r["cycle_period_s"]),
        })

    cons = [p for p in per_pair if p["consistent"]]
    inc = [p for p in per_pair if not p["consistent"]]

    def med(rs, k):
        vs = [r[k] for r in rs if r[k] is not None]
        if not vs:
            return "n/a"
        return (f"median={np.median(vs):.2f} "
                f"IQR=[{np.percentile(vs, 25):.2f}, {np.percentile(vs, 75):.2f}] "
                f"max={max(vs):.2f}")

    print(f"Within-study pairs: {len(per_pair)}  "
          f"(consistent={len(cons)}  inconsistent={len(inc)})")
    print()
    print("Hypothesis 1 — HR drift between clips of the same study:")
    print(f"  consistent   pairs: |HR_a - HR_b|  {med(cons, 'hr_delta')}")
    print(f"  inconsistent pairs: |HR_a - HR_b|  {med(inc, 'hr_delta')}")
    print()
    print("Hypothesis 2 — scanner / frame-time mismatch within a study:")
    for tag, rs in [("consistent", cons), ("inconsistent", inc)]:
        if not rs:
            continue
        model_mm = sum(1 for r in rs if not r["model_match"])
        manu_mm = sum(1 for r in rs if not r["manu_match"])
        ft_mm = sum(1 for r in rs
                    if r["ft_match"] is False)
        print(f"  {tag:12s} n={len(rs)}:  "
              f"model_mismatch={model_mm}/{len(rs)}  "
              f"manufacturer_mismatch={manu_mm}/{len(rs)}  "
              f"frame_time_mismatch={ft_mm}/{len(rs)}")

    print()
    print("Inconsistent-pair detail (pair, HR_a, HR_b, deltaHR, lag_mod_cycle, "
          "cycle, peak_corr, model):")
    for p in sorted(inc, key=lambda r: -(r["hr_delta"] or 0)):
        hd = f"{p['hr_delta']:.0f}" if p["hr_delta"] is not None else "?"
        lmc = f"{p['lag_mod_cycle']:+.2f}" if p["lag_mod_cycle"] is not None else "?"
        cyc = f"{p['cycle_period_s']:.2f}" if p["cycle_period_s"] is not None else "?"
        corr = f"{p['peak_corr']:+.3f}" if p["peak_corr"] is not None else "?"
        print(f"  {p['pair']}  HR={p['hr_a']}/{p['hr_b']} dHR={hd}  "
              f"lag%cyc={lmc}s cyc={cyc}s  corr={corr}  model={p['model']}")

    print()
    print("Consistent-pair HR deltas (for contrast):")
    for p in sorted(cons, key=lambda r: -(r["hr_delta"] or 0))[:10]:
        hd = f"{p['hr_delta']:.0f}" if p["hr_delta"] is not None else "?"
        print(f"  {p['pair']}  HR={p['hr_a']}/{p['hr_b']} dHR={hd}")

    # Retrospective: would using pair-mean HR reclassify any inconsistent
    # pair as consistent?
    print()
    print("Retrospective: pair-mean-HR tolerance check (|lag % pair-cycle| < cycle/4):")
    reclass = 0
    for p in inc:
        if not (p["hr_a"] and p["hr_b"] and p["lag_mod_cycle"] is not None):
            continue
        # We have lag % (cycle from HR_a). Compute the raw lag seconds from
        # the original csv column? We don't — but peak_lag_seconds is in the
        # full CSV. Approximate: if HR delta is large, cycle periods differ,
        # wrapping the same peak lag under a different cycle can flip status.
        # For an exact answer we'd redo with pair-mean HR; here we just
        # report the bound.
        hr_mean = 0.5 * (p["hr_a"] + p["hr_b"])
        cyc_mean = 60.0 / hr_mean
        # The original wrapped lag is in (-cyc_a/2, cyc_a/2]. With cyc_mean
        # slightly different, |wrap| could drop below cyc_mean/4 only if
        # |cyc_mean - cyc_a| moves the threshold; compare.
        thr = cyc_mean / 4.0
        if abs(p["lag_mod_cycle"]) < thr:
            reclass += 1
    print(f"  Inconsistent pairs that would pass with pair-mean HR "
          f"(loose approximation): {reclass}/{len(inc)}")


if __name__ == "__main__":
    main()
