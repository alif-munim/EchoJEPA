"""50-clip validation gate for phase_annotations.parquet.

Pure parquet-level check (no video decoding, no DataLoader). Verifies that
per-frame phase arrays are self-consistent and that phase-matched frames
across clips in the same study land within +/- 1 frame of each other after
the full round-trip through the pipeline.

Systematic offset > 1 frame means there is a bias in R-peak detection or
column->frame mapping to fix upstream before committing GPU-hours.

Usage:
    python phase_matched_validation.py \
        --parquet phase_annotations/phase_annotations.parquet \
        --n-studies 50 --seed 0
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


def _decode_phase_arrays(row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    phase = np.array(
        [np.nan if v is None else float(v) for v in json.loads(row.per_frame_phase_json)],
        dtype=np.float64,
    )
    confident = np.array(json.loads(row.confident_mask_json), dtype=bool)
    return phase, confident


def _nearest_confident_frame(phase: np.ndarray, confident: np.ndarray, target_phi: float) -> int | None:
    """Return the frame index whose phase is closest to target_phi on the unit
    circle, among frames flagged confident. None if no confident frames."""
    if not confident.any():
        return None
    idx = np.where(confident)[0]
    ph = phase[idx]
    d = np.abs(ph - target_phi)
    # wrap-around: treat 0 and 1 as equal
    d = np.minimum(d, 1.0 - d)
    return int(idx[np.argmin(d)])


def sample_studies(df: pd.DataFrame, n_studies: int, min_rpeaks: int, seed: int) -> pd.DataFrame:
    """Keep studies with >=2 clips where each clip has >=min_rpeaks in-video,
    then sample n_studies of them."""
    eligible = df[(df.n_rpeaks_in_video >= min_rpeaks) & (df.quality_tier.isin(["high", "medium"]))]
    per_study = eligible.groupby("study_id").size()
    multi = per_study[per_study >= 2].index
    eligible = eligible[eligible.study_id.isin(multi)]
    rng = np.random.default_rng(seed)
    studies = eligible.study_id.unique()
    rng.shuffle(studies)
    picked = studies[:n_studies]
    return eligible[eligible.study_id.isin(picked)].reset_index(drop=True)


def validate(df: pd.DataFrame, anchors: np.ndarray) -> dict:
    """For each study, pick all ordered clip pairs. For each anchor phase,
    find the nearest-confident-phase frame in each clip. Record the absolute
    phase error at that match (should be ~0) and the normalized frame
    discrepancy (|i_a/n_a - i_b/n_b| in cycle fractions, should be ~0 for
    bias-free pipeline)."""
    rows = []
    for study_id, sub in df.groupby("study_id"):
        clips = []
        for _, r in sub.iterrows():
            phase, conf = _decode_phase_arrays(r)
            clips.append((r.dicom_id, int(r.n_video_frames), phase, conf))
        for i in range(len(clips)):
            for j in range(i + 1, len(clips)):
                id_a, n_a, ph_a, c_a = clips[i]
                id_b, n_b, ph_b, c_b = clips[j]
                for phi in anchors:
                    fa = _nearest_confident_frame(ph_a, c_a, phi)
                    fb = _nearest_confident_frame(ph_b, c_b, phi)
                    if fa is None or fb is None:
                        continue
                    phase_err_a = abs(ph_a[fa] - phi)
                    phase_err_a = min(phase_err_a, 1.0 - phase_err_a)
                    phase_err_b = abs(ph_b[fb] - phi)
                    phase_err_b = min(phase_err_b, 1.0 - phase_err_b)
                    rows.append(
                        {
                            "study_id": study_id,
                            "dicom_a": id_a,
                            "dicom_b": id_b,
                            "phi": phi,
                            "frame_a": fa,
                            "frame_b": fb,
                            "n_frames_a": n_a,
                            "n_frames_b": n_b,
                            "phase_err_a": phase_err_a,
                            "phase_err_b": phase_err_b,
                            "phase_at_a": ph_a[fa],
                            "phase_at_b": ph_b[fb],
                        }
                    )
    out = pd.DataFrame(rows)
    if not len(out):
        return {"n_pairs": 0}
    # Normalized frame discrepancy expressed as cycle fractions. Two clips at
    # the same phase should have |phase_at_a - phase_at_b| ~= 0 (wrap-aware).
    wrap_err = np.minimum(
        np.abs(out.phase_at_a - out.phase_at_b),
        1.0 - np.abs(out.phase_at_a - out.phase_at_b),
    )
    out["cross_clip_phase_err"] = wrap_err
    return {
        "n_pairs": len(out),
        "n_studies": out.study_id.nunique(),
        "median_phase_err_a": float(out.phase_err_a.median()),
        "median_phase_err_b": float(out.phase_err_b.median()),
        "median_cross_clip_phase_err": float(out.cross_clip_phase_err.median()),
        "p90_cross_clip_phase_err": float(out.cross_clip_phase_err.quantile(0.9)),
        "max_cross_clip_phase_err": float(out.cross_clip_phase_err.max()),
        "table": out,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", type=Path, required=True)
    ap.add_argument("--n-studies", type=int, default=50)
    ap.add_argument("--min-rpeaks", type=int, default=3)
    ap.add_argument("--n-anchors", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-csv", type=Path, default=None)
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)
    sample = sample_studies(df, args.n_studies, args.min_rpeaks, args.seed)
    print(f"sampled {sample.study_id.nunique()} studies, {len(sample)} clips")

    anchors = np.linspace(0.0, 1.0, args.n_anchors, endpoint=False)
    result = validate(sample, anchors)

    print(f"n pairs: {result['n_pairs']}")
    print(f"n studies with pairs: {result['n_studies']}")
    print(f"median phase-err clip A: {result['median_phase_err_a']:.4f} cycles")
    print(f"median phase-err clip B: {result['median_phase_err_b']:.4f} cycles")
    print(f"median cross-clip phase disagreement: {result['median_cross_clip_phase_err']:.4f} cycles")
    print(f"p90  cross-clip phase disagreement: {result['p90_cross_clip_phase_err']:.4f} cycles")
    print(f"max  cross-clip phase disagreement: {result['max_cross_clip_phase_err']:.4f} cycles")

    # Express cross-clip error as frames at representative fps (avg of pair).
    tbl = result["table"]
    fps = (tbl.n_frames_a + tbl.n_frames_b) / 2.0  # loose proxy; actual phase->frame needs RR
    rr_frames = fps / 1.0  # RR span in frames equals n_video_frames when 1 cycle, loose
    frame_err = tbl.cross_clip_phase_err * rr_frames
    print(
        f"approx frame-equivalent discrepancy: "
        f"median={float(frame_err.median()):.2f} frames, "
        f"p90={float(frame_err.quantile(0.9)):.2f}, "
        f"max={float(frame_err.max()):.2f}"
    )

    # Gate: median cross-clip phase error should be < 1/min(n_frames) for
    # a well-behaved pipeline. Threshold: 0.05 cycles (5% of a cardiac
    # cycle), which is roughly 1-2 frames on ~30fps 2-3 second clips.
    gate = 0.05
    passed = result["median_cross_clip_phase_err"] < gate
    print(f"\ngate (median cross-clip err < {gate}): {'PASS' if passed else 'FAIL'}")

    if args.out_csv is not None:
        tbl.to_csv(args.out_csv, index=False)
        print(f"wrote per-pair table to {args.out_csv}")


if __name__ == "__main__":
    main()
