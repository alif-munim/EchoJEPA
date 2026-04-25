"""Preprocess the Gate 1 full-corpus phase metadata CSV for phi-JEPA Run D.

Input:  clip_phase_metadata.csv (from extract_dicom_phase_metadata.py, N_STUDIES=-1)
Output: mimic_clip_phase_metadata.csv with added `is_irregular` column.

Operations:
  1. Compute per-study HR stdev.
  2. Flag studies with stdev > --irregular-stdev (default 15 bpm) as irregular.
  3. Emit an augmented CSV consumed by VideoDataset.phase_metadata_csv.

Also writes a short summary report to stdout.

Usage:
  python scripts/neurips/phase/preprocess_phase_metadata.py \
      --in  /opt/dlami/nvme/clip_phase_metadata.csv \
      --out /opt/dlami/nvme/mimic_clip_phase_metadata.csv
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Raw Gate 1 CSV.")
    ap.add_argument("--out", dest="out_path", required=True, help="Augmented CSV.")
    ap.add_argument("--irregular-stdev", type=float, default=15.0,
                    help="Within-study HR stdev threshold (bpm). Above -> irregular.")
    args = ap.parse_args()

    print(f"[load] {args.in_path}", flush=True)
    df = pd.read_csv(args.in_path)
    n0 = len(df)
    print(f"       {n0} clips, columns: {list(df.columns)}", flush=True)

    # Drop zero-row guard.
    if "hr_bpm" not in df.columns or "study_id" not in df.columns:
        print("[fatal] input CSV missing hr_bpm or study_id columns", file=sys.stderr)
        return 1

    # Clip valid-HR mask used for study-level stdev only.
    valid = df["hr_bpm"].notna() & df["hr_bpm"].between(40.0, 180.0)

    # Per-study HR stdev from VALID clips.
    stdev = (
        df.loc[valid]
        .groupby("study_id")["hr_bpm"]
        .std(ddof=0)
        .rename("study_hr_stdev")
    )
    study_ct = df.loc[valid].groupby("study_id").size().rename("study_valid_ct")
    stdev_df = pd.concat([stdev, study_ct], axis=1)

    # Studies with fewer than 2 valid clips: stdev is NaN; mark as "insufficient data"
    # and default to is_irregular=False (regular) so a single-clip study still contributes.
    stdev_df["is_irregular"] = (stdev_df["study_hr_stdev"] > args.irregular_stdev).fillna(False)

    # Join back onto clip-level.
    df = df.merge(stdev_df, how="left", left_on="study_id", right_index=True)
    df["is_irregular"] = df["is_irregular"].fillna(False).astype(bool)

    n_irr_clips = int(df["is_irregular"].sum())
    n_irr_studies = int(stdev_df["is_irregular"].sum())
    n_studies = len(stdev_df)
    n_cine = int(df.get("present_ft", pd.Series(dtype=bool)).astype(bool).sum()) if "present_ft" in df.columns else -1

    print(f"[stats] {n_studies} studies with >=2 valid-HR clips", flush=True)
    print(f"        {n_irr_studies} irregular-rhythm studies "
          f"(stdev > {args.irregular_stdev} bpm)  ->  {n_irr_clips}/{n0} clips flagged",
          flush=True)
    if n_cine >= 0:
        print(f"        {n_cine} clips have FrameTime (cine — usable for pretraining)", flush=True)

    # Reorder columns for readability.
    preferred = ["study_id", "clip_id", "hr_bpm", "frame_time_ms", "num_frames",
                 "fps", "is_irregular", "study_hr_stdev", "study_valid_ct",
                 "present_hr", "present_ft", "present_nf", "dicom_path",
                 "acquisition_dt", "sop_class", "error"]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    df = df[cols]

    df.to_csv(args.out_path, index=False)
    print(f"[out]   wrote {args.out_path} ({len(df)} rows, {len(df.columns)} cols)", flush=True)

    # FrameTime variance sanity: confirms the sampled-fps framing decision.
    if "frame_time_ms" in df.columns:
        ft = df["frame_time_ms"].dropna()
        if len(ft) > 0:
            print(f"[ft-variance]  n={len(ft)}", flush=True)
            print(f"               mean={ft.mean():.3f} ms  median={ft.median():.3f}", flush=True)
            print(f"               p05={np.percentile(ft, 5):.3f}  p95={np.percentile(ft, 95):.3f}", flush=True)
            print(f"               range=[{ft.min():.3f}, {ft.max():.3f}]", flush=True)
            print(f"               sampled-fps framing OK if p05..p95 is tight (within ~1 ms)", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
