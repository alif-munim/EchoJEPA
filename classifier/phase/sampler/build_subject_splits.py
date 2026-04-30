"""Build subject-level train/val/test splits from phase_annotations.parquet.

Splits on ``subject_id`` (NOT ``study_id``), because the parquet audit
showed 1,408 subjects appearing in multiple studies. Any study-level
split would leak the same patient into both sides of the pre/post
Δ_within comparison.

Filters eligible clips using ``quality_tiers`` and ``rr_filter_mode``
(same filters the phase-matched sampler applies) before splitting, so
the reported per-split clip counts reflect what training will actually
see.

Usage:
    python build_subject_splits.py \\
        --parquet phase_annotations/phase_annotations.parquet \\
        --out-dir splits/ \\
        --val-frac 0.05 --test-frac 0.10 \\
        --seed 0 \\
        --quality-tiers high \\
        --rr-filter-mode strict

Writes:
    <out-dir>/train_subjects.txt
    <out-dir>/val_subjects.txt
    <out-dir>/test_subjects.txt
    <out-dir>/subjects_split.csv         (subject_id,split)
    <out-dir>/dicoms_split.csv           (dicom_id,subject_id,study_id,split)
    <out-dir>/split_summary.json

The CSVs are the canonical input to ``check_phase_split_integrity.py
--split-csv`` which will fail loudly if any subject appears in more
than one split.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from rr_consistency import rr_consistent  # noqa: E402


def filter_eligible(
    df: pd.DataFrame,
    quality_tiers: list[str],
    rr_filter_mode: str,
    rr_meta_ratio_range: tuple[float, float] = (0.80, 1.25),
    rr_max_min_ratio: float = 1.40,
) -> pd.DataFrame:
    sub = df[df.quality_tier.isin(quality_tiers)].copy()
    if rr_filter_mode == "strict":
        mask = sub.apply(
            lambda r: rr_consistent(r, rr_meta_ratio_range, rr_max_min_ratio),
            axis=1,
        )
    elif rr_filter_mode == "permissive_afib":
        mask = sub.apply(
            lambda r: rr_consistent(r, rr_meta_ratio_range, None), axis=1
        )
    elif rr_filter_mode == "off":
        mask = pd.Series(True, index=sub.index)
    else:
        raise ValueError(f"unknown rr_filter_mode {rr_filter_mode}")
    return sub[mask.values].reset_index(drop=True)


def split_subjects(
    subjects: list[str],
    val_frac: float,
    test_frac: float,
    seed: int,
) -> dict[str, list[str]]:
    rng = np.random.default_rng(seed)
    arr = np.array(sorted(subjects))
    rng.shuffle(arr)
    n = len(arr)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    n_train = n - n_test - n_val
    if n_train <= 0:
        raise ValueError(
            f"Splits too large: {n} subjects -> train={n_train}, val={n_val}, test={n_test}"
        )
    return {
        "train": arr[:n_train].tolist(),
        "val": arr[n_train : n_train + n_val].tolist(),
        "test": arr[n_train + n_val :].tolist(),
    }


def summarize_split(df: pd.DataFrame, subj_to_split: dict[str, str]) -> dict:
    df = df.copy()
    df["split"] = df.subject_id.astype(str).map(subj_to_split)
    # Unassigned rows (shouldn't happen after filtering subjects to dfsubjects).
    n_unassigned = int(df.split.isna().sum())
    out = {"n_rows_unassigned": n_unassigned, "per_split": {}}
    for split in ("train", "val", "test"):
        sub = df[df.split == split]
        if not len(sub):
            continue
        subjects = sub.subject_id.dropna().astype(str).unique()
        studies = sub.study_id.dropna().astype(str).unique()
        clips_per_study = sub.groupby("study_id").size()
        multi = int((clips_per_study >= 2).sum())
        out["per_split"][split] = {
            "n_subjects": int(len(subjects)),
            "n_studies": int(len(studies)),
            "n_clips": int(len(sub)),
            "n_studies_multi_clip": multi,
            "median_clips_per_study": float(clips_per_study.median()) if len(clips_per_study) else 0.0,
            "p90_clips_per_study": float(clips_per_study.quantile(0.9)) if len(clips_per_study) else 0.0,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--test-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quality-tiers", nargs="+", default=["high"])
    ap.add_argument("--rr-filter-mode", choices=("strict", "permissive_afib", "off"),
                    default="strict")
    args = ap.parse_args()

    need = [
        "dicom_id", "subject_id", "study_id", "quality_tier",
        "hr_metadata", "fps_video", "r_peaks_video_json",
    ]
    df = pd.read_parquet(args.parquet, columns=need)
    df = df[df.subject_id.notna()].copy()
    df["subject_id"] = df.subject_id.astype(str)

    n_all = len(df)
    elig = filter_eligible(
        df, args.quality_tiers, args.rr_filter_mode
    )
    n_elig = len(elig)

    subjects = sorted(elig.subject_id.unique().tolist())
    splits = split_subjects(subjects, args.val_frac, args.test_frac, args.seed)

    # Safety: no subject in two splits.
    seen = set()
    for s in ("train", "val", "test"):
        overlap = seen & set(splits[s])
        if overlap:
            raise RuntimeError(f"INTERNAL: subject_id leakage in {s} split: {sorted(overlap)[:5]}")
        seen |= set(splits[s])
    assert seen == set(subjects), "subject coverage mismatch"

    subj_to_split = {s: split for split, subs in splits.items() for s in subs}
    summary = summarize_split(elig, subj_to_split)
    summary.update({
        "parquet": str(args.parquet),
        "quality_tiers": list(args.quality_tiers),
        "rr_filter_mode": args.rr_filter_mode,
        "seed": args.seed,
        "val_frac": args.val_frac,
        "test_frac": args.test_frac,
        "n_rows_in_parquet": int(n_all),
        "n_rows_eligible": int(n_elig),
        "n_subjects_total": int(len(subjects)),
    })

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    for split, subs in splits.items():
        (out / f"{split}_subjects.txt").write_text("\n".join(subs) + "\n")
    subj_csv = pd.DataFrame(
        {"subject_id": list(subj_to_split.keys()), "split": list(subj_to_split.values())}
    )
    subj_csv.to_csv(out / "subjects_split.csv", index=False)
    dicom_csv = elig[["dicom_id", "subject_id", "study_id"]].copy()
    dicom_csv["split"] = dicom_csv.subject_id.map(subj_to_split)
    dicom_csv.to_csv(out / "dicoms_split.csv", index=False)

    # Final leak check on the saved CSV.
    per_subj = subj_csv.groupby("subject_id").split.nunique()
    leaked = per_subj[per_subj > 1]
    if len(leaked):
        raise RuntimeError(
            f"LEAKAGE: {len(leaked)} subjects appear in multiple splits: {leaked.head(5).to_dict()}"
        )

    (out / "split_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
