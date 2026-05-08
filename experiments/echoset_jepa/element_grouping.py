"""Build ``study_element_manifest.parquet`` from ``study_clip_manifest.parquet``.

Groups clips by ``(view_family, modality, phase_bucket)`` per study (plan §3.1,
PR-N1). Does **not** aggregate ``c_clip`` here — aggregation happens at
dataloader time so the same element manifest can be reused under different
``element_agg`` settings.

Quality is NOT part of the element key. It is:
  - a diagnostic stratifier (``mean_quality_score`` column)
  - a bucket for context-side meta tokens (``quality_bucket_context``)
  - an aggregation weight when ``element_agg=quality_weighted`` at train time

Measurement site is a fourth manifest column but also not part of the key for
v1 — element_grouping follows the plan's 3-tuple identity. A TR CW Doppler
clip is still grouped with other CW Doppler same-view clips at the MVP level;
measurement-site splits become a Phase 6 refinement once Doppler coverage is
large enough to warrant it.

Usage
-----
    python -m experiments.echoset_jepa.element_grouping \\
        --clip_manifest s3://.../study_clip_manifest.parquet \\
        --out s3://.../study_element_manifest.parquet \\
        --max_M 64
"""

from __future__ import annotations

import argparse
import logging
from typing import Iterable

logger = logging.getLogger(__name__)

MAX_M_DEFAULT = 64


def build_element_manifest(
    clip_manifest_path: str,
    out_path: str,
    max_M: int = MAX_M_DEFAULT,
) -> None:
    import pandas as pd

    df = pd.read_parquet(clip_manifest_path)
    required = {
        "patient_id",
        "study_id",
        "clip_id",
        "view_family",
        "modality",
        "phase_bucket",
        "measurement_site",
        "quality_score",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"clip manifest missing columns: {sorted(missing)}")

    group_cols = ["study_id", "view_family", "modality", "phase_bucket"]
    grouped = (
        df.groupby(group_cols)
        .agg(
            patient_id=("patient_id", "first"),
            clip_ids=("clip_id", list),
            n_clips_in_element=("clip_id", "count"),
            mean_quality_score=("quality_score", "mean"),
            measurement_sites=("measurement_site", lambda s: sorted(set(s))),
        )
        .reset_index()
    )
    grouped["element_id"] = [
        f"{s}::{v}.{m}.{p}"
        for s, v, m, p in zip(
            grouped["study_id"],
            grouped["view_family"],
            grouped["modality"],
            grouped["phase_bucket"],
        )
    ]

    # Context-side quality bucket token: from mean_quality_score tertiles over
    # elements (cohort-wide). This is orthogonal to the per-clip
    # quality_bucket in the clip manifest.
    grouped = _add_element_quality_bucket(grouped)

    # Cap M_elements per study with diversity-preserving drop.
    grouped = _cap_per_study(grouped, max_M=max_M)

    logger.info(
        "built element manifest: %d studies → %d elements (capped at M<=%d)",
        grouped["study_id"].nunique(),
        len(grouped),
        max_M,
    )
    grouped.to_parquet(out_path, index=False)


def _add_element_quality_bucket(df) -> "pandas.DataFrame":
    import pandas as pd

    scores = df["mean_quality_score"].to_numpy()
    if len(scores) < 3:
        df["quality_bucket_context"] = "unknown"
        return df
    q33, q66 = pd.Series(scores).quantile([1 / 3, 2 / 3]).tolist()

    def _bucket(s: float) -> str:
        if s != s:
            return "unknown"
        if s < q33:
            return "low"
        if s < q66:
            return "med"
        return "high"

    df["quality_bucket_context"] = [_bucket(s) for s in scores]
    return df


def _cap_per_study(df, max_M: int) -> "pandas.DataFrame":
    """Drop lowest-priority elements so every study has <= max_M.

    Priority order (keep first):
      1. Unique view_family × modality pairs (diversity first)
      2. Then by mean_quality_score descending
      3. Then by n_clips_in_element descending
    """
    def _rank(group):
        # Mark one representative element per (view_family, modality) pair to
        # keep unconditionally; then rank the rest by quality × clip count.
        seen: set = set()
        primary: list[int] = []
        secondary: list[int] = []
        g = group.sort_values(
            ["mean_quality_score", "n_clips_in_element"], ascending=[False, False]
        )
        for idx, row in g.iterrows():
            key = (row["view_family"], row["modality"])
            if key not in seen:
                seen.add(key)
                primary.append(idx)
            else:
                secondary.append(idx)
        order = primary + secondary
        return order[:max_M]

    keep: list[int] = []
    for _, group in df.groupby("study_id"):
        keep.extend(_rank(group))
    return df.loc[keep].reset_index(drop=True)


def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip_manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_M", type=int, default=MAX_M_DEFAULT)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build_element_manifest(args.clip_manifest, args.out, max_M=args.max_M)


if __name__ == "__main__":
    _main()


__all__ = ["build_element_manifest", "MAX_M_DEFAULT"]
