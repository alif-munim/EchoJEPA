"""Same-study pair manifest builder for MCC-JEPA.

Given a flat manifest of per-clip rows with columns ``study_id``, ``view``,
``modality``, and ``path``, draw one ``(clip_A, clip_B)`` pair per study
per epoch under a controlled mixture of view-pair classes:

    same_view        0.40   (e.g. A4C ↔ A4C different clip / phase)
    same_broad_family 0.30  (apical ↔ apical, parasternal_* ↔ parasternal_*)
    cross_view       0.20   (A4C ↔ PLAX, PLAX ↔ PSAX, etc.)
    cross_modality   0.10   (B-mode ↔ color Doppler, if available)

Falls back to any same-study distinct pair when a bucket is dry, or to
``(clip, clip)`` for single-clip studies.

The ``shuffle_source`` flag swaps clip_A with a clip from a *different*
study (matched on view if possible). Used by the shuffled-A diagnostic.

This module returns a pandas DataFrame with columns
``path_a, path_b, view_a, view_b, modality_a, modality_b, bucket,
fallback, shuffled_source``, which is then wired into the existing
``VideoGroupDataset.set_pair_dataframe(...)`` scaffold.
"""

from __future__ import annotations

import logging
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from classifier.phase.sampler.phase_matched_sampler import VIEW_FAMILIES, view_pair_class

logger = logging.getLogger(__name__)

DEFAULT_MIXTURE = {
    "same_view": 0.40,
    "same_broad_family": 0.30,
    "cross_view": 0.20,
    "cross_modality": 0.10,
}


@dataclass(frozen=True)
class PairRow:
    path_a: str
    path_b: str
    view_a: str
    view_b: str
    modality_a: str
    modality_b: str
    study_id: str
    bucket: str
    fallback: bool
    shuffled_source: bool


def _family(view: Optional[str]) -> str:
    v = (view or "UNKNOWN").upper()
    if v == "SUBCOSTAL":
        v = "Subcostal"
    return VIEW_FAMILIES.get(v, "other")


def _clip_pair_bucket(view_a: Optional[str], view_b: Optional[str], mod_a: str, mod_b: str) -> str:
    """Classify a clip pair into one of the four MVP buckets."""
    if mod_a != mod_b:
        return "cross_modality"
    cls = view_pair_class(view_a, view_b)
    if cls == "same_view":
        return "same_view"
    if cls == "same_family":
        return "same_broad_family"
    return "cross_view"


def _sample_bucket(rng: random.Random, mixture: dict) -> str:
    names = list(mixture.keys())
    weights = list(mixture.values())
    return rng.choices(names, weights=weights, k=1)[0]


def _pick_partner(
    anchor: pd.Series,
    study_clips: pd.DataFrame,
    want_bucket: str,
    rng: random.Random,
) -> Optional[pd.Series]:
    """Try to pick a same-study partner clip matching ``want_bucket``."""
    candidates = study_clips[study_clips.index != anchor.name]
    if len(candidates) == 0:
        return None

    def _bucket_for(r: pd.Series) -> str:
        return _clip_pair_bucket(
            anchor.get("view"),
            r.get("view"),
            anchor.get("modality", "bmode"),
            r.get("modality", "bmode"),
        )

    buckets = candidates.apply(_bucket_for, axis=1)
    matching = candidates[buckets == want_bucket]
    if len(matching) > 0:
        return matching.sample(n=1, random_state=rng.randint(0, 2**31 - 1)).iloc[0]
    return None


def build_pair_manifest(
    clip_df: pd.DataFrame,
    mixture: Optional[dict] = None,
    seed: int = 0,
    shuffle_source: bool = False,
) -> pd.DataFrame:
    """Build one (clip_A, clip_B) pair per study under the MVP mixture.

    :param clip_df: DataFrame with columns
        ``study_id, path, view, modality`` (at minimum).
    :param mixture: dict of bucket -> weight. Defaults to DEFAULT_MIXTURE.
    :param seed: RNG seed.
    :param shuffle_source: if True, clip_A is replaced by a view-matched clip
        from a *different* study. Used for the anti-hallucination diagnostic.
    """
    mixture = dict(mixture or DEFAULT_MIXTURE)
    if not np.isclose(sum(mixture.values()), 1.0):
        total = sum(mixture.values())
        mixture = {k: v / total for k, v in mixture.items()}
    required = {"study_id", "path", "view"}
    missing = required - set(clip_df.columns)
    if missing:
        raise ValueError(f"clip_df missing columns: {missing}")
    df = clip_df.copy()
    if "modality" not in df.columns:
        df["modality"] = "bmode"
    df["modality"] = df["modality"].fillna("bmode")

    rng = random.Random(seed)
    rows: list[PairRow] = []
    by_view: dict[str, list[int]] = defaultdict(list)
    for idx, row in df.iterrows():
        by_view[(row.get("view") or "UNKNOWN").upper()].append(idx)

    study_groups = dict(tuple(df.groupby("study_id", sort=False)))

    for study_id, study_clips in study_groups.items():
        if len(study_clips) == 0:
            continue
        anchor_idx = rng.randrange(len(study_clips))
        anchor = study_clips.iloc[anchor_idx]
        fallback = False
        if len(study_clips) == 1:
            partner = anchor
            fallback = True
            bucket = "fallback_single_clip"
        else:
            want = _sample_bucket(rng, mixture)
            partner = _pick_partner(anchor, study_clips, want, rng)
            if partner is None:
                fallback = True
                partner_row = (
                    study_clips[study_clips.index != anchor.name]
                    .sample(n=1, random_state=rng.randint(0, 2**31 - 1))
                    .iloc[0]
                )
                partner = partner_row
                bucket = "fallback_any"
            else:
                bucket = want

        if shuffle_source:
            anchor_view = (anchor.get("view") or "UNKNOWN").upper()
            pool = by_view.get(anchor_view, [])
            other_study_pool = [i for i in pool if df.loc[i, "study_id"] != study_id]
            if len(other_study_pool) == 0:
                other_study_pool = [i for i in df.index if df.loc[i, "study_id"] != study_id]
            if len(other_study_pool) == 0:
                shuffled_src = anchor
            else:
                shuffled_src = df.loc[rng.choice(other_study_pool)]
            source_row = shuffled_src
            shuffled_flag = True
        else:
            source_row = anchor
            shuffled_flag = False
        target_row = partner

        rows.append(
            PairRow(
                path_a=str(source_row["path"]),
                path_b=str(target_row["path"]),
                view_a=str(source_row.get("view", "UNKNOWN")),
                view_b=str(target_row.get("view", "UNKNOWN")),
                modality_a=str(source_row.get("modality", "bmode")),
                modality_b=str(target_row.get("modality", "bmode")),
                study_id=str(study_id),
                bucket=bucket,
                fallback=fallback,
                shuffled_source=shuffled_flag,
            )
        )

    return pd.DataFrame([r.__dict__ for r in rows])


def sampler_diagnostics(pair_df: pd.DataFrame) -> dict:
    """Summarize a pair manifest for CSV logging."""
    n = max(len(pair_df), 1)
    bucket_counts = Counter(pair_df["bucket"])
    view_a = Counter(pair_df["view_a"])
    view_b = Counter(pair_df["view_b"])
    mod_a = Counter(pair_df["modality_a"])
    mod_b = Counter(pair_df["modality_b"])

    pair_same_study_rate = float((~pair_df["shuffled_source"]).mean())
    pair_distinct_clip_rate = float((pair_df["path_a"] != pair_df["path_b"]).mean())
    fallback_fraction = float(pair_df["fallback"].mean())

    return {
        "n_pairs": int(n),
        "pair_same_study_rate": pair_same_study_rate,
        "pair_distinct_clip_rate": pair_distinct_clip_rate,
        "fallback_fraction": fallback_fraction,
        "same_view_fraction": bucket_counts.get("same_view", 0) / n,
        "same_broad_family_fraction": bucket_counts.get("same_broad_family", 0) / n,
        "cross_view_fraction": bucket_counts.get("cross_view", 0) / n,
        "cross_modality_fraction": bucket_counts.get("cross_modality", 0) / n,
        "bucket_counts": dict(bucket_counts),
        "view_A_top": dict(view_a.most_common(8)),
        "view_B_top": dict(view_b.most_common(8)),
        "modality_A": dict(mod_a),
        "modality_B": dict(mod_b),
    }


def dry_run(clip_df: pd.DataFrame, seed: int = 0) -> dict:
    """Convenience entry for the launch helper: build + summarize."""
    pair_df = build_pair_manifest(clip_df, seed=seed)
    return sampler_diagnostics(pair_df)
