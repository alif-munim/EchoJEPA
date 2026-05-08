"""Fixed-K view/modality-stratified K-clip sampler.

Emits one ``study_clip_sample_K{K}_seed{S}.parquet`` file that is the single
source of truth for what clips every method (EchoSet-JEPA + Controls A–E)
sees at train/eval time. Plan §3.6.

Sampling policy (plan §I):
  K = 8 default
  target mix per study:
    4-5 slots for B-mode view diversity (apical / parasternal / RV / subcostal)
    1-2 slots for color Doppler if available
    1-2 slots for spectral Doppler / M-mode / TDI if available
    fill remaining by quality-weighted diversity

All baselines read this manifest — identical clip selection across methods.
"""

from __future__ import annotations

import argparse
import logging
import random
from typing import Dict, List, Optional, Sequence

from .taxonomy import normalize_view_family

logger = logging.getLogger(__name__)


def _quality_sort_key(row: Dict) -> tuple:
    # Higher quality first; clip_id is the deterministic tie-break.
    return (-float(row.get("quality_score", 0.0)), row["clip_id"])


def sample_view_stratified(
    clip_rows: Sequence[Dict],
    K: int,
    rng: random.Random,
) -> List[Dict]:
    """Round-robin across view_family, taking the highest-quality clip per pass.

    Kept for backwards compatibility with earlier tests. New code should use
    :func:`sample_mixed_modality` for full K=8 stratification.
    """
    by_family: Dict[str, List[Dict]] = {}
    for r in clip_rows:
        vf = r.get("view_family") or normalize_view_family(r.get("view_label"))
        by_family.setdefault(vf, []).append(r)

    for rows in by_family.values():
        rows.sort(key=_quality_sort_key)

    families = sorted(by_family.keys())
    rng.shuffle(families)

    picked: List[Dict] = []
    cursors = {f: 0 for f in families}
    while len(picked) < K:
        progressed = False
        for f in families:
            if len(picked) >= K:
                break
            rows = by_family[f]
            if cursors[f] < len(rows):
                picked.append(rows[cursors[f]])
                cursors[f] += 1
                progressed = True
        if not progressed:
            break
    return picked


def sample_mixed_modality(
    clip_rows: Sequence[Dict],
    K: int,
    rng: random.Random,
    bmode_budget: int = 6,
    color_budget: int = 2,
    spectral_budget: int = 0,
) -> List[Dict]:
    """Plan §I stratified sampler: view diversity + modality diversity.

    Budgets are soft: if a modality is absent from the study, its budget is
    reallocated to the next priority bucket (B-mode view diversity → color →
    spectral → other). The final `K` clips are guaranteed to:
      - cover as many distinct view_family values as possible,
      - include color Doppler if present and ``color_budget > 0``,
      - include spectral Doppler / M-mode / TDI if present and
        ``spectral_budget > 0``.

    Defaults are tuned for MIMIC (62% B-mode / 38% color Doppler; no spectral):
    K=8 → 6 B-mode slots + 2 color slots, spectral disabled. To re-enable
    spectral for a future cohort, pass ``spectral_budget>0``.
    """
    by_modality: Dict[str, List[Dict]] = {}
    for r in clip_rows:
        by_modality.setdefault(r.get("modality", "b_mode"), []).append(r)
    for rows in by_modality.values():
        rows.sort(key=_quality_sort_key)

    picked_ids: set = set()
    picked: List[Dict] = []

    def _take(pool: List[Dict], budget: int, stratify_by: str = "view_family") -> None:
        if budget <= 0 or not pool:
            return
        # Round-robin over stratify_by for diversity
        by_key: Dict[str, List[Dict]] = {}
        for r in pool:
            if r["clip_id"] in picked_ids:
                continue
            key = r.get(stratify_by) or "unknown"
            by_key.setdefault(key, []).append(r)
        keys = sorted(by_key.keys())
        rng.shuffle(keys)
        cursors = {k: 0 for k in keys}
        taken = 0
        while taken < budget and len(picked) < K:
            progressed = False
            for k in keys:
                if taken >= budget or len(picked) >= K:
                    break
                if cursors[k] < len(by_key[k]):
                    cand = by_key[k][cursors[k]]
                    cursors[k] += 1
                    if cand["clip_id"] in picked_ids:
                        continue
                    picked.append(cand)
                    picked_ids.add(cand["clip_id"])
                    taken += 1
                    progressed = True
            if not progressed:
                break

    # Priority 1: B-mode view diversity
    _take(by_modality.get("b_mode", []), bmode_budget)
    # Priority 2: color Doppler
    _take(by_modality.get("color_doppler", []), color_budget)
    # Priority 3: spectral Doppler / M-mode / TDI
    spectral_pool: List[Dict] = []
    for m in ("cw_doppler", "pw_doppler", "m_mode", "tdi"):
        spectral_pool.extend(by_modality.get(m, []))
    spectral_pool.sort(key=_quality_sort_key)
    _take(spectral_pool, spectral_budget)
    # Fill remaining by quality-weighted diversity across all clips
    remaining_pool = [r for r in clip_rows if r["clip_id"] not in picked_ids]
    remaining_pool.sort(key=_quality_sort_key)
    _take(remaining_pool, K - len(picked))
    return picked


def _annotate_k_sample(picked: List[Dict], K: int, seed: int) -> List[Dict]:
    """Add per-row sampler diagnostic columns."""
    view_set = {r.get("view_family", "unknown") for r in picked}
    mod_set = {r.get("modality", "b_mode") for r in picked}
    n_color = sum(1 for r in picked if r.get("modality") == "color_doppler")
    n_spectral = sum(1 for r in picked if r.get("modality") in {"cw_doppler", "pw_doppler", "m_mode", "tdi"})
    n_bmode = sum(1 for r in picked if r.get("modality") == "b_mode")
    # Elements this selection would produce
    element_keys = {
        (r.get("view_family", "unknown"), r.get("modality", "b_mode"), r.get("phase_bucket", "unknown"))
        for r in picked
    }
    rows = []
    for slot, r in enumerate(picked):
        rows.append(
            {
                "study_id": r.get("study_id"),
                "clip_id": r["clip_id"],
                "s3_uri": r.get("s3_uri", ""),
                "view_family": r.get("view_family", "unknown"),
                "modality": r.get("modality", "b_mode"),
                "phase_bucket": r.get("phase_bucket", "unknown"),
                "measurement_site": r.get("measurement_site", "none"),
                "quality_score": float(r.get("quality_score", 0.0)),
                "slot": slot,
                "K": K,
                "seed": seed,
                "cached_cclip_s3": r.get("cached_cclip_s3", ""),
                "n_unique_view_families": len(view_set),
                "n_unique_modalities": len(mod_set),
                "n_bmode": n_bmode,
                "n_color": n_color,
                "n_spectral": n_spectral,
                "n_elements": len(element_keys),
            }
        )
    return rows


def build(
    clip_manifest_path: str,
    out_path: str,
    K: int,
    seed: int,
    policy: str = "mixed_modality",
    bmode_budget: int = 6,
    color_budget: int = 2,
    spectral_budget: int = 0,
    split_filter: Optional[str] = None,
) -> None:
    import pandas as pd

    df = pd.read_parquet(clip_manifest_path)
    if split_filter is not None:
        if "split" not in df.columns:
            raise ValueError("manifest has no 'split' column; run splits.build_split first")
        df = df[df["split"] == split_filter].reset_index(drop=True)
        logger.info("filtered to split=%r: %d rows", split_filter, len(df))
    if "view_family" not in df.columns:
        df["view_family"] = [normalize_view_family(v) for v in df.get("view_label", [])]

    rng = random.Random(seed)
    out_rows: List[Dict] = []
    for study_id, group in df.groupby("study_id"):
        clip_rows = group.to_dict("records")
        if policy == "mixed_modality":
            picked = sample_mixed_modality(
                clip_rows, K=K, rng=rng,
                bmode_budget=bmode_budget, color_budget=color_budget,
                spectral_budget=spectral_budget,
            )
        elif policy == "view_stratified":
            picked = sample_view_stratified(clip_rows, K=K, rng=rng)
        else:
            raise ValueError(f"unknown policy={policy!r}")
        out_rows.extend(_annotate_k_sample(picked, K=K, seed=seed))

    out = pd.DataFrame(out_rows)
    out.to_parquet(out_path, index=False)
    logger.info("wrote %d rows (≤K=%d per study, policy=%s) -> %s", len(out), K, policy, out_path)


def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip_manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--policy", choices=["mixed_modality", "view_stratified"], default="mixed_modality")
    ap.add_argument("--bmode_budget", type=int, default=6)
    ap.add_argument("--color_budget", type=int, default=2)
    ap.add_argument("--spectral_budget", type=int, default=0)
    ap.add_argument("--split", choices=["train", "val", "test"], default=None,
                    help="if manifest has a 'split' column, filter to this split")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build(
        args.clip_manifest, args.out, args.K, args.seed, policy=args.policy,
        bmode_budget=args.bmode_budget, color_budget=args.color_budget,
        spectral_budget=args.spectral_budget, split_filter=args.split,
    )


if __name__ == "__main__":
    _main()


__all__ = ["sample_view_stratified", "sample_mixed_modality", "build"]
