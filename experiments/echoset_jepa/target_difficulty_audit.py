"""Target-difficulty audit — P2 gate (plan §10, PR-N3).

For a sample of studies from the val split, construct ``(context, target)``
mask instances and compute four cosines per target element:

  1. cos(target_element, nearest_context_element)
  2. cos(target_element, same-view other-study mean)
  3. cos(target_element, metadata-only prediction)
  4. cos(target_element, same-study different-view mean)

Outputs:
  reports/echoset_jepa/target_difficulty.{md,json}

Flag thresholds (plan §10):
  - cos(target, nearest_context) > 0.9 for > 10% of cases → dedup too loose
  - cos(target, metadata_pred)    > 0.8 for > 20% of cases → target trivially meta-solvable

Exit code:
  0  = all gates pass (PR-N3 gate)
  1  = one or more gate thresholds exceeded; operator must investigate

This script does not touch GPU — it operates on the cached c_clip .npy
files produced by ``cache_cclip.py``. Runtime is minutes on a single CPU
over ~5k sampled studies.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


GATE = {
    "max_near_context_frac": 0.10,
    "max_metadata_trivial_frac": 0.20,
}


def _load_c_clip(cache_prefix: str, study_id: str, clip_id: str) -> Optional[np.ndarray]:
    if cache_prefix.startswith("s3://"):
        # For audit we expect a local mirror; callers should set cache_prefix
        # to the local mount after syncing. Avoid per-clip S3 reads.
        return None
    p = Path(cache_prefix) / str(study_id) / f"{clip_id}.npy"
    if not p.exists():
        return None
    return np.load(p)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a) + 1e-12
    nb = np.linalg.norm(b) + 1e-12
    return float(np.dot(a, b) / (na * nb))


def _element_vectors(
    clip_manifest_rows: Sequence[dict],
    cache_prefix: str,
) -> Tuple[List[Tuple[str, str, str]], np.ndarray, List[List[str]]]:
    """Aggregate a study's clips into element vectors (mean pool).

    Returns (element_keys, element_matrix [M, D], element_clip_ids).
    Missing c_clips are silently dropped. Elements with zero successful
    loads are skipped.
    """
    buckets: Dict[Tuple[str, str, str], List[Tuple[str, np.ndarray]]] = defaultdict(list)
    for r in clip_manifest_rows:
        vec = _load_c_clip(cache_prefix, r["study_id"], r["clip_id"])
        if vec is None:
            continue
        key = (r.get("view_family", "unknown"),
               r.get("modality", "b_mode"),
               r.get("phase_bucket", "unknown"))
        buckets[key].append((r["clip_id"], vec))

    keys: List[Tuple[str, str, str]] = []
    rows: List[np.ndarray] = []
    clip_ids: List[List[str]] = []
    for key, entries in buckets.items():
        if not entries:
            continue
        vecs = np.stack([v for _, v in entries], axis=0)
        keys.append(key)
        rows.append(vecs.mean(axis=0))
        clip_ids.append([cid for cid, _ in entries])
    if not rows:
        return [], np.empty((0, 0)), []
    return keys, np.stack(rows, axis=0), clip_ids


def _metadata_only_prediction(
    target_key: Tuple[str, str, str],
    meta_prototypes: Dict[Tuple[str, str, str], np.ndarray],
) -> Optional[np.ndarray]:
    """Look up the global-mean element vector for this (view, modality, phase)
    key. Returns None if the key never appeared in the prototype pool."""
    return meta_prototypes.get(target_key)


def _build_meta_prototypes(
    clip_manifest_df,
    cache_prefix: str,
    max_clips_per_key: int = 500,
    seed: int = 0,
) -> Dict[Tuple[str, str, str], np.ndarray]:
    """One vector per (view_family, modality, phase_bucket) = mean of up to
    max_clips_per_key c_clips from held-out audit studies.
    """
    rng = random.Random(seed)
    by_key: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)
    rows = clip_manifest_df.to_dict("records")
    rng.shuffle(rows)
    for r in rows:
        key = (r.get("view_family", "unknown"),
               r.get("modality", "b_mode"),
               r.get("phase_bucket", "unknown"))
        if len(by_key[key]) < max_clips_per_key:
            by_key[key].append((r["study_id"], r["clip_id"]))

    protos: Dict[Tuple[str, str, str], np.ndarray] = {}
    for key, clip_list in by_key.items():
        vecs = []
        for sid, cid in clip_list:
            v = _load_c_clip(cache_prefix, sid, cid)
            if v is not None:
                vecs.append(v)
        if vecs:
            protos[key] = np.stack(vecs, axis=0).mean(axis=0)
    logger.info("built %d metadata prototypes", len(protos))
    return protos


def _sample_mask(M: int, rng: random.Random) -> Tuple[List[int], List[int]]:
    """Pick a simple random-element mask: context + 1-2 targets."""
    if M < 2:
        return list(range(M)), []
    n_target = max(1, min(2, M - 1))
    targets = rng.sample(range(M), n_target)
    context = [i for i in range(M) if i not in targets]
    return context, targets


def run_audit(
    clip_manifest_path: str,
    element_manifest_path: str,
    cache_prefix: str,
    out_dir: str,
    *,
    audit_split: str = "val",
    proto_split: str = "train",
    num_studies: int = 5000,
    seed: int = 0,
) -> Dict:
    import pandas as pd

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    clip_df = pd.read_parquet(clip_manifest_path)
    elem_df = pd.read_parquet(element_manifest_path)  # noqa: F841 — reserved for stratification

    if "split" in clip_df.columns:
        audit_df = clip_df[clip_df["split"] == audit_split]
        proto_df = clip_df[clip_df["split"] == proto_split]
    else:
        audit_df = clip_df
        proto_df = clip_df
        logger.warning("manifest has no 'split' column; using all rows for both audit and prototypes")

    logger.info("audit_split=%s: %d clips, %d studies", audit_split,
                len(audit_df), audit_df["study_id"].nunique())
    logger.info("proto_split=%s: %d clips", proto_split, len(proto_df))

    # --- metadata prototypes (from proto_split) ----------------------------
    meta_protos = _build_meta_prototypes(proto_df, cache_prefix, seed=seed)

    # --- same-view other-study pool (from proto_split) ---------------------
    # Collect per-view mean from the proto pool; use for cos #2
    same_view_protos: Dict[str, np.ndarray] = {}
    by_view: Dict[str, List[np.ndarray]] = defaultdict(list)
    for r in proto_df.sample(min(len(proto_df), 20000), random_state=seed).to_dict("records"):
        v = _load_c_clip(cache_prefix, r["study_id"], r["clip_id"])
        if v is not None:
            by_view[r.get("view_family", "unknown")].append(v)
    for vf, vs in by_view.items():
        same_view_protos[vf] = np.stack(vs, axis=0).mean(axis=0)
    logger.info("built %d view-family prototypes", len(same_view_protos))

    # --- iterate audit studies ---------------------------------------------
    rng = random.Random(seed)
    studies = list(audit_df["study_id"].unique())
    rng.shuffle(studies)
    studies = studies[:num_studies]

    per_row: List[Dict] = []
    for sid in studies:
        rows = audit_df[audit_df["study_id"] == sid].to_dict("records")
        keys, elem_mat, _ = _element_vectors(rows, cache_prefix)
        M = len(keys)
        if M < 2:
            continue
        ctx_idx, tgt_idx = _sample_mask(M, rng)
        if not tgt_idx:
            continue
        ctx_mat = elem_mat[ctx_idx]

        for ti in tgt_idx:
            tgt_vec = elem_mat[ti]
            tgt_key = keys[ti]
            tgt_vf, tgt_mod, tgt_phase = tgt_key

            # 1. nearest context element
            cos_nc = float(max(_cos(tgt_vec, ctx_mat[j]) for j in range(ctx_mat.shape[0])))

            # 2. same-view other-study mean
            sv = same_view_protos.get(tgt_vf)
            cos_sv = _cos(tgt_vec, sv) if sv is not None else float("nan")

            # 3. metadata-only prediction
            mp = _metadata_only_prediction(tgt_key, meta_protos)
            cos_mp = _cos(tgt_vec, mp) if mp is not None else float("nan")

            # 4. same-study different-view mean
            other_ctx = [j for j in range(ctx_mat.shape[0]) if keys[ctx_idx[j]][0] != tgt_vf]
            if other_ctx:
                ss_dv = ctx_mat[other_ctx].mean(axis=0)
                cos_ssdv = _cos(tgt_vec, ss_dv)
            else:
                cos_ssdv = float("nan")

            per_row.append({
                "study_id": sid,
                "target_view_family": tgt_vf,
                "target_modality": tgt_mod,
                "target_phase_bucket": tgt_phase,
                "M_elements": M,
                "cos_nearest_context": cos_nc,
                "cos_same_view_other_study": cos_sv,
                "cos_metadata_only": cos_mp,
                "cos_same_study_different_view": cos_ssdv,
            })

    df = pd.DataFrame(per_row)
    if df.empty:
        logger.error("no audit rows produced — c_clip cache may be empty or unreadable")
        sys.exit(2)
    df.to_csv(out / "target_difficulty_per_row.csv", index=False)
    logger.info("wrote %d audit rows → %s", len(df), out / "target_difficulty_per_row.csv")

    # --- summary + gate evaluation -----------------------------------------
    def _frac(series, threshold):
        s = series.dropna()
        return float((s > threshold).mean()) if len(s) else float("nan")

    summary: Dict = {
        "audit_split": audit_split,
        "proto_split": proto_split,
        "n_rows": int(len(df)),
        "n_studies": int(df["study_id"].nunique()),
        "cos_nearest_context": {
            "mean": float(df["cos_nearest_context"].mean()),
            "median": float(df["cos_nearest_context"].median()),
            "frac_gt_0p9": _frac(df["cos_nearest_context"], 0.9),
        },
        "cos_same_view_other_study": {
            "mean": float(df["cos_same_view_other_study"].mean()),
            "median": float(df["cos_same_view_other_study"].median()),
        },
        "cos_metadata_only": {
            "mean": float(df["cos_metadata_only"].mean()),
            "median": float(df["cos_metadata_only"].median()),
            "frac_gt_0p8": _frac(df["cos_metadata_only"], 0.8),
        },
        "cos_same_study_different_view": {
            "mean": float(df["cos_same_study_different_view"].mean()),
            "median": float(df["cos_same_study_different_view"].median()),
        },
        "gate_thresholds": GATE,
    }

    # Per-target-view breakdown
    by_view = {}
    for vf, sub in df.groupby("target_view_family"):
        by_view[str(vf)] = {
            "n": int(len(sub)),
            "cos_nearest_context_mean": float(sub["cos_nearest_context"].mean()),
            "cos_metadata_only_mean": float(sub["cos_metadata_only"].mean()),
            "cos_same_study_different_view_mean": float(sub["cos_same_study_different_view"].mean()),
        }
    summary["per_target_view"] = by_view

    gate_pass = (
        summary["cos_nearest_context"]["frac_gt_0p9"] <= GATE["max_near_context_frac"]
        and summary["cos_metadata_only"]["frac_gt_0p8"] <= GATE["max_metadata_trivial_frac"]
    )
    summary["gate_passed"] = bool(gate_pass)
    summary["gate_diagnostics"] = {
        "nearest_context_fraction": summary["cos_nearest_context"]["frac_gt_0p9"],
        "metadata_only_fraction": summary["cos_metadata_only"]["frac_gt_0p8"],
    }

    (out / "target_difficulty.json").write_text(json.dumps(summary, indent=2, default=str))
    (out / "target_difficulty.md").write_text(_render_markdown(summary))
    logger.info("target-difficulty audit: gate_passed=%s", gate_pass)
    return summary


def _render_markdown(s: Dict) -> str:
    lines = [
        "# EchoSet-JEPA target-difficulty audit (P2 gate)",
        "",
        f"- audit_split: `{s['audit_split']}`",
        f"- proto_split: `{s['proto_split']}`",
        f"- rows: {s['n_rows']:,}, studies: {s['n_studies']:,}",
        "",
        "## Gate status",
        "",
        f"- **passed**: {s['gate_passed']}",
        f"- cos(target, nearest_context) > 0.9 frac = {s['cos_nearest_context']['frac_gt_0p9']:.3f}  (threshold ≤ {s['gate_thresholds']['max_near_context_frac']})",
        f"- cos(target, metadata_only)    > 0.8 frac = {s['cos_metadata_only']['frac_gt_0p8']:.3f}  (threshold ≤ {s['gate_thresholds']['max_metadata_trivial_frac']})",
        "",
        "## Cosine distributions (mean / median)",
        "",
        f"- nearest_context:          {s['cos_nearest_context']['mean']:.3f} / {s['cos_nearest_context']['median']:.3f}",
        f"- same_view_other_study:    {s['cos_same_view_other_study']['mean']:.3f} / {s['cos_same_view_other_study']['median']:.3f}",
        f"- metadata_only:            {s['cos_metadata_only']['mean']:.3f} / {s['cos_metadata_only']['median']:.3f}",
        f"- same_study_different_view:{s['cos_same_study_different_view']['mean']:.3f} / {s['cos_same_study_different_view']['median']:.3f}",
        "",
        "## Per-target-view breakdown",
        "",
        "| target_view | n | nearest_ctx | meta_only | same_study_dv |",
        "|---|---|---|---|---|",
    ]
    for vf, row in s["per_target_view"].items():
        lines.append(
            f"| {vf} | {row['n']} | {row['cos_nearest_context_mean']:.3f} | "
            f"{row['cos_metadata_only_mean']:.3f} | {row['cos_same_study_different_view_mean']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip_manifest", required=True)
    ap.add_argument("--element_manifest", required=True)
    ap.add_argument("--cache_prefix", required=True,
                    help="local directory containing {study_id}/{clip_id}.npy")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--audit_split", default="val")
    ap.add_argument("--proto_split", default="train")
    ap.add_argument("--num_studies", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    s = run_audit(
        clip_manifest_path=args.clip_manifest,
        element_manifest_path=args.element_manifest,
        cache_prefix=args.cache_prefix,
        out_dir=args.out_dir,
        audit_split=args.audit_split,
        proto_split=args.proto_split,
        num_studies=args.num_studies,
        seed=args.seed,
    )
    sys.exit(0 if s["gate_passed"] else 1)


if __name__ == "__main__":
    _main()
