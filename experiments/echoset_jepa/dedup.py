"""Near-duplicate clip clustering (plan §11.3).

Two modes:

- **metadata-only** (PR-N2): predicate drops the embedding-cosine rule. Used
  before c_clip cache lands. Catches obvious duplicates (same study + view +
  modality + near-identical frame count / duration) but misses content-level
  duplicates that differ in metadata.
- **full** (PR-N3+): includes c_clip cosine > 0.98 as the final gate.

Annotates ``study_clip_manifest`` with:
  - ``n_duplicates``: cluster-size-minus-one per row
  - ``is_duplicate_of``: clip_id of cluster representative (or "" if row is rep)
  - ``dedup_mode``: "metadata_only" | "full" — records which predicate was used
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DedupConfig:
    view_conf_threshold: float = 0.8
    max_frame_diff: int = 3
    max_duration_diff_s: float = 0.2
    cosine_threshold: float = 0.98
    require_cosine: bool = True     # False = metadata-only mode


def _pairwise_cosine(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    Xn = X / norms
    return Xn @ Xn.T


def find_near_dup_clusters(
    clip_ids: Sequence[str],
    view_labels: Sequence[str],
    view_confs: Sequence[float],
    modalities: Sequence[str],
    n_frames: Sequence[int],
    durations_s: Sequence[float],
    quality_scores: Sequence[float],
    c_clip: Optional[np.ndarray],
    cfg: DedupConfig,
) -> Tuple[List[int], List[Optional[str]]]:
    """Cluster near-dup clips within a single study.

    If ``cfg.require_cosine=True`` and ``c_clip`` is None, raises. If
    ``cfg.require_cosine=False``, the cosine rule is skipped.

    Returns ``(n_duplicates[i], is_duplicate_of[i])`` where n_duplicates is
    the cluster size minus one and is_duplicate_of is the cluster
    representative's ``clip_id`` (or ``None`` if this row IS the rep).
    """
    N = len(clip_ids)
    if cfg.require_cosine:
        if c_clip is None:
            raise ValueError("cfg.require_cosine=True but c_clip is None")
        assert c_clip.shape[0] == N, "c_clip rows must match clip count"

    parent = list(range(N))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    cos = _pairwise_cosine(c_clip.astype(np.float64)) if cfg.require_cosine else None

    for i in range(N):
        for j in range(i + 1, N):
            if view_labels[i] != view_labels[j]:
                continue
            if view_confs[i] <= cfg.view_conf_threshold or view_confs[j] <= cfg.view_conf_threshold:
                continue
            if modalities[i] != modalities[j]:
                continue
            if abs(n_frames[i] - n_frames[j]) >= cfg.max_frame_diff:
                continue
            if abs(durations_s[i] - durations_s[j]) >= cfg.max_duration_diff_s:
                continue
            if cfg.require_cosine:
                if cos[i, j] <= cfg.cosine_threshold:
                    continue
            union(i, j)

    # Representative = highest quality in cluster (tie-break by clip_id)
    cluster_members: dict = {}
    for i in range(N):
        cluster_members.setdefault(find(i), []).append(i)

    n_duplicates = [0] * N
    is_dup_of: List[Optional[str]] = [None] * N
    for members in cluster_members.values():
        if len(members) == 1:
            continue
        rep = max(members, key=lambda i: (quality_scores[i], -_stable_hash(clip_ids[i])))
        for m in members:
            n_duplicates[m] = len(members) - 1
            if m != rep:
                is_dup_of[m] = clip_ids[rep]
    return n_duplicates, is_dup_of


def _stable_hash(s: str) -> int:
    return sum(ord(c) * (i + 1) for i, c in enumerate(s))


def dedup_manifest(
    manifest_path: str,
    out_path: str,
    cfg: Optional[DedupConfig] = None,
    cclip_cache_dir: Optional[str] = None,
) -> None:
    """Apply ``find_near_dup_clusters`` per study, write annotated parquet.

    If ``cclip_cache_dir`` is None or cfg.require_cosine=False, runs in
    metadata-only mode. Otherwise loads c_clip .npy per clip from
    ``cclip_cache_dir/{study_id}/{clip_id}.npy``.
    """
    import pandas as pd

    cfg = cfg or DedupConfig()
    df = pd.read_parquet(manifest_path)
    required = {
        "study_id", "clip_id", "view_label", "view_conf", "modality",
        "n_frames", "clip_duration_s", "quality_score",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"manifest missing columns: {sorted(missing)}")

    use_cosine = cfg.require_cosine and cclip_cache_dir is not None
    mode = "full" if use_cosine else "metadata_only"
    logger.info("running dedup in mode=%s", mode)

    df["n_duplicates"] = 0
    df["is_duplicate_of"] = ""
    df["dedup_mode"] = mode
    total_dup_clips = 0
    total_clusters = 0

    for study_id, group in df.groupby("study_id"):
        idx = group.index.tolist()
        if len(idx) < 2:
            continue
        c_clip_arr = None
        if use_cosine:
            cache_dir = Path(cclip_cache_dir) / str(study_id)
            vecs = []
            for cid in group["clip_id"]:
                p = cache_dir / f"{cid}.npy"
                if not p.exists():
                    logger.warning("missing c_clip for %s/%s; skipping study dedup", study_id, cid)
                    c_clip_arr = None
                    break
                vecs.append(np.load(p))
            else:
                c_clip_arr = np.stack(vecs)
        local_cfg = DedupConfig(**{**cfg.__dict__, "require_cosine": use_cosine and c_clip_arr is not None})
        n_dup, is_dup_of = find_near_dup_clusters(
            clip_ids=group["clip_id"].tolist(),
            view_labels=group["view_label"].tolist(),
            view_confs=group["view_conf"].tolist(),
            modalities=group["modality"].tolist(),
            n_frames=group["n_frames"].astype(int).tolist(),
            durations_s=group["clip_duration_s"].astype(float).tolist(),
            quality_scores=group["quality_score"].astype(float).tolist(),
            c_clip=c_clip_arr,
            cfg=local_cfg,
        )
        df.loc[idx, "n_duplicates"] = n_dup
        df.loc[idx, "is_duplicate_of"] = [s if s is not None else "" for s in is_dup_of]
        total_dup_clips += sum(1 for s in is_dup_of if s is not None)
        total_clusters += sum(1 for n in n_dup if n > 0) // 2 if total_dup_clips else 0

    df.to_parquet(out_path, index=False)
    _emit_audit(df, Path(out_path).with_suffix(".dedup.json"), mode=mode, total_dup_clips=total_dup_clips)


def _emit_audit(df, path, *, mode: str, total_dup_clips: int) -> None:
    import pandas as pd

    audit = {
        "mode": mode,
        "total_clips": int(len(df)),
        "total_duplicate_clips": int(total_dup_clips),
        "duplicate_rate": float(total_dup_clips) / max(len(df), 1),
        "duplicate_rate_by_view_family": df[df.is_duplicate_of != ""]
            .groupby("view_family").size().to_dict()
            if (df.is_duplicate_of != "").any() else {},
        "duplicate_rate_by_modality": df[df.is_duplicate_of != ""]
            .groupby("modality").size().to_dict()
            if (df.is_duplicate_of != "").any() else {},
        "cluster_size_distribution": df[df["n_duplicates"] > 0]["n_duplicates"].add(1).value_counts().to_dict(),
    }
    Path(path).write_text(json.dumps(audit, indent=2, default=str))
    logger.info("dedup audit: %s", path)


def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cclip_cache_dir", default=None,
                    help="Path to c_clip cache. If omitted, runs in metadata-only mode.")
    ap.add_argument("--cosine_threshold", type=float, default=0.98)
    ap.add_argument("--require_cosine", action="store_true",
                    help="If set, fail if cache dir is missing or empty; else fall back to metadata-only.")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = DedupConfig(cosine_threshold=args.cosine_threshold,
                      require_cosine=args.require_cosine and args.cclip_cache_dir is not None)
    dedup_manifest(args.manifest, args.out, cfg, cclip_cache_dir=args.cclip_cache_dir)


if __name__ == "__main__":
    _main()
