#!/usr/bin/env python3
"""MV2SV fused-pool coverage audit.

Runs the PhaseMatchedStudySampler with ``mv2sv_sampler.fused_pool.enabled
= true`` for N epochs and reports how often the sampler can actually fill
the fused pool with >=2 same-study distinct-view clips. This is the gate
that the Stage C smoke (job 653) tripped — fused_valid_mask mean valid
views was only 1.44-1.75, below the hard >=2 guard in
``forward_privileged_multiview``.

The audit is offline: it iterates ``MatchRecord`` objects without
decoding video. We only need ``target_view`` / ``fused_views`` /
``clip_a.view``.

Outputs:
    fused_valid_views mean / median / p10 / p25 / p75
    fraction of rows with >=2 valid fused views
    fraction of rows with >=3 valid fused views
    source_view counts
    target_view counts
    source x target pair counts
    fused_valid_views count buckets by (source_view, target_view)

Usage:
    python scripts/neurips/phase/mv2sv_fused_coverage_audit.py \
        --config configs/train/vitl16/smoke/mv2sv-smoke-v5-fused.yaml \
        --epochs 2 \
        --output /tmp/mv2sv_fused_audit.json

The audit does NOT require GPU. Runs on rank 0 only.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "classifier" / "phase" / "sampler"))


def _percentile(values, q):
    if not values:
        return float("nan")
    return float(np.percentile(np.array(values, dtype=np.float64), q))


def _coerce_fused_enabled(cfg: dict) -> dict:
    """Force fused_pool.enabled=true for the audit; keep everything else
    as the supplied config dictates. We want to audit what the sampler
    would deliver if we *did* turn fused on."""
    pm = cfg.setdefault("phase_multiview", {})
    priv = pm.setdefault("privileged_multiview", {})
    mv = priv.setdefault("mv2sv_sampler", {})
    mv["enabled"] = True
    fp = mv.setdefault("fused_pool", {})
    fp["enabled"] = True
    # Keep whatever n_fused_min / n_fused_max the config specifies; fall
    # back to 2 / 4 if missing.
    fp.setdefault("n_fused_min", 2)
    fp.setdefault("n_fused_max", 4)
    return cfg


def _build_sampler(cfg: dict):
    """Minimal sampler construction — mirrors the relevant pieces of
    ``app/vjepa_multiview/train.py::main`` without touching torch / DDP.
    """
    from phase_matched_sampler import PhaseMatchedStudySampler  # type: ignore

    pmv = cfg.get("phase_multiview", {}) or {}
    priv = pmv.get("privileged_multiview", {}) or {}
    mv2sv = priv.get("mv2sv_sampler", {}) or {}

    sampler = PhaseMatchedStudySampler(
        phase_annotations_path=pmv["phase_annotations_path"],
        num_replicas=1,
        rank=0,
        seed=int(cfg.get("meta", {}).get("seed", 345)),
        pairs_per_study=int(pmv.get("pairs_per_study", 8)),
        frames_per_clip=int(pmv.get("frames_per_clip", 16)),
        frame_step=int(pmv.get("frame_step", 1)),
        fps=int(cfg.get("data", {}).get("fps", 8)),
        tubelet_size=int(cfg.get("data", {}).get("tubelet_size", 2)),
        sampling_mode=pmv.get("sampling_mode", "uniform_phase"),
        phase_tolerance=float(pmv.get("phase_tolerance", 0.15)),
        quality_tiers=pmv.get("quality_tiers", ["high", "medium"]),
        rr_filter_mode=pmv.get("rr_filter_mode", "strict"),
        require_rr_consistent=bool(pmv.get("require_rr_consistent", True)),
        allow_frame_step_gt1=bool(pmv.get("allow_frame_step_gt1", False)),
        same_session_only=bool(pmv.get("same_session_only", False)),
        video_uri_mode=pmv.get("video_uri_mode", "mp4"),
        raw_bucket_prefix=pmv.get("raw_bucket_prefix", ""),
        mp4_bucket_prefix=pmv.get("mp4_bucket_prefix", ""),
        view_pair_policy=pmv.get("view_pair_policy"),
        delta_phase_mode=pmv.get("delta_phase_mode"),
        delta_phase_buckets=pmv.get("delta_phase_buckets"),
        delta_phase_bucket_probs=pmv.get("delta_phase_bucket_probs"),
        view_labels_path=pmv.get("view_labels_path"),
        view_label_column=pmv.get("view_label_column", "view"),
        view_confidence_column=pmv.get("view_confidence_column", "view_confidence"),
        min_view_confidence=float(pmv.get("min_view_confidence", 0.0)),
        rel_require_same_study_wrong_phase_negative=bool(
            pmv.get("rel_require_same_study_wrong_phase_negative", False)
        ),
        rel_wrong_phase_min_delta=float(pmv.get("rel_wrong_phase_min_delta", 0.25)),
        rel_wrong_phase_strategy=pmv.get("rel_wrong_phase_strategy", "same_view_then_same_family"),
        rel_allow_missing_hard_negative=bool(pmv.get("rel_allow_missing_hard_negative", False)),
        rel_hard_negative_fallback=pmv.get("rel_hard_negative_fallback", "resample_anchor"),
        rel_max_hard_neg_attempts=int(pmv.get("rel_max_hard_neg_attempts", 16)),
        multiview_objective=pmv.get("multiview_objective", "privileged_multiview"),
        mv2sv_config=mv2sv,
        total_epochs=int(pmv.get("total_epochs", 100)),
    )
    return sampler


def audit_one_epoch(sampler, epoch: int) -> dict:
    sampler.epoch = epoch
    records = sampler.build_records()

    per_row_valid: list[int] = []
    src_counter: Counter = Counter()
    tgt_counter: Counter = Counter()
    pair_counter: Counter = Counter()
    valid_by_pair: dict = defaultdict(list)  # (src_view, tgt_view) -> [valid_count, ...]
    records_without_target = 0
    records_without_fused = 0

    for r in records:
        src_v = getattr(r.clip_a, "view", None) or "UNK"
        tgt_v = r.target_view or "UNK"
        if r.target_view is None:
            records_without_target += 1
            continue
        src_counter[src_v] += 1
        tgt_counter[tgt_v] += 1
        pair_counter[(src_v, tgt_v)] += 1

        fused = list(r.fused_views or ())
        if not fused:
            records_without_fused += 1
            valid = 0
        else:
            # fused_clips[0] IS the target_clip; the fused-pool "other
            # views" count is len(fused_clips) - 1. The forward path's
            # fused_valid_mask[i, k] is True iff clip k is present, so the
            # maximum achievable mean-valid-views is n_fused_max (the
            # padded width).
            #
            # Here we count distinct-view clips actually delivered,
            # matching how ``fused_valid_mask`` gets populated in
            # ``phase_matched_pair_dataset.py`` (real clips -> True,
            # MISSING_TOKEN padding -> False).
            valid = len(fused)
        per_row_valid.append(valid)
        valid_by_pair[(src_v, tgt_v)].append(valid)

    if not per_row_valid:
        return {
            "n_records": len(records),
            "records_without_target": records_without_target,
            "records_without_fused": records_without_fused,
            "note": "no records had target_view set",
        }

    arr = np.array(per_row_valid, dtype=np.float64)
    result = {
        "epoch": epoch,
        "n_records": len(records),
        "n_rows_with_target": len(per_row_valid),
        "records_without_target": records_without_target,
        "records_without_fused": records_without_fused,
        "fused_valid_views": {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "p10": _percentile(per_row_valid, 10),
            "p25": _percentile(per_row_valid, 25),
            "p75": _percentile(per_row_valid, 75),
            "p90": _percentile(per_row_valid, 90),
            "min": int(arr.min()),
            "max": int(arr.max()),
        },
        "frac_rows_ge_2_valid": float((arr >= 2).mean()),
        "frac_rows_ge_3_valid": float((arr >= 3).mean()),
        "frac_rows_ge_4_valid": float((arr >= 4).mean()),
        "source_view_counts": dict(src_counter.most_common()),
        "target_view_counts": dict(tgt_counter.most_common()),
        "source_target_pair_counts": {
            f"{s}->{t}": c for (s, t), c in pair_counter.most_common()
        },
        "fused_valid_mean_by_pair": {
            f"{s}->{t}": float(np.mean(vs))
            for (s, t), vs in sorted(valid_by_pair.items())
        },
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    with args.config.open() as fh:
        cfg = yaml.safe_load(fh)
    cfg = _coerce_fused_enabled(cfg)

    print(f"[audit] config: {args.config}")
    print(
        f"[audit] fused_pool: enabled=True, "
        f"n_fused_min={cfg['phase_multiview']['privileged_multiview']['mv2sv_sampler']['fused_pool']['n_fused_min']}, "
        f"n_fused_max={cfg['phase_multiview']['privileged_multiview']['mv2sv_sampler']['fused_pool']['n_fused_max']}"
    )

    sampler = _build_sampler(cfg)

    per_epoch: list[dict] = []
    for e in range(args.epochs):
        print(f"[audit] building records for epoch {e} ...")
        r = audit_one_epoch(sampler, e)
        per_epoch.append(r)
        print(f"[audit] epoch {e} — summary:")
        if "fused_valid_views" in r:
            fv = r["fused_valid_views"]
            print(
                f"  n_records={r['n_records']}  with_target={r['n_rows_with_target']}  "
                f"without_target={r['records_without_target']}"
            )
            print(
                f"  fused_valid_views mean={fv['mean']:.3f}  median={fv['median']:.3f}  "
                f"p10={fv['p10']:.2f} p25={fv['p25']:.2f} p75={fv['p75']:.2f}"
            )
            print(
                f"  frac rows with >=2 valid fused views: {r['frac_rows_ge_2_valid']:.3f}"
            )
            print(
                f"  frac rows with >=3 valid fused views: {r['frac_rows_ge_3_valid']:.3f}"
            )
            print(f"  source_view top: {list(r['source_view_counts'].items())[:5]}")
            print(f"  target_view top: {list(r['target_view_counts'].items())[:5]}")
        else:
            print(f"  {r}")

    # Aggregate across epochs.
    all_means = [e["fused_valid_views"]["mean"] for e in per_epoch if "fused_valid_views" in e]
    all_ge2 = [e["frac_rows_ge_2_valid"] for e in per_epoch if "frac_rows_ge_2_valid" in e]
    aggregate = {
        "epochs_audited": len(per_epoch),
        "mean_fused_valid_views_across_epochs": float(np.mean(all_means)) if all_means else float("nan"),
        "mean_frac_ge_2_across_epochs": float(np.mean(all_ge2)) if all_ge2 else float("nan"),
    }
    print("[audit] aggregate:", aggregate)

    full_report = {"per_epoch": per_epoch, "aggregate": aggregate}
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as fh:
            json.dump(full_report, fh, indent=2, default=str)
        print(f"[audit] wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
