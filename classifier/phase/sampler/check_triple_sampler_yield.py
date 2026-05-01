"""Production sampler-yield gate for phase_relational_jepa.

Metadata-only test: uses the real phase_annotations parquet and the
paper-ready production sampler config, but does NOT decode any video
clips. Validates that the hard-negative draw is feasible on the full
eligible population, with the view-preference strategy the paper run
uses (``same_view_then_same_family``).

Gate criteria:

    hard_neg_available_frac >= 0.95
    same_study_all_three_frac == 1.0
    phase_distance_min >= 0.25 - eps
    pair-dataframe schema includes the 16 required columns
    positive Δφ is within bucket half-width + phase_tolerance of its bucket center

If this fails, the sampler / hard-negative search needs debugging —
the local DataLoader smoke (check_triple_clip_smoke.py) cannot
substitute because the on-disk fixture is too sparse and its view
labels are mostly null.

Usage:
    python classifier/phase/sampler/check_triple_sampler_yield.py --n 1024
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from phase_matched_sampler import (  # noqa: E402
    PhaseMatchedStudySampler,
    VIEW_FAMILIES,
    circular_phase_distance,
)
from phase_matched_pair_dataset import (  # noqa: E402
    _records_to_pair_dataframe,
)


def _view_family(v):
    if v is None:
        return "other"
    u = v.upper()
    if u == "SUBCOSTAL":
        u = "Subcostal"
    return VIEW_FAMILIES.get(u if u in VIEW_FAMILIES else v, "other")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--parquet", type=Path,
        default=Path("classifier/phase/phase_annotations/phase_annotations.parquet"),
    )
    ap.add_argument(
        "--view-labels-csv", type=Path,
        default=Path("classifier/output/mimic_view_predictions.csv"),
        help=(
            "CSV with per-clip view labels. Required because the parquet "
            "has no view column. Must contain either dicom_id+view or "
            "s3_uri+view."
        ),
    )
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--frames-per-clip", type=int, default=16)
    ap.add_argument("--frame-step", type=int, default=1)
    ap.add_argument("--wrong-phase-min-delta", type=float, default=0.25)
    ap.add_argument("--phase-tolerance", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # ---- Load view labels (required — parquet has no view column) ---- #
    if not args.view_labels_csv.exists():
        print(
            f"[FAIL] view-labels CSV not found: {args.view_labels_csv}\n"
            "       Production sampler requires view labels for the "
            "same_view_then_same_family strategy. Pass --view-labels-csv "
            "explicitly or regenerate mimic_view_predictions.csv."
        )
        raise SystemExit(2)
    vdf = pd.read_csv(args.view_labels_csv)
    # Accept either dicom_id+view or s3_uri+view schemas.
    if "dicom_id" in vdf.columns and "view" in vdf.columns:
        view_labels = dict(zip(vdf.dicom_id.astype(str), vdf.view.astype(str)))
    elif "s3_uri" in vdf.columns and "view" in vdf.columns:
        # Derive dicom_id from the s3_uri stem.
        view_labels = dict(
            zip(
                vdf.s3_uri.astype(str).map(lambda u: Path(u).stem),
                vdf.view.astype(str),
            )
        )
    else:
        print(
            f"[FAIL] {args.view_labels_csv} must have (dicom_id,view) "
            "or (s3_uri,view) columns; got " + ", ".join(vdf.columns[:6])
        )
        raise SystemExit(2)
    print(f"[INFO] loaded {len(view_labels):,} view labels from {args.view_labels_csv}")
    # Distribution for sanity
    from collections import Counter as _C
    _vc = _C(view_labels.values())
    top_views = ", ".join(f"{k}={v}" for k, v in _vc.most_common(8))
    print(f"[INFO] top view labels: {top_views}")

    # ---- Paper-ready production config (NO any_same_study here) ---- #
    sampler = PhaseMatchedStudySampler(
        parquet_path=args.parquet,
        view_labels=view_labels,
        tiers=("high", "medium"),
        rr_filter_mode="strict",
        require_rr_consistent=True,
        sampling_mode="uniform_phase",
        phase_tolerance=args.phase_tolerance,
        frames_per_clip=args.frames_per_clip,
        frame_step=args.frame_step,
        pairs_per_study=1,
        seed=args.seed,
        view_pair_policy={
            "enabled": True,
            "same_view_prob": 0.35,
            "same_family_prob": 0.45,
            "cross_family_prob": 0.20,
            "require_different_dicom": True,
            "allow_same_view": True,
            "resample_attempts": 8,
        },
        delta_phase_mode="controlled_buckets",
        delta_phase_buckets=(0.0, 0.125, 0.25, 0.5),
        delta_phase_bucket_probs=(0.40, 0.30, 0.20, 0.10),
        require_same_study_wrong_phase_negative=True,
        wrong_phase_min_delta=args.wrong_phase_min_delta,
        wrong_phase_strategy="same_view_then_same_family",   # PRODUCTION
        allow_missing_hard_negative=False,
        hard_negative_fallback="resample_anchor",
        max_hard_neg_attempts=16,
    )

    print(f"[INFO] parquet rows: {len(sampler._df):,}")
    print(f"[INFO] multi-clip studies: {sampler.n_studies}")
    print(f"[INFO] target N triples: {args.n}")
    print("[INFO] wrong_phase_strategy=same_view_then_same_family (production)")

    # ---- Draw records across the full eligible study set ---- #
    rng = np.random.default_rng(args.seed)
    records = []
    n_attempts = 0
    max_attempts = args.n * 4
    while len(records) < args.n and n_attempts < max_attempts:
        n_attempts += 1
        sid = str(sampler.study_keys[int(rng.integers(0, sampler.n_studies))])
        r = sampler._draw_pair(sid, rng)
        if r is None:
            continue
        records.append(r)
    print(f"[INFO] drew {len(records)} records after {n_attempts} attempts")

    if len(records) == 0:
        print("[FAIL] sampler produced 0 records — sampler is broken.")
        raise SystemExit(2)

    # ---- Assertions ---- #
    failures: list[str] = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)
            print(f"[FAIL] {msg}")
        else:
            print(f"[pass] {msg[:80]}")

    # (a) Pair-dataframe schema (via _records_to_pair_dataframe; we don't
    # need the local URI rewrite — just verify columns exist).
    pair_df = _records_to_pair_dataframe(records, sampler._df, video_uri_mode="mp4")
    required_cols = {
        "view_0", "view_1", "view_2", "label",
        "clip_b_neg_dicom_id", "clip_b_neg_anchor_frame",
        "clip_b_neg_phase_at_anchor", "clip_b_neg_phase_error",
        "clip_b_neg_view", "target_phi_b_neg",
        "delta_phase_bucket_pos", "delta_phase_bucket_neg",
        "view_pair_class_pos", "view_pair_class_neg",
        "hard_neg_available", "hard_neg_resample_count",
    }
    missing_cols = required_cols - set(pair_df.columns)
    check(not missing_cols, f"pair_df schema complete (missing={missing_cols})")

    # (b) Study identity — anchor rows all from same study
    same_study_count = 0
    for r in records:
        sa = str(sampler._df.loc[r.clip_a.row_idx, "study_id"])
        sb = str(sampler._df.loc[r.clip_b.row_idx, "study_id"])
        sn = (
            str(sampler._df.loc[r.clip_b_neg_phase.row_idx, "study_id"])
            if r.clip_b_neg_phase is not None else None
        )
        if sa == sb == sn == r.study_id:
            same_study_count += 1
    same_study_frac = same_study_count / max(1, len(records))
    check(same_study_frac == 1.0, f"same_study_all_three_frac == 1.0 (got {same_study_frac:.4f})")

    # (c) Hard-negative availability
    n_avail = sum(1 for r in records if r.hard_neg_available)
    hn_avail_frac = n_avail / max(1, len(records))
    check(hn_avail_frac >= 0.95, f"hard_neg_available_frac >= 0.95 (got {hn_avail_frac:.3f})")

    # (d) Phase distance: Δφ_neg vs Δφ_pos ≥ wrong_phase_min_delta
    phase_dists = []
    for r in records:
        if r.clip_b_neg_phase is None:
            continue
        dp = (r.target_phi_b - r.target_phi_a) % 1.0
        dn = (r.target_phi_b_neg - r.target_phi_a) % 1.0
        phase_dists.append(circular_phase_distance(dp, dn))
    phase_dists = np.asarray(phase_dists, dtype=np.float64)
    phase_min = float(phase_dists.min()) if len(phase_dists) else 0.0
    phase_mean = float(phase_dists.mean()) if len(phase_dists) else 0.0
    EPS = 1e-6
    check(
        phase_min >= args.wrong_phase_min_delta - EPS,
        f"phase_distance_min >= {args.wrong_phase_min_delta} (got {phase_min:.4f})",
    )

    # (e) Positive satisfies requested Δφ bucket within half-width + 2*phase_tolerance.
    # The sampler randomizes the sign of δφ, so actual Δφ ∈ {+center, -center} ≡
    # {+center, 1-center} when wrapped into [0,1). Bucket centers are *magnitudes*,
    # so the right check is |Δφ|_cyclic ≤ bucket_center + drift_tolerance.
    # |Δφ|_cyclic = min(Δφ, 1-Δφ) when Δφ ∈ [0, 1).
    # Drift tolerance = half-width + 2*phase_tolerance (one tolerance per anchor snap).
    half = sampler._delta_phase_half_width
    bucket_bound = half + 2.0 * args.phase_tolerance
    bucket_ok = 0
    bucket_total = 0
    for r in records:
        if r.delta_phase_bucket_pos is None:
            continue
        bucket_total += 1
        center = sampler.delta_phase_bucket_centers[r.delta_phase_bucket_pos]
        dpos = (r.target_phi_b - r.target_phi_a) % 1.0
        # Cyclic magnitude: folds 0.75 back to 0.25, 0.875 back to 0.125, etc.
        dpos_mag = min(dpos, 1.0 - dpos)
        if abs(dpos_mag - center) <= bucket_bound:
            bucket_ok += 1
    bucket_frac = bucket_ok / max(1, bucket_total)
    check(
        bucket_frac >= 0.99,
        f"Δφ_pos bucket consistency ≥ 0.99 on |Δφ|_cyclic ≤ center+{bucket_bound:.3f} "
        f"(got {bucket_frac:.3f} over {bucket_total})",
    )

    # (f) View-preference breakdown under same_view_then_same_family
    sv = 0      # same view
    sf_only = 0  # same family but not same view
    other = 0
    for r in records:
        if r.clip_b_neg_phase is None:
            continue
        vp = r.clip_b.view
        vn = r.clip_b_neg_phase.view
        if vp is not None and vn is not None and vp == vn:
            sv += 1
        elif _view_family(vp) == _view_family(vn) and _view_family(vp) != "other":
            sf_only += 1
        else:
            other += 1
    hn_sv_frac = sv / max(1, n_avail)
    hn_sf_frac = (sv + sf_only) / max(1, n_avail)
    # With "same_view_then_same_family" strategy, the vast majority of hard
    # negatives should be same-view OR same-family (strategy cannot return
    # cross-family unless study has no same-view/same-family candidates).
    # 'other' here means "cross-family" — which the strategy should NOT
    # return at all. Allow a small slack for edge cases where the strategy's
    # `any_same_study` fallback kicks in (it shouldn't, by config).
    check(
        hn_sf_frac >= 0.99,
        f"same_view+same_family coverage ≥ 0.99 under production strategy "
        f"(got {hn_sf_frac:.3f}; cross-family slippage = {other})",
    )

    # ---- Histograms / diagnostics ---- #
    def _hist(labels):
        counts = Counter(labels)
        total = sum(counts.values())
        return ", ".join(f"{k}={v}({100*v/max(1, total):.0f}%)" for k, v in sorted(counts.items()))

    bpos_hist = _hist(
        [r.delta_phase_bucket_pos for r in records if r.delta_phase_bucket_pos is not None]
    )
    bneg_hist = _hist(
        [r.delta_phase_bucket_neg for r in records if r.delta_phase_bucket_neg is not None]
    )
    vp_pos_hist = _hist([r.view_pair_class_pos or "?" for r in records])
    vp_neg_hist = _hist(
        [r.view_pair_class_neg or "?" for r in records if r.clip_b_neg_phase is not None]
    )
    resample_counts = [r.hard_neg_resample_count for r in records]

    # ---- Final summary ---- #
    print()
    print("=" * 72)
    print("PRODUCTION SAMPLER-YIELD GATE — summary (metadata only)")
    print("strategy: same_view_then_same_family | tiers: [high,medium] | rr: strict")
    print("=" * 72)
    print(f"{'n_requested':<40} {args.n}")
    print(f"{'n_drawn':<40} {len(records)}")
    print(f"{'hard_neg_available_frac':<40} {hn_avail_frac:.3f}")
    print(f"{'same_study_all_three_frac':<40} {same_study_frac:.3f}")
    print(f"{'hard_neg_same_view_frac':<40} {hn_sv_frac:.3f}")
    print(f"{'hard_neg_same_family_frac (incl.sv)':<40} {hn_sf_frac:.3f}")
    print(f"{'hard_neg_phase_distance_min':<40} {phase_min:.4f}")
    print(f"{'hard_neg_phase_distance_mean':<40} {phase_mean:.4f}")
    print(f"{'hard_neg_resample_count_mean':<40} {np.mean(resample_counts):.2f}")
    print(f"{'hard_neg_resample_count_max':<40} {max(resample_counts) if resample_counts else 0}")
    print(f"{'Δφ_pos bucket check (≤half+tol) frac':<40} {bucket_frac:.3f}  ({bucket_ok}/{bucket_total})")
    print(f"{'delta_phase_bucket_pos_histogram':<40} {bpos_hist}")
    print(f"{'delta_phase_bucket_neg_histogram':<40} {bneg_hist}")
    print(f"{'view_pair_class_pos_histogram':<40} {vp_pos_hist}")
    print(f"{'view_pair_class_neg_histogram':<40} {vp_neg_hist}")
    print("=" * 72)

    if failures:
        print()
        print("FAILURES:")
        for m in failures:
            print(f"  - {m}")
        print()
        print("DO NOT PROCEED to train.py — debug sampler hard-negative search.")
        raise SystemExit(1)

    print()
    print("PRODUCTION SAMPLER GATE PASSED.")


if __name__ == "__main__":
    main()
