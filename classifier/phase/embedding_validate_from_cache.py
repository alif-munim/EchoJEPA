#!/usr/bin/env python3
"""Run the embedding-substrate analysis on cached embeddings, with regime-aware
phase assignment.

For each clip we keep the precomputed embeddings but recompute phase /
confident / regime under the current HR-extrapolation settings. Then the
Δ_within / Δ_specificity distributions are reported pooled *and stratified by
regime* (strict / permissive / hr_extrap), per the precision caveat.

Also dumps a per-pair count of anchors that used pre-video vs in-video
R-peaks, so we can see how much of any signal depends on the extrapolation
rescue.

Usage:
    python embedding_validate_from_cache.py --hr-extrap-cycles 0.5
    python embedding_validate_from_cache.py --hr-extrap-cycles 0.25
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import embedding_substrate_validation as ev

HERE = Path(__file__).resolve().parent

REGIME_PRECEDENCE = {
    ev.REGIME_STRICT: 0,
    ev.REGIME_PERMISSIVE: 1,
    ev.REGIME_HR_EXTRAP: 2,
}


def pair_regime(ra: str, rb: str) -> str:
    """Regime for a pair: the *less precise* of the two anchor frames."""
    pa = REGIME_PRECEDENCE.get(ra, 99)
    pb = REGIME_PRECEDENCE.get(rb, 99)
    return ra if pa >= pb else rb


def load_cached_clips(clips_meta: dict, hr_extrap_cycles: float) -> dict[str, dict]:
    cache: dict[str, dict] = {}
    for p in sorted(ev.EMBED_CACHE.glob("*.npz")):
        try:
            d = dict(np.load(p))
        except Exception as e:
            print(f"[cache] failed to load {p.name}: {e}")
            continue
        # Rebuild phase/confident/regime under current rules.
        if p.stem not in clips_meta:
            continue
        fps = float(d["fps"])
        n = len(d["embeddings"])
        phase, conf, regime, r_peaks = ev.compute_phase_for_clip(
            p.stem, clips_meta, n, fps, hr_extrap_cycles
        )
        if not conf.any():
            continue
        d["phase"] = phase
        d["confident"] = conf
        d["regime"] = regime
        d["r_peaks_ecg"] = r_peaks
        # Pre-compute pre-video r-peaks vs in-video r-peaks for provenance.
        sig = ev.load_processed_signal(p.stem)
        if sig is None:
            continue
        strip_width = int(sig["width"])
        x0 = int(sig["x0"]); x1 = int(sig["x1"])
        sr = clips_meta[p.stem]["sr_ecg"]
        r_peaks_video_all = np.array([
            ev.ecg_col_to_video_frame(int(c), strip_width, sr, n, fps,
                                      x0=x0, x1=x1)
            for c in r_peaks
        ], dtype=int)
        in_video = ((r_peaks_video_all >= 0) & (r_peaks_video_all < n)).sum()
        pre_video = (r_peaks_video_all < 0).sum()
        d["n_rpeaks_in_video"] = int(in_video)
        d["n_rpeaks_pre_video"] = int(pre_video)
        cache[p.stem] = d
    return cache


def find_phase_matched_with_regime(
    cache_entry: dict, target_phase: float,
) -> tuple[int | None, str]:
    phase = cache_entry["phase"]
    conf = cache_entry["confident"]
    regime = cache_entry["regime"]
    valid = conf & ~np.isnan(phase)
    if not valid.any():
        return None, ""
    diffs = np.abs(((phase - target_phase) + 0.5) % 1.0 - 0.5)
    diffs = np.where(valid, diffs, np.inf)
    idx = int(np.argmin(diffs))
    if diffs[idx] >= 0.5:
        return None, ""
    r = regime[idx]
    if isinstance(r, (bytes, np.bytes_)):
        r = r.decode("ascii")
    return idx, r


def _dist(vals: list[float]) -> str:
    if not vals:
        return "n/a"
    return (f"median={np.median(vals):+.3f}  "
            f"IQR=[{np.percentile(vals, 25):+.3f}, {np.percentile(vals, 75):+.3f}]  "
            f"frac>0={(np.array(vals) > 0).mean() * 100:.0f}%  n={len(vals)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--hr-extrap-cycles", type=float, default=0.5,
                    help="Max cycles from any R-peak to trust HR-extrapolated phase")
    ap.add_argument("--n-anchors", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-csv", type=Path,
                    default=HERE / "embedding_validation_results.csv")
    ap.add_argument("--out-summary", type=Path,
                    default=HERE / "embedding_validation_summary.txt")
    ap.add_argument("--tag", default="default",
                    help="Tag appended to output filenames to separate runs")
    args = ap.parse_args()

    if args.tag != "default":
        args.out_csv = args.out_csv.with_name(
            args.out_csv.stem + f".{args.tag}" + args.out_csv.suffix
        )
        args.out_summary = args.out_summary.with_name(
            args.out_summary.stem + f".{args.tag}" + args.out_summary.suffix
        )

    clips_meta = ev.load_clip_data()
    cache = load_cached_clips(clips_meta, args.hr_extrap_cycles)
    print(f"Cached clips with confident frames @ hr_extrap={args.hr_extrap_cycles}: "
          f"{len(cache)}")

    by_study: dict[str, list[str]] = defaultdict(list)
    for c in cache:
        by_study[c.split("_")[0]].append(c)
    multi = {s: cs for s, cs in by_study.items() if len(cs) >= 2}
    within_pairs: list[tuple[str, str]] = []
    for s in sorted(multi):
        cs = multi[s]
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                within_pairs.append((cs[i], cs[j]))
    print(f"multi-clip studies: {len(multi)}  within pairs (exhaustive): {len(within_pairs)}")

    rng = np.random.default_rng(args.seed)
    rng.shuffle(within_pairs)
    if len(within_pairs) > 40:
        within_pairs = within_pairs[:40]

    usable_by_study: dict[str, list[str]] = defaultdict(list)
    for c in cache:
        usable_by_study[c.split("_")[0]].append(c)

    records = []
    for pair_idx, (a, b) in enumerate(within_pairs):
        ca = cache[a]; cb = cache[b]
        anchors_all = np.where(ca["confident"])[0]
        if len(anchors_all) < 2:
            continue
        n_sel = min(args.n_anchors, len(anchors_all))
        sel = np.linspace(0, len(anchors_all) - 1, n_sel).astype(int)
        anchor_idxs = anchors_all[sel]

        # Cross-study negative clip: different study with confident frames.
        neg_studies = [s for s in usable_by_study
                       if s != a.split("_")[0] and s != b.split("_")[0]]
        if not neg_studies:
            continue
        neg_study = rng.choice(neg_studies)
        neg_clip = rng.choice(usable_by_study[neg_study])
        cc = cache[neg_clip]
        b_confident = np.where(cb["confident"])[0]
        if len(b_confident) == 0:
            continue
        anchor_regime = ca["regime"]
        for ai in anchor_idxs:
            p = float(ca["phase"][ai])
            bi, reg_b = find_phase_matched_with_regime(cb, p)
            ci, reg_c = find_phase_matched_with_regime(cc, p)
            if bi is None or ci is None:
                continue
            ri = int(rng.choice(b_confident))
            emb_a = ca["embeddings"][ai]
            emb_b_match = cb["embeddings"][bi]
            emb_b_rand = cb["embeddings"][ri]
            emb_c_match = cc["embeddings"][ci]
            reg_a = anchor_regime[ai]
            if isinstance(reg_a, (bytes, np.bytes_)):
                reg_a = reg_a.decode("ascii")
            pair_r_within = pair_regime(reg_a, reg_b)
            pair_r_cross = pair_regime(reg_a, reg_c)
            records.append({
                "pair_idx": pair_idx,
                "anchor_clip": a, "partner_clip": b, "cross_clip": neg_clip,
                "anchor_frame": int(ai),
                "anchor_phase": round(p, 4),
                "regime_anchor": reg_a,
                "regime_within": pair_r_within,
                "regime_cross": pair_r_cross,
                "anchor_pre_video_rpeaks": int(ca["n_rpeaks_pre_video"]),
                "anchor_in_video_rpeaks": int(ca["n_rpeaks_in_video"]),
                "sim_phase_within": round(ev.cos(emb_a, emb_b_match), 4),
                "sim_random_within": round(ev.cos(emb_a, emb_b_rand), 4),
                "sim_phase_cross": round(ev.cos(emb_a, emb_c_match), 4),
            })

    print(f"anchor records: {len(records)}")
    if not records:
        return

    keys = list(records[0].keys())
    with args.out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(records)
    print(f"wrote {args.out_csv}")

    sim_phase = np.array([r["sim_phase_within"] for r in records])
    sim_rand = np.array([r["sim_random_within"] for r in records])
    sim_cross = np.array([r["sim_phase_cross"] for r in records])
    d_within = sim_phase - sim_rand
    d_spec = sim_phase - sim_cross

    lines = []
    lines.append(f"### Run tag={args.tag}  hr_extrap_cycles={args.hr_extrap_cycles}")
    lines.append(f"clips usable: {len(cache)}   "
                 f"within_study studies: {len(multi)}   "
                 f"within_pairs tested: {len({r['pair_idx'] for r in records})}   "
                 f"anchor records: {len(records)}")
    lines.append("")
    lines.append(f"sim_phase_within (pooled):  {_dist(list(sim_phase))}")
    lines.append(f"sim_random_within (pooled): {_dist(list(sim_rand))}")
    lines.append(f"sim_phase_cross (pooled):   {_dist(list(sim_cross))}")
    lines.append("")
    lines.append(f"Δ_within (pooled):        {_dist(list(d_within))}")
    lines.append(f"Δ_specificity (pooled):   {_dist(list(d_spec))}")

    try:
        from scipy.stats import wilcoxon
        w_w = wilcoxon(d_within, alternative="greater")
        w_s = wilcoxon(d_spec, alternative="greater")
        lines.append(f"Wilcoxon Δ_within > 0:       p = {w_w.pvalue:.2e}")
        lines.append(f"Wilcoxon Δ_specificity > 0:  p = {w_s.pvalue:.2e}")
    except Exception as e:
        lines.append(f"Wilcoxon failed: {e}")

    # --- Stratified by regime ---
    lines.append("")
    lines.append("Stratified by regime_within (pair regime = less precise of the two anchors):")
    reg_counts = Counter(r["regime_within"] for r in records)
    lines.append(f"  regime distribution: {dict(reg_counts)}")
    for regime_label in [ev.REGIME_STRICT, ev.REGIME_PERMISSIVE, ev.REGIME_HR_EXTRAP]:
        mask = [r["regime_within"] == regime_label for r in records]
        sub = [r for r, m in zip(records, mask) if m]
        if not sub:
            continue
        dw = np.array([r["sim_phase_within"] - r["sim_random_within"] for r in sub])
        ds = np.array([r["sim_phase_within"] - r["sim_phase_cross"] for r in sub])
        n_pairs = len({r["pair_idx"] for r in sub})
        lines.append(f"  [{regime_label}] n={len(sub)} anchors, {n_pairs} pairs")
        lines.append(f"     Δ_within:      {_dist(list(dw))}")
        lines.append(f"     Δ_specificity: {_dist(list(ds))}")

    # --- Phase buckets, within strict-only (highest-precision subset) ---
    strict_records = [r for r in records if r["regime_within"] == ev.REGIME_STRICT]
    if strict_records:
        lines.append("")
        lines.append("Δ_within by anchor phase bucket (strict-only, highest precision):")
        phases = np.array([r["anchor_phase"] for r in strict_records])
        dws = np.array([r["sim_phase_within"] - r["sim_random_within"]
                        for r in strict_records])
        for lo, hi in zip([0, 0.2, 0.4, 0.6, 0.8], [0.2, 0.4, 0.6, 0.8, 1.0]):
            m = (phases >= lo) & (phases < hi)
            if m.any():
                lines.append(f"  phase [{lo:.1f},{hi:.1f}):  "
                             f"n={m.sum()}  median Δ={np.median(dws[m]):+.3f}  "
                             f"frac>0={(dws[m] > 0).mean() * 100:.0f}%")

    # --- Anchor provenance: pre-video vs in-video R-peak count per pair ---
    lines.append("")
    lines.append("Anchor clip R-peak provenance (counts are per anchor clip):")
    in_vid = [r["anchor_in_video_rpeaks"] for r in records]
    pre_vid = [r["anchor_pre_video_rpeaks"] for r in records]
    lines.append(f"  in-video R-peaks per anchor: "
                 f"median={int(np.median(in_vid))}  "
                 f"IQR=[{int(np.percentile(in_vid, 25))}, "
                 f"{int(np.percentile(in_vid, 75))}]  "
                 f"max={max(in_vid)}")
    lines.append(f"  pre-video R-peaks per anchor: "
                 f"median={int(np.median(pre_vid))}  "
                 f"IQR=[{int(np.percentile(pre_vid, 25))}, "
                 f"{int(np.percentile(pre_vid, 75))}]  "
                 f"max={max(pre_vid)}")

    # --- Decision verdict ---
    med_dw = float(np.median(d_within))
    med_ds = float(np.median(d_spec))
    frac_dw = float((d_within > 0).mean())
    frac_ds = float((d_spec > 0).mean())
    if med_dw >= 0.10 and med_ds >= 0.05 and frac_dw > 0.85 and frac_ds > 0.85:
        verdict = "PASS — strongly positive both deltas."
    elif med_dw >= 0.05 and frac_dw > 0.70 and (med_ds < 0.03 or frac_ds < 0.70):
        verdict = ("PARTIAL — Δ_within positive, Δ_specificity weak. "
                   "Encoder picks up cardiac phase but not patient-specific structure.")
    elif med_dw < 0:
        verdict = "BROKEN — Δ_within negative; check mapping."
    elif med_dw <= 0.03 or frac_dw < 0.70:
        verdict = "WEAK — Δ_within too small. Encoder not phase-sensitive, or substrate not predictive."
    else:
        verdict = "INTERMEDIATE — above-zero but below strong thresholds."

    lines.append("")
    lines.append(f"Decision (pooled): {verdict}")

    args.out_summary.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
