#!/usr/bin/env python3
"""R-peak-based phase alignment as a comparison substrate to xcorr.

Mirrors ``test_xcorr_alignment.py``: same 30+30 pair sample (seed 42), same
strip/calibration/HR sources, same study-membership convention. Differences:
  - detect R-peaks per clip with NeuroKit2 (fallback: Pan-Tompkins)
  - assign per-sample cardiac phase in [0, 1) between consecutive R-peaks
  - align pairs by the offset that maximizes mean cos(2*pi * phase_diff)

Writes ``rpeak_test_results.csv`` and an NPZ bundle of per-pair phase arrays
for post-hoc plotting.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.signal import butter, filtfilt

from ecg_signal import cropped_to_signal
from rpeak_detectors import robust_rpeaks


def study_id_from_dicom(dicom_id: str) -> str:
    return dicom_id.split("_")[0]


# ---------------------------------------------------------------------------
# R-peak detection
# ---------------------------------------------------------------------------

def _pan_tompkins(amplitude: np.ndarray, sr: float) -> np.ndarray:
    """Minimal Pan-Tompkins fallback. Used only if neurokit2 is unavailable."""
    nyq = sr / 2.0
    b, a = butter(2, [5.0 / nyq, 15.0 / nyq], btype="band")
    x = filtfilt(b, a, amplitude.astype(np.float64))
    dx = np.diff(x, prepend=x[0])
    sq = dx * dx
    win = max(1, int(0.150 * sr))
    kernel = np.ones(win) / win
    mwi = np.convolve(sq, kernel, mode="same")

    refrac = max(1, int(0.250 * sr))
    snap = max(1, int(0.050 * sr))
    thr = 0.3 * float(np.max(mwi))
    peaks: list[int] = []
    i = 0
    n = len(mwi)
    while i < n:
        if mwi[i] > thr:
            j = min(n, i + refrac)
            local = int(np.argmax(mwi[i:j])) + i
            lo = max(0, local - snap)
            hi = min(n, local + snap + 1)
            snap_idx = int(np.argmax(amplitude[lo:hi])) + lo
            peaks.append(snap_idx)
            i = local + refrac
        else:
            i += 1
    return np.asarray(peaks, dtype=np.int64)


def _largest_valid_segment(valid_mask: np.ndarray) -> tuple[int, int]:
    """Return (start, end) of the longest contiguous True run in valid_mask."""
    if not valid_mask.any():
        return 0, 0
    best = (0, 0)
    cur_start = None
    for i in range(len(valid_mask)):
        if valid_mask[i]:
            if cur_start is None:
                cur_start = i
        else:
            if cur_start is not None:
                if i - cur_start > best[1] - best[0]:
                    best = (cur_start, i)
                cur_start = None
    if cur_start is not None and len(valid_mask) - cur_start > best[1] - best[0]:
        best = (cur_start, len(valid_mask))
    return best


def detect_r_peaks(
    amplitude: np.ndarray,
    valid_mask: np.ndarray,
    sr: float,
    hr_metadata: float | None = None,
) -> tuple[np.ndarray, str, float]:
    """Return (peaks, method_used, ratio_distance).

    When ``hr_metadata`` is given, uses the metadata-supervised
    ``robust_rpeaks`` ensemble; otherwise falls back to neurokit2-only.
    Peaks are returned in the clip's original index frame.
    """
    start, end = _largest_valid_segment(valid_mask)
    if end - start < int(0.5 * sr):
        return np.array([], dtype=np.int64), "empty_segment", float("inf")
    seg = amplitude[start:end].astype(np.float64)
    if np.isnan(seg).any():
        seg = np.nan_to_num(seg, nan=float(np.nanmean(seg)))

    if hr_metadata and hr_metadata > 0:
        peaks, method, dist = robust_rpeaks(seg, sr, hr_metadata)
    else:
        try:
            import neurokit2 as nk
            _, info = nk.ecg_peaks(seg, sampling_rate=sr, method="neurokit")
            peaks = np.asarray(info.get("ECG_R_Peaks", []), dtype=np.int64)
        except Exception:
            peaks = _pan_tompkins(seg, sr)
        method = "neurokit_only"
        dist = float("nan")

    return peaks + start, method, dist


# ---------------------------------------------------------------------------
# Phase assignment and alignment
# ---------------------------------------------------------------------------

def assign_phase(n_samples: int, r_peaks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    phase = np.full(n_samples, np.nan, dtype=np.float64)
    confident = np.zeros(n_samples, dtype=bool)
    if len(r_peaks) < 2:
        return phase, confident
    rr_intervals = np.diff(r_peaks)
    for i in range(len(r_peaks) - 1):
        start = int(r_peaks[i])
        end = int(r_peaks[i + 1])
        rr = end - start
        if rr <= 0:
            continue
        idx = np.arange(start, end)
        phase[idx] = (idx - start) / rr
        confident[idx] = True
    median_rr = float(np.median(rr_intervals))
    if median_rr <= 0:
        return phase, confident
    if r_peaks[0] > 0:
        idx = np.arange(0, int(r_peaks[0]))
        phase[idx] = ((idx - int(r_peaks[0])) / median_rr) % 1.0
    if r_peaks[-1] < n_samples - 1:
        idx = np.arange(int(r_peaks[-1]), n_samples)
        phase[idx] = ((idx - int(r_peaks[-1])) / median_rr) % 1.0
    return phase, confident


def rpeak_alignment_score(
    phase_a: np.ndarray, conf_a: np.ndarray,
    phase_b: np.ndarray, conf_b: np.ndarray,
    sr: float, hr: float,
) -> dict:
    n = min(len(phase_a), len(phase_b))
    cycle_samples = max(1, int(60.0 / hr * sr))
    max_offset = 2 * cycle_samples
    best_score = -np.inf
    best_offset = 0
    min_overlap = int(0.5 * cycle_samples)
    for offset in range(-max_offset, max_offset + 1):
        if offset >= 0:
            lo_a = 0
            hi_a = n - offset
            lo_b = offset
            hi_b = n
        else:
            lo_a = -offset
            hi_a = n
            lo_b = 0
            hi_b = n + offset
        if hi_a - lo_a < min_overlap:
            continue
        pa = phase_a[lo_a:hi_a]
        pb = phase_b[lo_b:hi_b]
        ca = conf_a[lo_a:hi_a]
        cb = conf_b[lo_b:hi_b]
        valid = ca & cb & ~np.isnan(pa) & ~np.isnan(pb)
        if valid.sum() < min_overlap:
            continue
        delta = (pa[valid] - pb[valid]) % 1.0
        score = float(np.mean(np.cos(2 * np.pi * delta)))
        if score > best_score:
            best_score = score
            best_offset = offset

    off_s = best_offset / sr
    cycle_s = 60.0 / hr
    wrapped = off_s % cycle_s
    if wrapped > cycle_s / 2:
        wrapped -= cycle_s
    consistent = bool(abs(wrapped) < cycle_s / 4)

    return {
        "best_offset_seconds": float(off_s),
        "phase_agreement": float(best_score if np.isfinite(best_score) else float("nan")),
        "consistent_with_hr": consistent,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_signal(
    strip_path: Path,
    processed_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Load 1D amplitude + valid mask, preferring the PCHIP-smoothed NPZ when
    available."""
    if processed_dir is not None:
        npz_path = processed_dir / (strip_path.stem + ".npz")
        if npz_path.exists():
            try:
                data = np.load(npz_path)
                full_y = data["full_y"].astype(np.float64)
                span = data["trace_span_mask"].astype(bool)
                valid = span & ~np.isnan(full_y)
                if valid.sum() < 100:
                    return None
                amp = np.where(valid, full_y, 0.0)
                return amp, valid
            except Exception as e:
                print(f"  failed to load {npz_path.name}: {e}")
    try:
        img = np.asarray(Image.open(strip_path).convert("RGB"), dtype=np.uint8)
        amp, valid = cropped_to_signal(img)
        if valid.sum() < 100:
            return None
        return amp, valid
    except Exception as e:
        print(f"  failed to load {strip_path.name}: {e}")
        return None


def expected_peaks(hr: float, duration_s: float) -> float:
    return hr / 60.0 * duration_s


def detection_ok(n_peaks: int, hr: float, duration_s: float,
                 tol_frac: float = 0.25, abs_tol_beats: float = 1.5) -> bool:
    """detected peaks within max(tol_frac*expected, abs_tol_beats) AND >=2 peaks.

    Absolute tolerance floor matters when the valid segment is short: at
    2 expected beats, a 25% relative tolerance leaves no margin.
    """
    if n_peaks < 2:
        return False
    if hr is None or hr <= 0 or duration_s <= 0:
        return True
    exp = expected_peaks(hr, duration_s)
    if exp <= 0:
        return False
    tol = max(tol_frac * exp, abs_tol_beats)
    return abs(n_peaks - exp) <= tol


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--strip-dir", type=Path, default=here / "lastframe" / "waveform")
    ap.add_argument("--calibration-csv", type=Path, default=here / "calibration_results.csv")
    ap.add_argument("--metadata-csv", type=Path, default=here / "dicom_metadata.csv")
    ap.add_argument("--n-within-study", type=int, default=30)
    ap.add_argument("--n-cross-study", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=here / "rpeak_test_results.csv")
    ap.add_argument("--npz-out", type=Path, default=here / "rpeak_phase_bundle.npz")
    ap.add_argument("--processed-dir", type=Path,
                    default=here / "lastframe" / "waveform_processed",
                    help="Dir of .npz from process_waveform.py (preferred signal source).")
    args = ap.parse_args()
    processed_dir = args.processed_dir if str(args.processed_dir) else None

    sr_by: dict[str, float] = {}
    with args.calibration_csv.open() as f:
        for row in csv.DictReader(f):
            try:
                k = row["dicom_id"].replace(".dcm", "")
                sr_by[k] = float(row["sampling_rate_hz"])
            except (ValueError, KeyError):
                pass

    hr_by: dict[str, float] = {}
    with args.metadata_csv.open() as f:
        for row in csv.DictReader(f):
            try:
                hr = float(row.get("heart_rate", "") or 0)
                if hr > 0:
                    k = (row.get("dicom", "") or
                         row.get("dicom_id", "")).replace(".dcm", "")
                    hr_by[k] = hr
            except ValueError:
                pass

    clips_by_study: dict[str, list[str]] = defaultdict(list)
    for p in sorted(args.strip_dir.glob("*.png")):
        c = p.stem
        if c in sr_by:
            clips_by_study[study_id_from_dicom(c)].append(c)

    multi = {s: cs for s, cs in clips_by_study.items() if len(cs) >= 2}
    print(f"Multi-clip studies: {len(multi)}, total clips with calib: "
          f"{sum(len(cs) for cs in clips_by_study.values())}")

    rng = np.random.default_rng(args.seed)
    within_pairs: list[tuple[str, str, str]] = []
    studies = list(multi.keys())
    rng.shuffle(studies)
    for study in studies:
        if len(within_pairs) >= args.n_within_study:
            break
        clips = multi[study]
        if len(clips) >= 2:
            i, j = rng.choice(len(clips), size=2, replace=False)
            within_pairs.append((clips[i], clips[j], "within_study"))

    all_clips = [c for cs in clips_by_study.values() for c in cs]
    cross_pairs: list[tuple[str, str, str]] = []
    attempts = 0
    while len(cross_pairs) < args.n_cross_study and len(all_clips) >= 2 and attempts < 10000:
        attempts += 1
        i, j = rng.choice(len(all_clips), size=2, replace=False)
        a, b = all_clips[i], all_clips[j]
        if study_id_from_dicom(a) != study_id_from_dicom(b):
            cross_pairs.append((a, b, "cross_study"))

    print(f"Within pairs: {len(within_pairs)}, Cross pairs: {len(cross_pairs)}")

    # --------------------------------------------------------------
    # Cache per-clip detection so each clip is analysed once, not per pair.
    # --------------------------------------------------------------
    all_pair_clips = {c for pair in within_pairs + cross_pairs for c in pair[:2]}
    clip_cache: dict[str, dict] = {}
    for c in sorted(all_pair_clips):
        sig = load_signal(args.strip_dir / f"{c}.png",
                          processed_dir=processed_dir)
        if sig is None:
            clip_cache[c] = {"ok": False, "reason": "load_failed"}
            continue
        amp, valid = sig
        sr = sr_by.get(c, 213.0)
        hr = hr_by.get(c)
        r_peaks, method_used, ratio_dist = detect_r_peaks(amp, valid, sr, hr)
        seg_lo, seg_hi = _largest_valid_segment(valid)
        seg_dur_s = max(1e-6, (seg_hi - seg_lo) / sr)
        n_exp = expected_peaks(hr, seg_dur_s) if hr else float("nan")
        ok = detection_ok(len(r_peaks), hr, seg_dur_s)
        duration_s = len(amp) / sr
        phase, conf = assign_phase(len(amp), r_peaks)
        clip_cache[c] = {
            "ok": bool(ok),
            "amp": amp, "valid": valid, "sr": sr, "hr": hr,
            "duration_s": duration_s, "r_peaks": r_peaks,
            "n_exp": n_exp, "phase": phase, "conf": conf,
            "method_used": method_used, "ratio_dist": ratio_dist,
        }

    # Detection quality report.
    n_clips = sum(1 for v in clip_cache.values() if "amp" in v)
    n_within_10 = 0
    for v in clip_cache.values():
        if "amp" not in v or not v.get("hr"):
            continue
        exp = v["n_exp"]
        if exp <= 0:
            continue
        if abs(len(v["r_peaks"]) - exp) / exp <= 0.10:
            n_within_10 += 1
    n_det_ok = sum(1 for v in clip_cache.values() if v.get("ok"))
    print()
    print("=== Detection quality ===")
    print(f"  clips analysed: {n_clips}")
    print(f"  detection_ok (within 25% of expected): {n_det_ok}/{n_clips} "
          f"({100 * n_det_ok / max(1, n_clips):.0f}%)")
    print(f"  detection within 10% of expected:      {n_within_10}/{n_clips} "
          f"({100 * n_within_10 / max(1, n_clips):.0f}%)")
    from collections import Counter
    method_counts = Counter(v.get("method_used", "?") for v in clip_cache.values()
                            if "amp" in v)
    print("  method used:")
    for m, n in method_counts.most_common():
        print(f"    {m:25s}  {n}/{n_clips}  "
              f"({100 * n / max(1, n_clips):.0f}%)")

    # --------------------------------------------------------------
    # Pair alignment.
    # --------------------------------------------------------------
    results: list[dict] = []
    pair_bundle: dict[str, dict] = {}
    for clip_a, clip_b, kind in within_pairs + cross_pairs:
        va = clip_cache.get(clip_a, {})
        vb = clip_cache.get(clip_b, {})
        det_a = bool(va.get("ok"))
        det_b = bool(vb.get("ok"))
        usable = det_a and det_b

        hr = va.get("hr") or vb.get("hr") or 75.0
        sr = va.get("sr") or 213.0

        if not usable:
            results.append({
                "clip_a": clip_a, "clip_b": clip_b, "kind": kind,
                "usable": usable, "det_a": det_a, "det_b": det_b,
                "phase_agreement": float("nan"),
                "best_offset_seconds": float("nan"),
                "consistent_with_hr": None,
                "n_peaks_a": len(va.get("r_peaks", [])),
                "n_peaks_b": len(vb.get("r_peaks", [])),
                "hr_a": va.get("hr"), "hr_b": vb.get("hr"),
                "sr": sr,
                "method_a": va.get("method_used", "?"),
                "method_b": vb.get("method_used", "?"),
            })
            continue

        score = rpeak_alignment_score(
            va["phase"], va["conf"], vb["phase"], vb["conf"], sr, hr
        )
        if not np.isfinite(score["phase_agreement"]):
            # Search window emptied out; treat as unusable.
            results.append({
                "clip_a": clip_a, "clip_b": clip_b, "kind": kind,
                "usable": False, "det_a": det_a, "det_b": det_b,
                "phase_agreement": float("nan"),
                "best_offset_seconds": float("nan"),
                "consistent_with_hr": None,
                "n_peaks_a": len(va.get("r_peaks", [])),
                "n_peaks_b": len(vb.get("r_peaks", [])),
                "hr_a": va.get("hr"), "hr_b": vb.get("hr"),
                "sr": sr,
            })
            continue
        results.append({
            "clip_a": clip_a, "clip_b": clip_b, "kind": kind,
            "usable": usable, "det_a": det_a, "det_b": det_b,
            "phase_agreement": score["phase_agreement"],
            "best_offset_seconds": score["best_offset_seconds"],
            "consistent_with_hr": score["consistent_with_hr"],
            "n_peaks_a": len(va["r_peaks"]),
            "n_peaks_b": len(vb["r_peaks"]),
            "hr_a": va.get("hr"), "hr_b": vb.get("hr"),
            "sr": sr,
            "method_a": va.get("method_used", "?"),
            "method_b": vb.get("method_used", "?"),
        })
        tag = f"{clip_a}__{clip_b}"
        pair_bundle[tag] = {
            "kind": kind,
            "phase_a": va["phase"], "phase_b": vb["phase"],
            "conf_a": va["conf"], "conf_b": vb["conf"],
            "r_peaks_a": va["r_peaks"], "r_peaks_b": vb["r_peaks"],
            "amp_a": va["amp"], "amp_b": vb["amp"],
            "sr": sr, "hr_a": va.get("hr", np.nan), "hr_b": vb.get("hr", np.nan),
            "phase_agreement": score["phase_agreement"],
            "best_offset_seconds": score["best_offset_seconds"],
        }
        cons_str = "?" if score["consistent_with_hr"] is None else (
            "yes" if score["consistent_with_hr"] else "no")
        print(f"  {kind:13s} {clip_a}<->{clip_b}: "
              f"phase_agreement={score['phase_agreement']:+.3f} "
              f"offset={score['best_offset_seconds']:+.2f}s "
              f"consistent_with_hr={cons_str}  "
              f"(n_peaks {len(va['r_peaks'])}/{len(vb['r_peaks'])})")

    within = [r for r in results if r["kind"] == "within_study"]
    cross = [r for r in results if r["kind"] == "cross_study"]

    def _stats(vals):
        if not vals:
            return "n/a"
        return (f"median={np.median(vals):+.3f} "
                f"IQR=[{np.percentile(vals, 25):+.3f}, "
                f"{np.percentile(vals, 75):+.3f}] "
                f"range=[{min(vals):+.3f}, {max(vals):+.3f}]")

    def _summarize(rs, label):
        usable_rs = [r for r in rs if r["usable"]]
        full_vals = [r["phase_agreement"] if r["usable"] else 0.0 for r in rs]
        inter_vals = [r["phase_agreement"] for r in usable_rs]
        checked = [r for r in usable_rs if r["consistent_with_hr"] is not None]
        n_cons = sum(1 for r in checked if r["consistent_with_hr"])
        print(f"\n{label} (n={len(rs)}, usable={len(usable_rs)}):")
        print(f"  phase_agreement (intersection): {_stats(inter_vals)}")
        print(f"  phase_agreement (full set, fail=0): {_stats(full_vals)}")
        if checked:
            print(f"  HR-consistent: {n_cons}/{len(checked)} "
                  f"({100 * n_cons / len(checked):.0f}%)")

    print()
    print("=" * 70)
    _summarize(within, "Within-study")
    _summarize(cross, "Cross-study")

    # Comparison table
    def _med(rs, inter=True):
        if inter:
            vals = [r["phase_agreement"] for r in rs if r["usable"]]
        else:
            vals = [r["phase_agreement"] if r["usable"] else 0.0 for r in rs]
        return float(np.median(vals)) if vals else float("nan")

    w_med = _med(within, inter=True)
    c_med = _med(cross, inter=True)
    gap = w_med - c_med
    w_usable_checked = [r for r in within if r["usable"]
                        and r["consistent_with_hr"] is not None]
    w_cons = sum(1 for r in w_usable_checked if r["consistent_with_hr"])
    n_usable_w = sum(1 for r in within if r["usable"])
    n_usable_c = sum(1 for r in cross if r["usable"])

    print()
    print("=" * 70)
    print("Comparison vs xcorr baseline (intersection-only for R-peak):")
    print(f"  {'Metric':<35}  {'xcorr':>10}  {'R-peak':>10}")
    print(f"  {'Within-study median':<35}  {'+0.564':>10}  {w_med:+10.3f}")
    print(f"  {'Cross-study median':<35}  {'+0.339':>10}  {c_med:+10.3f}")
    print(f"  {'Discrimination gap':<35}  {'+0.220':>10}  {gap:+10.3f}")
    hr_str = (f"{w_cons}/{len(w_usable_checked)} "
              f"({100 * w_cons / max(1, len(w_usable_checked)):.0f}%)"
              if w_usable_checked else "n/a")
    print(f"  {'HR-consistent (within)':<35}  {'23/29 (79%)':>10}  {hr_str:>10}")
    print(f"  {'Pairs usable (within/cross)':<35}  "
          f"{'30/30':>10}  {f'{n_usable_w}/{n_usable_c}':>10}")
    print(f"  {'Full-set within median (fail=0)':<35}  "
          f"{'':>10}  {_med(within, inter=False):+10.3f}")
    print(f"  {'Full-set cross median (fail=0)':<35}  "
          f"{'':>10}  {_med(cross, inter=False):+10.3f}")

    # Write CSV.
    keys = ["clip_a", "clip_b", "kind", "usable", "det_a", "det_b",
            "phase_agreement", "best_offset_seconds", "consistent_with_hr",
            "n_peaks_a", "n_peaks_b", "hr_a", "hr_b", "sr",
            "method_a", "method_b"]
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k) for k in keys})
    print(f"\nWrote {args.out}")

    # NPZ bundle for the 3 best / 3 worst within-study pairs and 3 random cross.
    if pair_bundle:
        usable_within = [r for r in within if r["usable"]]
        usable_within_sorted = sorted(
            usable_within, key=lambda r: r["phase_agreement"]
        )
        worst3 = usable_within_sorted[:3]
        best3 = usable_within_sorted[-3:][::-1]
        cross_usable = [r for r in cross if r["usable"]]
        sel_cross = rng.choice(len(cross_usable),
                               size=min(3, len(cross_usable)),
                               replace=False) if cross_usable else []
        cross_pick = [cross_usable[int(i)] for i in sel_cross]

        subset = {}
        for r in best3 + worst3 + cross_pick:
            tag = f"{r['clip_a']}__{r['clip_b']}"
            if tag in pair_bundle:
                subset[tag] = pair_bundle[tag]

        if subset:
            np.savez_compressed(
                args.npz_out,
                pairs=np.array(list(subset.keys())),
                **{f"{tag}/{k}": v for tag, d in subset.items()
                   for k, v in d.items()
                   if isinstance(v, np.ndarray)},
                meta=np.array([
                    (tag, d["kind"], d["sr"], d["hr_a"], d["hr_b"],
                     d["phase_agreement"], d["best_offset_seconds"])
                    for tag, d in subset.items()
                ], dtype=object),
            )
            print(f"Wrote {args.npz_out} "
                  f"(3 best-within + 3 worst-within + {len(cross_pick)} cross)")


if __name__ == "__main__":
    main()
