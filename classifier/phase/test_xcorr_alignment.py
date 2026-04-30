#!/usr/bin/env python3
"""Test cross-correlation alignment between pairs of ECG signals.

For within-study pairs: alignment should be sharp and consistent with HR.
For cross-study pairs: alignment should be noisier (negative control).
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.signal import correlate

from ecg_signal import cropped_to_signal


def study_id_from_dicom(dicom_id: str) -> str:
    """Extract study ID. Convention: 'NNNNNNNN_NNNN' where the leading
    8 digits are study, trailing 4 are clip index."""
    return dicom_id.split("_")[0]


def normalize_signal(amplitude: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Normalize a 1D ECG signal for cross-correlation.

    Computes mean/std on valid samples only, applies to valid samples only,
    leaves invalid samples at zero. Zero-valued invalid regions contribute
    nothing to the raw correlation sum.
    """
    out = np.zeros_like(amplitude, dtype=np.float64)
    if not valid_mask.any():
        return out
    valid_vals = amplitude[valid_mask].astype(np.float64)
    mean = float(np.mean(valid_vals))
    std = float(np.std(valid_vals))
    if std < 1e-6:
        return out
    out[valid_mask] = (valid_vals - mean) / std
    return out


def masked_normalized_xcorr(
    a: np.ndarray,
    b: np.ndarray,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cross-correlation normalized by valid-sample overlap at each lag.

    Zero-pads both signals (and their masks) to N = max(len_a, len_b) so no
    data from the longer signal is discarded. The per-lag overlap denominator
    from the masks naturally handles padded positions — padding contributes
    zero to numerator and denominator.
    """
    n_a = len(a)
    n_b = len(b)
    N = max(n_a, n_b)

    a_padded = np.zeros(N, dtype=np.float64)
    a_padded[:n_a] = a
    b_padded = np.zeros(N, dtype=np.float64)
    b_padded[:n_b] = b

    ma_padded = np.zeros(N, dtype=np.float64)
    ma_padded[:n_a] = mask_a.astype(np.float64)
    mb_padded = np.zeros(N, dtype=np.float64)
    mb_padded[:n_b] = mask_b.astype(np.float64)

    xc_raw = correlate(a_padded, b_padded, mode="full")
    valid_overlap = correlate(ma_padded, mb_padded, mode="full")

    xc = np.zeros_like(xc_raw)
    nz = valid_overlap > 0.5
    xc[nz] = xc_raw[nz] / valid_overlap[nz]

    lags = np.arange(-(N - 1), N)
    return xc, lags, valid_overlap


def phase_only_correlation(
    a: np.ndarray,
    b: np.ndarray,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Phase-only (amplitude-invariant) cross-correlation.

    Zeros magnitude of each frequency component of A * conj(B); only the
    relative phase of each component is preserved. Invalid regions are
    already zero in the input (``normalize_signal`` ensures this); the
    mask is still used to produce a valid-overlap array for downstream
    eligibility filtering.
    """
    n_a = len(a)
    n_b = len(b)
    N = max(n_a, n_b)

    a_padded = np.zeros(N, dtype=np.float64)
    a_padded[:n_a] = a
    b_padded = np.zeros(N, dtype=np.float64)
    b_padded[:n_b] = b

    fft_size = 2 * N  # zero-pad to avoid circular-correlation wrap
    A = np.fft.fft(a_padded, fft_size)
    B = np.fft.fft(b_padded, fft_size)

    cross_spectrum = A * np.conj(B)
    magnitude = np.abs(cross_spectrum)
    magnitude[magnitude < 1e-10] = 1.0
    phase_only_spectrum = cross_spectrum / magnitude

    xc_full = np.real(np.fft.ifft(phase_only_spectrum))
    # xc_full is circular; lag 0 at index 0, positive lags ascending,
    # negative lags at the tail. Reassemble a centered array of length 2N-1.
    xc = np.concatenate([xc_full[-(N - 1):], xc_full[:N]])
    lags = np.arange(-(N - 1), N)

    ma_padded = np.zeros(N, dtype=np.float64)
    ma_padded[:n_a] = mask_a.astype(np.float64)
    mb_padded = np.zeros(N, dtype=np.float64)
    mb_padded[:n_b] = mask_b.astype(np.float64)
    valid_overlap = correlate(ma_padded, mb_padded, mode="full")

    return xc, lags, valid_overlap


def xcorr_alignment(
    sig_a: np.ndarray,
    sig_b: np.ndarray,
    sampling_rate_hz: float,
    hr_bpm: float | None = None,
    mask_a: np.ndarray | None = None,
    mask_b: np.ndarray | None = None,
    min_overlap_frac: float = 0.8,
    lag_constraint_cycles: float | None = None,
    correlation_method: str = "masked_normalized",
) -> dict:
    """Compute cross-correlation between two 1D signals and characterize the peak.

    Uses per-lag normalization by valid-sample overlap. Only lags with at
    least ``min_overlap_frac * max_overlap`` contributing samples are
    considered when picking the peak. If ``lag_constraint_cycles`` is set
    and ``hr_bpm`` is known, peak search is restricted to lags within
    ``±lag_constraint_cycles`` cardiac periods of zero. Alternative
    ``correlation_method='phase_only'`` uses amplitude-invariant correlation.
    """
    n = min(len(sig_a), len(sig_b))
    if mask_a is None:
        mask_a = np.ones(len(sig_a), dtype=bool)
    if mask_b is None:
        mask_b = np.ones(len(sig_b), dtype=bool)

    if correlation_method == "phase_only":
        xc, lags, valid_overlap = phase_only_correlation(
            sig_a, sig_b, mask_a, mask_b
        )
    else:
        xc, lags, valid_overlap = masked_normalized_xcorr(
            sig_a, sig_b, mask_a, mask_b
        )

    # Peak selection: restrict to lags with enough effective overlap and,
    # optionally, within a physiologically meaningful lag window.
    if valid_overlap.max() <= 0:
        peak_idx = (n - 1)
    else:
        thr = min_overlap_frac * float(valid_overlap.max())
        eligible = valid_overlap >= thr
        if (lag_constraint_cycles is not None
                and hr_bpm is not None and hr_bpm > 0):
            cycle_period_s = 60.0 / hr_bpm
            max_lag_samples = int(
                lag_constraint_cycles * cycle_period_s * sampling_rate_hz
            )
            eligible = eligible & (np.abs(lags) <= max_lag_samples)
        if not eligible.any():
            # Fall back to overlap-only if the constraint emptied the set.
            eligible = valid_overlap >= thr
        masked_xc = np.where(eligible, np.abs(xc), -np.inf)
        peak_idx = int(np.argmax(masked_xc))
    peak_lag = int(lags[peak_idx])
    peak_corr = float(xc[peak_idx])
    peak_overlap = int(valid_overlap[peak_idx])

    peak_lag_seconds = peak_lag / sampling_rate_hz

    near_window = max(5, int(0.05 * sampling_rate_hz))  # ~50ms
    far_lo = max(0, peak_idx - 5 * near_window)
    far_hi = min(len(xc), peak_idx + 5 * near_window)
    baseline_region = np.concatenate([
        xc[far_lo:max(far_lo, peak_idx - near_window)],
        xc[min(far_hi, peak_idx + near_window):far_hi],
    ])
    baseline = np.mean(np.abs(baseline_region)) if len(baseline_region) > 0 else 0.0
    sharpness = abs(peak_corr) / max(baseline, 1e-6)

    result = {
        "peak_lag_samples": int(peak_lag),
        "peak_lag_seconds": float(peak_lag_seconds),
        "peak_correlation": float(peak_corr),
        "peak_sharpness": float(sharpness),
        "peak_overlap": peak_overlap,
        "cycle_period_s": None,
        "peak_lag_mod_cycle": None,
        "consistent_with_hr": None,
    }

    if hr_bpm is not None and hr_bpm > 0:
        cycle_period_s = 60.0 / hr_bpm
        wrapped = peak_lag_seconds % cycle_period_s
        if wrapped > cycle_period_s / 2:
            wrapped -= cycle_period_s
        result["cycle_period_s"] = float(cycle_period_s)
        result["peak_lag_mod_cycle"] = float(wrapped)
        result["consistent_with_hr"] = bool(abs(wrapped) < cycle_period_s / 4)

    return result


def load_signal_for_clip(
    strip_path: Path,
    processed_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Load a 1D ECG signal + valid mask for a clip.

    If ``processed_dir`` is given and contains ``{stem}.npz`` from
    ``process_waveform.py``, use that (PCHIP-smoothed, full-strip-width
    coordinate system). Otherwise fall back to on-the-fly extraction from the
    cropped strip PNG.
    """
    if processed_dir is not None:
        npz_path = processed_dir / (strip_path.stem + ".npz")
        if npz_path.exists():
            try:
                data = np.load(npz_path)
                full_y = data["full_y"].astype(np.float64)
                observed = data["observed_mask"].astype(bool)
                span = data["trace_span_mask"].astype(bool)
                # Treat the entire PCHIP-filled span as "valid" for correlation.
                valid = span & ~np.isnan(full_y)
                if valid.sum() < 100:
                    return None
                amp = np.where(valid, full_y, 0.0)
                sig = normalize_signal(amp, valid)
                return sig, valid
            except Exception as e:
                print(f"  failed to load {npz_path.name}: {e}")
                # fall through to PNG fallback
    try:
        img = np.asarray(Image.open(strip_path).convert("RGB"), dtype=np.uint8)
        amplitude, valid_mask = cropped_to_signal(img)
        if valid_mask.sum() < 100:
            return None
        sig = normalize_signal(amplitude, valid_mask)
        return sig, valid_mask
    except Exception as e:
        print(f"  failed to load {strip_path.name}: {e}")
        return None


def _dist(vals: list[float]) -> str:
    if not vals:
        return "n/a"
    return (f"median={np.median(vals):+.3f} "
            f"IQR=[{np.percentile(vals, 25):+.3f}, {np.percentile(vals, 75):+.3f}] "
            f"range=[{min(vals):+.3f}, {max(vals):+.3f}]")


def _bucket(vals: list[float], edges: list[float], labels: list[str]) -> dict[str, list[int]]:
    """Return {label: [indices]}. edges is strictly increasing."""
    out: dict[str, list[int]] = {lb: [] for lb in labels}
    for idx, v in enumerate(vals):
        if v is None:
            continue
        for i, hi in enumerate(edges):
            if v < hi:
                out[labels[i]].append(idx)
                break
        else:
            out[labels[-1]].append(idx)
    return out


def _filter_summary(results: list[dict], label: str) -> None:
    within = [r for r in results if r["kind"] == "within_study"]
    cross = [r for r in results if r["kind"] == "cross_study"]
    w_corr = [r["peak_correlation"] for r in within]
    c_corr = [r["peak_correlation"] for r in cross]
    w_checked = [r for r in within if r.get("consistent_with_hr") is not None]
    w_cons = sum(1 for r in w_checked if r["consistent_with_hr"])
    w_med = float(np.median(w_corr)) if w_corr else float("nan")
    c_med = float(np.median(c_corr)) if c_corr else float("nan")
    ratio = w_med / c_med if c_corr and abs(c_med) > 1e-6 else float("inf")
    print(f"  [{label}] within n={len(within)} cross n={len(cross)}")
    print(f"    within median corr: {w_med:+.3f}  cross median corr: {c_med:+.3f}  "
          f"ratio: {ratio:+.2f}x")
    if w_checked:
        print(f"    HR-consistent within: {w_cons}/{len(w_checked)} "
              f"({100 * w_cons / len(w_checked):.0f}%)")


def length_stratified_report(
    results: list[dict],
    out_csv: Path,
    sr_by_clip: dict[str, float],
    clips_by_study: dict[str, list[str]],
) -> None:
    if not results:
        return
    out_dir = out_csv.parent

    # Length distribution across all pairs (not per-clip; gives pair-level context).
    dur_all = [r["min_duration_s"] for r in results]
    cyc_in_shorter = [r["cycles_in_shorter"] for r in results
                      if r.get("cycles_in_shorter") is not None]
    print()
    print("=" * 70)
    print("Length distribution (over tested pairs):")
    print(f"  min_duration_s:     {_dist(dur_all)}")
    print(f"  cycles_in_shorter:  {_dist(cyc_in_shorter)}")

    # ------------------------------------------------------------------
    # Stratified by cycles_in_shorter.
    # ------------------------------------------------------------------
    cycle_edges = [2.0, 3.0, 5.0, float("inf")]
    cycle_labels = ["<2", "2-3", "3-5", "5+"]
    ratio_edges = [0.5, 0.7, 1.0001]
    ratio_labels = ["<0.5", "0.5-0.7", "0.7-1.0"]

    for kind in ("within_study", "cross_study"):
        rs = [r for r in results if r["kind"] == kind]
        if not rs:
            continue
        print()
        print(f"Stratified by cycles_in_shorter ({kind}, n={len(rs)}):")
        cycles = [r.get("cycles_in_shorter") for r in rs]
        buckets = _bucket(cycles, cycle_edges, cycle_labels)
        for lb in cycle_labels:
            idxs = buckets[lb]
            corrs = [rs[i]["peak_correlation"] for i in idxs]
            print(f"  cycles {lb:7s} n={len(idxs):3d}  corr: {_dist(corrs)}")

        print(f"Stratified by length_ratio ({kind}, n={len(rs)}):")
        ratios = [r.get("length_ratio") for r in rs]
        buckets = _bucket(ratios, ratio_edges, ratio_labels)
        for lb in ratio_labels:
            idxs = buckets[lb]
            corrs = [rs[i]["peak_correlation"] for i in idxs]
            print(f"  ratio {lb:9s} n={len(idxs):3d}  corr: {_dist(corrs)}")

    # ------------------------------------------------------------------
    # Scatter plots.
    # ------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for field, xlabel, fname in [
            ("cycles_in_shorter", "Cycles in shorter clip",
             "xcorr_vs_cycles.png"),
            ("length_ratio", "Length ratio (min/max)",
             "xcorr_vs_length_ratio.png"),
        ]:
            fig, ax = plt.subplots(figsize=(7, 5), dpi=120)
            for kind, color, marker in [
                ("within_study", "tab:blue", "o"),
                ("cross_study", "tab:orange", "x"),
            ]:
                rs = [r for r in results if r["kind"] == kind
                      and r.get(field) is not None]
                xs = [r[field] for r in rs]
                ys = [r["peak_correlation"] for r in rs]
                ax.scatter(xs, ys, s=40, c=color, marker=marker,
                           alpha=0.75, label=kind)
            ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Peak correlation")
            ax.set_title(f"Peak correlation vs {xlabel}")
            ax.legend()
            fig.tight_layout()
            fig.savefig(out_dir / fname)
            plt.close(fig)
            print(f"\nWrote scatter -> {out_dir / fname}")
    except Exception as e:
        print(f"(scatter plots skipped: {e})")

    # ------------------------------------------------------------------
    # Filtered discrimination on three subsets.
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("Filtered discrimination:")
    _filter_summary(results, "all pairs")
    subset_2 = [r for r in results if (r.get("cycles_in_shorter") or 0) >= 3]
    _filter_summary(subset_2, "cycles_in_shorter >= 3")
    subset_3 = [r for r in subset_2 if (r.get("length_ratio") or 0) >= 0.7]
    _filter_summary(subset_3, "cycles_in_shorter >= 3 AND length_ratio >= 0.7")

    # ------------------------------------------------------------------
    # Dataset-wide coverage: what fraction of all within-study pairs meet
    # the quality threshold? Uses all multi-clip studies in the strip set,
    # not just the random sample we tested. For each clip we need a
    # duration estimate -> load the strip width once.
    # ------------------------------------------------------------------
    print()
    print("Dataset-wide quality-threshold coverage (cycles_shorter >= 3 AND "
          "length_ratio >= 0.7):")
    strip_dir = Path(list({str(Path(out_csv).parent)})[0]) / "lastframe" / "waveform"
    # Actually strip_dir isn't trivially derivable; we cached widths via CSV run.
    # Rebuild clip duration table directly from clips_by_study + sampling rates
    # by reading strip widths from disk.
    clip_dur: dict[str, float] = {}
    clip_hr: dict[str, float] = {}
    # Need HR + duration. Load HR from metadata CSV via existing callable scope.
    # We already passed sr_by_clip + clips_by_study; load HR inline.
    here = Path(__file__).resolve().parent
    md_path = here / "dicom_metadata.csv"
    if md_path.exists():
        with md_path.open() as f:
            for row in csv.DictReader(f):
                try:
                    hr = float(row.get("heart_rate", "") or 0)
                    if hr > 0:
                        k = (row.get("dicom", "") or
                             row.get("dicom_id", "")).replace(".dcm", "")
                        clip_hr[k] = hr
                except ValueError:
                    pass

    sdir = here / "lastframe" / "waveform"
    for study_clips in clips_by_study.values():
        for c in study_clips:
            p = sdir / f"{c}.png"
            if p.exists() and c in sr_by_clip:
                with Image.open(p) as im:
                    w = im.size[0]
                clip_dur[c] = w / sr_by_clip[c]

    multi = {s: cs for s, cs in clips_by_study.items() if len(cs) >= 2}
    n_total = n_pass = 0
    for cs in multi.values():
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                a, b = cs[i], cs[j]
                if a not in clip_dur or b not in clip_dur:
                    continue
                hr = clip_hr.get(a) or clip_hr.get(b)
                if not hr:
                    continue
                da, db = clip_dur[a], clip_dur[b]
                mn, mx = min(da, db), max(da, db)
                cycles = mn * hr / 60.0
                ratio = mn / mx if mx > 0 else 0
                n_total += 1
                if cycles >= 3 and ratio >= 0.7:
                    n_pass += 1
    if n_total:
        print(f"  Within-study pairs available: {n_total}")
        print(f"  Pairs passing threshold:      {n_pass} "
              f"({100 * n_pass / n_total:.1f}%)")
    else:
        print("  No multi-clip pairs available to count.")


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--strip-dir", type=Path,
                    default=here / "lastframe" / "waveform")
    ap.add_argument("--calibration-csv", type=Path,
                    default=here / "calibration_results.csv")
    ap.add_argument("--metadata-csv", type=Path,
                    default=here / "dicom_metadata.csv")
    ap.add_argument("--n-within-study", type=int, default=30,
                    help="Number of within-study pairs to test")
    ap.add_argument("--n-cross-study", type=int, default=30,
                    help="Number of cross-study pairs as negative control")
    ap.add_argument("--out", type=Path, default=here / "xcorr_test_results.csv")
    ap.add_argument("--lag-constraint-cycles", type=float, default=None,
                    help="Restrict peak search to |lag| <= cycles*period")
    ap.add_argument("--correlation-method", default="masked_normalized",
                    choices=["masked_normalized", "phase_only"])
    ap.add_argument("--min-overlap-frac", type=float, default=0.8)
    ap.add_argument("--processed-dir", type=Path,
                    default=here / "lastframe" / "waveform_processed",
                    help="Dir of .npz files from process_waveform.py. "
                         "Pass --processed-dir '' to force PNG-based extraction.")
    args = ap.parse_args()
    processed_dir = args.processed_dir if str(args.processed_dir) else None

    sr_by_clip: dict[str, float] = {}
    with open(args.calibration_csv) as f:
        for row in csv.DictReader(f):
            try:
                key = row["dicom_id"].replace(".dcm", "")
                sr_by_clip[key] = float(row["sampling_rate_hz"])
            except (ValueError, KeyError):
                pass

    hr_by_clip: dict[str, float] = {}
    with open(args.metadata_csv) as f:
        for row in csv.DictReader(f):
            try:
                hr = float(row.get("heart_rate", "") or 0)
                if hr > 0:
                    key = row.get("dicom", "") or row.get("dicom_id", "")
                    hr_by_clip[key.replace(".dcm", "")] = hr
            except ValueError:
                pass

    clips_by_study: dict[str, list[str]] = defaultdict(list)
    for strip_path in sorted(args.strip_dir.glob("*.png")):
        clip_id = strip_path.stem
        if clip_id in sr_by_clip:
            clips_by_study[study_id_from_dicom(clip_id)].append(clip_id)

    multi_clip_studies = {s: c for s, c in clips_by_study.items() if len(c) >= 2}
    print(f"Studies with 2+ clips: {len(multi_clip_studies)}")
    print(f"Total clips with calibration: {sum(len(c) for c in clips_by_study.values())}")

    rng = np.random.default_rng(42)
    within_pairs: list[tuple[str, str, str]] = []
    studies = list(multi_clip_studies.keys())
    rng.shuffle(studies)
    for study in studies:
        if len(within_pairs) >= args.n_within_study:
            break
        clips = multi_clip_studies[study]
        if len(clips) >= 2:
            i, j = rng.choice(len(clips), size=2, replace=False)
            within_pairs.append((clips[i], clips[j], "within_study"))

    all_clips = [c for clips in clips_by_study.values() for c in clips]
    cross_pairs: list[tuple[str, str, str]] = []
    attempts = 0
    while len(cross_pairs) < args.n_cross_study and len(all_clips) >= 2 and attempts < 10000:
        attempts += 1
        i, j = rng.choice(len(all_clips), size=2, replace=False)
        c_i, c_j = all_clips[i], all_clips[j]
        if study_id_from_dicom(c_i) != study_id_from_dicom(c_j):
            cross_pairs.append((c_i, c_j, "cross_study"))

    print(f"Within-study pairs: {len(within_pairs)}")
    print(f"Cross-study pairs:  {len(cross_pairs)}")

    results: list[dict] = []
    for clip_a, clip_b, kind in within_pairs + cross_pairs:
        path_a = args.strip_dir / f"{clip_a}.png"
        path_b = args.strip_dir / f"{clip_b}.png"
        if not (path_a.exists() and path_b.exists()):
            continue

        load_a = load_signal_for_clip(path_a, processed_dir=processed_dir)
        load_b = load_signal_for_clip(path_b, processed_dir=processed_dir)
        if load_a is None or load_b is None:
            continue
        sig_a, mask_a = load_a
        sig_b, mask_b = load_b

        hr = hr_by_clip.get(clip_a)
        sr = sr_by_clip.get(clip_a, 213.0)

        r = xcorr_alignment(
            sig_a, sig_b, sr, hr,
            mask_a=mask_a, mask_b=mask_b,
            min_overlap_frac=args.min_overlap_frac,
            lag_constraint_cycles=args.lag_constraint_cycles,
            correlation_method=args.correlation_method,
        )
        r["clip_a"] = clip_a
        r["clip_b"] = clip_b
        r["kind"] = kind
        r["hr_bpm"] = hr
        r["sampling_rate_hz"] = sr

        # Length-related fields: clip duration, length ratio, cycle counts.
        len_a_s = len(sig_a) / sr
        len_b_s = len(sig_b) / sr
        min_dur = min(len_a_s, len_b_s)
        max_dur = max(len_a_s, len_b_s)
        r["len_a_seconds"] = float(len_a_s)
        r["len_b_seconds"] = float(len_b_s)
        r["min_duration_s"] = float(min_dur)
        r["length_ratio"] = float(min_dur / max_dur) if max_dur > 0 else 0.0
        if hr and hr > 0:
            r["cycles_in_shorter"] = float(min_dur * hr / 60.0)
            r["cycles_in_longer"] = float(max_dur * hr / 60.0)
        else:
            r["cycles_in_shorter"] = None
            r["cycles_in_longer"] = None
        results.append(r)
        cons = r.get("consistent_with_hr")
        cons_str = "?" if cons is None else ("yes" if cons else "no")
        print(f"  {kind:13s} {clip_a}<->{clip_b}: "
              f"corr={r['peak_correlation']:+.3f} "
              f"sharpness={r['peak_sharpness']:.2f} "
              f"lag={r['peak_lag_seconds']:+.2f}s "
              f"consistent_with_hr={cons_str}")

    within = [r for r in results if r["kind"] == "within_study"]
    cross = [r for r in results if r["kind"] == "cross_study"]

    def stats(rs: list[dict], key: str) -> str:
        vals = [r[key] for r in rs if key in r and r[key] is not None]
        if not vals:
            return "n/a"
        return (f"median={np.median(vals):.3f} "
                f"IQR=[{np.percentile(vals, 25):.3f}, "
                f"{np.percentile(vals, 75):.3f}]")

    print()
    print("=" * 70)
    print("Within-study pairs:")
    print(f"  peak correlation:  {stats(within, 'peak_correlation')}")
    print(f"  peak sharpness:    {stats(within, 'peak_sharpness')}")
    if within and any(r.get("consistent_with_hr") is not None for r in within):
        checked = [r for r in within if r.get("consistent_with_hr") is not None]
        n_cons = sum(1 for r in checked if r["consistent_with_hr"])
        if checked:
            print(f"  consistent with HR: {n_cons}/{len(checked)} "
                  f"({100 * n_cons / len(checked):.0f}%)")

    print()
    print("Cross-study pairs (negative control):")
    print(f"  peak correlation:  {stats(cross, 'peak_correlation')}")
    print(f"  peak sharpness:    {stats(cross, 'peak_sharpness')}")

    print()
    print("Discrimination check:")
    if within and cross:
        within_med_sharpness = float(np.median([r["peak_sharpness"] for r in within]))
        cross_med_sharpness = float(np.median([r["peak_sharpness"] for r in cross]))
        print(f"  Within-study median sharpness:  {within_med_sharpness:.2f}")
        print(f"  Cross-study median sharpness:   {cross_med_sharpness:.2f}")
        ratio = within_med_sharpness / max(cross_med_sharpness, 1e-6)
        print(f"  Ratio: {ratio:.2f}x")
        if ratio > 2.0:
            print(f"  -> Cross-correlation IS discriminative "
                  f"(within > cross by {ratio:.1f}x)")
        else:
            print(f"  -> Cross-correlation is NOT clearly discriminative")

    if results:
        keys = ["clip_a", "clip_b", "kind", "hr_bpm", "sampling_rate_hz",
                "peak_lag_samples", "peak_lag_seconds", "peak_correlation",
                "peak_sharpness", "peak_overlap", "cycle_period_s",
                "peak_lag_mod_cycle", "consistent_with_hr",
                "len_a_seconds", "len_b_seconds", "min_duration_s",
                "length_ratio", "cycles_in_shorter", "cycles_in_longer"]
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in results:
                w.writerow({k: r.get(k) for k in keys})
        print(f"\nWrote {len(results)} pair results -> {args.out}")

    # ------------------------------------------------------------------
    # Length distribution + stratified analysis + filtered discrimination.
    # ------------------------------------------------------------------
    length_stratified_report(results, args.out, sr_by_clip, clips_by_study)


if __name__ == "__main__":
    main()
