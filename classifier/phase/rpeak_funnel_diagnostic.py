#!/usr/bin/env python3
"""Diagnose where clips are lost between R-peak detection and phase-confident
frames for the embedding validation experiment.

For each cached clip in ``embedding_cache/``:
  - decode the processed ECG NPZ to get R-peaks in strip-column coordinates
  - map to video-frame indices via the right-edge ("now") convention
  - count how many R-peaks survive each filter, and how many frames end up
    confident under both the strict (between-R-peaks) and permissive
    (within 1 median-RR of a detected R-peak) rules

Outputs a per-clip CSV + printed funnel summary.

Also probes 5 clips where R-peaks exist on the strip but fall outside the
video window, to classify the loss (duration mismatch vs start-time offset).
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

import embedding_substrate_validation as ev
from rpeak_detectors import robust_rpeaks

HERE = Path(__file__).resolve().parent
CACHE = HERE / "embedding_cache"
PROCESSED = HERE / "lastframe" / "waveform_processed"


def confident_mask_strict(n_video_frames: int, r_peaks_video: np.ndarray) -> np.ndarray:
    mask = np.zeros(n_video_frames, dtype=bool)
    if len(r_peaks_video) < 2:
        return mask
    for s, e in zip(r_peaks_video[:-1], r_peaks_video[1:]):
        mask[int(s):int(e)] = True
    return mask


def confident_mask_permissive(
    n_video_frames: int,
    r_peaks_video: np.ndarray,
    max_extrapolation_cycles: float = 1.0,
) -> np.ndarray:
    """Frames between detected R-peaks, or within ``max_extrapolation_cycles``
    median-RR of the first/last detected R-peak."""
    mask = np.zeros(n_video_frames, dtype=bool)
    if len(r_peaks_video) < 2:
        return mask
    rr = np.diff(r_peaks_video.astype(int))
    median_rr = float(np.median(rr))
    if median_rr <= 0:
        return mask
    for s, e in zip(r_peaks_video[:-1], r_peaks_video[1:]):
        mask[int(s):int(e)] = True
    extrap = int(max_extrapolation_cycles * median_rr)
    first, last = int(r_peaks_video[0]), int(r_peaks_video[-1])
    mask[max(0, first - extrap):first] = True
    mask[last:min(n_video_frames, last + extrap)] = True
    return mask


def confident_mask_hr_extrapolated(
    n_video_frames: int,
    r_peaks_video_all: np.ndarray,
    fps_video: float,
    hr_bpm: float,
    max_dist_cycles: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Phase from *any* detected R-peak plus the metadata-HR cycle.

    A video frame is confident if it sits within ``max_dist_cycles`` of the
    nearest R-peak anchor (anchors may lie *before* the video starts — the
    scrolling-ECG strip legitimately captures R-peaks that predate the cine).
    Phase of frame `i` is ``((i - anchor) * fps_video / cycle_frames) mod 1``
    with the nearest anchor chosen per frame.

    Returns (phase_array [0,1) or NaN, confident_mask).
    """
    phase = np.full(n_video_frames, np.nan, dtype=np.float64)
    conf = np.zeros(n_video_frames, dtype=bool)
    if hr_bpm is None or hr_bpm <= 0 or len(r_peaks_video_all) == 0:
        return phase, conf
    cycle_frames = 60.0 / float(hr_bpm) * float(fps_video)
    if cycle_frames <= 0:
        return phase, conf
    idx = np.arange(n_video_frames)
    anchors = r_peaks_video_all.astype(np.float64)
    dists = np.abs(idx[:, None] - anchors[None, :])
    nearest = dists.min(axis=1)
    nearest_anchor = anchors[dists.argmin(axis=1)]
    within = nearest <= max_dist_cycles * cycle_frames
    # Phase from the anchor wraps into [0, 1)
    ph = ((idx - nearest_anchor) / cycle_frames) % 1.0
    phase = np.where(within, ph, np.nan)
    conf = within
    return phase, conf


def analyse_clip(clip_id: str, clips_meta: dict) -> dict | None:
    cache_path = CACHE / f"{clip_id}.npz"
    if not cache_path.exists():
        return None
    cached = dict(np.load(cache_path))
    fps = float(cached["fps"])
    strip_width = int(cached["strip_width"])
    sr_ecg = float(cached["sr_ecg"])
    n_video_frames = len(cached["phase"])
    # R-peaks in ECG strip coords: re-run from processed NPZ to be sure.
    proc_path = PROCESSED / f"{clip_id}.npz"
    proc = dict(np.load(proc_path))
    full_y = proc["full_y"].astype(np.float64)
    x0, x1 = int(proc["x0"]), int(proc["x1"])
    seg = np.nan_to_num(full_y[x0:x1 + 1], nan=0.0)
    hr = clips_meta.get(clip_id, {}).get("hr") or 75.0
    try:
        peaks_rel, _method, _dist = robust_rpeaks(seg, sr_ecg, hr)
        r_peaks_ecg = peaks_rel.astype(int) + x0
    except Exception:
        r_peaks_ecg = np.array([], dtype=int)
    # Map to video frame indices (right-edge = "now" convention).
    r_peaks_video_all = np.array([
        ev.ecg_col_to_video_frame(int(c), strip_width, sr_ecg,
                                  n_video_frames, fps,
                                  x0=x0, x1=x1)
        for c in r_peaks_ecg
    ], dtype=int) if len(r_peaks_ecg) else np.array([], dtype=int)
    in_window = (r_peaks_video_all >= 0) & (r_peaks_video_all < n_video_frames)
    r_peaks_video = np.unique(r_peaks_video_all[in_window])
    # Counts at each loss bucket.
    loss_before = int((r_peaks_video_all < 0).sum())
    loss_after = int((r_peaks_video_all >= n_video_frames).sum())
    strict = confident_mask_strict(n_video_frames, r_peaks_video)
    permissive = confident_mask_permissive(n_video_frames, r_peaks_video)
    hr = clips_meta.get(clip_id, {}).get("hr") or 0.0
    _phase_hr, hr_extrap = confident_mask_hr_extrapolated(
        n_video_frames, r_peaks_video_all, fps, hr,
    )
    return {
        "clip_id": clip_id,
        "n_video_frames": n_video_frames,
        "fps_video": round(fps, 2),
        "ecg_duration_s": round(strip_width / sr_ecg, 2),
        "video_duration_s": round(n_video_frames / fps, 2),
        "n_rpeaks_total": int(len(r_peaks_ecg)),
        "n_rpeaks_in_video": int(len(r_peaks_video)),
        "rpeaks_lost_before_start": loss_before,
        "rpeaks_lost_after_end": loss_after,
        "n_confident_strict": int(strict.sum()),
        "n_confident_permissive": int(permissive.sum()),
        "n_confident_hr_extrap": int(hr_extrap.sum()),
        "r_peaks_ecg": r_peaks_ecg,
        "r_peaks_video_all": r_peaks_video_all,
        "sr_ecg": sr_ecg,
        "strip_width": strip_width,
    }


def main() -> None:
    clips_meta = ev.load_clip_data()
    cached = sorted(p.stem for p in CACHE.glob("*.npz"))
    print(f"Embedded clips on disk: {len(cached)}")

    records = []
    for c in cached:
        r = analyse_clip(c, clips_meta)
        if r is not None:
            records.append(r)
    print(f"Analysed: {len(records)}")

    # Funnel
    n_total = len(records)
    n_rp_total = sum(1 for r in records if r["n_rpeaks_total"] >= 2)
    n_rp_video = sum(1 for r in records if r["n_rpeaks_in_video"] >= 2)
    n_conf_strict = sum(1 for r in records if r["n_confident_strict"] > 0)
    n_conf_perm = sum(1 for r in records if r["n_confident_permissive"] > 0)
    n_conf_hr = sum(1 for r in records if r["n_confident_hr_extrap"] > 0)

    print()
    print("Funnel:")
    print(f"  embedded clips:                       {n_total}")
    print(f"  >=2 R-peaks detected on strip:        {n_rp_total}  "
          f"(lost {n_total - n_rp_total})")
    print(f"  >=2 R-peaks inside video window:      {n_rp_video}  "
          f"(lost {n_rp_total - n_rp_video})")
    print(f"  >=1 confident frame (strict):         {n_conf_strict}")
    print(f"  >=1 confident frame (permissive):     {n_conf_perm}  "
          f"(gain over strict: {n_conf_perm - n_conf_strict})")
    print(f"  >=1 confident frame (HR-extrapolated): {n_conf_hr}  "
          f"(gain over permissive: {n_conf_hr - n_conf_perm})")

    # Study-level pair counts
    from collections import defaultdict
    def _pair_yield(key: str) -> tuple[int, int]:
        by_study: dict[str, list[str]] = defaultdict(list)
        for r in records:
            if r[key] > 0:
                by_study[r["clip_id"].split("_")[0]].append(r["clip_id"])
        multi = sum(1 for cs in by_study.values() if len(cs) >= 2)
        pairs = sum(len(cs) * (len(cs) - 1) // 2
                    for cs in by_study.values() if len(cs) >= 2)
        return multi, pairs

    print()
    print("Within-study pair yield (combinatorial):")
    for label, key in [("strict           ", "n_confident_strict"),
                       ("permissive       ", "n_confident_permissive"),
                       ("HR-extrapolated  ", "n_confident_hr_extrap")]:
        m, p = _pair_yield(key)
        print(f"  {label}: {m:3d} studies, {p:4d} pairs possible")

    # Step 2 — probe clips where R-peaks exist on strip but fall outside video
    print()
    print("Step 2: clips with R-peaks on strip but <2 inside video window:")
    candidates = [r for r in records
                  if r["n_rpeaks_total"] >= 2 and r["n_rpeaks_in_video"] < 2]
    print(f"  candidates: {len(candidates)}")
    rng = np.random.default_rng(7)
    sample = [candidates[int(i)] for i in rng.choice(
        len(candidates), size=min(5, len(candidates)), replace=False
    )] if candidates else []
    before_tot = 0; after_tot = 0
    for r in sample:
        print(f"  -- {r['clip_id']}: ecg_dur={r['ecg_duration_s']}s "
              f"video_dur={r['video_duration_s']}s  "
              f"sr_ecg={r['sr_ecg']:.0f}  n_frames={r['n_video_frames']}")
        print(f"     strip cols: {list(r['r_peaks_ecg'])[:8]}")
        print(f"     mapped frames: {list(r['r_peaks_video_all'])[:8]}")
        print(f"     lost_before_start={r['rpeaks_lost_before_start']}  "
              f"lost_after_end={r['rpeaks_lost_after_end']}")
    for r in candidates:
        before_tot += r["rpeaks_lost_before_start"]
        after_tot += r["rpeaks_lost_after_end"]
    print(f"  aggregate loss across {len(candidates)} clips: "
          f"before_start={before_tot}  after_end={after_tot}")

    # Duration ratio summary across all clips
    ratios = [r["ecg_duration_s"] / r["video_duration_s"]
              for r in records if r["video_duration_s"] > 0]
    if ratios:
        print()
        print("ECG-duration / video-duration ratio:")
        print(f"  median={np.median(ratios):.2f}  "
              f"IQR=[{np.percentile(ratios, 25):.2f}, "
              f"{np.percentile(ratios, 75):.2f}]  "
              f"max={max(ratios):.2f}")

    # Per-clip CSV
    keys = ["clip_id", "n_video_frames", "fps_video", "ecg_duration_s",
            "video_duration_s", "n_rpeaks_total", "n_rpeaks_in_video",
            "rpeaks_lost_before_start", "rpeaks_lost_after_end",
            "n_confident_strict", "n_confident_permissive",
            "n_confident_hr_extrap"]
    out = HERE / "rpeak_funnel.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in records:
            w.writerow({k: r[k] for k in keys})
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
