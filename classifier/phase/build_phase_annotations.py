#!/usr/bin/env python3
"""Build the per-clip phase-annotation table for multi-view phase-aligned
pretraining, over every DICOM in ``s3://echodata25/mimic-raw-staging/``.

Output: one Parquet shard per worker, concatenated into a single
``phase_annotations.parquet`` at the end. Each row is one clip with:

  dicom_id, subject_id, study_id, dicom_filepath, s3_uri,
  manufacturer, model, sop_class_uid,
  n_video_frames, fps_video, video_duration_s,
  sr_ecg, x0, x1, trace_span_dur_s, duration_ratio,
  coverage_frac,  # from process_waveform.py
  hr_metadata, hr_detected, detection_method, rpeak_ratio_dist,
  n_rpeaks_total, n_rpeaks_in_video, r_peaks_video_json, rr_median_frames,
  per_frame_phase_json, confident_mask_json,   # arrays as JSON strings
  quality_tier,   # 'high' | 'medium' | 'low' | 'reject'
  reject_reason,  # when applicable

The pipeline per DICOM is the one documented in ``README.md`` step 1-6:

  1. Read DICOM (header + pixel data) from S3.
  2. Skip singleton / missing-FrameTime DICOMs.
  3. Extract last frame; run ``find_waveform_band`` + green-family trace
     isolation to produce the white-background strip image.
  4. Run ``process_waveform.extract_ecg_signal`` on that strip to get the
     1D signal, coverage_frac, and (x0, x1) trace span.
  5. Calibrate sampling rate via scanner-modal default.
  6. Detect R-peaks with the HR-supervised ensemble.
  7. Map R-peaks to video-frame indices via the ``[x0, x1]`` linear mapping.
  8. Assign per-frame phase under the three-regime rule.

Does not write any images to S3 — only the Parquet shard. DICOM staging
lives in ``--tmpdir`` (default: ``/opt/dlami/nvme/tmp/phase_build/<pid>``)
and is cleaned per clip.

Usage (single-process, local; for small ranges):
    python build_phase_annotations.py --shard-id 0 --n-shards 100 \
        --out-dir /opt/dlami/nvme/phase_anno

SLURM array usage (each task picks up one shard via SLURM_ARRAY_TASK_ID):
    sbatch --array=0-255 scripts/phase_annotations.sbatch
"""

from __future__ import annotations

import argparse
import csv
import gc
import io
import json
import os
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from PIL import Image

# --- local helpers from this directory ------------------------------------
from crop_waveform_frame import find_waveform_band
from process_waveform import extract_ecg_signal as _extract_ecg_signal_file
from process_waveform import save_npz as _pwf_save_npz  # noqa: F401  (kept for parity)
from rpeak_detectors import robust_rpeaks
from ecg_calibration import calibrate_sampling_rate, load_scanner_defaults


HERE = Path(__file__).resolve().parent

DEFAULT_RECORD_LIST = Path(
    "/mnt/custom-file-systems/efs/fs-0049217cdf69186d7_fsap-0fa7145b64eaa046b"
    "/vjepa2/uhn_echo/nature_medicine/data_exploration/mimic/mimic-iv-echo"
    "/echo-record-list.csv"
)
S3_BUCKET = "s3://echodata25/mimic-raw-staging"
SCANNER_DEFAULTS_PATH = HERE / "scanner_defaults.json"

# Quality-tier thresholds. Shared with the README step 3.
HIGH_MIN_IN_VIDEO_RPEAKS = 3
HIGH_MIN_COVERAGE = 0.90
HIGH_MAX_RPEAK_RATIO_DIST = 0.10
MEDIUM_MIN_IN_VIDEO_RPEAKS = 2
MEDIUM_MIN_COVERAGE = 0.80
MEDIUM_MAX_RPEAK_RATIO_DIST = 0.25


# ---------------------------------------------------------------------------
# In-memory adapter of process_waveform.extract_ecg_signal
# ---------------------------------------------------------------------------

def extract_ecg_signal_from_rgb(
    img: np.ndarray,
    lum_threshold: int = 200,
    dilate_iters: int = 2,
    median_size: int = 3,
) -> dict:
    """In-memory twin of ``process_waveform.extract_ecg_signal`` (which reads
    from disk). Same algorithm — thresholds on luminance, keeps the largest
    connected component, per-column centroid, PCHIP fill, median filter.
    """
    from scipy.interpolate import PchipInterpolator
    from scipy.ndimage import binary_dilation, label, median_filter
    H, W, _ = img.shape
    empty = {
        "xs": np.array([], dtype=int),
        "ys": np.array([], dtype=float),
        "full_y": np.full(W, np.nan, dtype=np.float32),
        "observed_mask": np.zeros(W, dtype=bool),
        "trace_span_mask": np.zeros(W, dtype=bool),
        "interpolated_mask": np.zeros(W, dtype=bool),
        "width": int(W), "height": int(H),
        "x0": -1, "x1": -1,
        "n_observed": 0, "coverage_frac": 0.0,
    }
    mask = img.mean(axis=2) < lum_threshold
    if not mask.any():
        return empty
    labeled, _ = label(binary_dilation(mask, iterations=dilate_iters))
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    if sizes.max() == 0:
        return empty
    keep = labeled == sizes.argmax()
    mask = mask & keep
    raw_ys = np.full(W, np.nan, dtype=np.float64)
    for x in range(W):
        col = np.where(mask[:, x])[0]
        if len(col):
            raw_ys[x] = col.mean()
    valid = ~np.isnan(raw_ys)
    if not valid.any():
        return empty
    x0, x1 = int(np.where(valid)[0][0]), int(np.where(valid)[0][-1])
    xs = np.arange(x0, x1 + 1)
    yt_raw = raw_ys[xs]
    observed_in_span = ~np.isnan(yt_raw)
    if observed_in_span.sum() < 2:
        yt = -np.where(observed_in_span, yt_raw, 0.0)
    else:
        yt = PchipInterpolator(xs[observed_in_span], yt_raw[observed_in_span])(xs)
        yt = -median_filter(yt, size=median_size)
    full_y = np.full(W, np.nan, dtype=np.float32)
    full_y[xs] = yt
    observed_mask = np.zeros(W, dtype=bool)
    observed_mask[xs] = observed_in_span
    trace_span_mask = np.zeros(W, dtype=bool)
    trace_span_mask[xs] = True
    n_obs = int(observed_in_span.sum())
    span = x1 - x0 + 1
    coverage = float(n_obs / span) if span > 0 else 0.0
    return {
        "xs": xs,
        "ys": yt.astype(np.float32),
        "full_y": full_y,
        "observed_mask": observed_mask,
        "trace_span_mask": trace_span_mask,
        "interpolated_mask": trace_span_mask & ~observed_mask,
        "width": int(W), "height": int(H),
        "x0": x0, "x1": x1,
        "n_observed": n_obs, "coverage_frac": coverage,
    }


# ---------------------------------------------------------------------------
# Phase assignment (three-regime, same logic as embedding_substrate_validation)
# ---------------------------------------------------------------------------

REGIME_STRICT = "strict"
REGIME_PERMISSIVE = "permissive"
REGIME_HR_EXTRAP = "hr_extrap"


def ecg_col_to_video_frame(col: int, x0: int, x1: int, n_frames: int) -> int:
    if x1 <= x0:
        return -1
    return int(round((col - x0) / (x1 - x0) * (n_frames - 1)))


def assign_phase(
    n_video_frames: int,
    fps_video: float,
    r_peaks_ecg: np.ndarray,
    x0: int, x1: int,
    hr_metadata: float | None = None,
    hr_extrap_max_cycles: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (phase, confident, regime_bytes, r_peaks_video_all)."""
    r_peaks_video_all = np.array([
        ecg_col_to_video_frame(int(c), x0, x1, n_video_frames)
        for c in r_peaks_ecg
    ], dtype=int)
    in_window = (r_peaks_video_all >= 0) & (r_peaks_video_all < n_video_frames)
    r_peaks_video = np.unique(r_peaks_video_all[in_window])

    phase = np.full(n_video_frames, np.nan, dtype=np.float32)
    confident = np.zeros(n_video_frames, dtype=bool)
    regime = np.full(n_video_frames, "", dtype=object)

    # regime 1: strict
    if len(r_peaks_video) >= 2:
        for i in range(len(r_peaks_video) - 1):
            s, e = int(r_peaks_video[i]), int(r_peaks_video[i + 1])
            if e <= s:
                continue
            idx = np.arange(s, e)
            phase[idx] = (idx - s) / (e - s)
            confident[idx] = True
            regime[idx] = REGIME_STRICT

    # regime 2: permissive extrapolation ±1 median-RR
    if len(r_peaks_video) >= 2:
        rr = np.diff(r_peaks_video.astype(int))
        median_rr = float(np.median(rr))
        if median_rr > 0:
            extrap = int(1.0 * median_rr)
            first, last = int(r_peaks_video[0]), int(r_peaks_video[-1])
            for i in range(max(0, first - extrap), first):
                if not confident[i]:
                    phase[i] = ((i - first) / median_rr) % 1.0
                    confident[i] = True
                    regime[i] = REGIME_PERMISSIVE
            for i in range(last, min(n_video_frames, last + extrap)):
                if not confident[i]:
                    phase[i] = ((i - last) / median_rr) % 1.0
                    confident[i] = True
                    regime[i] = REGIME_PERMISSIVE

    # regime 3: HR-extrapolated from any R-peak
    if (hr_metadata and hr_metadata > 0 and len(r_peaks_video_all) > 0):
        cycle_frames = 60.0 / float(hr_metadata) * float(fps_video)
        if cycle_frames > 0:
            max_dist = hr_extrap_max_cycles * cycle_frames
            idx = np.arange(n_video_frames)
            anchors = r_peaks_video_all.astype(np.float64)
            dists = np.abs(idx[:, None] - anchors[None, :])
            nearest = dists.min(axis=1)
            nearest_anchor = anchors[dists.argmin(axis=1)]
            can_fill = (~confident) & (nearest <= max_dist)
            if can_fill.any():
                ph = ((idx - nearest_anchor) / cycle_frames) % 1.0
                phase[can_fill] = ph[can_fill].astype(np.float32)
                confident[can_fill] = True
                regime[can_fill] = REGIME_HR_EXTRAP

    regime_bytes = np.array([s.encode("ascii") for s in regime], dtype="S12")
    return phase, confident, regime_bytes, r_peaks_video_all


# ---------------------------------------------------------------------------
# Last-frame + waveform isolation (in-memory, adapted from extract_lastframe)
# ---------------------------------------------------------------------------

def decode_last_frame_rgb(ds) -> np.ndarray:
    """Return the last frame of a DICOM as an HxWx3 uint8 RGB array.

    Uses pydicom's built-in pixel decoders (JPEG baseline / YBR_FULL_422
    auto-converts; PALETTE needs apply_color_lut).
    """
    pa = ds.pixel_array
    pi = str(getattr(ds, "PhotometricInterpretation", ""))
    if "PALETTE" in pi:
        from pydicom.pixels.processing import apply_color_lut
        pa = apply_color_lut(pa, ds)
        if pa.dtype == np.uint16:
            pa = (pa / 256).astype(np.uint8)
    n_frames = int(getattr(ds, "NumberOfFrames", 1))
    if n_frames > 1:
        last = pa[-1]
    elif pa.ndim == 4 and pa.shape[0] == 1:
        last = pa[0]
    elif pa.ndim == 3 and pa.shape[0] == 1 and pa.shape[-1] not in (3, 4):
        last = pa[0]
    else:
        last = pa
    last = np.ascontiguousarray(last, dtype=np.uint8)
    if last.ndim == 2:
        last = np.stack([last, last, last], axis=-1)
    return last


def waveform_on_white(frame_rgb: np.ndarray, sat_thresh: int = 60) -> np.ndarray:
    """Isolate trace-on-white using the strip band + green-family mask.

    Same algorithm as extract_lastframe.extract_waveform_on_white.
    """
    y0, y1 = find_waveform_band(frame_rgb)
    crop = frame_rgb[y0:y1].astype(np.int16)
    R = crop[..., 0]
    G = crop[..., 1]
    B = crop[..., 2]
    sat = crop.max(axis=-1) - crop.min(axis=-1)
    trace_mask = (sat > sat_thresh) & (G > R) & (G > B)
    out = np.full_like(crop, 255, dtype=np.uint8)
    out[trace_mask] = crop[trace_mask].astype(np.uint8)
    return out


# ---------------------------------------------------------------------------
# Per-DICOM worker
# ---------------------------------------------------------------------------

@dataclass
class Row:
    dicom_id: str
    subject_id: str
    study_id: str
    dicom_filepath: str
    s3_uri: str
    manufacturer: str = ""
    model: str = ""
    sop_class_uid: str = ""
    n_video_frames: int = 0
    fps_video: float = float("nan")
    video_duration_s: float = float("nan")
    sr_ecg: float = float("nan")
    sr_source: str = ""
    sr_confidence: str = ""
    x0: int = -1
    x1: int = -1
    trace_span_dur_s: float = float("nan")
    duration_ratio: float = float("nan")
    coverage_frac: float = float("nan")
    hr_metadata: float = float("nan")
    hr_detected: float = float("nan")
    detection_method: str = ""
    rpeak_ratio_dist: float = float("nan")
    n_rpeaks_total: int = 0
    n_rpeaks_in_video: int = 0
    r_peaks_video_json: str = ""
    rr_median_frames: float = float("nan")
    per_frame_phase_json: str = ""
    confident_mask_json: str = ""
    regime_summary: str = ""   # e.g. "strict:48,permissive:12,hr_extrap:0"
    # Raw 1D ECG signal (full width, NaN outside [x0, x1]).
    # Stored as a uint8 base85-encoded float16 byte string for compactness:
    # ~1000 samples × 2 bytes * 1.25 base85 overhead ≈ 2.5 KB per clip.
    # Decode with build_phase_annotations.decode_full_y(row).
    full_y_b85: str = ""
    quality_tier: str = "reject"
    reject_reason: str = ""
    elapsed_s: float = float("nan")


def _as_json_short(arr: np.ndarray, decimals: int = 4) -> str:
    if arr.dtype == bool:
        # Run-length encode for space efficiency.
        flat = arr.astype(np.uint8).tolist()
        return json.dumps(flat)
    if np.issubdtype(arr.dtype, np.integer):
        return json.dumps(arr.tolist())
    rounded = np.round(arr.astype(np.float64), decimals)
    return json.dumps([None if np.isnan(v) else float(v) for v in rounded])


def _encode_full_y(full_y: np.ndarray) -> str:
    """Pack a float array as a base85 byte string (float16 for compactness).

    NaN preserved. Decode with ``decode_full_y``.
    """
    import base64
    a = full_y.astype(np.float16)
    return base64.b85encode(a.tobytes()).decode("ascii")


def decode_full_y(row_or_b85: "str | dict | pd.Series") -> np.ndarray:
    """Inverse of ``_encode_full_y``. Accepts either the b85 string or a
    DataFrame row dict with ``full_y_b85`` key."""
    import base64
    if isinstance(row_or_b85, str):
        s = row_or_b85
    else:
        s = row_or_b85.get("full_y_b85", "") if hasattr(row_or_b85, "get") \
            else row_or_b85["full_y_b85"]
    if not s:
        return np.array([], dtype=np.float32)
    raw = base64.b85decode(s.encode("ascii"))
    return np.frombuffer(raw, dtype=np.float16).astype(np.float32)


def classify_tier(row: Row) -> tuple[str, str]:
    """Return (quality_tier, reject_reason)."""
    if row.n_video_frames < 2:
        return "reject", "single_frame_or_no_decode"
    if not np.isfinite(row.fps_video) or row.fps_video <= 0:
        return "reject", "no_fps"
    if row.x0 < 0 or row.x1 <= row.x0:
        return "reject", "no_trace_span"
    if not np.isfinite(row.sr_ecg) or row.sr_ecg <= 0:
        return "reject", "no_calibration"
    if row.n_rpeaks_in_video < 1:
        return "reject", "no_in_video_rpeaks"

    cov = row.coverage_frac if np.isfinite(row.coverage_frac) else 0.0
    rd = row.rpeak_ratio_dist if np.isfinite(row.rpeak_ratio_dist) else 1.0

    if (row.n_rpeaks_in_video >= HIGH_MIN_IN_VIDEO_RPEAKS
            and cov >= HIGH_MIN_COVERAGE
            and rd <= HIGH_MAX_RPEAK_RATIO_DIST):
        return "high", ""
    if (row.n_rpeaks_in_video >= MEDIUM_MIN_IN_VIDEO_RPEAKS
            and cov >= MEDIUM_MIN_COVERAGE
            and rd <= MEDIUM_MAX_RPEAK_RATIO_DIST):
        return "medium", ""
    return "low", ""


def process_one(
    record: dict,
    tmpdir: Path,
    scanner_defaults: dict,
) -> Row:
    """Run the full pipeline on one DICOM. Always returns a Row (tier=reject
    on failure). Never raises."""
    t0 = time.time()
    rel = record["dicom_filepath"]
    fname = Path(rel).name
    row = Row(
        dicom_id=fname.replace(".dcm", ""),
        subject_id=record.get("subject_id", ""),
        study_id=record.get("study_id", ""),
        dicom_filepath=rel,
        s3_uri=f"{S3_BUCKET}/{rel}",
    )

    # 1. download
    local = tmpdir / fname
    try:
        r = subprocess.run(
            ["aws", "s3", "cp", row.s3_uri, str(local), "--quiet"],
            capture_output=True, text=True, timeout=120,
        )
        if r.returncode != 0 or not local.exists() or local.stat().st_size == 0:
            row.reject_reason = f"s3_cp_fail:{(r.stderr or '').strip()[:80]}"
            return _finalize(row, t0)
    except Exception as e:
        row.reject_reason = f"s3_cp_exc:{str(e)[:80]}"
        return _finalize(row, t0)

    # 2. read DICOM header + metadata
    try:
        ds = pydicom.dcmread(str(local))
    except Exception as e:
        row.reject_reason = f"dcmread:{str(e)[:80]}"
        _cleanup(local)
        return _finalize(row, t0)

    row.manufacturer = str(getattr(ds, "Manufacturer", "") or "")
    row.model = str(getattr(ds, "ManufacturerModelName", "") or "")
    row.sop_class_uid = str(getattr(ds, "SOPClassUID", "") or "")
    n_frames = int(getattr(ds, "NumberOfFrames", 1))
    row.n_video_frames = n_frames
    ft = getattr(ds, "FrameTime", None)
    if ft is None or n_frames < 2:
        row.reject_reason = "single_or_no_frame_time"
        _cleanup(local)
        return _finalize(row, t0)
    try:
        row.fps_video = 1000.0 / float(ft)
    except Exception:
        row.reject_reason = "bad_frame_time"
        _cleanup(local)
        return _finalize(row, t0)
    row.video_duration_s = n_frames / row.fps_video
    try:
        row.hr_metadata = float(getattr(ds, "HeartRate", 0) or 0)
    except Exception:
        row.hr_metadata = float("nan")

    # 3. decode last frame
    try:
        last_rgb = decode_last_frame_rgb(ds)
    except Exception as e:
        row.reject_reason = f"decode:{str(e)[:80]}"
        _cleanup(local)
        return _finalize(row, t0)
    if last_rgb.ndim != 3 or last_rgb.shape[-1] != 3:
        row.reject_reason = "not_rgb_last_frame"
        _cleanup(local)
        return _finalize(row, t0)

    # 4. waveform isolate + 1D signal
    try:
        strip_rgb = waveform_on_white(last_rgb)
    except Exception as e:
        row.reject_reason = f"waveform_band:{str(e)[:80]}"
        _cleanup(local)
        return _finalize(row, t0)
    sig = extract_ecg_signal_from_rgb(strip_rgb)
    if sig["x0"] < 0 or sig["x1"] <= sig["x0"]:
        row.reject_reason = "no_trace_span"
        row.coverage_frac = 0.0
        _cleanup(local)
        return _finalize(row, t0)
    row.x0 = int(sig["x0"])
    row.x1 = int(sig["x1"])
    row.coverage_frac = float(sig["coverage_frac"])
    # Persist the raw 1D signal so downstream can re-run phase assignment,
    # alternative detectors, or xcorr without re-decoding the DICOM.
    row.full_y_b85 = _encode_full_y(sig["full_y"])

    # 5. calibration
    calib = calibrate_sampling_rate(local, scanner_defaults=scanner_defaults)
    if calib["sampling_rate_hz"] is None or calib["sampling_rate_hz"] <= 0:
        row.sr_source = calib.get("source", "fallback")
        row.sr_confidence = calib.get("confidence", "low")
        row.reject_reason = "no_sr"
        _cleanup(local)
        return _finalize(row, t0)
    row.sr_ecg = float(calib["sampling_rate_hz"])
    row.sr_source = calib["source"]
    row.sr_confidence = calib["confidence"]

    row.trace_span_dur_s = (row.x1 - row.x0 + 1) / row.sr_ecg
    if row.video_duration_s > 0:
        row.duration_ratio = row.trace_span_dur_s / row.video_duration_s

    # 6. R-peak detection (on the strict trace span only)
    full_y = sig["full_y"]
    seg = np.nan_to_num(full_y[row.x0:row.x1 + 1].astype(np.float64), nan=0.0)
    hr_for_rpeaks = row.hr_metadata if row.hr_metadata > 0 else 75.0
    try:
        peaks_rel, method, dist = robust_rpeaks(seg, row.sr_ecg, hr_for_rpeaks)
        row.detection_method = method
        row.rpeak_ratio_dist = float(dist) if np.isfinite(dist) else float("nan")
    except Exception as e:
        row.reject_reason = f"rpeak:{str(e)[:80]}"
        _cleanup(local)
        return _finalize(row, t0)
    r_peaks_ecg = peaks_rel.astype(int) + row.x0
    row.n_rpeaks_total = int(len(r_peaks_ecg))

    if row.n_rpeaks_total >= 2:
        rr_ecg = np.diff(r_peaks_ecg)
        row.hr_detected = 60.0 * row.sr_ecg / float(np.median(rr_ecg))

    # 7/8. phase assignment
    phase, confident, regime_bytes, rpv_all = assign_phase(
        n_video_frames=n_frames,
        fps_video=row.fps_video,
        r_peaks_ecg=r_peaks_ecg,
        x0=row.x0, x1=row.x1,
        hr_metadata=row.hr_metadata if row.hr_metadata > 0 else None,
        hr_extrap_max_cycles=0.5,
    )
    in_video_rpeaks = rpv_all[(rpv_all >= 0) & (rpv_all < n_frames)]
    row.n_rpeaks_in_video = int(len(in_video_rpeaks))
    row.r_peaks_video_json = _as_json_short(in_video_rpeaks)
    if len(in_video_rpeaks) >= 2:
        row.rr_median_frames = float(np.median(np.diff(in_video_rpeaks)))
    row.per_frame_phase_json = _as_json_short(phase, decimals=4)
    row.confident_mask_json = _as_json_short(confident)
    regimes = [s.decode("ascii") for s in regime_bytes]
    from collections import Counter
    c = Counter(regimes)
    row.regime_summary = ",".join(f"{k}:{v}" for k, v in c.most_common()
                                  if k)

    tier, reason = classify_tier(row)
    row.quality_tier = tier
    row.reject_reason = reason
    _cleanup(local)
    return _finalize(row, t0)


def _cleanup(p: Path) -> None:
    try:
        p.unlink(missing_ok=True)
    except Exception:
        pass


def _finalize(row: Row, t0: float) -> Row:
    row.elapsed_s = round(time.time() - t0, 3)
    return row


# ---------------------------------------------------------------------------
# Shard runner
# ---------------------------------------------------------------------------

def slice_records(
    record_list: Path,
    shard_id: int,
    n_shards: int,
    limit: int | None = None,
) -> list[dict]:
    """Round-robin assignment so each shard sees a representative slice."""
    rows: list[dict] = []
    with record_list.open() as f:
        rdr = csv.DictReader(f)
        for i, r in enumerate(rdr):
            if (i % n_shards) == shard_id:
                rows.append(r)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def shard_output_name(out_dir: Path, shard_id: int, n_shards: int) -> Path:
    return out_dir / f"phase_annotations_shard_{shard_id:05d}_of_{n_shards:05d}.parquet"


def run_shard(
    shard_id: int,
    n_shards: int,
    record_list: Path,
    out_dir: Path,
    tmpdir: Path,
    limit: int | None = None,
    report_every: int = 50,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    tmpdir.mkdir(parents=True, exist_ok=True)

    out_path = shard_output_name(out_dir, shard_id, n_shards)
    if out_path.exists():
        print(f"[shard {shard_id}] already complete: {out_path}", flush=True)
        return out_path

    sd = load_scanner_defaults(SCANNER_DEFAULTS_PATH)
    records = slice_records(record_list, shard_id, n_shards, limit=limit)
    print(f"[shard {shard_id}/{n_shards}] {len(records)} records", flush=True)

    rows: list[Row] = []
    t0 = time.time()
    for i, rec in enumerate(records, 1):
        try:
            row = process_one(rec, tmpdir, sd)
        except Exception as e:
            # Should never happen — process_one already swallows.
            row = Row(
                dicom_id=Path(rec["dicom_filepath"]).name.replace(".dcm", ""),
                subject_id=rec.get("subject_id", ""),
                study_id=rec.get("study_id", ""),
                dicom_filepath=rec["dicom_filepath"],
                s3_uri=f"{S3_BUCKET}/{rec['dicom_filepath']}",
                reject_reason=f"uncaught:{str(e)[:80]}",
            )
        rows.append(row)
        if i % report_every == 0 or i == len(records):
            n_good = sum(1 for r in rows if r.quality_tier != "reject")
            elapsed = time.time() - t0
            rate = i / max(1e-6, elapsed)
            eta = (len(records) - i) / max(1e-6, rate)
            print(f"[shard {shard_id}] {i}/{len(records)}  "
                  f"tier!=reject: {n_good}  "
                  f"rate={rate:.1f}/s  eta={eta/60:.1f}m",
                  flush=True)
        # Periodic GC: DICOM pixel arrays can be heavy.
        if i % 200 == 0:
            gc.collect()

    df = pd.DataFrame([asdict(r) for r in rows])
    tmp_out = out_path.with_suffix(".tmp.parquet")
    df.to_parquet(tmp_out, index=False)
    tmp_out.rename(out_path)
    print(f"[shard {shard_id}] wrote {out_path} "
          f"({len(df)} rows, {df['quality_tier'].value_counts().to_dict()})",
          flush=True)
    return out_path


def aggregate(out_dir: Path, aggregated: Path) -> int:
    """Concatenate all shard parquets under ``out_dir`` into ``aggregated``."""
    shards = sorted(out_dir.glob("phase_annotations_shard_*.parquet"))
    print(f"Aggregating {len(shards)} shards -> {aggregated}")
    frames = []
    for s in shards:
        try:
            frames.append(pd.read_parquet(s))
        except Exception as e:
            print(f"  skip {s.name}: {e}")
    if not frames:
        print("No shards to aggregate.")
        return 0
    df = pd.concat(frames, ignore_index=True)
    aggregated.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(aggregated, index=False)
    print(f"Wrote {aggregated}: {len(df)} rows, "
          f"tier breakdown: {df['quality_tier'].value_counts().to_dict()}")
    return len(df)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--record-list", type=Path, default=DEFAULT_RECORD_LIST,
                    help="MIMIC-IV-Echo record list CSV "
                         "(subject_id, study_id, acquisition_datetime, dicom_filepath).")
    ap.add_argument("--shard-id", type=int, default=None,
                    help="Shard to process. Defaults to SLURM_ARRAY_TASK_ID if set.")
    ap.add_argument("--n-shards", type=int, default=256)
    ap.add_argument("--out-dir", type=Path,
                    default=Path("/opt/dlami/nvme/phase_anno/shards"))
    ap.add_argument("--tmpdir", type=Path,
                    default=Path(f"/opt/dlami/nvme/tmp/phase_build/{os.getpid()}"))
    ap.add_argument("--limit", type=int, default=None,
                    help="Limit records per shard (for debugging).")
    ap.add_argument("--aggregate", action="store_true",
                    help="Instead of running a shard, concat all shard "
                         "parquets in --out-dir into --aggregated-out.")
    ap.add_argument("--aggregated-out", type=Path,
                    default=Path("/opt/dlami/nvme/phase_anno/phase_annotations.parquet"))
    args = ap.parse_args()

    if args.aggregate:
        aggregate(args.out_dir, args.aggregated_out)
        return

    shard_id = args.shard_id
    if shard_id is None:
        env = os.environ.get("SLURM_ARRAY_TASK_ID")
        if env is None:
            print("No --shard-id and no SLURM_ARRAY_TASK_ID; abort.", file=sys.stderr)
            sys.exit(2)
        shard_id = int(env)

    try:
        run_shard(shard_id, args.n_shards, args.record_list,
                  args.out_dir, args.tmpdir, limit=args.limit)
    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
