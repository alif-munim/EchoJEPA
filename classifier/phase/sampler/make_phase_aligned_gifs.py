"""Write two phase-aligned GIFs for a within-study clip pair.

Picks a grid of target phases phi_0 .. phi_{N-1} spanning one cardiac cycle,
and for each clip emits the frame whose confident per-frame phase is closest
to phi_i under wrap-aware distance. Both GIFs end up with the same number of
frames, and frame i in GIF A and frame i in GIF B both sit at phi_i.

Usage::

    python make_phase_aligned_gifs.py --study 90210289 \\
        --clip-a 90210289_0015 --clip-b 90210289_0036 \\
        --out-dir examples/phase_aligned \\
        --n-grid 30 --fps 15

DICOMs are read from classifier/phase/dicoms/<clip>.dcm; the ECG strip at
the bottom of each frame is masked with the aggregated region-of-ultrasound
boundary if present, otherwise we fall back to the `x0, x1` trace-span metadata
row. Output: <out-dir>/<study>/<clip_a>.gif and <clip_b>.gif.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from PIL import Image


def _decode_pixels(ds) -> np.ndarray:
    pa = ds.pixel_array
    pi = str(getattr(ds, "PhotometricInterpretation", ""))
    if "PALETTE" in pi:
        from pydicom.pixels.processing import apply_color_lut
        pa = apply_color_lut(pa, ds)
        if pa.dtype == np.uint16:
            pa = (pa / 256).astype(np.uint8)
    pa = np.ascontiguousarray(pa, dtype=np.uint8)
    if pa.ndim == 3:  # single-frame RGB
        pa = pa[None, ...]
    if pa.shape[-1] == 1:
        pa = np.repeat(pa, 3, axis=-1)
    return pa  # [n_frames, H, W, 3]


def _mask_below_sector(frames: np.ndarray, ds) -> np.ndarray:
    """Zero out pixels below the ultrasound sector's bottom edge so the ECG
    strip is hidden in the output GIF."""
    y_bottom = None
    seq = getattr(ds, "SequenceOfUltrasoundRegions", None)
    if seq is not None:
        for region in seq:
            y = int(getattr(region, "RegionLocationMaxY1", 0) or 0)
            if y > (y_bottom or 0):
                y_bottom = y
    if y_bottom is None or y_bottom <= 0 or y_bottom >= frames.shape[1]:
        # Conservative fallback: hide bottom 15% of the frame.
        y_bottom = int(frames.shape[1] * 0.85)
    out = frames.copy()
    out[:, y_bottom:, :, :] = 0
    return out


def _nearest_confident_frame(
    phase: np.ndarray,
    confident: np.ndarray,
    target_phi: float,
    restrict_to: np.ndarray | None = None,
) -> int | None:
    """Restrict search to ``restrict_to`` indices when provided, else all
    confident frames. Ties broken by smallest index."""
    mask = confident.copy()
    if restrict_to is not None:
        r = np.zeros_like(confident)
        r[restrict_to] = True
        mask &= r
    if not mask.any():
        return None
    idx = np.where(mask)[0]
    d = np.abs(phase[idx] - target_phi)
    d = np.minimum(d, 1.0 - d)
    return int(idx[int(np.argmin(d))])


def _load_phase_row(parquet_path: Path, dicom_id: str) -> pd.Series:
    df = pd.read_parquet(
        parquet_path,
        columns=[
            "dicom_id", "study_id", "n_video_frames", "fps_video",
            "quality_tier", "n_rpeaks_in_video", "coverage_frac",
            "r_peaks_video_json", "per_frame_phase_json", "confident_mask_json",
        ],
    )
    row = df[df.dicom_id == dicom_id]
    if not len(row):
        raise ValueError(f"dicom_id {dicom_id} not in parquet")
    return row.iloc[0]


def _annotate_frame(frame: np.ndarray, phi: float, frame_idx: int, n_frames: int) -> np.ndarray:
    """Burn a small phase/frame indicator into the top-left corner."""
    from PIL import Image, ImageDraw
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    txt = f"phi={phi:.2f}  f={frame_idx}/{n_frames - 1}"
    # White text with black outline for legibility regardless of background.
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        draw.text((8 + dx, 8 + dy), txt, fill=(0, 0, 0))
    draw.text((8, 8), txt, fill=(255, 255, 255))
    return np.array(img)


def _pick_rr_interval(phase_row: pd.Series, strategy: str = "median") -> np.ndarray | None:
    """Return the frame indices [r_s, r_e) of one RR interval.

    ``strategy`` choices:
      - "median"   : pick the RR whose length is the median length (default;
                     robust when one interval spans a missed beat).
      - "nearest_meta": pick the RR whose length is closest to 60*fps/HR_metadata.
      - "longest"  : pick the longest RR (legacy; can select an interval that
                     contains a missed beat).
      - "first"    : first interval.
    """
    r_peaks = json.loads(phase_row.r_peaks_video_json)
    if len(r_peaks) < 2:
        return None
    rr = [(s, e) for s, e in zip(r_peaks[:-1], r_peaks[1:]) if e > s]
    if not rr:
        return None
    if strategy == "longest":
        s, e = max(rr, key=lambda p: p[1] - p[0])
    elif strategy == "first":
        s, e = rr[0]
    elif strategy == "nearest_meta":
        hr = phase_row.hr_metadata if phase_row.hr_metadata and phase_row.hr_metadata > 0 else None
        fps = phase_row.fps_video
        if hr is None or not (fps and fps > 0):
            lens = sorted(e - s for s, e in rr)
            tgt = lens[len(lens) // 2]
            s, e = min(rr, key=lambda p: abs((p[1] - p[0]) - tgt))
        else:
            target = 60.0 * float(fps) / float(hr)
            s, e = min(rr, key=lambda p: abs((p[1] - p[0]) - target))
    else:  # "median"
        lens = sorted(e - s for s, e in rr)
        tgt = lens[len(lens) // 2]
        s, e = min(rr, key=lambda p: abs((p[1] - p[0]) - tgt))
    return np.arange(int(s), int(e))


def build_aligned_clip(
    dicom_path: Path,
    phase_row: pd.Series,
    anchor_phases: np.ndarray,
    annotate: bool = True,
    restrict_to_single_cycle: bool = True,
) -> tuple[list[np.ndarray], list[int], list[float]]:
    ds = pydicom.dcmread(str(dicom_path))
    pixels = _decode_pixels(ds)
    pixels = _mask_below_sector(pixels, ds)

    phase = np.array(
        [np.nan if v is None else float(v) for v in json.loads(phase_row.per_frame_phase_json)],
        dtype=np.float64,
    )
    confident = np.array(json.loads(phase_row.confident_mask_json), dtype=bool)
    n = int(phase_row.n_video_frames)
    assert len(phase) == n and len(confident) == n, "phase-array length mismatch"

    restrict_idx = None
    if restrict_to_single_cycle:
        restrict_idx = _pick_rr_interval(phase_row, strategy="nearest_meta")

    out_frames, picked_idx, picked_phi = [], [], []
    for phi in anchor_phases:
        f = _nearest_confident_frame(phase, confident, float(phi), restrict_to=restrict_idx)
        if f is None:
            continue
        frame = pixels[f]
        if annotate:
            frame = _annotate_frame(frame, float(phi), f, n)
        out_frames.append(frame)
        picked_idx.append(f)
        picked_phi.append(float(phi))
    return out_frames, picked_idx, picked_phi


def write_gif(frames: list[np.ndarray], path: Path, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imgs = [Image.fromarray(f) for f in frames]
    duration_ms = int(1000.0 / fps)
    imgs[0].save(
        path,
        save_all=True,
        append_images=imgs[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )


def main() -> None:
    here = Path(__file__).resolve().parent
    phase_dir = here.parent  # classifier/phase
    default_parquet = phase_dir / "phase_annotations" / "phase_annotations.parquet"
    default_dicom_dir = phase_dir / "dicoms"
    default_out = phase_dir / "examples" / "phase_aligned"

    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", type=Path, default=default_parquet)
    ap.add_argument("--dicom-dir", type=Path, default=default_dicom_dir)
    ap.add_argument("--study", required=True)
    ap.add_argument("--clip-a", required=True)
    ap.add_argument("--clip-b", required=True)
    ap.add_argument("--out-dir", type=Path, default=default_out)
    ap.add_argument("--n-grid", type=int, default=30,
                    help="Number of phases in the aligned grid (== # GIF frames).")
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--no-annotate", action="store_true")
    ap.add_argument("--all-frames", action="store_true",
                    help="Allow picks from anywhere in the clip (default: "
                         "restrict to the longest in-video RR interval so "
                         "the GIF reads like a natural single cycle).")
    args = ap.parse_args()

    anchors = np.linspace(0.0, 1.0, args.n_grid, endpoint=False)
    out_root = args.out_dir / args.study

    for clip_id in (args.clip_a, args.clip_b):
        dicom = args.dicom_dir / f"{clip_id}.dcm"
        if not dicom.exists():
            raise FileNotFoundError(dicom)
        row = _load_phase_row(args.parquet, clip_id)
        if row.study_id != args.study:
            raise ValueError(f"{clip_id} belongs to study {row.study_id}, not {args.study}")
        frames, idxs, phis = build_aligned_clip(
            dicom, row, anchors,
            annotate=not args.no_annotate,
            restrict_to_single_cycle=not args.all_frames,
        )
        out = out_root / f"{clip_id}.gif"
        write_gif(frames, out, args.fps)
        print(f"wrote {out}  ({len(frames)} frames, {row.n_video_frames} source, "
              f"fps_src={row.fps_video:.1f}, tier={row.quality_tier})")
        # Per-phase picks for debugging / demo transparency.
        dump = pd.DataFrame(
            {"phi": phis, "frame_idx": idxs}
        )
        dump.to_csv(out.with_suffix(".csv"), index=False)


if __name__ == "__main__":
    main()
