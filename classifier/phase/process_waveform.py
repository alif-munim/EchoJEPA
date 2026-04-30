#!/usr/bin/env python3
"""Post-process extracted ECG strips into clean 1D signals + high-res renders.

Takes the white-background strips in ``phase/lastframe/waveform_extracted/``
(trace pixels preserved, everything else white) and produces:

  * ``phase/lastframe/waveform_processed/{stem}.png`` — re-rendered trace at
    300 DPI (for visual inspection).
  * ``phase/lastframe/waveform_processed/{stem}.npz`` — the 1D signal plus
    alignment metadata. This is the canonical ECG representation used by
    downstream phase / alignment analysis; the image rendering is disposable.

NPZ contents:

    xs               : int array, columns in [x0, x1] (detected trace span)
    ys               : float array, PCHIP-interpolated, median-filtered,
                       sign-flipped amplitude over xs (positive = up)
    full_y           : float array of length W, NaN outside the trace span;
                       preserves strip-column timing for cross-clip alignment
    observed_mask    : bool array of length W, True where a trace pixel was
                       detected directly in that column (pre-interpolation)
    trace_span_mask  : bool array of length W, True for columns inside
                       [x0, x1] — the region covered by the PCHIP fit
    interpolated_mask: bool array of length W, trace_span_mask & ~observed_mask
    width, height    : source-image dimensions (int)
    x0, x1           : first / last detected-trace column (int)
    n_observed       : int, trace pixels detected directly
    coverage_frac    : float, n_observed / (x1 - x0 + 1) — confidence score

Pipeline per clip:
  1. Luminance threshold (``mean(RGB) < lum_threshold``) segments the
     teal/green trace against white background.
  2. ``binary_dilation`` bridges broken segments; largest connected component
     is kept to drop stray text / scale markers.
  3. Per-column centroid of the mask is the raw ``y(x)``.
  4. ``observed_mask`` records which columns had any trace pixel at all
     (pre-PCHIP), so downstream xcorr / R-peak can weight or gate on real vs
     interpolated samples.
  5. PCHIP interpolation fills internal gaps inside the trace span without
     overshooting (monotonic cubic).
  6. 3-px median filter smooths residual jitter; y is flipped (positive up).

Caveat: vertical resolution of the source raster is the hard amplitude
ceiling. Only ~50 distinct y-values exist — stair-stepping in flat segments
is expected and not a bug.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import binary_dilation, label, median_filter

HERE = Path(__file__).resolve().parent
INPUT_DIR = HERE / "lastframe" / "waveform_extracted"
OUTPUT_DIR = HERE / "lastframe" / "waveform_processed"


def extract_ecg_signal(
    image_path: Path,
    lum_threshold: int = 200,
    dilate_iters: int = 2,
    median_size: int = 3,
) -> dict:
    """Return a dict with xs, ys, full_y, masks, geometry. See module docstring."""
    img = np.array(Image.open(image_path).convert("RGB"))
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
        "img": img,
    }

    mask = img.mean(axis=2) < lum_threshold
    if not mask.any():
        return empty

    labeled, _ = label(binary_dilation(mask, iterations=dilate_iters))
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
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
        # Not enough points to interpolate; emit the raw (possibly sparse) array.
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

    interpolated_mask = trace_span_mask & ~observed_mask

    n_obs = int(observed_in_span.sum())
    span = x1 - x0 + 1
    coverage = float(n_obs / span) if span > 0 else 0.0

    return {
        "xs": xs,
        "ys": yt.astype(np.float32),
        "full_y": full_y,
        "observed_mask": observed_mask,
        "trace_span_mask": trace_span_mask,
        "interpolated_mask": interpolated_mask,
        "width": int(W), "height": int(H),
        "x0": x0, "x1": x1,
        "n_observed": n_obs, "coverage_frac": coverage,
        "img": img,
    }


def render_high_res(
    xs: np.ndarray,
    ys: np.ndarray,
    out_path: Path,
    color: str = "#2a7f7f",
    dpi: int = 300,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 2), dpi=dpi)
    ax.plot(xs, ys, color=color, linewidth=1.4, antialiased=True)
    ax.set_axis_off()
    ax.margins(x=0.005, y=0.08)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close()


def save_npz(out_path: Path, data: dict) -> None:
    np.savez_compressed(
        out_path,
        xs=data["xs"],
        ys=data["ys"],
        full_y=data["full_y"],
        observed_mask=data["observed_mask"],
        trace_span_mask=data["trace_span_mask"],
        interpolated_mask=data["interpolated_mask"],
        width=np.int32(data["width"]),
        height=np.int32(data["height"]),
        x0=np.int32(data["x0"]),
        x1=np.int32(data["x1"]),
        n_observed=np.int32(data["n_observed"]),
        coverage_frac=np.float32(data["coverage_frac"]),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input-dir", type=Path, default=INPUT_DIR,
                    help=f"Directory of extracted strips (default {INPUT_DIR})")
    ap.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                    help=f"Output directory (default {OUTPUT_DIR})")
    ap.add_argument("--lum-threshold", type=int, default=200)
    ap.add_argument("--dilate-iters", type=int, default=2)
    ap.add_argument("--median-size", type=int, default=3)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--no-save-signal", action="store_true",
                    help="Skip the .npz signal dump (rendering only).")
    ap.add_argument("--no-render", action="store_true",
                    help="Skip the .png rendering (signal dump only).")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pngs = sorted(args.input_dir.glob("*.png"))
    print(f"Processing {len(pngs)} strips from {args.input_dir} -> {args.output_dir}")

    n_ok = n_empty = n_err = 0
    for i, p in enumerate(pngs, 1):
        try:
            data = extract_ecg_signal(
                p,
                lum_threshold=args.lum_threshold,
                dilate_iters=args.dilate_iters,
                median_size=args.median_size,
            )
            if len(data["xs"]) < 2:
                n_empty += 1
                print(f"  [{i:3d}/{len(pngs)}] skip  {p.name}  (no trace)")
                continue
            if not args.no_render:
                render_high_res(data["xs"], data["ys"],
                                args.output_dir / p.name, dpi=args.dpi)
            if not args.no_save_signal:
                save_npz(args.output_dir / (p.stem + ".npz"), data)
            n_ok += 1
            if i % 25 == 0 or i == len(pngs):
                cov = data["coverage_frac"]
                span = data["x1"] - data["x0"] + 1
                print(f"  [{i:3d}/{len(pngs)}] ok    {p.name}  "
                      f"span={span:4d}/{data['width']}  coverage={cov:.2f}")
        except Exception as e:
            n_err += 1
            print(f"  [{i:3d}/{len(pngs)}] FAIL  {p.name}  {e}")

    print(f"\nDone: {n_ok} ok, {n_empty} empty, {n_err} errors")


if __name__ == "__main__":
    main()
