#!/usr/bin/env python3
"""Crop the ECG waveform strip from a single still frame (PNG/JPG).

Same band-detection heuristic as `crop_waveform.py`, but operates on a single
image instead of an animated cine — intended for the last-frame PNGs produced
by `extract_lastframe.py`, where the ECG sweep shows the full cardiac cycle of
the clip in one view.

Usage
    python crop_waveform_frame.py INPUT.png [-o OUTPUT.png]
                                            [--min-y FRAC] [--sat-thresh INT]
                                            [--pad INT]

    # Batch mode — all PNGs under a directory:
    python crop_waveform_frame.py --batch lastframe --out-dir lastframe/waveform
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def _density_band(
    row_density: np.ndarray,
    y_start: int,
    core_thresh: int,
    edge_thresh: int,
    edge_gap_px: int,
    edge_max_px: int,
    min_band_px: int,
    max_band_px: int,
) -> tuple[int, int] | None:
    """Locate the tallest contiguous high-density run below `y_start`,
    then expand the edges using a looser density threshold.

    Returns (y0, y1) of the band or None if no qualifying run exists.
    """
    H = len(row_density)
    mask = np.zeros(H, dtype=bool)
    mask[y_start:] = row_density[y_start:] >= core_thresh

    runs: list[tuple[int, int]] = []
    i = 0
    while i < H:
        if mask[i]:
            j = i
            while j < H and mask[j]:
                j += 1
            h = j - i
            if min_band_px <= h <= max_band_px:
                runs.append((i, j))
            i = j
        else:
            i += 1

    if not runs:
        return None

    y0, y1 = max(runs, key=lambda r: r[1] - r[0])
    core_y0, core_y1 = y0, y1

    gap = 0
    y = core_y0 - 1
    while y >= 0 and gap <= edge_gap_px and (core_y0 - y) <= edge_max_px:
        if row_density[y] >= edge_thresh:
            y0 = y
            gap = 0
        else:
            gap += 1
        y -= 1

    gap = 0
    y = core_y1
    while y < H and gap <= edge_gap_px and (y - core_y1) < edge_max_px:
        if row_density[y] >= edge_thresh:
            y1 = y + 1
            gap = 0
        else:
            gap += 1
        y += 1

    return y0, y1


def find_waveform_band(
    img: np.ndarray,
    min_y_frac: float = 0.82,
    sat_thresh: int = 60,
    core_density: int = 10,
    edge_density: int = 2,
    edge_gap_px: int = 3,
    edge_max_px: int = 20,
    min_band_px: int = 8,
    max_band_px: int = 120,
    breathing_room: int = 22,
) -> tuple[int, int]:
    """Find the vertical extent of the ECG trace in a single RGB image.

    Per-row density with a green-family color prior. Count per row the
    pixels that are saturated (`sat > sat_thresh`) **and** green-dominant
    (`G > R and G > B`) — the color family echo ECG traces always fall
    into. Take the tallest contiguous run of rows with density
    `>= core_density` in the bottom `1 - min_y_frac` of the frame, then
    edge-expand to `edge_density` (gap `edge_gap_px`, capped at
    `edge_max_px` rows per side). Color-Doppler blue (B-dominant), red
    annotations (R-dominant), and grayscale sector content (low
    saturation) all fail the green-dominant test, so the density signal
    is essentially pure ECG.

    Finally, add `breathing_room` rows on each side (unconditional, up to
    the frame edges). The detected band sometimes sits a few pixels inside
    the true extent of R-peak tips and trough nadirs — those are sparse,
    anti-aliased, and naturally have lower colored-pixel density *because*
    they are peaks/troughs far from the main trace line. Rather than
    trying to detect these tips (fragile — the density signal genuinely
    tapers off), just pad outward by a fixed number of rows. The pad is
    bounded and small, so even in the worst case it only adds a thin sliver
    of adjacent sector/overlay rather than leaking into something meaningful.
    """
    assert img.ndim == 3 and img.shape[-1] == 3, f"expected HxWx3, got {img.shape}"
    H, _, _ = img.shape
    img_i = img.astype(np.int16)
    sat = img_i.max(axis=-1) - img_i.min(axis=-1)  # (H, W)
    R = img_i[..., 0]; G = img_i[..., 1]; B = img_i[..., 2]

    green_mask = (sat > sat_thresh) & (G > R) & (G > B)
    row_density = green_mask.sum(axis=1)
    y_start = int(H * min_y_frac)
    band = _density_band(
        row_density, y_start,
        core_thresh=core_density, edge_thresh=edge_density,
        edge_gap_px=edge_gap_px, edge_max_px=edge_max_px,
        min_band_px=min_band_px, max_band_px=max_band_px,
    )
    if band is None:
        raise RuntimeError(
            f"No waveform band found below y={y_start} with green-density>={core_density} "
            f"(sat>{sat_thresh}) and height in [{min_band_px}, {max_band_px}]. "
            f"Try lowering --core-density, --sat-thresh, or --min-y."
        )

    y0, y1 = band
    y0 = max(0, y0 - breathing_room)
    y1 = min(H, y1 + breathing_room)
    return y0, y1


def crop_one(in_path: Path, out_path: Path, **kw) -> tuple[int, int]:
    img = np.asarray(Image.open(in_path).convert("RGB"), dtype=np.uint8)
    y0, y1 = find_waveform_band(img, **kw)
    Image.fromarray(img[y0:y1]).save(out_path)
    return y0, y1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("input", type=Path, nargs="?",
                    help="Input single-frame image (PNG/JPG). Ignored with --batch.")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="Output image (default: INPUT.waveform.png)")
    ap.add_argument("--batch", type=Path, default=None,
                    help="Directory of input images; processes all PNGs within")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Output directory for --batch mode "
                         "(default: <batch>/waveform)")
    ap.add_argument("--min-y", type=float, default=0.82,
                    help="Search below this fraction of frame height (default 0.82)")
    ap.add_argument("--sat-thresh", type=int, default=60,
                    help="Min (max_ch - min_ch) to count a pixel as colored (default 60)")
    ap.add_argument("--core-density", type=int, default=10,
                    help="Min colored pixels per row to count toward the core band (default 10)")
    ap.add_argument("--edge-density", type=int, default=2,
                    help="Min colored pixels per row during edge expansion past peak/trough tips (default 2)")
    ap.add_argument("--edge-gap-px", type=int, default=3,
                    help="Max consecutive below-threshold rows tolerated during edge expansion (default 3)")
    ap.add_argument("--edge-max-px", type=int, default=20,
                    help="Max rows of edge expansion per side — prevents running into color-Doppler (default 20)")
    ap.add_argument("--breathing-room", type=int, default=25,
                    help="Extra rows added above and below the detected band to capture "
                         "peak/trough tips that have sparse colored pixels (default 22)")
    ap.add_argument("--min-band-px", type=int, default=8,
                    help="Minimum band height in pixels (default 8)")
    ap.add_argument("--max-band-px", type=int, default=120,
                    help="Maximum band height in pixels (default 120)")
    args = ap.parse_args()

    kw = dict(
        min_y_frac=args.min_y,
        sat_thresh=args.sat_thresh,
        core_density=args.core_density,
        edge_density=args.edge_density,
        edge_gap_px=args.edge_gap_px,
        edge_max_px=args.edge_max_px,
        breathing_room=args.breathing_room,
        min_band_px=args.min_band_px,
        max_band_px=args.max_band_px,
    )

    if args.batch is not None:
        out_dir = args.out_dir or (args.batch / "waveform")
        out_dir.mkdir(parents=True, exist_ok=True)
        pngs = sorted(p for p in args.batch.glob("*.png") if p.parent == args.batch)
        print(f"Cropping {len(pngs)} PNGs from {args.batch} → {out_dir}")
        n_ok = n_err = 0
        for i, p in enumerate(pngs, 1):
            out = out_dir / p.name
            try:
                y0, y1 = crop_one(p, out, **kw)
                n_ok += 1
                print(f"  [{i:3d}/{len(pngs)}] ok    {p.name}  band=[{y0},{y1}]  h={y1 - y0}")
            except Exception as e:
                n_err += 1
                print(f"  [{i:3d}/{len(pngs)}] FAIL  {p.name}  {e}")
        print(f"\nDone: {n_ok} cropped, {n_err} errors")
        return

    if args.input is None:
        ap.error("input is required unless --batch is given")
    out = args.output or args.input.with_suffix(".waveform.png")
    y0, y1 = crop_one(args.input, out, **kw)
    print(f"Input:  {args.input}")
    print(f"Band:   y0={y0} y1={y1}  (height={y1 - y0})")
    print(f"Output: {out}")


if __name__ == "__main__":
    main()
