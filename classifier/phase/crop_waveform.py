#!/usr/bin/env python3
"""Crop the ECG waveform strip out of an echo cine GIF.

Vertical-only crop (full width preserved). Locates the bottom-of-frame ECG band by
finding rows with a stable colored trace — the ECG is rendered as a saturated
(non-gray) line over a near-black strip, unlike the greyscale sector above it.

Usage
    python crop_waveform.py INPUT.gif [-o OUTPUT.gif] [--min-y FRAC]
                                       [--sat-thresh INT] [--pad INT]

Example
    python crop_waveform.py 94106955_0008.gif
    # writes 94106955_0008.waveform.gif
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageSequence


def load_gif_rgb(path: Path) -> np.ndarray:
    """Load an animated GIF as a (T, H, W, 3) uint8 array."""
    im = Image.open(path)
    frames = [np.asarray(f.convert("RGB"), dtype=np.uint8) for f in ImageSequence.Iterator(im)]
    return np.stack(frames, axis=0)


def read_durations(path: Path) -> list[int]:
    """Per-frame durations (ms) from the source GIF, for re-saving at native rate."""
    im = Image.open(path)
    out = []
    for f in ImageSequence.Iterator(im):
        out.append(int(f.info.get("duration", 33)))
    return out


def find_waveform_band(
    rgb: np.ndarray,
    min_y_frac: float = 0.82,
    sat_thresh: int = 60,
    min_band_px: int = 8,
    max_band_px: int = 120,
    pad: int = 4,
) -> tuple[int, int]:
    """Find the vertical extent of the ECG trace.

    Strategy: compute a per-row "colored" score = max-across-frames of
    (max_channel - min_channel), i.e. how saturated the most-saturated pixel
    in that row ever gets. The greyscale sector scores near 0; the ECG trace
    spikes well above `sat_thresh`. Restrict search to the lower portion of
    the frame (below `min_y_frac`) to skip color Doppler inside the sector.

    Band selection: within the bottom search window, take the *tallest*
    contiguous colored run whose height is within [min_band_px, max_band_px].
    `max_band_px` keeps us from latching onto tall color-Doppler regions that
    bleed into the lower half; among the remaining (thin) runs, the ECG
    trace is the most prominent.
    """
    T, H, W, _ = rgb.shape
    rgb_i = rgb.astype(np.int16)
    sat = rgb_i.max(axis=-1) - rgb_i.min(axis=-1)  # (T, H, W)
    row_sat = sat.max(axis=(0, 2))                  # (H,)

    y_start = int(H * min_y_frac)
    mask = np.zeros(H, dtype=bool)
    mask[y_start:] = row_sat[y_start:] > sat_thresh

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
        raise RuntimeError(
            f"No waveform band found below y={y_start} with sat>{sat_thresh} "
            f"and height in [{min_band_px}, {max_band_px}]. "
            f"Try adjusting --min-y, --sat-thresh, or --max-band-px."
        )

    # Tallest qualifying run — within the bottom search window, the ECG trace
    # is the dominant thin colored band (small label/text bands score lower).
    y0, y1 = max(runs, key=lambda r: r[1] - r[0])
    y0 = max(0, y0 - pad)
    y1 = min(H, y1 + pad)
    return y0, y1


def save_gif(frames: np.ndarray, out_path: Path, durations: list[int]) -> None:
    """Save a (T, H, W, 3) uint8 stack as an animated GIF."""
    imgs = [Image.fromarray(frames[t]) for t in range(frames.shape[0])]
    # PIL's `duration` can be a list (per-frame); it falls back to a scalar if
    # the list has length 1.
    dur = durations if len(durations) == len(imgs) else durations[0]
    imgs[0].save(
        out_path,
        save_all=True,
        append_images=imgs[1:],
        duration=dur,
        loop=0,
        disposal=2,
        optimize=False,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("input", type=Path, help="Input echo cine GIF")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="Output GIF (default: INPUT.waveform.gif)")
    ap.add_argument("--min-y", type=float, default=0.82,
                    help="Search below this fraction of frame height (default 0.82)")
    ap.add_argument("--sat-thresh", type=int, default=60,
                    help="Min (max_ch - min_ch) to count a pixel as colored (default 60)")
    ap.add_argument("--pad", type=int, default=4,
                    help="Extra rows of padding around the detected band (default 4)")
    ap.add_argument("--min-band-px", type=int, default=8,
                    help="Minimum band height in pixels (default 8)")
    ap.add_argument("--max-band-px", type=int, default=120,
                    help="Maximum band height in pixels (default 120)")
    args = ap.parse_args()

    out = args.output or args.input.with_suffix(".waveform.gif")

    rgb = load_gif_rgb(args.input)
    durations = read_durations(args.input)
    T, H, W, _ = rgb.shape

    y0, y1 = find_waveform_band(
        rgb,
        min_y_frac=args.min_y,
        sat_thresh=args.sat_thresh,
        pad=args.pad,
        min_band_px=args.min_band_px,
        max_band_px=args.max_band_px,
    )
    print(f"Input:  {args.input}  ({T} frames, {H}x{W})")
    print(f"Band:   rows y0={y0} y1={y1}  (height={y1 - y0})")

    cropped = rgb[:, y0:y1, :, :]
    save_gif(cropped, out, durations)
    print(f"Output: {out}  ({cropped.shape[0]} frames, {cropped.shape[1]}x{cropped.shape[2]})")


if __name__ == "__main__":
    main()
