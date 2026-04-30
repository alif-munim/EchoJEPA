#!/usr/bin/env python3
"""Save the final frame of every DICOM under phase/dicoms/ as a PNG.

- Multi-frame cines → last frame.
- Single-frame stills → that frame.
- Output goes to phase/lastframe/{stem}.png.

With ``--extract-waveform``, also write a processed version that crops to the
ECG strip and isolates only the trace pixels on a white background. Saved to
phase/lastframe/waveform_extracted/{stem}.png.
"""

import argparse
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image

from crop_waveform_frame import find_waveform_band

HERE = Path(__file__).resolve().parent
DICOM_DIR = HERE / "dicoms"
OUT_DIR = HERE / "lastframe"
WAVEFORM_EXTRACTED_DIR = HERE / "lastframe" / "waveform_extracted"


def extract_waveform_on_white(frame_rgb: np.ndarray, sat_thresh: int = 60) -> np.ndarray:
    """Isolate the ECG trace on a white background.

    1. Locate the ECG band via `find_waveform_band` (green-family density
       detector, same defaults as `crop_waveform_frame.py`).
    2. Inside the band, keep pixels that are saturated and green-dominant
       (the ECG trace). Everything else becomes pure white.
    3. Return an image the same dimensions as the cropped band.
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


def extract(
    dcm_path: Path,
    out_dir: Path,
    waveform_out_dir: Path | None = None,
) -> tuple[str, str | None]:
    try:
        ds = pydicom.dcmread(str(dcm_path))
    except Exception as e:
        return "read_error", str(e)[:120]

    try:
        pa = ds.pixel_array
    except Exception as e:
        return "decode_error", str(e)[:120]

    pi = str(getattr(ds, "PhotometricInterpretation", ""))
    if "PALETTE" in pi:
        from pydicom.pixels.processing import apply_color_lut
        pa = apply_color_lut(pa, ds)
        if pa.dtype == np.uint16:
            pa = (pa / 256).astype(np.uint8)

    # Normalize to a single 2D/3D frame.
    n_frames = int(getattr(ds, "NumberOfFrames", 1))
    if n_frames > 1:
        last = pa[-1]
    elif pa.ndim == 4 and pa.shape[0] == 1:
        last = pa[0]
    elif pa.ndim == 3 and pa.shape[0] == 1 and pa.shape[-1] not in (3, 4):
        # (1, H, W) grayscale
        last = pa[0]
    else:
        last = pa

    last = np.ascontiguousarray(last, dtype=np.uint8)
    out = out_dir / (dcm_path.stem + ".png")
    Image.fromarray(last).save(out, "PNG")

    if waveform_out_dir is not None:
        # Waveform isolation only makes sense for color 3-channel frames.
        if last.ndim != 3 or last.shape[-1] != 3:
            return "ok_no_waveform", None
        try:
            wf = extract_waveform_on_white(last)
        except Exception as e:
            return "ok_waveform_fail", str(e)[:120]
        wf_out = waveform_out_dir / (dcm_path.stem + ".png")
        Image.fromarray(wf).save(wf_out, "PNG")

    return "ok", None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dicom-dir", type=Path, default=DICOM_DIR)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--extract-waveform", action="store_true",
                    help="Also write a waveform-isolated version (trace on "
                         "white background, cropped to the ECG band) to "
                         "--waveform-out-dir.")
    ap.add_argument("--waveform-out-dir", type=Path, default=WAVEFORM_EXTRACTED_DIR,
                    help=f"Output dir for --extract-waveform (default: {WAVEFORM_EXTRACTED_DIR})")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.extract_waveform:
        args.waveform_out_dir.mkdir(parents=True, exist_ok=True)

    dcms = sorted(args.dicom_dir.glob("*.dcm"))
    print(f"Extracting last frame for {len(dcms)} DICOMs → {args.out_dir}")
    if args.extract_waveform:
        print(f"  + waveform extraction → {args.waveform_out_dir}")

    n_ok = n_err = n_wf_skip = n_wf_fail = 0
    for i, p in enumerate(dcms, 1):
        wdir = args.waveform_out_dir if args.extract_waveform else None
        status, err = extract(p, args.out_dir, waveform_out_dir=wdir)
        if status == "ok":
            n_ok += 1
            print(f"  [{i:3d}/{len(dcms)}] ok    {p.name}")
        elif status == "ok_no_waveform":
            n_ok += 1
            n_wf_skip += 1
            print(f"  [{i:3d}/{len(dcms)}] ok    {p.name}  (waveform skip: not RGB)")
        elif status == "ok_waveform_fail":
            n_ok += 1
            n_wf_fail += 1
            print(f"  [{i:3d}/{len(dcms)}] ok    {p.name}  (waveform fail: {err})")
        else:
            n_err += 1
            print(f"  [{i:3d}/{len(dcms)}] FAIL  {p.name}  [{status}]: {err}")

    msg = f"\nDone: {n_ok} PNGs, {n_err} errors"
    if args.extract_waveform:
        msg += f"  (waveform: {n_ok - n_wf_skip - n_wf_fail} ok, {n_wf_skip} non-RGB skip, {n_wf_fail} fail)"
    print(msg)


if __name__ == "__main__":
    main()
