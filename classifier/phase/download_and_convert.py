#!/usr/bin/env python3
"""Sample random MIMIC-IV-Echo DICOMs from S3 and convert each to a GIF.

- Samples N random rows from `echo-record-list.csv`.
- Downloads each DICOM from `s3://echodata25/mimic-raw-staging/<dicom_filepath>`
  into `phase/dicoms/`, flattened as `{study_id}_{NNNN}.dcm`.
- Converts multi-frame DICOMs to animated GIFs at the DICOM frame rate into
  `phase/gif/`. Single-frame stills (Doppler spectra, static measurements) are
  skipped — we want cines for ECG-phase work.
"""

import argparse
import csv
import random
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image

HERE = Path(__file__).resolve().parent
DICOM_DIR = HERE / "dicoms"
GIF_DIR = HERE / "gif"
S3_BUCKET = "s3://echodata25/mimic-raw-staging"
RECORD_LIST = Path(
    "/mnt/custom-file-systems/efs/fs-0049217cdf69186d7_fsap-0fa7145b64eaa046b"
    "/vjepa2/uhn_echo/nature_medicine/data_exploration/mimic/mimic-iv-echo"
    "/echo-record-list.csv"
)


def sample_records(csv_path: Path, n: int, seed: int) -> list[dict]:
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    rng = random.Random(seed)
    return rng.sample(rows, min(n, len(rows)))


def download(row: dict) -> tuple[dict, Path | None, str | None]:
    """Download a single DICOM. Returns (row, local_path, error)."""
    rel = row["dicom_filepath"]            # e.g. files/p10/p10002221/s94106955/94106955_0001.dcm
    fname = Path(rel).name                 # 94106955_0001.dcm
    local = DICOM_DIR / fname
    if local.exists() and local.stat().st_size > 0:
        return row, local, None
    s3_uri = f"{S3_BUCKET}/{rel}"
    r = subprocess.run(
        ["aws", "s3", "cp", s3_uri, str(local), "--quiet"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        return row, None, r.stderr.strip() or f"exit {r.returncode}"
    return row, local, None


def dcm_to_gif(dcm_path: Path, out_dir: Path) -> tuple[str, int, str | None]:
    """Convert a DICOM cine to an animated GIF. Returns (status, n_frames, err)."""
    try:
        ds = pydicom.dcmread(str(dcm_path))
    except Exception as e:
        return "read_error", 0, str(e)[:120]

    n_frames = int(getattr(ds, "NumberOfFrames", 1))
    if n_frames <= 1:
        return "single_frame", n_frames, None

    try:
        pa = ds.pixel_array
    except Exception as e:
        return "decode_error", n_frames, str(e)[:120]

    # YBR_FULL / YBR_FULL_422 are auto-converted to RGB by pydicom on decode.
    # PALETTE COLOR needs the LUT applied.
    pi = str(getattr(ds, "PhotometricInterpretation", ""))
    if "PALETTE" in pi:
        from pydicom.pixels.processing import apply_color_lut
        pa = apply_color_lut(pa, ds)
        if pa.dtype == np.uint16:
            pa = (pa / 256).astype(np.uint8)

    # Grayscale → expand to 3-channel for consistent GIF palettes.
    if pa.ndim == 3:            # (T, H, W)
        pa = np.stack([pa, pa, pa], axis=-1)
    pa = np.ascontiguousarray(pa, dtype=np.uint8)

    frame_time_ms = getattr(ds, "FrameTime", None)
    if frame_time_ms is None:
        fps = float(getattr(ds, "RecommendedDisplayFrameRate", None)
                    or getattr(ds, "CineRate", None) or 30)
        frame_time_ms = 1000.0 / fps
    else:
        frame_time_ms = float(frame_time_ms)

    frames = [Image.fromarray(pa[i]) for i in range(pa.shape[0])]
    out = out_dir / (dcm_path.stem + ".gif")
    frames[0].save(
        out,
        save_all=True,
        append_images=frames[1:],
        duration=frame_time_ms,
        loop=0,
        disposal=2,
        optimize=False,
    )
    return "ok", n_frames, None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-n", "--num", type=int, default=20,
                    help="Number of DICOMs to sample (default 20)")
    ap.add_argument("--seed", type=int, default=42,
                    help="RNG seed for the sample (default 42)")
    ap.add_argument("--workers", type=int, default=8,
                    help="Parallel S3 downloads (default 8)")
    args = ap.parse_args()

    DICOM_DIR.mkdir(parents=True, exist_ok=True)
    GIF_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Sampling {args.num} records from {RECORD_LIST.name} (seed={args.seed})")
    records = sample_records(RECORD_LIST, args.num, args.seed)

    print(f"Downloading to {DICOM_DIR}/  (workers={args.workers})")
    downloaded: list[Path] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(download, r) for r in records]
        for i, fut in enumerate(as_completed(futs), 1):
            row, local, err = fut.result()
            tag = row["dicom_filepath"]
            if err:
                print(f"  [{i:3d}/{len(records)}] FAIL  {tag}: {err}")
            else:
                downloaded.append(local)
                print(f"  [{i:3d}/{len(records)}] ok    {local.name}  ({local.stat().st_size:>9} B)")

    print(f"\nConverting {len(downloaded)} DICOMs to GIF → {GIF_DIR}/")
    n_ok = n_skip = n_err = 0
    for i, p in enumerate(sorted(downloaded), 1):
        status, nf, err = dcm_to_gif(p, GIF_DIR)
        if status == "ok":
            n_ok += 1
            print(f"  [{i:3d}/{len(downloaded)}] ok    {p.name}  ({nf} frames)")
        elif status == "single_frame":
            n_skip += 1
            print(f"  [{i:3d}/{len(downloaded)}] skip  {p.name}  (single-frame still)")
        else:
            n_err += 1
            print(f"  [{i:3d}/{len(downloaded)}] FAIL  {p.name}  [{status}]: {err}")

    print(f"\nDone: {n_ok} GIFs, {n_skip} single-frame skipped, {n_err} errors")
    print(f"DICOMs: {DICOM_DIR}")
    print(f"GIFs:   {GIF_DIR}")


if __name__ == "__main__":
    main()
