"""
Convert DICOM echocardiograms to MP4 video files.

Handles grayscale, RGB, and multi-frame DICOMs. Normalizes pixel values to 0-255.
Preserves directory structure for recursive input directories.

Usage:
    # Native resolution (recommended)
    python preprocessing/convert_dicom.py \
        --input_dir /path/to/dicoms --output_dir /path/to/mp4s --workers 8

    # Fixed resolution (for pretraining I/O savings)
    python preprocessing/convert_dicom.py \
        --input_dir /path/to/dicoms --output_dir /path/to/mp4s_224 \
        --resolution 224 --workers 8
"""

import argparse
import os
from multiprocessing import Pool, cpu_count
from pathlib import Path

import cv2
import numpy as np
import pydicom
from tqdm import tqdm


def convert_single(args):
    """Convert a single DICOM file to MP4."""
    input_path, output_path, resolution, fps = args

    if os.path.exists(output_path):
        return "skipped"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        ds = pydicom.dcmread(input_path)
        if "PixelData" not in ds:
            return "no_pixels"

        pixel_array = ds.pixel_array

        # Normalize shape to (Frames, H, W) or (Frames, H, W, C)
        if pixel_array.ndim == 2:
            # Single grayscale frame
            pixel_array = pixel_array[np.newaxis, ...]
        elif pixel_array.ndim == 3:
            # Could be (Frames, H, W) grayscale or (H, W, C) single RGB
            n_frames = getattr(ds, "NumberOfFrames", 1)
            if int(n_frames) > 1 and pixel_array.shape[0] == int(n_frames):
                pass  # (Frames, H, W) grayscale — correct
            elif pixel_array.shape[2] in (3, 4):
                pixel_array = pixel_array[np.newaxis, ...]  # (1, H, W, C)

        # Normalize to 0-255
        pixel_array = pixel_array.astype(np.float32)
        p_min, p_max = pixel_array.min(), pixel_array.max()
        if p_max - p_min > 0:
            pixel_array = ((pixel_array - p_min) / (p_max - p_min)) * 255.0
        else:
            pixel_array = np.zeros_like(pixel_array)
        pixel_array = pixel_array.astype(np.uint8)

        # Determine output size
        if resolution is not None:
            target_size = (resolution, resolution)
        else:
            # Native resolution
            if pixel_array.ndim == 4:
                target_size = (pixel_array.shape[2], pixel_array.shape[1])  # (W, H)
            else:
                target_size = (pixel_array.shape[2], pixel_array.shape[1])  # (W, H)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, target_size, isColor=True)

        for i in range(pixel_array.shape[0]):
            frame = pixel_array[i]

            # Resize if needed
            if (frame.shape[1], frame.shape[0]) != target_size:
                frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)

            # Ensure 3-channel BGR
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            elif frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

            out.write(frame)

        out.release()
        return "success"

    except Exception as e:
        print(f"Error converting {input_path}: {e}")
        return "error"


def find_dicoms(input_dir):
    """Recursively find DICOM files."""
    dicoms = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            lower = f.lower()
            if lower.endswith(".dcm") or lower.endswith(".dicom"):
                dicoms.append(os.path.join(root, f))
            elif "." not in f:
                # DICOM files often have no extension
                full = os.path.join(root, f)
                try:
                    pydicom.dcmread(full, stop_before_pixels=True)
                    dicoms.append(full)
                except Exception:
                    pass
    return dicoms


def main():
    parser = argparse.ArgumentParser(description="Convert DICOM echocardiograms to MP4")
    parser.add_argument("--input_dir", required=True, help="Directory containing DICOM files (recursive)")
    parser.add_argument("--output_dir", required=True, help="Output directory for MP4 files")
    parser.add_argument("--resolution", type=int, default=None, help="Target resolution (e.g., 224). None=native")
    parser.add_argument("--fps", type=int, default=30, help="Output FPS (default: 30)")
    parser.add_argument("--workers", type=int, default=max(1, cpu_count() // 2), help="Parallel workers")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Scanning for DICOM files in {args.input_dir}...")
    dicoms = find_dicoms(args.input_dir)
    print(f"Found {len(dicoms)} DICOM files.")

    if not dicoms:
        return

    # Build tasks preserving directory structure
    input_root = Path(args.input_dir)
    output_root = Path(args.output_dir)
    tasks = []
    for dcm_path in dicoms:
        rel = Path(dcm_path).relative_to(input_root)
        out_path = str(output_root / rel.with_suffix(".mp4"))
        tasks.append((dcm_path, out_path, args.resolution, args.fps))

    # Process
    counts = {"success": 0, "skipped": 0, "error": 0, "no_pixels": 0}

    if args.workers <= 1:
        for task in tqdm(tasks, desc="Converting"):
            result = convert_single(task)
            counts[result] = counts.get(result, 0) + 1
    else:
        with Pool(args.workers) as pool:
            for result in tqdm(pool.imap_unordered(convert_single, tasks), total=len(tasks), desc="Converting"):
                counts[result] = counts.get(result, 0) + 1

    res_str = f"{args.resolution}x{args.resolution}" if args.resolution else "native"
    print(f"\nDone. Resolution: {res_str}")
    print(f"  Success: {counts['success']}")
    print(f"  Skipped (exists): {counts['skipped']}")
    print(f"  No pixels: {counts['no_pixels']}")
    print(f"  Errors: {counts['error']}")


if __name__ == "__main__":
    main()
