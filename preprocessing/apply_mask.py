"""
Apply sector masks to echocardiogram MP4s to black out non-imaging regions.

Blacks out ECG traces, patient info, measurement annotations, and machine UI
elements. The mask is defined as proportional box coordinates (reference
640x480) so it scales to any input resolution. The same mask was used for
both UHN and MIMIC data.

Usage:
    python preprocessing/apply_mask.py \
        --input_dir /path/to/mp4s --output_dir /path/to/mp4s_masked

    # Skip masking (just copy files)
    python preprocessing/apply_mask.py \
        --input_dir /path/to/mp4s --output_dir /path/to/mp4s_masked --no_mask
"""

import argparse
import os
import shutil

import cv2
import numpy as np
from tqdm import tqdm

# Blackout boxes as (x_frac, y_frac, w_frac, h_frac) relative to image dims.
# Derived from the GE Voxel Cone layout at 640x480 reference resolution.
# Covers: top bar (hospital/date), bottom bar (ECG trace),
# left stack (TGC/gain controls, depth markers),
# right stack (mode indicators, scale bars).
MASK_BOXES = [
    # Top & bottom bars
    (0, 0, 1.0, 35 / 480),
    (0, 1.0 - 90 / 480, 1.0, 120 / 480),
    # Left-side stack
    (0, 0, 60 / 640, 1.0),
    (0, 0, 90 / 640, 0.55),
    (0, 0, 77 / 640, 0.62),
    (0, 0, 130 / 640, 0.3),
    (0, 0, 150 / 640, 0.26),
    (0, 0, 220 / 640, 0.20),
    (0, 0.72, 105 / 640, 0.3),
    # Right-side stack
    (1.0 - 220 / 640, 0, 220 / 640, 0.20),
    (1.0 - 145 / 640, 0, 120 / 640, 0.49),
    (1.0 - 130 / 640, 0, 120 / 640, 0.51),
    (1.0 - 115 / 640, 0, 120 / 640, 0.53),
    (1.0 - 90 / 640, 0, 120 / 640, 1.0),
    (1.0 - 105 / 640, 0.68, 105 / 640, 0.3),
    (1.0 - 115 / 640, 0.72, 105 / 640, 0.3),
]


def create_sector_mask(h, w):
    """Create a binary mask (0=black, 255=keep) scaled to (h, w)."""
    mask = np.ones((h, w), dtype=np.uint8) * 255
    for x_frac, y_frac, w_frac, h_frac in MASK_BOXES:
        x1 = max(0, int(x_frac * w))
        y1 = max(0, int(y_frac * h))
        x2 = min(w, int((x_frac + w_frac) * w))
        y2 = min(h, int((y_frac + h_frac) * h))
        mask[y1:y2, x1:x2] = 0
    return mask


def mask_video(input_path, output_path, skip_mask):
    """Apply sector mask to a single video."""
    if os.path.exists(output_path):
        return "skipped"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if skip_mask:
        shutil.copy2(input_path, output_path)
        return "copied"

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return "error"

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    mask = create_sector_mask(h, w)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h), isColor=True)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(cv2.bitwise_and(frame, frame, mask=mask))
    finally:
        cap.release()
        out.release()

    return "success"


def main():
    parser = argparse.ArgumentParser(description="Apply sector masks to echocardiogram MP4s")
    parser.add_argument("--input_dir", required=True, help="Directory of MP4 files")
    parser.add_argument("--output_dir", required=True, help="Output directory for masked MP4s")
    parser.add_argument("--no_mask", action="store_true", help="Skip masking (just copy files)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.no_mask:
        print("Masking disabled — copying files unchanged.")
    else:
        print("Applying sector mask (Voxel Cone layout, 640x480 reference).")

    # Find all MP4s recursively
    mp4s = []
    for root, _, files in os.walk(args.input_dir):
        for f in files:
            if f.lower().endswith(".mp4"):
                mp4s.append(os.path.join(root, f))

    print(f"Found {len(mp4s)} MP4 files.")

    if not mp4s:
        return

    counts = {"success": 0, "skipped": 0, "copied": 0, "error": 0}
    input_root = os.path.abspath(args.input_dir)
    output_root = os.path.abspath(args.output_dir)

    for mp4 in tqdm(mp4s, desc="Masking"):
        rel = os.path.relpath(mp4, input_root)
        out_path = os.path.join(output_root, rel)
        result = mask_video(mp4, out_path, args.no_mask)
        counts[result] = counts.get(result, 0) + 1

    print(f"\nDone.")
    print(f"  Masked: {counts['success']}")
    print(f"  Copied (no mask): {counts['copied']}")
    print(f"  Skipped (exists): {counts['skipped']}")
    print(f"  Errors: {counts['error']}")


if __name__ == "__main__":
    main()
