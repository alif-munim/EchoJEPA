"""
Print resolution, frame count, FPS, and duration statistics for MP4 files.

Usage:
    python preprocessing/check_videos.py --input_dir /path/to/mp4s
    python preprocessing/check_videos.py --input_dir /path/to/mp4s --sample 100
"""

import argparse
import os
import random
from collections import Counter

import cv2


def main():
    parser = argparse.ArgumentParser(description="QC check MP4 echocardiogram files")
    parser.add_argument("--input_dir", required=True, help="Directory of MP4 files")
    parser.add_argument("--sample", type=int, default=None, help="Sample N files (default: all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    # Find MP4s
    mp4s = []
    for root, _, files in os.walk(args.input_dir):
        for f in files:
            if f.lower().endswith(".mp4"):
                mp4s.append(os.path.join(root, f))

    if not mp4s:
        print(f"No .mp4 files found in {args.input_dir}")
        return

    if args.sample and args.sample < len(mp4s):
        random.seed(args.seed)
        mp4s = random.sample(mp4s, args.sample)
        print(f"Sampling {args.sample} of {len(mp4s)} files.")

    print(f"Checking {len(mp4s)} files...\n")

    frame_counts = []
    durations = []
    fps_values = []
    resolutions = []
    errors = 0

    for path in mp4s:
        try:
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                errors += 1
                continue

            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            resolutions.append(f"{w}x{h}")
            fps_values.append(fps)
            frame_counts.append(n_frames)
            if fps > 0:
                durations.append(n_frames / fps)

        except Exception:
            errors += 1

    valid = len(frame_counts)
    if valid == 0:
        print("Could not read any files.")
        return

    # Resolution distribution
    res_counts = Counter(resolutions)
    print("Resolution distribution:")
    for res, count in res_counts.most_common(10):
        print(f"  {res}: {count} ({100 * count / valid:.1f}%)")

    print(f"\nFPS:          min={min(fps_values):.1f}  mean={sum(fps_values) / valid:.1f}  max={max(fps_values):.1f}")
    print(f"Frame count:  min={min(frame_counts)}  mean={sum(frame_counts) / valid:.1f}  max={max(frame_counts)}")
    if durations:
        print(
            f"Duration (s): min={min(durations):.2f}  mean={sum(durations) / len(durations):.2f}  max={max(durations):.2f}"
        )

    print(f"\nTotal files checked: {valid}")
    print(f"Errors: {errors}")


if __name__ == "__main__":
    main()
