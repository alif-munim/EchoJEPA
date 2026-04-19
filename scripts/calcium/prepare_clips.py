"""
Extract 16-frame clips from Allen Brain Observatory calcium imaging HDF5 movie.

Converts the ~162K-frame, 512×512 grayscale movie into a directory of
16-frame clips saved as individual .npy files (for fast loading during training).

Also generates a CSV in the format expected by VideoDataset:
  path_to_clip.npy 0

Usage:
    python scripts/calcium/prepare_clips.py \
        --input data/calcium_imaging/ophys_501794235.h5 \
        --output_dir data/calcium_imaging/clips_501794235 \
        --stride 4 --num_frames 16
"""

import argparse
import os

import h5py
import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description="Extract calcium imaging clips")
    parser.add_argument("--input", required=True, help="Path to HDF5 movie")
    parser.add_argument("--output_dir", required=True, help="Output directory for clips")
    parser.add_argument("--csv_output", default=None, help="Output CSV path")
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--stride", type=int, default=4, help="Stride between clip starts")
    parser.add_argument("--resolution", type=int, default=224, help="Resize to this resolution")
    parser.add_argument("--max_clips", type=int, default=None, help="Max clips to extract")
    parser.add_argument("--subsample_temporal", type=int, default=3,
                        help="Take every Nth frame (30Hz→10Hz at N=3)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.csv_output is None:
        args.csv_output = os.path.join(args.output_dir, "clips.csv")

    print(f"Opening {args.input}...")
    f = h5py.File(args.input, "r")
    data = f["data"]
    total_frames = data.shape[0]
    H, W = data.shape[1], data.shape[2]
    print(f"Movie: {total_frames} frames, {H}×{W}")

    # Effective frame step (temporal subsampling × clip stride)
    frame_step = args.subsample_temporal
    needed_per_clip = args.num_frames * frame_step
    clip_stride = args.stride * frame_step  # stride in raw frames

    n_clips = (total_frames - needed_per_clip) // clip_stride + 1
    if args.max_clips:
        n_clips = min(n_clips, args.max_clips)
    print(f"Extracting {n_clips} clips (stride={args.stride}, subsample={frame_step})")

    csv_lines = []
    for i in range(n_clips):
        start = i * clip_stride
        indices = list(range(start, start + needed_per_clip, frame_step))
        if indices[-1] >= total_frames:
            break

        # Load frames: [T, H, W] uint16 or uint8
        frames = data[indices]  # [T, H, W]

        # Normalize to 0-255 uint8
        if frames.dtype != np.uint8:
            # Calcium movies are often uint16, normalize to uint8
            fmin, fmax = frames.min(), frames.max()
            if fmax > fmin:
                frames = ((frames.astype(np.float32) - fmin) / (fmax - fmin) * 255).astype(np.uint8)
            else:
                frames = np.zeros_like(frames, dtype=np.uint8)

        # Resize to target resolution
        if H != args.resolution or W != args.resolution:
            resized = []
            for t in range(frames.shape[0]):
                img = Image.fromarray(frames[t]).resize(
                    (args.resolution, args.resolution), Image.BILINEAR
                )
                resized.append(np.array(img))
            frames = np.stack(resized)

        # Convert grayscale to 3-channel (repeat)
        frames_3ch = np.stack([frames, frames, frames], axis=-1)  # [T, H, W, 3]

        # Save as AVI (compatible with decord/VideoDataset)
        clip_path = os.path.join(args.output_dir, f"clip_{i:06d}.avi")
        import cv2
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(clip_path, fourcc, 10, (args.resolution, args.resolution))
        for t in range(frames_3ch.shape[0]):
            writer.write(frames_3ch[t])  # cv2 expects BGR but grayscale 3ch is same
        writer.release()
        csv_lines.append(f"{clip_path} 0")

        if (i + 1) % 5000 == 0:
            print(f"  {i + 1}/{n_clips}")

    f.close()

    # Write CSV
    with open(args.csv_output, "w") as csvf:
        csvf.write("\n".join(csv_lines) + "\n")

    print(f"\nExtracted {len(csv_lines)} clips to {args.output_dir}")
    print(f"CSV: {args.csv_output}")
    print(f"Clip shape: [{args.num_frames}, {args.resolution}, {args.resolution}, 3]")


if __name__ == "__main__":
    main()
