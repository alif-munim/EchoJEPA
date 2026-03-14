"""
Convert masked Samsung MP4s to GIFs with view prediction overlay.

Reads view_predictions.csv and creates GIFs with the predicted view label
in white text in the top-left corner.

Usage:
    python data/samsung/make_prediction_gifs.py

Output:
    data/samsung/gifs/  — annotated GIFs
"""

import os

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MASKED_DIR = os.path.join(SCRIPT_DIR, "mp4_masked")
GIF_DIR = os.path.join(SCRIPT_DIR, "gifs")
PREDICTIONS_CSV = os.path.join(SCRIPT_DIR, "view_predictions.csv")


def mp4_to_gif(video_path, gif_path, label, confidence, fps=10, max_frames=60):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  Error opening {video_path}")
        return

    frames = []
    count = 0
    # Sample every n-th frame to keep GIF size reasonable
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, int(video_fps / fps))

    frame_idx = 0
    while count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            # BGR → RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(frame_rgb)

            # Draw label
            draw = ImageDraw.Draw(pil_frame)
            text = label
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
            except OSError:
                font = ImageFont.load_default()
            draw.text((8, 8), text, fill="white", font=font, stroke_width=1, stroke_fill="black")

            frames.append(pil_frame)
            count += 1
        frame_idx += 1

    cap.release()

    if frames:
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=int(1000 / fps),
            loop=0,
        )


def main():
    os.makedirs(GIF_DIR, exist_ok=True)

    df = pd.read_csv(PREDICTIONS_CSV)
    print(f"Creating GIFs for {len(df)} predictions...")

    for _, row in tqdm(df.iterrows(), total=len(df), desc="GIFs"):
        video_file = row["video_file"]
        video_path = os.path.join(MASKED_DIR, video_file)
        gif_path = os.path.join(GIF_DIR, video_file.replace(".mp4", ".gif"))

        if not os.path.exists(video_path):
            print(f"  Missing: {video_path}")
            continue

        mp4_to_gif(video_path, gif_path, row["predicted_view"], row["confidence"])

    print(f"Done! GIFs saved to {GIF_DIR}")


if __name__ == "__main__":
    main()
