"""
Process Samsung DICOM data: unzip → convert to 224x224 MP4 → apply sector masking.

Usage:
    python data/scripts/process_samsung.py

Outputs:
    data/samsung/mp4/          — unmasked 224x224 MP4s
    data/samsung/mp4_masked/   — masked 224x224 MP4s
"""

import glob
import os
import zipfile

import cv2
import numpy as np
import pydicom
from tqdm import tqdm

# --- Configuration ---
BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "samsung")
ZIP_DIR = BASE_DIR
DICOM_DIR = os.path.join(BASE_DIR, "dcm")
MP4_DIR = os.path.join(BASE_DIR, "mp4")
MASKED_DIR = os.path.join(BASE_DIR, "mp4_masked")

TARGET_SIZE = (224, 224)  # (Width, Height)
FPS = 30


# --- Step 1: Unzip ---
def unzip_all(zip_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    zips = glob.glob(os.path.join(zip_dir, "*.zip"))
    if not zips:
        print(f"No .zip files found in {zip_dir}")
        return
    for zp in zips:
        print(f"Unzipping {os.path.basename(zp)} ...")
        with zipfile.ZipFile(zp, "r") as zf:
            zf.extractall(out_dir)
    print(f"Unzipped {len(zips)} archive(s) to {out_dir}")


# --- Step 2: DICOM → 224x224 MP4 ---
def convert_dicom(file_path, out_dir):
    filename = os.path.basename(file_path)
    save_path = os.path.join(out_dir, os.path.splitext(filename)[0] + ".mp4")
    if os.path.exists(save_path):
        return save_path

    try:
        ds = pydicom.dcmread(file_path)
        if "PixelData" not in ds:
            print(f"  Skipping {filename}: No pixel data.")
            return None

        pixel_array = ds.pixel_array
        if pixel_array.ndim == 2:
            pixel_array = pixel_array[np.newaxis, ...]

        # Normalize to 0-255
        pixel_array = pixel_array.astype(float)
        p_min, p_max = pixel_array.min(), pixel_array.max()
        if p_max - p_min != 0:
            pixel_array = ((pixel_array - p_min) / (p_max - p_min)) * 255.0
        else:
            pixel_array = np.zeros_like(pixel_array)
        pixel_array = pixel_array.astype(np.uint8)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(save_path, fourcc, FPS, TARGET_SIZE, isColor=True)

        for frame in pixel_array:
            resized = cv2.resize(frame, TARGET_SIZE, interpolation=cv2.INTER_AREA)
            if resized.ndim == 2:
                bgr = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
            elif resized.shape[2] == 3:
                bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
            else:
                bgr = resized
            out.write(bgr)

        out.release()
        return save_path
    except Exception as e:
        print(f"  Error converting {filename}: {e}")
        return None


def convert_all_dicoms(dcm_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    dcm_files = []
    for root, _, files in os.walk(dcm_dir):
        for f in files:
            if f.lower().endswith(".dcm") or not os.path.splitext(f)[1]:
                dcm_files.append(os.path.join(root, f))

    if not dcm_files:
        print(f"No DICOM files found in {dcm_dir}")
        return []

    converted = []
    for fp in tqdm(dcm_files, desc="DICOM → MP4"):
        result = convert_dicom(fp, out_dir)
        if result:
            converted.append(result)

    print(f"Converted {len(converted)}/{len(dcm_files)} files to {out_dir}")
    return converted


# --- Step 3: Sector masking (from apply_masking.py) ---
def create_sector_mask(h, w):
    """Voxel Cone sector mask. 0 = masked (black), 255 = visible (keep)."""
    mask = np.ones((h, w), dtype=np.uint8) * 255

    def draw_box(x, y, box_w, box_h):
        x1, y1 = max(0, int(x)), max(0, int(y))
        x2, y2 = min(w, int(x + box_w)), min(h, int(y + box_h))
        mask[y1:y2, x1:x2] = 0

    # Top & Bottom bars
    draw_box(0, 0, w, h * (35 / 480))
    draw_box(0, h - h * (90 / 480), w, h * (120 / 480))

    # Left-side stack
    draw_box(0, 0, w * (60 / 640), h)
    draw_box(0, 0, w * (90 / 640), h * 0.55)
    draw_box(0, 0, w * (77 / 640), h * 0.62)
    draw_box(0, 0, w * (130 / 640), h * 0.3)
    draw_box(0, 0, w * (150 / 640), h * 0.26)
    draw_box(0, 0, w * (220 / 640), h * 0.20)
    draw_box(0, h * 0.72, w * (105 / 640), h * 0.3)

    # Right-side stack
    draw_box(w - w * (220 / 640), 0, w * (220 / 640), h * 0.20)
    draw_box(w - w * (145 / 640), 0, w * (120 / 640), h * 0.49)
    draw_box(w - w * (130 / 640), 0, w * (120 / 640), h * 0.51)
    draw_box(w - w * (115 / 640), 0, w * (120 / 640), h * 0.53)
    draw_box(w - w * (90 / 640), 0, w * (120 / 640), h)
    draw_box(w - w * (105 / 640), h * 0.68, w * (105 / 640), h * 0.3)
    draw_box(w - w * (115 / 640), h * 0.72, w * (105 / 640), h * 0.3)

    return mask


def mask_video(file_path, out_dir):
    filename = os.path.basename(file_path)
    save_path = os.path.join(out_dir, filename)
    if os.path.exists(save_path):
        return

    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        print(f"  Error opening {filename}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    mask = create_sector_mask(height, width)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height), isColor=True)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(cv2.bitwise_and(frame, frame, mask=mask))
    finally:
        cap.release()
        out.release()


def mask_all_videos(mp4_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    mp4s = glob.glob(os.path.join(mp4_dir, "*.mp4"))
    if not mp4s:
        print(f"No .mp4 files found in {mp4_dir}")
        return

    for fp in tqdm(mp4s, desc="Masking"):
        mask_video(fp, out_dir)

    print(f"Masked {len(mp4s)} videos to {out_dir}")


# --- Main ---
if __name__ == "__main__":
    print("=== Step 1: Unzip ===")
    unzip_all(ZIP_DIR, DICOM_DIR)

    print("\n=== Step 2: DICOM → 224x224 MP4 ===")
    convert_all_dicoms(DICOM_DIR, MP4_DIR)

    print("\n=== Step 3: Sector Masking ===")
    mask_all_videos(MP4_DIR, MASKED_DIR)

    print("\nDone!")
    print(f"  Unmasked MP4s: {MP4_DIR}")
    print(f"  Masked MP4s:   {MASKED_DIR}")
