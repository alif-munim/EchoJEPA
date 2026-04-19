"""
CMR-Multi to MP4 conversion pipeline.

Converts CINE_MULTI NIfTI data (SAX, 2CH, 4CH) into H.264 MP4 clips compatible
with V-JEPA and VideoMAE training pipelines.

CMR-Multi stores 3D NIfTI volumes where the third dimension is slices * frames
interleaved. For SAX: ~11 slices × 25 frames. For 2CH/4CH: 1 slice × N frames.
Slice boundaries are detected via intensity discontinuities.

LVEF labels are available for all 105 SAX patients (from dataset_train.xlsx).

Usage:
    # Sanity check (3 patients, SAX only)
    python scripts/neurips/cmr/convert_cmr_multi.py --sanity-check

    # SAX only (most useful for temporal experiments)
    python scripts/neurips/cmr/convert_cmr_multi.py --sax

    # All CINE views (SAX + 2CH + 4CH)
    python scripts/neurips/cmr/convert_cmr_multi.py --all

    # Append to existing processed/ manifests
    python scripts/neurips/cmr/convert_cmr_multi.py --all --append
"""

import argparse
import json
import os
import warnings
from pathlib import Path

import imageio
import nibabel as nib
import numpy as np
import pandas as pd
from PIL import Image

# ============================================================
# Constants
# ============================================================

DATA_ROOT = Path("/home/sagemaker-user/user-default-efs/vjepa2/data/cmr")
CMR_MULTI_ROOT = DATA_ROOT / "cmr-multi"
OUTPUT_DIR = DATA_ROOT / "processed" / "clips"
PROCESSED_DIR = DATA_ROOT / "processed"

# Minimum frames per clip to keep (skip degenerate slices)
MIN_FRAMES = 8


# ============================================================
# Slice boundary detection
# ============================================================

def detect_slice_boundaries(vol_3d: np.ndarray, min_gap_ratio: float = 2.0) -> list[int]:
    """Detect slice boundaries in a 3D volume where dim2 = slices * frames.

    Returns list of boundary indices (start of each new slice).
    E.g., for 11 slices × 25 frames → [0, 25, 50, 75, ...].
    """
    z_dim = vol_3d.shape[2]

    # Compute mean absolute difference between adjacent z-planes
    diffs = np.array([
        np.mean(np.abs(vol_3d[:, :, z + 1].astype(np.float64) - vol_3d[:, :, z].astype(np.float64)))
        for z in range(z_dim - 1)
    ])

    if len(diffs) == 0:
        return [0]

    # Find large jumps (slice transitions vs temporal changes)
    median_diff = np.median(diffs)
    if median_diff < 1e-6:
        # Constant volume — treat as single slice
        return [0]

    threshold = median_diff * min_gap_ratio
    jump_indices = np.where(diffs > threshold)[0]

    # Build boundary list: start of volume + index after each jump
    boundaries = [0]
    for idx in jump_indices:
        boundaries.append(idx + 1)

    # Validate: slices should have roughly equal frame counts
    slice_lengths = []
    for i in range(len(boundaries)):
        start = boundaries[i]
        end = boundaries[i + 1] if i + 1 < len(boundaries) else z_dim
        slice_lengths.append(end - start)

    if len(slice_lengths) > 1:
        median_len = np.median(slice_lengths)
        # Filter out spurious boundaries (slices that are too short)
        filtered = [0]
        for i in range(1, len(boundaries)):
            start = boundaries[i]
            prev_start = filtered[-1]
            prev_len = start - prev_start
            if prev_len >= median_len * 0.5:
                filtered.append(start)
        boundaries = filtered

    return boundaries


def split_into_slices(vol_3d: np.ndarray) -> list[np.ndarray]:
    """Split a 3D (H, W, slices*frames) volume into per-slice temporal arrays.

    Returns list of [H, W, T] arrays, one per slice.
    """
    boundaries = detect_slice_boundaries(vol_3d)
    z_dim = vol_3d.shape[2]
    slices = []

    for i in range(len(boundaries)):
        start = boundaries[i]
        end = boundaries[i + 1] if i + 1 < len(boundaries) else z_dim
        if end - start >= MIN_FRAMES:
            slices.append(vol_3d[:, :, start:end])

    return slices


# ============================================================
# Post-processing: same as convert_cmr_to_mp4.py
# ============================================================

def percentile_clip_and_scale(arr: np.ndarray, p_low=1, p_high=99) -> np.ndarray:
    """Clip to [p_low, p_high] percentile and scale to uint8 [0, 255]."""
    lo = np.percentile(arr, p_low)
    hi = np.percentile(arr, p_high)
    if hi - lo < 1e-6:
        return np.zeros_like(arr, dtype=np.uint8)
    clipped = np.clip(arr, lo, hi)
    scaled = (clipped - lo) / (hi - lo) * 255.0
    return scaled.astype(np.uint8)


def resize_with_padding(frame: np.ndarray, target_size: int = 256) -> np.ndarray:
    """Resize to target_size x target_size with aspect-ratio-preserving padding."""
    h, w = frame.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)

    img = Image.fromarray(frame)
    img = img.resize((new_w, new_h), Image.BILINEAR)

    result = np.zeros((target_size, target_size), dtype=np.uint8)
    y_off = (target_size - new_h) // 2
    x_off = (target_size - new_w) // 2
    result[y_off:y_off + new_h, x_off:x_off + new_w] = np.array(img)
    return result


def slice_to_mp4(
    slice_data: np.ndarray,
    clip_name: str,
    output_dir: Path,
    target_size: int = 256,
    fps: int = 25,
    crf: int = 17,
) -> Path:
    """Write a [H, W, T] slice array as an MP4 clip. Returns output path."""
    scaled = percentile_clip_and_scale(slice_data)
    T = scaled.shape[2]

    clip_path = output_dir / clip_name
    writer = imageio.get_writer(
        str(clip_path),
        format="FFMPEG",
        mode="I",
        fps=fps,
        codec="libx264",
        output_params=["-crf", str(crf), "-pix_fmt", "yuv420p"],
    )
    for t in range(T):
        frame = resize_with_padding(scaled[:, :, t], target_size)
        frame_rgb = np.stack([frame, frame, frame], axis=-1)
        writer.append_data(frame_rgb)
    writer.close()

    return clip_path


# ============================================================
# Dataset loading
# ============================================================

def load_lvef_labels(xlsx_path: Path) -> dict[int, float]:
    """Load LVEF labels from dataset_train.xlsx SAX sheet.

    Returns {patient_id (int): ef_value (float)}.
    """
    if not xlsx_path.exists():
        return {}

    df = pd.read_excel(xlsx_path, sheet_name=0)  # first sheet = SAX

    # Find the LVEF column
    lvef_col = None
    for col in df.columns:
        if "LVEF" in str(col).upper():
            lvef_col = col
            break
    if lvef_col is None:
        warnings.warn(f"No LVEF column found in {xlsx_path}")
        return {}

    # Find the patient ID column
    pid_col = None
    for col in df.columns:
        if "patient" in str(col).lower() or col in ("编号",):
            pid_col = col
            break
    if pid_col is None:
        pid_col = df.columns[0]  # fall back to first column

    labels = {}
    for _, row in df.iterrows():
        pid = int(row[pid_col])
        ef_str = str(row[lvef_col]).strip().replace("%", "").strip()
        try:
            ef = float(ef_str)
            labels[pid] = round(ef, 1)
        except ValueError:
            continue

    return labels


def discover_cmr_multi(views=("SAX",)):
    """Discover CMR-Multi CINE patients for specified views.

    Returns list of dicts with keys: path, view, patient_num, split.
    """
    cine_dir = CMR_MULTI_ROOT / "CINE_MULTI"
    if not cine_dir.exists():
        print(f"ERROR: {cine_dir} not found")
        return []

    # Load LVEF labels
    xlsx_path = cine_dir / "dataset_train.xlsx"
    lvef_labels = load_lvef_labels(xlsx_path)
    if lvef_labels:
        print(f"  Loaded {len(lvef_labels)} LVEF labels from {xlsx_path.name}")

    patients = []
    for view in views:
        view_dir = cine_dir / f"{view}_TR" / "image"
        if not view_dir.exists():
            print(f"  WARNING: {view_dir} not found, skipping")
            continue

        nii_files = sorted(view_dir.glob("*.nii.gz"))
        print(f"  {view}: {len(nii_files)} files")

        for nii_path in nii_files:
            # Extract patient number from filename: CINE_SAX_001.nii.gz → 1
            fname = nii_path.stem.replace(".nii", "")
            patient_num = int(fname.split("_")[-1])

            patients.append({
                "path": nii_path,
                "view": view,
                "patient_num": patient_num,
                "ef_label": lvef_labels.get(patient_num) if view == "SAX" else None,
            })

    # Assign splits: deterministic 80/10/10 by patient number
    unique_pids = sorted(set(p["patient_num"] for p in patients))
    rng = np.random.RandomState(44)
    indices = rng.permutation(len(unique_pids))
    n_train = int(0.8 * len(unique_pids))
    n_val = int(0.1 * len(unique_pids))

    pid_to_split = {}
    for i, idx in enumerate(indices):
        pid = unique_pids[idx]
        if i < n_train:
            pid_to_split[pid] = "train"
        elif i < n_train + n_val:
            pid_to_split[pid] = "val"
        else:
            pid_to_split[pid] = "test"

    for p in patients:
        p["split"] = pid_to_split[p["patient_num"]]

    return patients


# ============================================================
# Conversion
# ============================================================

def convert_patient(patient: dict, output_dir: Path) -> list[dict]:
    """Convert one CMR-Multi patient NIfTI to MP4 clips.

    Returns list of clip info dicts.
    """
    nii_path = patient["path"]
    view = patient["view"]
    pnum = patient["patient_num"]
    split = patient["split"]
    ef_label = patient["ef_label"]

    img = nib.load(str(nii_path))
    vol = img.get_fdata().astype(np.float32)  # [H, W, Z]
    H, W, Z = vol.shape

    clip_infos = []

    if view == "SAX":
        # SAX: split into per-slice temporal clips
        slices = split_into_slices(vol)
        for s_idx, slice_data in enumerate(slices):
            T = slice_data.shape[2]
            clip_name = f"cmrmulti_{view.lower()}_{pnum:03d}_slice{s_idx:02d}.mp4"
            clip_path = slice_to_mp4(slice_data, clip_name, output_dir)

            clip_infos.append({
                "path": str(clip_path),
                "clip_name": clip_name,
                "dataset": "cmrmulti",
                "view": view,
                "patient_id": f"{pnum:03d}",
                "patient_num": pnum,
                "slice_idx": s_idx,
                "n_frames": T,
                "original_shape": f"{H}x{W}",
                "ef_label": ef_label,
                "split": split,
            })
    else:
        # 2CH / 4CH: entire 3rd dimension is time (single slice)
        T = Z
        if T < MIN_FRAMES:
            warnings.warn(f"Skipping {nii_path.name}: only {T} frames")
            return []

        clip_name = f"cmrmulti_{view.lower()}_{pnum:03d}.mp4"
        clip_path = slice_to_mp4(vol, clip_name, output_dir)

        clip_infos.append({
            "path": str(clip_path),
            "clip_name": clip_name,
            "dataset": "cmrmulti",
            "view": view,
            "patient_id": f"{pnum:03d}",
            "patient_num": pnum,
            "slice_idx": 0,
            "n_frames": T,
            "original_shape": f"{H}x{W}",
            "ef_label": ef_label,
            "split": split,
        })

    return clip_infos


def run_conversion(patients: list[dict], output_dir: Path, limit: int = None) -> list[dict]:
    """Convert all patients. Returns list of all clip infos."""
    output_dir.mkdir(parents=True, exist_ok=True)
    all_clips = []

    for i, patient in enumerate(patients):
        if limit is not None and i >= limit:
            break

        try:
            clips = convert_patient(patient, output_dir)
        except Exception as e:
            print(f"  ERROR: {patient['path'].name}: {e}")
            continue

        all_clips.extend(clips)
        view = patient["view"]
        pnum = patient["patient_num"]
        ef_str = f", EF={patient['ef_label']}%" if patient.get("ef_label") is not None else ""
        print(f"  [{i+1:3d}] {view}/{pnum:03d}: {len(clips)} clips{ef_str}")

    return all_clips


# ============================================================
# Manifest generation
# ============================================================

def write_manifests(all_clips: list[dict], append: bool = False):
    """Write pretrain and probe manifests + splits.json."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # --- Pretrain manifest (train split, dummy labels) ---
    pretrain_path = PROCESSED_DIR / "pretrain_manifest.csv"
    mode = "a" if append else "w"
    with open(pretrain_path, mode) as f:
        for c in all_clips:
            if c["split"] == "train":
                f.write(f"{c['path']} 0\n")
    n_pretrain = sum(1 for c in all_clips if c["split"] == "train")

    # --- Probe manifest (SAX clips with EF labels) ---
    probe_path = PROCESSED_DIR / "probe_manifest_cmrmulti.csv"
    with open(probe_path, "w") as f:
        f.write("path\tef\tsplit\tpatient_id\tslice_idx\tview\n")
        for c in all_clips:
            if c.get("ef_label") is not None:
                f.write(f"{c['path']}\t{c['ef_label']}\t{c['split']}\t"
                        f"{c['patient_id']}\t{c['slice_idx']}\t{c['view']}\n")
    n_probe = sum(1 for c in all_clips if c.get("ef_label") is not None)

    # --- Splits ---
    splits = {"train": [], "val": [], "test": []}
    seen = set()
    for c in all_clips:
        key = f"cmrmulti_{c['patient_id']}"
        if key not in seen:
            seen.add(key)
            splits[c["split"]].append(key)

    splits_path = PROCESSED_DIR / "splits_cmrmulti.json"
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=2)

    print(f"\nPretrain manifest: {n_pretrain} clips → {pretrain_path} ({'appended' if append else 'written'})")
    print(f"Probe manifest: {n_probe} clips with EF → {probe_path}")
    print(f"Splits: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])} → {splits_path}")


def print_summary(all_clips: list[dict]):
    """Print conversion summary."""
    print(f"\n{'=' * 60}")
    print(f"SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total clips: {len(all_clips)}")

    # Per-view breakdown
    by_view = {}
    for c in all_clips:
        by_view.setdefault(c["view"], []).append(c)
    for view, clips in sorted(by_view.items()):
        frames = [c["n_frames"] for c in clips]
        n_patients = len(set(c["patient_num"] for c in clips))
        ef_count = sum(1 for c in clips if c.get("ef_label") is not None)
        print(f"  {view:5s}: {len(clips):5d} clips from {n_patients} patients, "
              f"frames: {min(frames)}-{max(frames)} (mean {np.mean(frames):.0f})"
              f"{f', {ef_count} with EF' if ef_count else ''}")

    # Per-split breakdown
    by_split = {}
    for c in all_clips:
        by_split.setdefault(c["split"], []).append(c)
    print(f"\nBy split:")
    for split in ["train", "val", "test"]:
        clips = by_split.get(split, [])
        n_patients = len(set(c["patient_num"] for c in clips))
        print(f"  {split:5s}: {len(clips):5d} clips, {n_patients} patients")

    # EF distribution
    ef_values = [c["ef_label"] for c in all_clips if c.get("ef_label") is not None and c["slice_idx"] == 0]
    if ef_values:
        print(f"\nLVEF distribution ({len(ef_values)} patients):")
        print(f"  min={min(ef_values):.1f}%, max={max(ef_values):.1f}%, "
              f"mean={np.mean(ef_values):.1f}%, std={np.std(ef_values):.1f}%")


# ============================================================
# Entry points
# ============================================================

def sanity_check():
    """Process 3 SAX patients for quick verification."""
    print("=" * 60)
    print("SANITY CHECK: 3 CMR-Multi SAX patients")
    print("=" * 60)

    patients = discover_cmr_multi(views=("SAX",))[:3]
    all_clips = run_conversion(patients, OUTPUT_DIR, limit=3)
    print_summary(all_clips)

    # Save sample GIF
    if all_clips:
        mid = all_clips[len(all_clips) // 2]
        gif_path = PROCESSED_DIR / "sanity_cmrmulti.gif"
        try:
            from decord import VideoReader, cpu
            vr = VideoReader(mid["path"], num_threads=1, ctx=cpu(0))
            gif_frames = [vr[i].asnumpy() for i in range(len(vr))]
            imageio.mimsave(str(gif_path), gif_frames, duration=1000 / 25, loop=0)
            print(f"\nSaved GIF: {gif_path} ({len(gif_frames)} frames from {mid['clip_name']})")
        except Exception as e:
            print(f"\nGIF save failed: {e}")

    # Verify with decord
    print("\n--- Decord verification ---")
    try:
        from decord import VideoReader, cpu
        for clip in all_clips[:3]:
            vr = VideoReader(clip["path"], num_threads=1, ctx=cpu(0))
            frame = vr[0].asnumpy()
            print(f"  {clip['clip_name']}: {len(vr)} frames, shape={frame.shape}, "
                  f"range=[{frame.min()}, {frame.max()}]")
    except ImportError:
        print("  decord not available, skipping")


def full_conversion(views: tuple, append: bool = False):
    """Full conversion of specified views."""
    view_str = " + ".join(views)
    print("=" * 60)
    print(f"CMR-MULTI CONVERSION: {view_str}")
    print("=" * 60)

    patients = discover_cmr_multi(views=views)
    print(f"\nTotal: {len(patients)} files to convert")

    all_clips = run_conversion(patients, OUTPUT_DIR)
    print_summary(all_clips)
    write_manifests(all_clips, append=append)
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sanity-check", action="store_true",
                        help="Process 3 SAX patients for quick verification")
    parser.add_argument("--sax", action="store_true",
                        help="Convert SAX views only")
    parser.add_argument("--all", action="store_true",
                        help="Convert all CINE views (SAX + 2CH + 4CH)")
    parser.add_argument("--append", action="store_true",
                        help="Append to existing pretrain_manifest.csv instead of overwriting")
    args = parser.parse_args()

    if args.sanity_check:
        sanity_check()
    elif args.sax:
        full_conversion(views=("SAX",), append=args.append)
    elif args.all:
        full_conversion(views=("SAX", "2CH", "4CH"), append=args.append)
    else:
        parser.print_help()
