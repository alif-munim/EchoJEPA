"""
CMR-to-MP4 conversion pipeline for ACDC, M&Ms, M&Ms-2, and Sunnybrook.

Produces H.264 MP4 clips compatible with V-JEPA and VideoMAE training pipelines.
One MP4 per SAX slice per patient.

Usage:
    # Sanity check (3 ACDC patients)
    python scripts/neurips/cmr/convert_cmr_to_mp4.py --sanity-check

    # Full conversion (NIfTI datasets only)
    python scripts/neurips/cmr/convert_cmr_to_mp4.py --all

    # Include Sunnybrook (DICOM)
    python scripts/neurips/cmr/convert_cmr_to_mp4.py --all --include-sunnybrook
"""

import argparse
import json
import os
import warnings
from pathlib import Path

import imageio
import nibabel as nib
import numpy as np
from PIL import Image

try:
    import pydicom
except ImportError:
    pydicom = None

# ============================================================
# Dataset loaders — each returns (volume, metadata)
# volume: np.ndarray [H, W, S, T] float
# metadata: dict with patient_id, dataset, ef_label (optional)
# ============================================================

def load_acdc_patient(patient_dir: Path) -> tuple[np.ndarray, dict]:
    """Load one ACDC patient. Returns [H, W, S, T] float + metadata."""
    pid = patient_dir.name  # e.g., "patient001"
    nii_path = patient_dir / f"{pid}_4d.nii.gz"
    img = nib.load(str(nii_path))
    vol = img.get_fdata().astype(np.float32)  # [H, W, S, T]

    # Parse EF from Info.cfg
    meta = {"patient_id": pid, "dataset": "acdc", "ef_label": None}
    cfg_path = patient_dir / "Info.cfg"
    if cfg_path.exists():
        info = {}
        with open(cfg_path) as f:
            for line in f:
                if ":" in line:
                    k, v = line.strip().split(":", 1)
                    info[k.strip()] = v.strip()
        ed_frame = int(info.get("ED", 0)) - 1  # 1-indexed in cfg
        es_frame = int(info.get("ES", 0)) - 1
        meta["ed_frame"] = ed_frame
        meta["es_frame"] = es_frame
        meta["group"] = info.get("Group", "")
        meta["n_frames"] = int(info.get("NbFrame", vol.shape[3]))

        # Compute EF from ED/ES segmentation volumes if ground truth exists
        ed_gt_path = patient_dir / f"{pid}_frame{int(info.get('ED', 1)):02d}_gt.nii.gz"
        es_gt_path = patient_dir / f"{pid}_frame{int(info.get('ES', 1)):02d}_gt.nii.gz"
        if ed_gt_path.exists() and es_gt_path.exists():
            ed_gt = nib.load(str(ed_gt_path)).get_fdata()
            es_gt = nib.load(str(es_gt_path)).get_fdata()
            voxel_vol = np.prod(img.header.get_zooms()[:3])  # mm^3
            # LV cavity is label 3 in ACDC
            edv = np.sum(ed_gt == 3) * voxel_vol / 1000.0  # mL
            esv = np.sum(es_gt == 3) * voxel_vol / 1000.0
            if edv > 0:
                ef = (edv - esv) / edv * 100.0
                meta["ef_label"] = round(ef, 1)
                meta["edv_ml"] = round(edv, 1)
                meta["esv_ml"] = round(esv, 1)

    return vol, meta


def load_mnm_patient(patient_dir: Path) -> tuple[np.ndarray, dict]:
    """Load one M&Ms patient. Returns [H, W, S, T] float + metadata."""
    pid = patient_dir.name
    # M&Ms uses {pid}_sa.nii
    sa_path = patient_dir / f"{pid}_sa.nii"
    if not sa_path.exists():
        sa_path = patient_dir / f"{pid}_sa.nii.gz"
    img = nib.load(str(sa_path))
    vol = img.get_fdata().astype(np.float32)

    meta = {"patient_id": pid, "dataset": "mnm", "ef_label": None}
    return vol, meta


def load_mnm2_patient(patient_dir: Path) -> tuple[np.ndarray, dict]:
    """Load one M&Ms-2 patient. Returns [H, W, S, T] float + metadata."""
    pid = patient_dir.name
    sa_path = patient_dir / f"{pid}_SA_CINE.nii"
    if not sa_path.exists():
        sa_path = patient_dir / f"{pid}_SA_CINE.nii.gz"
    img = nib.load(str(sa_path))
    vol = img.get_fdata().astype(np.float32)

    meta = {"patient_id": pid, "dataset": "mnm2", "ef_label": None}
    return vol, meta


def load_dsb2_patient(patient_dir: Path, labels: dict = None) -> tuple[np.ndarray, dict]:
    """Load one DSB2 patient from DICOM. Returns [H, W, S, T] float + metadata.

    DSB2 stores DICOMs as: {patient_dir}/study/sax_N/*.dcm
    Each sax_N directory is one slice location. DICOMs within are time frames
    sorted by InstanceNumber.
    """
    if pydicom is None:
        raise ImportError("pydicom required for DSB2 DICOM loading")

    pid = patient_dir.name
    study_dir = patient_dir / "study"
    if not study_dir.exists():
        raise FileNotFoundError(f"No study directory in {patient_dir}")

    # Find SAX directories only
    sax_dirs = sorted([d for d in study_dir.iterdir()
                       if d.is_dir() and d.name.startswith("sax_")],
                      key=lambda d: int(d.name.split("_")[1]))

    if not sax_dirs:
        raise FileNotFoundError(f"No sax directories in {study_dir}")

    # Load each slice: sort DICOMs by InstanceNumber
    slice_data = []
    for sax_dir in sax_dirs:
        dcm_files = sorted(sax_dir.glob("*.dcm"))
        if not dcm_files:
            continue
        frames = []
        for dcm_path in dcm_files:
            ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=False)
            inst = int(getattr(ds, "InstanceNumber", 0))
            frames.append((inst, ds.pixel_array.astype(np.float32)))
        frames.sort(key=lambda x: x[0])
        slice_data.append([f[1] for f in frames])

    if not slice_data:
        raise FileNotFoundError(f"No DICOM data in {study_dir}")

    # Build 4D volume [H, W, S, T]
    n_frames = min(len(s) for s in slice_data)
    H, W = slice_data[0][0].shape
    n_slices = len(slice_data)

    vol = np.zeros((H, W, n_slices, n_frames), dtype=np.float32)
    for s_idx, frames in enumerate(slice_data):
        for t_idx in range(n_frames):
            frame = frames[t_idx]
            # Handle resolution mismatch across slices (rare but possible)
            if frame.shape == (H, W):
                vol[:, :, s_idx, t_idx] = frame
            else:
                # Resize to match first slice
                img = Image.fromarray(frame.astype(np.uint8) if frame.max() <= 255 else
                                      ((frame / frame.max()) * 255).astype(np.uint8))
                img = img.resize((W, H), Image.BILINEAR)
                vol[:, :, s_idx, t_idx] = np.array(img).astype(np.float32)

    meta = {"patient_id": pid, "dataset": "dsb2", "ef_label": None}

    # Add EF label if available
    if labels and pid in labels:
        esv, edv = labels[pid]
        if edv > 0:
            ef = (edv - esv) / edv * 100.0
            meta["ef_label"] = round(ef, 1)
            meta["esv_ml"] = round(esv, 1)
            meta["edv_ml"] = round(edv, 1)

    return vol, meta


def load_sunnybrook_patient(patient_dir: Path) -> tuple[np.ndarray, dict]:
    """Load one Sunnybrook patient from DICOM. Returns [H, W, S, T] float + metadata.

    Sunnybrook stores all SAX frames as individual DICOMs in a CINESAX_* directory.
    We group by SliceLocation and sort by TriggerTime to reconstruct the 4D volume.
    """
    if pydicom is None:
        raise ImportError("pydicom required for Sunnybrook DICOM loading")

    pid = patient_dir.name
    # Find the CINESAX directory
    sax_dirs = [d for d in patient_dir.iterdir() if d.is_dir() and "CINESAX" in d.name]
    if not sax_dirs:
        raise FileNotFoundError(f"No CINESAX directory found in {patient_dir}")

    sax_dir = sax_dirs[0]  # typically one SAX series per patient
    dcm_files = sorted([f for f in sax_dir.iterdir() if f.suffix == ".dcm"])

    # Group by slice location, sort by trigger time
    slices = {}
    for dcm_path in dcm_files:
        ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=False)
        loc = round(float(getattr(ds, "SliceLocation", 0)), 1)
        trig = float(getattr(ds, "TriggerTime", 0))
        pixel_array = ds.pixel_array.astype(np.float32)
        slices.setdefault(loc, []).append((trig, pixel_array))

    # Sort each slice by trigger time
    sorted_locs = sorted(slices.keys())
    for loc in sorted_locs:
        slices[loc].sort(key=lambda x: x[0])

    # Build 4D volume [H, W, S, T]
    n_slices = len(sorted_locs)
    n_frames = min(len(slices[loc]) for loc in sorted_locs)  # use min to handle uneven
    H, W = slices[sorted_locs[0]][0][1].shape

    vol = np.zeros((H, W, n_slices, n_frames), dtype=np.float32)
    for s_idx, loc in enumerate(sorted_locs):
        for t_idx in range(n_frames):
            vol[:, :, s_idx, t_idx] = slices[loc][t_idx][1]

    meta = {"patient_id": pid, "dataset": "sunnybrook", "ef_label": None}
    return vol, meta


# ============================================================
# Shared post-processing: volume → MP4 clips
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

    # Pad to square
    result = np.zeros((target_size, target_size), dtype=np.uint8)
    y_off = (target_size - new_h) // 2
    x_off = (target_size - new_w) // 2
    result[y_off:y_off + new_h, x_off:x_off + new_w] = np.array(img)
    return result


def volume_to_mp4s(
    vol: np.ndarray,
    meta: dict,
    output_dir: Path,
    target_size: int = 256,
    fps: int = 25,
    crf: int = 17,
) -> list[dict]:
    """
    Convert a 4D volume [H, W, S, T] to one MP4 per slice.

    Returns list of clip metadata dicts.
    """
    H, W, S, T = vol.shape
    dataset = meta["dataset"]
    pid = meta["patient_id"]
    clip_infos = []

    for s in range(S):
        slice_data = vol[:, :, s, :]  # [H, W, T]

        # Percentile clip across all frames of this slice
        scaled = percentile_clip_and_scale(slice_data)  # [H, W, T] uint8

        # Build frames: resize + pad + replicate to RGB
        frames = []
        for t in range(T):
            frame = resize_with_padding(scaled[:, :, t], target_size)
            frame_rgb = np.stack([frame, frame, frame], axis=-1)  # [H, W, 3]
            frames.append(frame_rgb)

        # Write MP4
        clip_name = f"{dataset}_{pid}_slice{s:02d}.mp4"
        clip_path = output_dir / clip_name

        writer = imageio.get_writer(
            str(clip_path),
            format="FFMPEG",
            mode="I",
            fps=fps,
            codec="libx264",
            output_params=["-crf", str(crf), "-pix_fmt", "yuv420p"],
        )
        for frame in frames:
            writer.append_data(frame)
        writer.close()

        info = {
            "path": str(clip_path),
            "clip_name": clip_name,
            "dataset": dataset,
            "patient_id": pid,
            "slice_idx": s,
            "n_frames": T,
            "original_shape": f"{H}x{W}",
            "ef_label": meta.get("ef_label"),
        }
        clip_infos.append(info)

        if T < 12:
            warnings.warn(f"Short clip: {clip_name} has only {T} frames")

    return clip_infos


# ============================================================
# Dataset discovery
# ============================================================

DATA_ROOT = Path("/home/sagemaker-user/user-default-efs/vjepa2/data/cmr")
OUTPUT_DIR = DATA_ROOT / "processed" / "clips"


def discover_acdc():
    """Return list of (patient_dir, split) for ACDC."""
    patients = []
    for split_name, split_dir in [
        ("train", DATA_ROOT / "acdc" / "ACDC" / "database" / "training"),
        ("test", DATA_ROOT / "acdc" / "ACDC" / "database" / "testing"),
    ]:
        if not split_dir.exists():
            continue
        for p in sorted(split_dir.iterdir()):
            if p.is_dir() and (p / f"{p.name}_4d.nii.gz").exists():
                patients.append((p, split_name, "acdc", load_acdc_patient))
    return patients


def discover_mnm():
    """Return list of (patient_dir, split) for M&Ms."""
    base = DATA_ROOT / "mnm" / "MnM"
    patients = []
    for split_name, split_subdir in [
        ("train", "Training/Labeled"),
        ("train", "Training/Unlabeled"),
        ("val", "Validation"),
        ("test", "Testing"),
    ]:
        split_dir = base / split_subdir
        if not split_dir.exists():
            continue
        for p in sorted(split_dir.iterdir()):
            if p.is_dir():
                sa_file = p / f"{p.name}_sa.nii"
                if sa_file.exists():
                    patients.append((p, split_name, "mnm", load_mnm_patient))
    return patients


def discover_mnm2():
    """Return list of (patient_dir, split) for M&Ms-2."""
    base = DATA_ROOT / "mnm2" / "MnM2" / "dataset"
    patients = []
    if not base.exists():
        return patients
    # MnM2 doesn't have explicit splits in directory structure.
    # Use a fixed seed 80/10/10 split.
    all_pids = sorted([p for p in base.iterdir() if p.is_dir()])
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(all_pids))
    n_train = int(0.8 * len(all_pids))
    n_val = int(0.1 * len(all_pids))
    for i, idx in enumerate(indices):
        p = all_pids[idx]
        sa_file = p / f"{p.name}_SA_CINE.nii"
        if not sa_file.exists():
            continue
        if i < n_train:
            split = "train"
        elif i < n_train + n_val:
            split = "val"
        else:
            split = "test"
        patients.append((p, split, "mnm2", load_mnm2_patient))
    return patients


def discover_dsb2():
    """Return list of (patient_dir, split) for DSB2."""
    base = DATA_ROOT / "dsb2"
    patients = []

    # Load labels from train.csv
    labels = {}
    train_csv = base / "train.csv"
    if train_csv.exists():
        import csv
        with open(train_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                labels[row["Id"]] = (float(row["Systole"]), float(row["Diastole"]))

    # Parse solution.csv for test/validate labels
    sol_csv = base / "solution.csv"
    sol_labels = {}
    if sol_csv.exists():
        import csv
        with open(sol_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Format: "1000_Diastole" or "1000_Systole"
                parts = row["Id"].rsplit("_", 1)
                pid = parts[0]
                vol_type = parts[1]  # Systole or Diastole
                sol_labels.setdefault(pid, {})[vol_type] = float(row["Volume"])
        for pid, vols in sol_labels.items():
            if "Systole" in vols and "Diastole" in vols:
                labels[pid] = (vols["Systole"], vols["Diastole"])

    # Train split: dsb2/train/train/{pid}/study/
    train_dir = base / "train" / "train"
    if train_dir.exists():
        for p in sorted(train_dir.iterdir(), key=lambda x: int(x.name) if x.name.isdigit() else 0):
            if p.is_dir() and (p / "study").exists():
                sax_dirs = [d for d in (p / "study").iterdir() if d.name.startswith("sax_")]
                if sax_dirs:
                    # Wrap loader with labels
                    def make_loader(lab):
                        return lambda pdir: load_dsb2_patient(pdir, lab)
                    patients.append((p, "train", "dsb2", make_loader(labels)))

    # Test split: dsb2/test/test/{pid}/study/
    test_dir = base / "test" / "test"
    if test_dir.exists():
        for p in sorted(test_dir.iterdir(), key=lambda x: int(x.name) if x.name.isdigit() else 0):
            if p.is_dir() and (p / "study").exists():
                sax_dirs = [d for d in (p / "study").iterdir() if d.name.startswith("sax_")]
                if sax_dirs:
                    def make_loader(lab):
                        return lambda pdir: load_dsb2_patient(pdir, lab)
                    patients.append((p, "test", "dsb2", make_loader(labels)))

    # Validate split: dsb2/validate/validate/{pid}/study/
    val_dir = base / "validate" / "validate"
    if val_dir.exists():
        for p in sorted(val_dir.iterdir(), key=lambda x: int(x.name) if x.name.isdigit() else 0):
            if p.is_dir() and (p / "study").exists():
                sax_dirs = [d for d in (p / "study").iterdir() if d.name.startswith("sax_")]
                if sax_dirs:
                    def make_loader(lab):
                        return lambda pdir: load_dsb2_patient(pdir, lab)
                    patients.append((p, "val", "dsb2", make_loader(labels)))

    return patients


def discover_sunnybrook():
    """Return list of (patient_dir, split) for Sunnybrook."""
    base = DATA_ROOT / "sunnybrook"
    patients = []
    if not base.exists():
        return patients

    all_pids = sorted([p for p in base.iterdir()
                       if p.is_dir() and p.name.startswith("SCD")
                       and any(d.name.startswith("CINESAX") for d in p.iterdir() if d.is_dir())])

    # No official splits; random 80/10/10
    rng = np.random.RandomState(43)
    indices = rng.permutation(len(all_pids))
    n_train = int(0.8 * len(all_pids))
    n_val = int(0.1 * len(all_pids))

    for i, idx in enumerate(indices):
        p = all_pids[idx]
        if i < n_train:
            split = "train"
        elif i < n_train + n_val:
            split = "val"
        else:
            split = "test"
        patients.append((p, split, "sunnybrook", load_sunnybrook_patient))
    return patients


# ============================================================
# Main
# ============================================================

def process_patients(patient_list, output_dir: Path, limit=None):
    """Process a list of patients and return all clip infos."""
    output_dir.mkdir(parents=True, exist_ok=True)
    all_clips = []
    frame_counts = []

    for i, (pdir, split, dataset, loader) in enumerate(patient_list):
        if limit is not None and i >= limit:
            break
        try:
            vol, meta = loader(pdir)
        except Exception as e:
            print(f"  ERROR loading {pdir}: {e}")
            continue

        meta["split"] = split
        clips = volume_to_mp4s(vol, meta, output_dir)

        for c in clips:
            c["split"] = split
            frame_counts.append(c["n_frames"])

        all_clips.extend(clips)
        n_slices = vol.shape[2]
        n_frames = vol.shape[3]
        ef_str = f", EF={meta.get('ef_label')}" if meta.get("ef_label") else ""
        print(f"  [{i+1}] {dataset}/{meta['patient_id']}: {n_slices} slices x {n_frames} frames{ef_str} → {len(clips)} clips")

    return all_clips, frame_counts


def sanity_check():
    """Process 3 ACDC patients, verify with decord, save a sample GIF."""
    print("=" * 60)
    print("SANITY CHECK: 3 ACDC patients")
    print("=" * 60)

    acdc_patients = discover_acdc()[:3]
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    all_clips, frame_counts = process_patients(acdc_patients, output_dir, limit=3)

    print(f"\nGenerated {len(all_clips)} clips")
    print(f"Frame count distribution: min={min(frame_counts)}, max={max(frame_counts)}, "
          f"mean={np.mean(frame_counts):.1f}")

    # Verify with decord
    print("\n--- Decord verification ---")
    try:
        from decord import VideoReader, cpu
        for clip in all_clips[:3]:
            vr = VideoReader(clip["path"], num_threads=1, ctx=cpu(0))
            frame = vr[0].asnumpy()
            print(f"  {clip['clip_name']}: {len(vr)} frames, shape={frame.shape}, "
                  f"dtype={frame.dtype}, range=[{frame.min()}, {frame.max()}]")
    except ImportError:
        print("  decord not available, skipping verification")

    # Save a sample GIF (mid-ventricular slice of first patient)
    print("\n--- Saving sample GIF ---")
    mid_clip = all_clips[len(all_clips) // 6]  # roughly mid-ventricle of first patient
    gif_path = DATA_ROOT / "processed" / "sanity_check.gif"

    try:
        from decord import VideoReader, cpu
        vr = VideoReader(mid_clip["path"], num_threads=1, ctx=cpu(0))
        gif_frames = [vr[i].asnumpy() for i in range(len(vr))]
        imageio.mimsave(str(gif_path), gif_frames, duration=1000 / 25, loop=0)
        print(f"  Saved GIF: {gif_path} ({len(gif_frames)} frames from {mid_clip['clip_name']})")
    except Exception as e:
        print(f"  GIF save failed: {e}")

    # Print EF labels
    print("\n--- EF labels ---")
    for clip in all_clips:
        if clip["ef_label"] is not None and clip["slice_idx"] == 0:
            print(f"  {clip['patient_id']}: EF={clip['ef_label']}%")

    print("\nSanity check complete.")
    return all_clips


def full_conversion(include_sunnybrook=False, include_dsb2=False):
    """Process all datasets."""
    datasets_label = "ACDC + M&Ms + M&Ms-2"
    if include_sunnybrook:
        datasets_label += " + Sunnybrook"
    if include_dsb2:
        datasets_label += " + DSB2"
    print("=" * 60)
    print(f"FULL CONVERSION: {datasets_label}")
    print("=" * 60)

    acdc = discover_acdc()
    mnm = discover_mnm()
    mnm2 = discover_mnm2()

    msg = f"Discovered: ACDC={len(acdc)}, MnM={len(mnm)}, MnM2={len(mnm2)}"
    all_patients = acdc + mnm + mnm2

    if include_sunnybrook:
        sb = discover_sunnybrook()
        msg += f", Sunnybrook={len(sb)}"
        all_patients += sb

    if include_dsb2:
        dsb = discover_dsb2()
        msg += f", DSB2={len(dsb)}"
        all_patients += dsb

    print(msg)

    output_dir = OUTPUT_DIR
    all_clips, frame_counts = process_patients(all_patients, output_dir)

    print(f"\n{'=' * 60}")
    print(f"Total clips: {len(all_clips)}")
    print(f"Frame counts: min={min(frame_counts)}, max={max(frame_counts)}, mean={np.mean(frame_counts):.1f}")
    short = sum(1 for f in frame_counts if f < 12)
    print(f"Clips with <12 frames: {short}")

    # Build splits.json
    splits = {"train": [], "val": [], "test": []}
    seen_patients = set()
    for c in all_clips:
        key = f"{c['dataset']}_{c['patient_id']}"
        if key not in seen_patients:
            seen_patients.add(key)
            splits[c["split"]].append(key)

    splits_path = DATA_ROOT / "processed" / "splits.json"
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"Splits: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")
    print(f"Saved: {splits_path}")

    # Build pretrain_manifest.csv (train split, dummy labels)
    pretrain_path = DATA_ROOT / "processed" / "pretrain_manifest.csv"
    with open(pretrain_path, "w") as f:
        for c in all_clips:
            if c["split"] == "train":
                f.write(f"{c['path']} 0\n")
    n_pretrain = sum(1 for c in all_clips if c["split"] == "train")
    print(f"Pretrain manifest: {n_pretrain} clips → {pretrain_path}")

    # Build probe_manifest.csv (ACDC only, with EF)
    probe_path = DATA_ROOT / "processed" / "probe_manifest.csv"
    with open(probe_path, "w") as f:
        f.write("path ef split patient_id slice_idx\n")
        for c in all_clips:
            if c["dataset"] == "acdc" and c["ef_label"] is not None:
                f.write(f"{c['path']} {c['ef_label']} {c['split']} {c['patient_id']} {c['slice_idx']}\n")
    n_probe = sum(1 for c in all_clips if c["dataset"] == "acdc" and c["ef_label"] is not None)
    print(f"Probe manifest: {n_probe} clips → {probe_path}")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sanity-check", action="store_true", help="Process 3 ACDC patients only")
    parser.add_argument("--all", action="store_true", help="Full conversion of all datasets")
    parser.add_argument("--include-sunnybrook", action="store_true", help="Include Sunnybrook DICOM dataset")
    parser.add_argument("--include-dsb2", action="store_true", help="Include DSB2 DICOM dataset")
    parser.add_argument("--dsb2-only", action="store_true", help="Convert DSB2 only (append to existing)")
    args = parser.parse_args()

    if args.sanity_check:
        sanity_check()
    elif args.dsb2_only:
        print("=" * 60)
        print("DSB2 CONVERSION ONLY")
        print("=" * 60)
        dsb = discover_dsb2()
        print(f"Discovered: DSB2={len(dsb)} patients")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        clips, frame_counts = process_patients(dsb, OUTPUT_DIR)
        print(f"\nDSB2: {len(clips)} clips, frames: min={min(frame_counts)}, max={max(frame_counts)}, mean={np.mean(frame_counts):.0f}")
        short = sum(1 for f in frame_counts if f < 12)
        print(f"Clips with <12 frames: {short}")
    elif args.all:
        full_conversion(include_sunnybrook=args.include_sunnybrook, include_dsb2=args.include_dsb2)
    else:
        parser.print_help()
