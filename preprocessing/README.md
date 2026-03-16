# Preprocessing Pipeline

End-to-end pipeline for preparing echocardiogram data (DICOM to model-ready MP4).

## Pipeline Overview

```
DICOM files
    │
    ▼
┌─────────────────────┐
│ 1. convert_dicom.py │  DICOM → MP4 (native resolution, 30 fps)
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ 2. apply_mask.py    │  Black out non-imaging regions (ECG, patient info, overlays)
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ 3. upload_s3.py     │  Parallel upload to S3 (optional — skip if local-only)
└─────────┬───────────┘
          │
          ▼
┌───────────────────────────┐
│ 4. classify_views.py      │  Run view + color classifiers (ConvNeXt-Small, 336px)
└─────────┬─────────────────┘
          │
          ▼
┌─────────────────────┐
│ 5. check_videos.py  │  QC: resolution, frame count, FPS, duration stats
└─────────────────────┘
```

After preprocessing, **task-specific label CSVs** are built by separate scripts that join
clip paths with clinical labels. These live in `experiments/`:

- `experiments/nature_medicine/uhn/build_probe_csvs.py` — UHN regression/classification
- `experiments/nature_medicine/uhn/build_viewfiltered_csvs.py` — view-filtered variants
- `experiments/nature_medicine/uhn/build_trajectory_csvs.py` — trajectory tasks
- `experiments/nature_medicine/mimic/build_probe_csvs.py` — MIMIC outcomes/biomarkers

Output CSVs go to `experiments/nature_medicine/{uhn,mimic}/probe_csvs/<task>/`.

## Quick Start (new dataset)

```bash
# 1. Convert DICOMs to MP4 (native resolution, no resize)
python preprocessing/convert_dicom.py \
    --input_dir /path/to/dicoms \
    --output_dir /path/to/mp4s \
    --workers 8

# 2. Apply sector mask (same mask used for UHN and MIMIC)
python preprocessing/apply_mask.py \
    --input_dir /path/to/mp4s \
    --output_dir /path/to/mp4s_masked

# 3. Upload to S3 (optional)
python preprocessing/upload_s3.py \
    --input_dir /path/to/mp4s_masked \
    --s3_prefix s3://bucket/dataset-name/ \
    --workers 16

# 4. Run view + color classification
python preprocessing/classify_views.py \
    --input_dir /path/to/mp4s_masked \
    --output_csv /path/to/classifications.csv \
    --view_checkpoint classifier/checkpoints/view_convnext_small_336px.pt \
    --color_checkpoint classifier/checkpoints/color_convnext_small_336px.pt \
    --num_frames 5 --batch_size 32

# 5. QC check
python preprocessing/check_videos.py --input_dir /path/to/mp4s_masked
```

## Resolution Strategy

**Store at native resolution. Resize at runtime.**

Each downstream pipeline handles its own resizing:
- **View/color classifiers**: resize to 336x336 internally (timm transforms)
- **JEPA probes**: resize to 224x224 internally (decord + torchvision)
- **Pretraining**: for repeated reads, pre-resizing to 224x224 saves I/O. Use `--resolution 224` in `convert_dicom.py`

## Scripts

### `convert_dicom.py`

Convert DICOM echocardiograms to MP4 video files.

```bash
# Native resolution (recommended)
python preprocessing/convert_dicom.py \
    --input_dir /path/to/dicoms --output_dir /path/to/mp4s --workers 8

# Fixed resolution (for pretraining I/O savings)
python preprocessing/convert_dicom.py \
    --input_dir /path/to/dicoms --output_dir /path/to/mp4s_224 --resolution 224 --workers 8
```

Handles grayscale, RGB, and multi-frame DICOMs. Normalizes pixel values to 0-255.
Preserves directory structure. Skip-if-exists for resumability.

### `apply_mask.py`

Black out non-imaging regions (ECG trace, patient info, overlays). The mask is defined
as proportional box coordinates from a 640x480 GE Voxel Cone reference layout and
scales to any input resolution. The same mask was applied to both UHN and MIMIC data.

```bash
python preprocessing/apply_mask.py \
    --input_dir /path/to/mp4s --output_dir /path/to/mp4s_masked

# Skip masking (just copy files)
python preprocessing/apply_mask.py \
    --input_dir /path/to/mp4s --output_dir /path/to/mp4s_masked --no_mask
```

### `classify_views.py`

Run ConvNeXt-Small view and color classifiers. Single-GPU, multi-frame voting.

```bash
python preprocessing/classify_views.py \
    --input_dir /path/to/mp4s_masked \
    --output_csv /path/to/classifications.csv \
    --view_checkpoint classifier/checkpoints/view_convnext_small_336px.pt \
    --color_checkpoint classifier/checkpoints/color_convnext_small_336px.pt \
    --num_frames 5 --batch_size 32
```

Output CSV columns: `path, view, view_confidence, color, color_confidence`

Classifiers were trained on UHN data (13 echo views, binary color Doppler).

### `upload_s3.py`

Parallel upload of local MP4 files to S3, preserving directory structure.

```bash
python preprocessing/upload_s3.py \
    --input_dir /path/to/mp4s_masked \
    --s3_prefix s3://bucket/dataset-name/ \
    --workers 16
```

### `check_videos.py`

Resolution, frame count, FPS, and duration statistics.

```bash
python preprocessing/check_videos.py --input_dir /path/to/mp4s --sample 100
```

## How the Data Was Prepared

### MIMIC-IV-Echo

1. **Download**: DICOMs from PhysioNet via `aws s3 sync`
2. **Convert**: DICOM → 224x224 MP4 (`data/scripts/convert_dicom.py`)
3. **Mask**: Same sector mask as UHN (`data/scripts/apply_masking.py`)
4. **Upload**: `aws s3 sync` to `s3://echodata25/mimic-echo-224px/`
5. **Index**: `data/csv/mimic_annotations_s3.csv` (525K clips)
6. **View/color classification**: not run on MIMIC

### UHN (18M)

1. **Source**: Native DICOM from Syngo/HeartLab (640x480 typical)
2. **Convert + Mask**: Institutional pipeline → MP4 at native resolution with sector masking
3. **Upload**: S3 prefixes `echo-study/`, `echo-study-1/`, `echo-study-2/`
4. **Index**: `indices/master_index_18M_cleaned.csv` (18M clips)
5. **Classify**: `classifier/inference_18m.py` → view + color per clip

## Legacy Scripts

Earlier versions of these scripts live in `data/scripts/` and are kept for reference.
The main ones are `convert_dicom.py` (DICOM → MP4), `apply_masking.py` (sector mask),
and `process_samsung.py` (all-in-one: unzip → convert → mask). The rest are one-off
SageMaker Processing Job scripts at various hardcoded resolutions.
