# Classifier Pipeline (`classifier/`)

Image-based ConvNeXt/Swin classifiers for echocardiogram view/color/quality/zoom classification, plus distributed inference on the 18M dataset. This is separate from the JEPA frozen probe system — it trains standalone supervised classifiers whose predictions are used as metadata labels for the 18M dataset.

## Purpose

The classifier pipeline serves two roles:
1. **Label the 18M dataset**: Run inference on all 18M echocardiograms to assign view, color Doppler, quality, and zoom labels. These labels are used to filter/stratify data for JEPA pretraining and evaluation.
2. **Baseline comparison**: ConvNeXt/Swin classifiers provide supervised baselines for echo view classification (best test F1: 0.835 ConvNeXt-Small, 0.830 Swin-Base). EchoJEPA frozen probes surpass these at 0.877 F1.

## Key Scripts

### Training
- **`train_convnext.py`** — distributed DDP training on S3-hosted videos. Supports `--mode view|color`. Uses `timm` ConvNeXt-Small, samples multiple frames per video, applies Mixup/Grayscale augmentation. Trains with `DistributedSampler` across 8 GPUs.
- **`cooldown.py`** — single-GPU low-LR fine-tuning (LR 1e-6). Local filesystem, image-based (PIL), no augmentation. Fundamentally different from `train_convnext.py` — not a simplified version, but a different training paradigm for polishing.

### Inference
- **`inference_18m.py`** — unified distributed inference script. Supports `--task view|color|quality|zoom|custom`. Built-in label maps via `BUILTIN_LABEL_MAPS` dict. Custom tasks use `--mapping_json`. Downloads videos from S3, samples frames, averages per-frame softmax probabilities, writes per-rank CSV outputs with incremental buffering.

### S3 Utilities
- **`s3_download_studies.py`** — batch download echo studies from S3 with progress tracking
- **`s3_list_studies.py`** — build S3 URI lists from study IDs

## Data Preparation Pipeline (`data_prep/`)

Sequential 6-step pipeline to go from raw annotations to JEPA-format splits:

1. **`data_prep_v3.py`** — initial label CSV generation from raw annotations
2. **`make_patient_split.py`** — patient-disjoint train/val/test splits using `syngo-deid.dedup.csv` for deidentified patient ID mapping. Ensures no patient appears in multiple splits.
3. **`map_labels_to_mp4.py`** — map image paths to MP4 video paths
4. **`rewrite_mp4_paths_to_s3.py`** — convert local paths to S3 URIs
5. **`sample_and_check_s3_mp4s.py`** + **`clean_dataset.py`** — verify S3 files exist, remove broken entries
6. **`make_view_labels_space_sep.py`** — convert to JEPA space-separated format (`<s3_uri> <int_label>`)

Additional: **`make_stratified_subset.py`** — create balanced class subsets for data efficiency experiments (e.g., 10% training data).

## Label Mappings (`mappings/`)

Canonical int→string JSON maps used by both training and inference:

| File | Classes |
|------|---------|
| `views.json` | 13 echo views: A2C, A3C, A4C, A5C, Exclude, PLAX, PSAX-AP, PSAX-AV, PSAX-MV, PSAX-PM, SSN, Subcostal, TEE |
| `color.json` | 2 classes: Yes, No (Color Doppler) |
| `quality.json` | 2 classes: keep, discard |
| `zoom.json` | 3 classes: Full, Large, Small |

## Utilities (`utils/`)

- `fix_inference_checkpoint.py` — remove `module.` prefix from DDP checkpoints for single-GPU loading
- `convert_to_parquet.py` — CSV → Parquet conversion for inference results
- `resize_dataset.py` — local ffmpeg video resizing
- `resize_dataset_s3.py` — S3 download → resize → upload

## Archive (`archive/`)

Superseded scripts kept for reference:
- `convnext_training.py` — old single-GPU local trainer, replaced by `train_convnext.py`
- `download_convnext.py` — one-liner timm weight download

## Experiment History

| Run | Model | Test F1 | Notes |
|-----|-------|---------|-------|
| 1 | ConvNeXt-Base | 0.801 | Severe overfitting |
| 2 | ConvNeXt-Small | 0.835 | Best CNN result |
| 3 | ConvNeXt-Small (cooldown) | 0.824 | Polished, confirmed ceiling |
| 4 | ConvNeXt-Base | 0.819 | High val F1 but failed to generalize |
| 5 | Swin-Base | 0.830 | Best transformer baseline |
| — | EchoJEPA (frozen probe) | 0.877 | Surpasses all supervised baselines |
