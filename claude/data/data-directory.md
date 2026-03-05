# Data Directory (`data/`)

Data assets for EchoJEPA training, evaluation, and analysis. This directory contains JEPA-format CSV splits, raw labels, scalers, parquet exports, notebooks, scripts, and sample data.

## Critical: CSV Split Files (`csv/`)

153 JEPA-format split files consumed directly by eval YAML configs in `configs/eval/`. **Do not rename files in `csv/` without updating the corresponding config YAML files.**

Formats (all space-separated):
- Classification: `<s3_uri> <int_label>`
- Regression: `<s3_uri> <z_score_float>` (Z-score normalized via scalers)
- Multi-view: `<view1_uri> <view2_uri> ... <float_label>`

Tasks covered: lvef, rvsp, tapse, view classification, color, quality, zoom, pacemaker, mvr, tvr, rvfx, la_dilation, pediatric EF, and more.

Additional CSV directories:
- `csv_pediatric/` — EchoNet Pediatric splits (train/val/test, local + S3 variants)
- `csv_rvsp/` — RVSP regression splits

## Scalers (`scalers/`)

Sklearn `StandardScaler` pickles for Z-score normalization of regression targets. Used at data prep time to transform raw values before writing to CSV splits.

| File | Target |
|------|--------|
| `ef_scaler.pkl` | Ejection fraction (general) |
| `lvef_scaler.pkl` | Left ventricular EF |
| `rvsp_scaler.pkl` | Right ventricular systolic pressure |
| `tapse_scaler.pkl` | TAPSE |
| `pediatric_ef_scaler.pkl` | Pediatric EF |

Apply with `scripts/normalize_lvef_external.py` or `scripts/normalize_rvsp_external.py`.

## Labels (`labels/`)

Raw label CSVs before JEPA formatting. These are the source-of-truth files from which `csv/` splits are derived.

- `master_video_list.csv` — full 18M video inventory
- `master_list_with_splits.csv` — 18M inventory with train/val/test splits
- `mimic_annotations.csv` — MIMIC-IV echo annotations (43MB)
- `pacemaker_labels.csv` — pacemaker detection labels (15MB)
- `mvr_labels.csv` — mitral valve regurgitation labels
- `a4c_LA_Dilated_Binary_*.csv` — LA dilation binary classification

## Parquet (`parquet/`)

Large structured data exports from echo study databases:
- `all_es_combined.parquet` (1.8GB) — combined echo studies
- `all_es_corrected.parquet` (793MB) — corrected echo studies
- `all_es_corrected_p2.parquet` (1.1GB) — corrected echo studies part 2

## Notebooks (`notebooks/`)

Jupyter notebooks for data exploration and split generation. Organized by task:

**Dataset preparation**: `lvef.ipynb`, `rvsp.ipynb`, `tapse.ipynb`, `echonet.ipynb`, `mimic.ipynb`
**Disease-specific splits**: `mvr_dataset.ipynb`, `mvr_splits.ipynb`, `tvr_dataset.ipynb`, `tvr_splits.ipynb`, `rvfx_splits.ipynb`, `rvfx_temporal.ipynb`, `pacemaker_splits.ipynb`, `la_dilation_temporal.ipynb`
**Data quality/linkage**: `deid_overlap.ipynb`, `heartlab_link.ipynb`, `identifiers.ipynb`, `measurement_cleaning_v2.ipynb`
**Analysis**: `results.ipynb`

## Scripts (`scripts/`)

Data processing and augmentation scripts:

**Augmentation**: `apply_depth_attenuation.py`, `apply_gaussian_shadow.py`, `apply_masking.py` (PHI masking), `batch_depth_attenuation.py`, `batch_gaussian_shadow.py`
**Preprocessing**: `convert_dicom.py` (DICOM → MP4), `preprocess.py`, `process.py`, `resize_script.py`
**Normalization**: `normalize_lvef_external.py`, `normalize_rvsp_external.py` (apply Z-score via scalers)
**Analysis**: `check_stats.py`, `plot_umap.py`, `plot_umap_lvef.py`, `lvef.py`, `la_dilation_temporal.py`
**Batch processing inputs**: `input_chunks/`, `processing_inputs/`, `processing_inputs_split/`

## Other Directories

- `training_logs/` — pretraining and cooldown loss curve CSVs (echojepa-l-pretrain, cooldown phases, run 7)
- `echojepa_cls/` — frozen probe classifier comparison results across models (echojepa, echoprime, panecho, videomae)
- `sample_data/` — sample datasets for development/testing (echonet, echonetpediatric, mimic_p10 variants, top_confidence_examples)
- `aws/` — AWS HeartLab/Syngo data exports
- `data/` — legacy nested data directory

## Docker

`Dockerfile` provides a container for DICOM processing (python:3.10, ffmpeg, pydicom).
