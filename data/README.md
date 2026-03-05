# data/

Data assets for EchoJEPA training, evaluation, and analysis.

## Directory Structure

```
data/
├── csv/                    # JEPA-format splits (space-separated: <s3_uri> <label>)
│                           # Referenced directly by configs/eval/ YAML files
│                           # 153 files covering all tasks (lvef, view, color, rvsp, etc.)
├── csv_pediatric/          # EchoNet Pediatric splits (train/val/test, local + S3 variants)
├── csv_rvsp/               # RVSP regression splits
├── labels/                 # Raw label CSVs before JEPA formatting
│   ├── master_video_list.csv          # Full 18M video inventory
│   ├── master_list_with_splits.csv    # 18M inventory with train/val/test splits
│   ├── mimic_annotations.csv         # MIMIC-IV echo annotations (43MB)
│   ├── pacemaker_labels.csv           # Pacemaker detection labels (15MB)
│   ├── mvr_labels.csv                 # Mitral valve regurgitation labels
│   └── a4c_LA_Dilated_Binary_*.csv    # LA dilation binary classification
├── scalers/                # Sklearn scalers for Z-score normalization
│   ├── ef_scaler.pkl                  # EF (ejection fraction)
│   ├── lvef_scaler.pkl                # LVEF (left ventricular EF)
│   ├── rvsp_scaler.pkl                # RVSP (right ventricular systolic pressure)
│   ├── tapse_scaler.pkl               # TAPSE
│   └── pediatric_ef_scaler.pkl        # Pediatric EF
├── parquet/                # Large structured data exports
│   ├── all_es_combined.parquet        # Combined echo studies (1.8GB)
│   ├── all_es_corrected.parquet       # Corrected echo studies (793MB)
│   └── all_es_corrected_p2.parquet    # Corrected echo studies part 2 (1.1GB)
├── training_logs/          # Pretraining and cooldown loss curves
│   ├── echojepa-l-pretrain.csv        # EchoJEPA-L initial pretrain
│   ├── echojepa-l-pretrain-cont*.csv  # Continued pretraining checkpoints
│   ├── echojepa-l-cooldown-*.csv      # Cooldown phase logs
│   └── log_r7.csv                     # Run 7 training log
├── notebooks/              # Data exploration and split generation
│   ├── lvef.ipynb                     # LVEF dataset preparation
│   ├── rvsp.ipynb                     # RVSP dataset preparation
│   ├── tapse.ipynb                    # TAPSE dataset preparation
│   ├── echonet.ipynb                  # EchoNet Dynamic processing
│   ├── mimic.ipynb                    # MIMIC-IV echo data exploration
│   ├── mvr_dataset.ipynb              # MVR dataset construction
│   ├── mvr_splits.ipynb               # MVR train/val/test splits
│   ├── tvr_dataset.ipynb              # TVR dataset construction
│   ├── tvr_splits.ipynb               # TVR train/val/test splits
│   ├── rvfx_splits.ipynb              # RV function splits
│   ├── rvfx_temporal.ipynb            # RV function temporal analysis
│   ├── pacemaker_splits.ipynb         # Pacemaker detection splits
│   ├── la_dilation_temporal.ipynb     # LA dilation temporal analysis
│   ├── deid_overlap.ipynb             # Deidentification overlap checks
│   ├── heartlab_link.ipynb            # HeartLab linkage
│   ├── identifiers.ipynb              # Study identifier mapping
│   ├── measurement_cleaning_v2.ipynb  # Measurement QC
│   └── results.ipynb                  # Results analysis
├── scripts/                # Data processing and augmentation
│   ├── apply_depth_attenuation.py     # Depth attenuation augmentation
│   ├── apply_gaussian_shadow.py       # Gaussian shadow augmentation
│   ├── apply_masking.py               # PHI masking for echo frames
│   ├── batch_depth_attenuation.py     # Batch depth attenuation
│   ├── batch_gaussian_shadow.py       # Batch Gaussian shadow
│   ├── convert_dicom.py               # DICOM → MP4 conversion
│   ├── preprocess.py                  # General preprocessing
│   ├── process.py                     # Video processing pipeline
│   ├── resize_script.py               # Video resizing
│   ├── lvef.py                        # LVEF data processing
│   ├── la_dilation_temporal.py        # LA dilation processing
│   ├── normalize_lvef_external.py     # LVEF Z-score normalization
│   ├── normalize_rvsp_external.py     # RVSP Z-score normalization
│   ├── check_stats.py                 # Dataset statistics
│   ├── plot_umap.py                   # UMAP visualization
│   ├── plot_umap_lvef.py              # UMAP visualization (LVEF)
│   ├── input_chunks/                  # Chunked input CSVs for batch processing
│   ├── processing_inputs/             # Processing input chunks
│   └── processing_inputs_split/       # Split processing chunks
├── echojepa_cls/           # Frozen probe classifier comparison results
│   └── ucmc22k-classifier-{echojepa,echoprime,panecho,videomae}-224px.csv
├── sample_data/            # Sample datasets for development/testing
│   ├── echonet/                       # EchoNet Dynamic subset
│   ├── echonetpediatric/              # EchoNet Pediatric subset
│   ├── mimic_p10_dcm/                 # MIMIC patient 10 DICOMs
│   ├── mimic_p10_masked/              # MIMIC patient 10 masked MP4s
│   ├── mimic_p10_mp4/                 # MIMIC patient 10 raw MP4s
│   └── top_confidence_examples/       # High-confidence view classification examples
├── aws/                    # AWS HeartLab/Syngo data exports
├── data/                   # Legacy nested data directory
├── Dockerfile              # Container for DICOM processing (python:3.10, ffmpeg, pydicom)
└── NOTICE.txt
```

## Key Conventions

**CSV formats** (consumed by eval configs):
- Classification: `<s3_uri> <int_label>` (space-separated)
- Regression: `<s3_uri> <z_score_float>` (space-separated, normalized via scalers)
- Multi-view: `<view1_uri> <view2_uri> ... <float_label>` (space-separated)

**Scalers**: Regression targets are Z-score normalized before training. The scaler `.pkl` files store the fitted `StandardScaler` for each target. Use `normalize_lvef_external.py` / `normalize_rvsp_external.py` to apply.

**Config references**: Eval YAML configs in `configs/eval/` reference files in `csv/` by path. Do not rename files in `csv/` without updating the corresponding configs.
