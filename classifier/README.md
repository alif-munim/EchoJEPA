# Echo View / Color / Quality / Zoom Classifier

Image-based ConvNeXt/Swin classifiers for echocardiogram classification, plus distributed inference on the 18M dataset.

## Directory Structure

```
classifier/
├── train_convnext.py          # Distributed training (DDP, S3, video-based)
├── cooldown.py                # Low-LR fine-tuning (single-GPU, local, image-based)
├── inference_18m.py           # Unified distributed inference (view/color/quality/zoom)
├── s3_download_studies.py     # Batch S3 download with progress
├── s3_list_studies.py         # Build S3 URI list from study IDs
├── data_prep/                 # Data preparation pipeline
│   ├── data_prep_v3.py        # Initial label CSV generation
│   ├── make_patient_split.py  # Patient-disjoint train/val/test splits
│   ├── map_labels_to_mp4.py   # Image path → MP4 path mapping
│   ├── rewrite_mp4_paths_to_s3.py  # Local → S3 URI conversion
│   ├── sample_and_check_s3_mp4s.py # Verify S3 files exist + metadata
│   ├── clean_dataset.py       # Remove broken S3 files from splits
│   ├── make_view_labels_space_sep.py # Format to JEPA space-separated
│   └── make_stratified_subset.py    # Create balanced class subsets
├── mappings/                  # Canonical label→int JSON maps
│   ├── views.json             # 13 echo views (A2C, A3C, A4C, ...)
│   ├── color.json             # Color Doppler (Yes/No)
│   ├── quality.json           # Quality (keep/discard)
│   └── zoom.json              # Zoom (Full/Large/Small)
├── utils/
│   ├── fix_inference_checkpoint.py  # Remove module. prefix from DDP checkpoints
│   ├── convert_to_parquet.py        # CSV → Parquet conversion
│   ├── resize_dataset.py           # Local ffmpeg video resizing
│   └── resize_dataset_s3.py        # S3 download → resize → upload
├── archive/                   # Superseded scripts (kept for reference)
│   ├── convnext_training.py   # Old single-GPU local trainer (replaced by train_convnext.py)
│   └── download_convnext.py   # One-liner timm weight download
├── data/                      # Intermediate label CSVs
├── output/                    # Training checkpoints, metrics, inference results
└── txt/                       # Patient lists, split summaries
```

## First Frame Experiments (Image-Based)

These used the old `archive/convnext_training.py` (single-GPU, local filesystem). See `train_convnext.py` for the current distributed trainer.

| Run   | Model & Architecture | Key Strategy                                                     | Best Validation F1 | Final Test F1 | Status                                                                      |
| ----- | -------------------- | ---------------------------------------------------------------- | ------------------ | ------------- | --------------------------------------------------------------------------- |
| Run 1 | ConvNeXt-Base (Raw)  | Baseline, Low Reg.                                               | 0.839              | 0.801         | Failed. Severe overfitting (Train > Test).                                  |
| Run 2 | ConvNeXt-Small (v2)  | High Reg. (DropPath 0.4), Mixup, Grayscale, Batch 128/LR 2e-4.  | 0.863              | 0.835         | Success. Overfitting resolved; highest initial F1.                          |
| Run 3 | ConvNeXt-Small (v3)  | Cooldown (LR 1e-6, No Mixup, No Grayscale).                     | 0.844              | 0.824         | Polished. Retained robustness, confirmed ceiling.                           |
| Run 4 | ConvNeXt-Base (v3)   | Extreme Reg. (DropPath 0.5), Mixup, Grayscale, Batch 64/LR 1e-4.| 0.864              | 0.819         | Capacity Test. Highest Val F1, failed to generalize.                        |
| Run 5 | Swin-Base            | Architecture Switch (CNN to Transformer), Batch 32/LR 5e-5.     | 0.855              | 0.83          | New Ceiling. Highest Test F1, struggled with A5C.                           |

With EchoJEPA, test F1 reached 0.877 (224px).

## Data Preparation Pipeline

### 1. Compile studies and download from S3

```bash
python s3_list_studies.py \
  --labels labels_masked_inplace.csv \
  --es es.txt --es1 es1.txt --es2 es2.txt \
  --out selected_study_dirs.csv

python s3_download_studies.py \
  --csv selected_study_dirs_607.csv \
  --dest /path/to/uhn_studies_22k_607 \
  --log download.log --skip-if-exists
```

### 2. Create patient-disjoint splits

Uses `syngo-deid.dedup.csv` to map deidentified study IDs to patient IDs.

```bash
python data_prep/make_patient_split.py \
  --labels labels_masked_inplace.csv \
  --syngo syngo-deid.dedup.csv \
  --out labels_patient_split.csv \
  --seed 42 \
  --train 0.80 --val 0.10 --test 0.10 \
  --min-val 45 --min-test 45 --max-tries 500
```

### 3. Map to MP4 paths and S3 URIs

```bash
python data_prep/map_labels_to_mp4.py \
  --in labels_patient_split.csv \
  --root /cluster/projects/bwanggroup/echo_reports/uhn_studies_22k_607/ \
  --out labels_patient_split_mp4.csv

python data_prep/rewrite_mp4_paths_to_s3.py \
  --in labels_patient_split_mp4.csv \
  --out labels_patient_split_mp4_s3.csv \
  --s3-prefix s3://echodata25/results/uhn_studies_22k_607 \
  --root-marker uhn_studies_22k_607
```

### 4. Verify and clean

```bash
python data_prep/sample_and_check_s3_mp4s.py \
  --csv labels_patient_split_mp4_s3.csv \
  --n 100 --seed 0 --out-prefix check100

# Remove broken files from final splits
python data_prep/clean_dataset.py \
  data/csv/uhn_views_22k_train.csv \
  data/csv/uhn_views_22k_val.csv \
  data/csv/uhn_views_22k_test.csv
```

### 5. Convert to JEPA format (space-separated)

```bash
for SPLIT in train val test; do
  python data_prep/make_view_labels_space_sep.py \
    --in labels_patient_split_mp4_s3.csv \
    --out ../data/csv/uhn_views_22k_${SPLIT}_336px.csv \
    --mapping mappings/views.json \
    --split $SPLIT
done
```

Same pipeline applies for color, quality, and zoom tasks — just swap the label CSV and mapping file.

### 6. Data efficiency subsets

```bash
python data_prep/make_stratified_subset.py \
  --input ../data/csv/uhn_views_22k_train.csv \
  --out ../data/csv/uhn_views_22k_train_10percent.csv \
  --percent 10.0 --min 3 --seed 42
```

## Training (Distributed)

```bash
# View classification (13 classes)
torchrun --nproc_per_node=8 train_convnext.py \
  --mode view \
  --train_csv "$TRAIN_DATA" --val_csv "$VAL_DATA" --test_csv "$TEST_DATA" \
  --batch_size 128 --lr 2e-4 --epochs 100 --img_size 336 \
  --output_dir "./output/run5_convnext_small_336px"

# Color classification (binary)
torchrun --nproc_per_node=8 --master_port=29501 train_convnext.py \
  --mode color \
  --train_csv "$TRAIN_DATA" --val_csv "$VAL_DATA" --test_csv "$TEST_DATA" \
  --batch_size 128 --lr 2e-4 --epochs 100 --img_size 336 \
  --output_dir "./output/color_run1_convnext_small_336px"
```

### Cooldown (single-GPU, low-LR fine-tuning)

```bash
python cooldown.py \
  --results_dir "results/cooldown_final" \
  --start_weights "path/to/best_model.pth" \
  --lr 1e-6 --epochs 10
```

## Inference on 18M Dataset

The unified `inference_18m.py` replaces the old task-specific scripts.

```bash
# View inference
torchrun --nproc_per_node=8 inference_18m.py \
  --task view \
  --input_csv "../indices/master_index_18M_cleaned.csv" \
  --output_dir "./output/view_inference_18m" \
  --num_frames 5 --batch_size 128

# Color inference
torchrun --nproc_per_node=8 inference_18m.py \
  --task color \
  --input_csv "../indices/master_index_18M_cleaned.csv" \
  --output_dir "./output/color_inference_18m" \
  --num_frames 5 --batch_size 128

# Custom task with external mapping
torchrun --nproc_per_node=8 inference_18m.py \
  --task custom --mapping_json mappings/quality.json \
  --checkpoint path/to/quality_model.pt \
  --input_csv "../indices/master_index_18M_cleaned.csv" \
  --output_dir "./output/quality_inference_18m"
```

### Post-inference

```bash
# Merge rank outputs
cd output/view_inference_18m
awk 'FNR==1 && NR!=1{next;}{print}' predictions_rank*.csv > master_predictions.csv

# Convert to parquet
python utils/convert_to_parquet.py
```

### Fix DDP checkpoint prefixes

```bash
python utils/fix_inference_checkpoint.py \
  --input path/to/latest.pt \
  --output path/to/latest_fixed.pt
```

## JEPA Probe Evaluation

After data prep, train frozen probes with the main JEPA eval system:

```bash
# View classification
python -m evals.main \
    --fname configs/eval/vitg-384/view/echojepa_view_classification_336px.yaml \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7

# Color, quality, zoom — same pattern with task-specific configs
```

## Map predictions to class names

```python
import json, pandas as pd

with open("mappings/views.json") as f:
    m = {int(k): v for k, v in json.load(f).items()}

df = pd.read_csv("predictions/uhn22k_predictions.csv")
df["true_label_name"] = df["true_label"].map(m)
df["predicted_class_name"] = df["predicted_class"].map(m)
df.to_csv("predictions/uhn22k_predictions_with_names.csv", index=False)
```
