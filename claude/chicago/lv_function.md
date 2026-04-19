# UCMC LV Function External Validation — Inference Guide

Instructions for running frozen LV function regression probes on UCMC echocardiography data. All probes were trained at UHN using frozen depth=1 attentive cross-attention heads. No model fine-tuning occurs — inference only.

**Contact:** Alif Munim (alif.munim@uhn.ca)

---

## 1. Overview

You will run **4 frozen encoder models** on **4 LV function regression tasks**:

| Model | Architecture | Params | Encoder File | Embed Dim |
|-------|-------------|--------|-------------|-----------|
| **EchoJEPA-G** | ViT-Giant | 1,012M | `pt-280-an81.pt` (15.3 GB) | 1408 |
| **EchoJEPA-L-K** | ViT-Large | 304M | `vitl-kinetics-pt220-an55.pt` (4.8 GB) | 1024 |
| **EchoPrime** | MViT-v2-S | ~35M | `echo_prime_encoder.pt` (~200 MB) | 512 |
| **PanEcho** | ConvNeXt-T + Transformer | ~30M | Auto-downloads from hub | 768 |

| Task | Type | Unit | Description |
|------|------|------|-------------|
| **LVEF** | Regression | % | Left ventricular ejection fraction |
| **EDV** | Regression | mL | End-diastolic volume |
| **ESV** | Regression | mL | End-systolic volume |
| **Cardiac Output** | Regression | L/min | Cardiac output |

That's **16 inference runs** total (4 models x 4 tasks).

---

## 2. Shared Google Drive

Encoder checkpoints:

```
gdrive:echo_foundation/nature_medicine/chicago/
└── checkpoints/
    └── echojepa_g_uhn_pt280_an81/
        └── pt-280-an81.pt                        # EchoJEPA-G encoder (15.3 GB)

gdrive:echo_foundation/nature_medicine/checkpoints/
└── vitl-kinetics-pt220-an55.pt                   # EchoJEPA-L-K encoder (4.8 GB)
```

Probes:

```
gdrive:echo_foundation/nature_medicine/chicago/probes/lv_function/
├── echojepa-g_lvef/best.pt                       # 3.0 GB
├── echojepa-g_edv/best.pt                        # 3.0 GB
├── echojepa-g_esv/best.pt                        # 3.0 GB
├── echojepa-g_cardiac_output/best.pt             # 3.0 GB
├── echojepa-l-k_lvef/best.pt                     # 1.6 GB
├── echojepa-l-k_edv/best.pt                      # 1.6 GB
├── echojepa-l-k_esv/best.pt                      # 1.6 GB
├── echojepa-l-k_cardiac_output/best.pt           # 1.6 GB
├── echoprime_lvef/best.pt                        # 398 MB
├── echoprime_edv/best.pt                         # 398 MB
├── echoprime_esv/best.pt                         # 398 MB
├── echoprime_cardiac_output/best.pt              # 398 MB
├── panecho_lvef/best.pt                          # 893 MB
├── panecho_edv/best.pt                           # 893 MB
├── panecho_esv/best.pt                           # 893 MB
└── panecho_cardiac_output/best.pt                # 893 MB
```

---

## 3. Environment Setup

Same as the valve severity guide — see `valve_severity.md` Sections 3.1–3.4 for repo clone, conda environment, EchoPrime weights, and PanEcho setup.

---

## 4. Organize Checkpoints

```bash
mkdir -p checkpoints/encoders
mkdir -p checkpoints/probes/{lvef,edv,esv,cardiac_output}/{echojepa-g,echojepa-l-k,echoprime,panecho}

# EchoJEPA-G encoder (from chicago/checkpoints/)
cp <gdrive>/chicago/checkpoints/echojepa_g_uhn_pt280_an81/pt-280-an81.pt checkpoints/encoders/

# EchoJEPA-L-K encoder (from parent checkpoints/)
cp <gdrive>/checkpoints/vitl-kinetics-pt220-an55.pt checkpoints/encoders/

# Probes (download all 16 from GDrive chicago/probes/lv_function/)
cp <gdrive>/probes/lv_function/echojepa-g_lvef/best.pt             checkpoints/probes/lvef/echojepa-g/
cp <gdrive>/probes/lv_function/echojepa-g_edv/best.pt              checkpoints/probes/edv/echojepa-g/
cp <gdrive>/probes/lv_function/echojepa-g_esv/best.pt              checkpoints/probes/esv/echojepa-g/
cp <gdrive>/probes/lv_function/echojepa-g_cardiac_output/best.pt   checkpoints/probes/cardiac_output/echojepa-g/
cp <gdrive>/probes/lv_function/echojepa-l-k_lvef/best.pt           checkpoints/probes/lvef/echojepa-l-k/
cp <gdrive>/probes/lv_function/echojepa-l-k_edv/best.pt            checkpoints/probes/edv/echojepa-l-k/
cp <gdrive>/probes/lv_function/echojepa-l-k_esv/best.pt            checkpoints/probes/esv/echojepa-l-k/
cp <gdrive>/probes/lv_function/echojepa-l-k_cardiac_output/best.pt checkpoints/probes/cardiac_output/echojepa-l-k/
cp <gdrive>/probes/lv_function/echoprime_lvef/best.pt              checkpoints/probes/lvef/echoprime/
cp <gdrive>/probes/lv_function/echoprime_edv/best.pt               checkpoints/probes/edv/echoprime/
cp <gdrive>/probes/lv_function/echoprime_esv/best.pt               checkpoints/probes/esv/echoprime/
cp <gdrive>/probes/lv_function/echoprime_cardiac_output/best.pt    checkpoints/probes/cardiac_output/echoprime/
cp <gdrive>/probes/lv_function/panecho_lvef/best.pt                checkpoints/probes/lvef/panecho/
cp <gdrive>/probes/lv_function/panecho_edv/best.pt                 checkpoints/probes/edv/panecho/
cp <gdrive>/probes/lv_function/panecho_esv/best.pt                 checkpoints/probes/esv/panecho/
cp <gdrive>/probes/lv_function/panecho_cardiac_output/best.pt      checkpoints/probes/cardiac_output/panecho/
```

---

## 5. Prepare Your Data

### 5.1 Video format

Same requirements as the valve severity guide — see `valve_severity.md` Section 5.1. MP4, 224x224, 8 fps preferred, B-mode only, sector-masked.

### 5.2 CSV format

Create a **space-delimited** CSV with **no header**. Two columns: video path and float label.

```
/data/ucmc/lv/study001_clip01.mp4 62.5
/data/ucmc/lv/study001_clip02.mp4 62.5
/data/ucmc/lv/study002_clip01.mp4 35.2
```

**Important:**
- Delimiter is **space** (not comma, not tab)
- No header row
- Labels are **raw float values** (not normalized — the pipeline Z-score normalizes at runtime)
- Units: LVEF in %, EDV/ESV in mL, cardiac output in L/min
- Each row is one video clip. Multiple clips per study is fine (prediction averaging pools them)

### 5.3 View filtering

Unlike valve severity, these probes were trained on **all B-mode views** per study (not view-filtered). The UHN training data includes ~25 clips per study across all standard echo views.

**Recommended:** Include all available B-mode clips per study. The prediction averaging pipeline will pool across all clips.

**Minimum:** At minimum, include apical views (A4C, A2C, A3C) which provide the strongest signal for LV function. PLAX views also contribute.

**Exclude:** Color Doppler, spectral Doppler, and tissue Doppler clips. Use the view/color classifiers from the valve severity guide (`valve_severity.md` Section 5.4, Option B) to filter if needed.

### 5.4 Create CSVs for each task

```
data/csv/ucmc_lvef_test.csv
data/csv/ucmc_edv_test.csv
data/csv/ucmc_esv_test.csv
data/csv/ucmc_cardiac_output_test.csv
```

All four CSVs can use the same video files — only the labels differ. If a study has a label for LVEF but not cardiac output, include it in the LVEF CSV only.

### 5.5 Study-level prediction averaging

Same as the valve severity guide — see `valve_severity.md` Section 5.6. Organize video paths so each study's clips are in a directory named with the study ID.

---

## 6. Inference Configs

Create configs in `configs/inference/chicago/`. All LV function tasks use `task_type: regression`.

### 6.1 EchoJEPA-G configs

**`configs/inference/chicago/echojepa_g_lvef.yaml`:**
```yaml
app: vjepa
eval_name: video_classification_frozen
val_only: true
resume_checkpoint: true
tag: ucmc-echojepa-g-lvef
probe_checkpoint: checkpoints/probes/lvef/echojepa-g/best.pt

experiment:
  classifier:
    num_heads: 16
    num_probe_blocks: 1

  data:
    dataset_type: VideoDataset
    task_type: regression
    dataset_train: data/csv/ucmc_lvef_test.csv
    dataset_val: data/csv/ucmc_lvef_test.csv
    num_classes: 1
    resolution: 224
    frames_per_clip: 16
    frame_step: 2
    num_segments: 2
    num_views_per_segment: 1
    study_sampling: true

  optimization:
    batch_size: 4
    num_epochs: 1
    use_bfloat16: true
    multihead_kwargs:
    - {lr: 0.0, start_lr: 0.0, final_lr: 0.0, warmup: 0.0, weight_decay: 0.0, final_weight_decay: 0.0}

model_kwargs:
  checkpoint: checkpoints/encoders/pt-280-an81.pt
  module_name: evals.video_classification_frozen.modelcustom.vit_encoder_multiclip
  pretrain_kwargs:
    encoder:
      checkpoint_key: target_encoder
      model_name: vit_giant_xformers
      patch_size: 16
      tubelet_size: 2
      uniform_power: true
      use_rope: true
  wrapper_kwargs:
    max_frames: 128
    use_pos_embed: false
```

**`configs/inference/chicago/echojepa_g_edv.yaml`:**
Same as LVEF but change:
```yaml
tag: ucmc-echojepa-g-edv
probe_checkpoint: checkpoints/probes/edv/echojepa-g/best.pt
dataset_train: data/csv/ucmc_edv_test.csv
dataset_val: data/csv/ucmc_edv_test.csv
```

**`configs/inference/chicago/echojepa_g_esv.yaml`:**
Same as LVEF but change:
```yaml
tag: ucmc-echojepa-g-esv
probe_checkpoint: checkpoints/probes/esv/echojepa-g/best.pt
dataset_train: data/csv/ucmc_esv_test.csv
dataset_val: data/csv/ucmc_esv_test.csv
```

**`configs/inference/chicago/echojepa_g_cardiac_output.yaml`:**
Same as LVEF but change:
```yaml
tag: ucmc-echojepa-g-cardiac-output
probe_checkpoint: checkpoints/probes/cardiac_output/echojepa-g/best.pt
dataset_train: data/csv/ucmc_cardiac_output_test.csv
dataset_val: data/csv/ucmc_cardiac_output_test.csv
```

### 6.2 EchoJEPA-L-K configs

**`configs/inference/chicago/echojepa_lk_lvef.yaml`:**
```yaml
app: vjepa
eval_name: video_classification_frozen
val_only: true
resume_checkpoint: true
tag: ucmc-echojepa-lk-lvef
probe_checkpoint: checkpoints/probes/lvef/echojepa-l-k/best.pt

experiment:
  classifier:
    num_heads: 16
    num_probe_blocks: 1

  data:
    dataset_type: VideoDataset
    task_type: regression
    dataset_train: data/csv/ucmc_lvef_test.csv
    dataset_val: data/csv/ucmc_lvef_test.csv
    num_classes: 1
    resolution: 224
    frames_per_clip: 16
    frame_step: 2
    num_segments: 2
    num_views_per_segment: 1
    study_sampling: true

  optimization:
    batch_size: 4
    num_epochs: 1
    use_bfloat16: true
    multihead_kwargs:
    - {lr: 0.0, start_lr: 0.0, final_lr: 0.0, warmup: 0.0, weight_decay: 0.0, final_weight_decay: 0.0}

model_kwargs:
  checkpoint: checkpoints/encoders/vitl-kinetics-pt220-an55.pt
  module_name: evals.video_classification_frozen.modelcustom.vit_encoder_multiclip
  pretrain_kwargs:
    encoder:
      checkpoint_key: target_encoder
      model_name: vit_large
      patch_size: 16
      tubelet_size: 2
      uniform_power: true
      use_rope: true
  wrapper_kwargs:
    max_frames: 128
    use_pos_embed: false
```

For EDV, ESV, cardiac output: same pattern — change `tag`, `probe_checkpoint`, and data paths.

### 6.3 EchoPrime configs

**`configs/inference/chicago/echoprime_lvef.yaml`:**
```yaml
app: vjepa
eval_name: video_classification_frozen
val_only: true
resume_checkpoint: true
tag: ucmc-echoprime-lvef
probe_checkpoint: checkpoints/probes/lvef/echoprime/best.pt

experiment:
  classifier:
    num_heads: 16
    num_probe_blocks: 1

  data:
    dataset_type: VideoDataset
    task_type: regression
    dataset_train: data/csv/ucmc_lvef_test.csv
    dataset_val: data/csv/ucmc_lvef_test.csv
    num_classes: 1
    resolution: 224
    frames_per_clip: 16
    frame_step: 2
    num_segments: 2
    num_views_per_segment: 1
    study_sampling: true

  optimization:
    batch_size: 4
    num_epochs: 1
    use_bfloat16: true
    multihead_kwargs:
    - {lr: 0.0, start_lr: 0.0, final_lr: 0.0, warmup: 0.0, weight_decay: 0.0, final_weight_decay: 0.0}

model_kwargs:
  checkpoint: null
  module_name: evals.video_classification_frozen.modelcustom.echo_prime_encoder
  pretrain_kwargs: {}
  wrapper_kwargs:
    echo_prime_root: evals/video_classification_frozen/modelcustom/EchoPrime
    force_fp32: true
    bin_size: 50
```

For EDV, ESV, cardiac output: same pattern — change `tag`, `probe_checkpoint`, and data paths.

### 6.4 PanEcho configs

**`configs/inference/chicago/panecho_lvef.yaml`:**
```yaml
app: vjepa
eval_name: video_classification_frozen
val_only: true
resume_checkpoint: true
tag: ucmc-panecho-lvef
probe_checkpoint: checkpoints/probes/lvef/panecho/best.pt

experiment:
  classifier:
    num_heads: 16
    num_probe_blocks: 1

  data:
    dataset_type: VideoDataset
    task_type: regression
    dataset_train: data/csv/ucmc_lvef_test.csv
    dataset_val: data/csv/ucmc_lvef_test.csv
    num_classes: 1
    resolution: 224
    frames_per_clip: 16
    frame_step: 2
    num_segments: 2
    num_views_per_segment: 1
    study_sampling: true

  optimization:
    batch_size: 4
    num_epochs: 1
    use_bfloat16: true
    multihead_kwargs:
    - {lr: 0.0, start_lr: 0.0, final_lr: 0.0, warmup: 0.0, weight_decay: 0.0, final_weight_decay: 0.0}

model_kwargs:
  checkpoint: null
  module_name: evals.video_classification_frozen.modelcustom.panecho_encoder
  pretrain_kwargs: {}
  wrapper_kwargs: {}
```

For EDV, ESV, cardiac output: same pattern — change `tag`, `probe_checkpoint`, and data paths.

---

## 7. Running Inference

### 7.1 Single run

```bash
cd /path/to/EchoJEPA

python -m evals.main \
  --fname configs/inference/chicago/echojepa_g_lvef.yaml \
  --devices cuda:0 \
  --val_only
```

**GPU memory requirements:**
- EchoJEPA-G: ~25-30 GB (needs A100/H100/A6000 or similar)
- EchoJEPA-L-K: ~12-16 GB (can run on V100/A100/RTX 3090)
- EchoPrime: ~8-10 GB (can run on V100/RTX 3090)
- PanEcho: ~6-8 GB (can run on V100/RTX 3090)

### 7.2 Run all 16 experiments

```bash
#!/bin/bash
# run_all_ucmc_lv.sh

MODELS=("echojepa_g" "echojepa_lk" "echoprime" "panecho")
TASKS=("lvef" "edv" "esv" "cardiac_output")

for model in "${MODELS[@]}"; do
  for task in "${TASKS[@]}"; do
    echo "Running ${model} on ${task}..."
    python -m evals.main \
      --fname "configs/inference/chicago/${model}_${task}.yaml" \
      --devices cuda:0 \
      --val_only \
      2>&1 | tee "logs/ucmc_${model}_${task}.log"
  done
done
```

### 7.3 Running on a SLURM cluster

```bash
#!/bin/bash
#SBATCH --job-name=ucmc-lv
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/ucmc_%j.out

source activate echojepa
cd /path/to/EchoJEPA

python -m evals.main \
  --fname configs/inference/chicago/${MODEL}_${TASK}.yaml \
  --devices cuda:0 \
  --val_only
```

Submit with:
```bash
for model in echojepa_g echojepa_lk echoprime panecho; do
  for task in lvef edv esv cardiac_output; do
    sbatch --export=MODEL=$model,TASK=$task run_ucmc_lv.sbatch
  done
done
```

---

## 8. Output

### 8.1 What gets saved

Each run saves to `<folder>/<tag>/`. Add `folder: results/ucmc` to configs to control output location.

**Study-level predictions (main output):**
```
results/ucmc/ucmc-echojepa-g-lvef/study_predictions.csv
```

For regression tasks, the CSV contains:
```
study_id, label, prediction, n_clips
s00123, 62.5, 60.3, 4
s00456, 35.2, 37.8, 3
```

### 8.2 Metrics

The pipeline automatically computes and prints:
- **MAE** (mean absolute error)
- **R²** (coefficient of determination)
- **Pearson r** (correlation)

### 8.3 Collecting results

After all 16 runs, you should have:

```
results/ucmc/
├── ucmc-echojepa-g-lvef/study_predictions.csv
├── ucmc-echojepa-g-edv/study_predictions.csv
├── ucmc-echojepa-g-esv/study_predictions.csv
├── ucmc-echojepa-g-cardiac-output/study_predictions.csv
├── ucmc-echojepa-lk-lvef/study_predictions.csv
├── ucmc-echojepa-lk-edv/study_predictions.csv
├── ucmc-echojepa-lk-esv/study_predictions.csv
├── ucmc-echojepa-lk-cardiac-output/study_predictions.csv
├── ucmc-echoprime-lvef/study_predictions.csv
├── ucmc-echoprime-edv/study_predictions.csv
├── ucmc-echoprime-esv/study_predictions.csv
├── ucmc-echoprime-cardiac-output/study_predictions.csv
├── ucmc-panecho-lvef/study_predictions.csv
├── ucmc-panecho-edv/study_predictions.csv
├── ucmc-panecho-esv/study_predictions.csv
└── ucmc-panecho-cardiac-output/study_predictions.csv
```

Please share these CSVs (and logs) back via GDrive.

---

## 9. Troubleshooting

See `valve_severity.md` Section 9 for common issues (CUDA OOM, missing checkpoints, decord, PanEcho weights). The same solutions apply.

**Additional regression-specific note:** If you see predictions that are all near zero or all near the same value, the Z-score normalization may be failing. Ensure your labels are raw float values (not pre-normalized).

---

## 10. Expected Results

For reference, here are the **UHN internal validation** results (not UCMC — these are from the Toronto training site's held-out test set):

| Task | Metric | EchoJEPA-G | EchoJEPA-L-K | EchoPrime | PanEcho |
|------|--------|-----------|-------------|----------|---------|
| LVEF | R² | 0.778 | 0.702 | 0.681 | 0.665 |
| LVEF | MAE | 4.78 | 5.69 | 5.81 | 5.85 |
| EDV | R² | 0.774 | 0.560 | 0.425 | 0.554 |
| EDV | MAE | 20.95 | 27.71 | 29.62 | 26.63 |
| ESV | R² | 0.853 | 0.721 | 0.589 | 0.675 |
| ESV | MAE | 12.21 | 16.38 | 17.50 | 16.14 |
| Cardiac Output | R² | 0.335 | 0.185 | 0.143 | 0.179 |
| Cardiac Output | MAE | 1.10 | 1.22 | 1.24 | 1.23 |

Cross-institution performance may differ due to population differences, measurement conventions, and ultrasound equipment. A moderate drop is typical for external validation.
