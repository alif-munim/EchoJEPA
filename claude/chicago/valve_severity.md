# UCMC Valve Severity External Validation — Inference Guide

Instructions for running frozen valve severity probes on UCMC echocardiography data. All probes were trained at UHN on ~130K-175K studies using frozen depth=1 attentive cross-attention heads. No model fine-tuning occurs — inference only.

**Contact:** Alif Munim (alif.munim@uhn.ca)

---

## 1. Overview

You will run **3 frozen encoder models** on **4 valve severity tasks**:

| Model | Architecture | Params | Encoder File | Embed Dim |
|-------|-------------|--------|-------------|-----------|
| **EchoJEPA-G** | ViT-Giant | 1,012M | `pt-280-an81.pt` (15.3 GB) | 1408 |
| **EchoJEPA-L-K** | ViT-Large | 304M | `vitl-kinetics-pt220-an55.pt` (4.8 GB) | 1024 |
| **EchoPrime** | MViT-v2-S | ~35M | `echo_prime_encoder.pt` (~200 MB) | 512 |
| **PanEcho** | ConvNeXt-T + Transformer | ~30M | Auto-downloads from hub | 768 |

| Task | Classes | Class Labels |
|------|---------|-------------|
| **MR Severity** | 5 | 0=none, 1=trace, 2=mild, 3=moderate, 4=severe |
| **TR Severity** | 5 | 0=none, 1=trivial/trace, 2=mild, 3=moderate, 4=severe |
| **AR Severity** | 5 | 0=none, 1=trace, 2=mild, 3=moderate, 4=severe |
| **AS Severity** | 4 | 0=none/sclerosis, 1=mild, 2=moderate, 3=severe |

That's **16 inference runs** total (4 models x 4 tasks).

---

## 2. Shared Google Drive

All checkpoints and probes are at:

```
gdrive:echo_foundation/nature_medicine/chicago/
├── checkpoints/
│   └── echojepa_g_uhn_pt280_an81/
│       └── pt-280-an81.pt                    # EchoJEPA-G encoder (15.3 GB)
└── probes/
    └── valve_severity/
        ├── echojepa-g_mr_severity/best.pt        # 3.0 GB
        ├── echojepa-g_tr_severity/best.pt        # 3.0 GB
        ├── echojepa-g_as_severity/best.pt        # 3.0 GB
        ├── echojepa-g_ar_severity/best.pt        # 3.0 GB
        ├── echojepa-l-k_mr_severity/best.pt      # 1.6 GB
        ├── echojepa-l-k_tr_severity/best.pt      # 1.6 GB
        ├── echojepa-l-k_as_severity/best.pt      # 1.6 GB
        ├── echojepa-l-k_ar_severity/best.pt      # 1.6 GB
        ├── echoprime_mr_severity/best.pt         # 398 MB
        ├── echoprime_tr_severity/best.pt         # 398 MB
        ├── echoprime_as_severity/best.pt         # 398 MB
        ├── echoprime_ar_severity/best.pt         # 398 MB
        ├── panecho_mr_severity/best.pt           # 894 MB
        ├── panecho_tr_severity/best.pt           # 894 MB
        ├── panecho_as_severity/best.pt           # 893 MB
        └── panecho_ar_severity/best.pt           # 894 MB

# EchoJEPA-L-K encoder is in the parent checkpoints directory:
gdrive:echo_foundation/nature_medicine/checkpoints/
└── vitl-kinetics-pt220-an55.pt               # EchoJEPA-L-K encoder (4.8 GB)
```

---

## 3. Environment Setup

### 3.1 Clone the repo

```bash
git clone <repo-url> EchoJEPA
cd EchoJEPA
```

### 3.2 Create conda environment

```bash
conda create -n echojepa python=3.12 -y
conda activate echojepa
pip install -e .
```

Key dependencies (installed by `pip install -e .`):
- `torch>=2.0`, `torchvision`
- `decord` (video decoding)
- `timm`, `transformers`
- `pandas`, `numpy`, `scikit-learn`
- `pyyaml`, `boto3`

If `decord` fails to install via pip, try:
```bash
pip install decord
# or: conda install -c conda-forge decord
```

### 3.3 Download EchoPrime encoder weights

EchoPrime requires its encoder weights downloaded separately:

```bash
cd evals/video_classification_frozen/modelcustom/EchoPrime
mkdir -p model_data/weights
wget https://github.com/echonet/EchoPrime/releases/download/v1.0.0/model_data.zip
unzip model_data.zip
cd ../../../..
```

This creates `model_data/weights/echo_prime_encoder.pt` inside the EchoPrime directory.

### 3.4 PanEcho

PanEcho source code is already included in the repo at:
```
evals/video_classification_frozen/modelcustom/PanEcho/
```

It downloads its own weights automatically on first run (requires internet). No manual setup needed.

---

## 4. Organize Checkpoints

Create a local checkpoint directory and download from GDrive:

```bash
mkdir -p checkpoints/encoders
mkdir -p checkpoints/probes/{mr_severity,tr_severity,as_severity,ar_severity}/{echojepa-g,echojepa-l-k,echoprime,panecho}

# EchoJEPA-G encoder (from chicago/checkpoints/)
cp <gdrive>/chicago/checkpoints/echojepa_g_uhn_pt280_an81/pt-280-an81.pt checkpoints/encoders/

# EchoJEPA-L-K encoder (from parent checkpoints/)
cp <gdrive>/checkpoints/vitl-kinetics-pt220-an55.pt checkpoints/encoders/

# Probes (download all 16 from GDrive chicago/probes/)
cp <gdrive>/probes/echojepa-g_mr_severity/best.pt    checkpoints/probes/mr_severity/echojepa-g/
cp <gdrive>/probes/echojepa-g_tr_severity/best.pt    checkpoints/probes/tr_severity/echojepa-g/
cp <gdrive>/probes/echojepa-g_as_severity/best.pt    checkpoints/probes/as_severity/echojepa-g/
cp <gdrive>/probes/echojepa-g_ar_severity/best.pt    checkpoints/probes/ar_severity/echojepa-g/
cp <gdrive>/probes/echojepa-l-k_mr_severity/best.pt  checkpoints/probes/mr_severity/echojepa-l-k/
cp <gdrive>/probes/echojepa-l-k_tr_severity/best.pt  checkpoints/probes/tr_severity/echojepa-l-k/
cp <gdrive>/probes/echojepa-l-k_as_severity/best.pt  checkpoints/probes/as_severity/echojepa-l-k/
cp <gdrive>/probes/echojepa-l-k_ar_severity/best.pt  checkpoints/probes/ar_severity/echojepa-l-k/
cp <gdrive>/probes/echoprime_mr_severity/best.pt     checkpoints/probes/mr_severity/echoprime/
cp <gdrive>/probes/echoprime_tr_severity/best.pt     checkpoints/probes/tr_severity/echoprime/
cp <gdrive>/probes/echoprime_as_severity/best.pt     checkpoints/probes/as_severity/echoprime/
cp <gdrive>/probes/echoprime_ar_severity/best.pt     checkpoints/probes/ar_severity/echoprime/
cp <gdrive>/probes/panecho_mr_severity/best.pt       checkpoints/probes/mr_severity/panecho/
cp <gdrive>/probes/panecho_tr_severity/best.pt       checkpoints/probes/tr_severity/panecho/
cp <gdrive>/probes/panecho_as_severity/best.pt       checkpoints/probes/as_severity/panecho/
cp <gdrive>/probes/panecho_ar_severity/best.pt       checkpoints/probes/ar_severity/panecho/
```

---

## 5. Prepare Your Data

### 5.1 Video format

- **Format:** MP4
- **Resolution:** 224x224 pixels (the pipeline will resize if different, but 224px is ideal)
- **Frame rate:** 8 fps preferred (matches training). Higher fps is fine — the pipeline samples frames with `frame_step: 2`
- **Content:** B-mode only. Exclude color Doppler, spectral Doppler, and tissue Doppler clips. If you have mixed views, only include B-mode grayscale clips
- **Masking:** If your videos have ECG traces, patient info overlays, or non-sector regions, apply sector masking (black out everything outside the ultrasound sector) before running inference. This matches the UHN preprocessing

### 5.2 CSV format

Create a **space-delimited** CSV with **no header**. Two columns: video path and integer label.

```
/data/ucmc/mr/study001_clip01.mp4 0
/data/ucmc/mr/study001_clip02.mp4 0
/data/ucmc/mr/study002_clip01.mp4 2
/data/ucmc/mr/study003_clip01.mp4 4
```

**Important:**
- Delimiter is **space** (not comma, not tab)
- No header row
- Labels are **0-indexed integers** matching the class tables in Section 1
- Each row is one video clip. Multiple clips per study is fine (and encouraged — prediction averaging pools them)
- Local paths only (no S3 paths unless you have AWS configured)

### 5.3 Label mapping for UCMC data

Your labels already match the UHN class scheme:

| Task | UCMC Classes | Matches UHN? |
|------|-------------|-------------|
| MR | 0=none, 1=trace, 2=mild, 3=moderate, 4=severe | Yes (5-class) |
| TR | 0=none, 1=trivial/trace, 2=mild, 3=moderate, 4=severe | Yes (5-class) |
| AR | 0=none, 1=trace, 2=mild, 3=moderate, 4=severe | Yes (5-class) |
| AS | 0=none/sclerosis, 1=mild, 2=moderate, 3=severe | Yes (4-class) |

No remapping needed.

### 5.4 View and color filtering (CRITICAL)

The probes were trained on **view-filtered, B-mode only** clips. You must filter your CSVs to include only task-relevant views and exclude all Doppler clips. Using unfiltered data will degrade performance.

**Required view filters per task:**

| Task | Allowed Views | Color Filter |
|------|--------------|-------------|
| MR Severity | A4C, A2C, A3C, PLAX | B-mode only (no color/spectral/tissue Doppler) |
| TR Severity | A4C, Subcostal, PLAX | B-mode only |
| AR Severity | A4C, A2C, A3C, PLAX | B-mode only |
| AS Severity | PLAX, PSAX-AV, A3C | B-mode only |

**How to filter your data:**

**Option A: Use DICOM metadata (preferred if available).** If your DICOM headers or reporting system has view labels, filter directly. Map your institution's view names to the categories above.

**Option B: Run our view + color classifiers.** We provide ConvNeXt-Small classifiers trained on 607 annotated UHN studies (27K clips). These classify each video into 13 echo views (A2C, A3C, A4C, A5C, PLAX, PSAX-AV, PSAX-PM, PSAX-MV, PSAX-AP, Subcostal, SSN, TEE, Exclude) and predict whether color Doppler is present (binary).

The classifier checkpoints are on GDrive:
```
gdrive:echo_foundation/nature_medicine/chicago/classifiers/
├── convnext_view_finetuned_chicago_best.pt   # 13-class view classifier, finetuned on UCMC data (189 MB) ← USE THIS
├── view_convnext_small_336px.pt              # 13-class view classifier, UHN-only (567 MB)
└── color_convnext_small_336px.pt             # Binary color classifier (567 MB)
```

**Use `convnext_view_finetuned_chicago_best.pt` for view classification** — it was finetuned on Chicago/UCMC data and will transfer better to your echo studies than the UHN-only model. The color classifier is the same for both.

Download and place them locally:
```bash
mkdir -p classifier/checkpoints
# Copy from GDrive to:
# classifier/checkpoints/convnext_view_finetuned_chicago_best.pt   (view — recommended)
# classifier/checkpoints/color_convnext_small_336px.pt             (color)
```

To run classification on your videos:
```bash
python preprocessing/classify_views.py \
    --input_dir /data/ucmc/mp4s \
    --output_csv /data/ucmc/classifications.csv \
    --view_checkpoint classifier/checkpoints/convnext_view_finetuned_chicago_best.pt \
    --color_checkpoint classifier/checkpoints/color_convnext_small_336px.pt \
    --num_frames 5 --batch_size 32
```

This produces a CSV with columns:
```
path, view, view_confidence, color, color_confidence
```

Then filter for each task. Example for MR severity:
```python
import pandas as pd

clf = pd.read_csv("/data/ucmc/classifications.csv")
labels = pd.read_csv("/data/ucmc/mr_labels.csv")  # your labels

# B-mode only (color = "No") + task-relevant views
mr_allowed = ["A4C", "A2C", "A3C", "PLAX"]
mr_clips = clf[(clf["color"] == "No") & (clf["view"].isin(mr_allowed))]

# Join with labels and write space-delimited CSV (no header)
merged = mr_clips.merge(labels, on="path")
with open("data/csv/ucmc_mr_severity_test.csv", "w") as f:
    for _, row in merged.iterrows():
        f.write(f"{row['path']} {row['label']}\n")
```

Repeat for each task with the appropriate view list.

**Option C: No filtering (not recommended).** If you cannot classify views, you can include all clips. Prediction averaging will dilute irrelevant views, but expect 3-8 pp lower AUROC compared to filtered results.

### 5.5 Create CSVs for each task

After filtering, create one CSV per task:
```
data/csv/ucmc_mr_severity_test.csv    # A4C, A2C, A3C, PLAX — B-mode only
data/csv/ucmc_tr_severity_test.csv    # A4C, Subcostal, PLAX — B-mode only
data/csv/ucmc_ar_severity_test.csv    # A4C, A2C, A3C, PLAX — B-mode only
data/csv/ucmc_as_severity_test.csv    # PLAX, PSAX-AV, A3C — B-mode only
```

### 5.6 Study-level prediction averaging (recommended)

To get study-level predictions (averaging across all clips per study), organize your video paths so each study's clips are in a directory named with the study ID:

```
/data/ucmc/studies/s00123/clip_0.mp4
/data/ucmc/studies/s00123/clip_1.mp4
/data/ucmc/studies/s00456/clip_0.mp4
```

The pipeline extracts study IDs from the path using the pattern `/s<digits>/`. If your paths don't follow this pattern, it falls back to using the parent directory name as the study ID.

---

## 6. Inference Configs

Below are the 12 config files you need. Create them in `configs/inference/chicago/`.

### 6.1 EchoJEPA-G configs

These 4 configs share the same encoder but point to different probe checkpoints and class counts.

**`configs/inference/chicago/echojepa_g_mr_severity.yaml`:**
```yaml
app: vjepa
eval_name: video_classification_frozen
val_only: true
resume_checkpoint: true
tag: ucmc-echojepa-g-mr-severity
probe_checkpoint: checkpoints/probes/mr_severity/echojepa-g/best.pt

experiment:
  classifier:
    num_heads: 16
    num_probe_blocks: 1

  data:
    dataset_type: VideoDataset
    dataset_train: data/csv/ucmc_mr_severity_test.csv
    dataset_val: data/csv/ucmc_mr_severity_test.csv
    num_classes: 5
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

**`configs/inference/chicago/echojepa_g_tr_severity.yaml`:**
Same as MR but change:
```yaml
tag: ucmc-echojepa-g-tr-severity
probe_checkpoint: checkpoints/probes/tr_severity/echojepa-g/best.pt
# data paths:
dataset_train: data/csv/ucmc_tr_severity_test.csv
dataset_val: data/csv/ucmc_tr_severity_test.csv
num_classes: 5
```

**`configs/inference/chicago/echojepa_g_ar_severity.yaml`:**
Same as MR but change:
```yaml
tag: ucmc-echojepa-g-ar-severity
probe_checkpoint: checkpoints/probes/ar_severity/echojepa-g/best.pt
dataset_train: data/csv/ucmc_ar_severity_test.csv
dataset_val: data/csv/ucmc_ar_severity_test.csv
num_classes: 5
```

**`configs/inference/chicago/echojepa_g_as_severity.yaml`:**
Same as MR but change:
```yaml
tag: ucmc-echojepa-g-as-severity
probe_checkpoint: checkpoints/probes/as_severity/echojepa-g/best.pt
dataset_train: data/csv/ucmc_as_severity_test.csv
dataset_val: data/csv/ucmc_as_severity_test.csv
num_classes: 4                 # AS is 4-class (none/sclerosis merged)
```

### 6.2 EchoJEPA-L-K configs

EchoJEPA-L-K is a ViT-Large model initialized from Kinetics-pretrained V-JEPA weights, then continued on MIMIC-IV-Echo. It uses the same config structure as EchoJEPA-G but with a different encoder and model name.

**`configs/inference/chicago/echojepa_lk_mr_severity.yaml`:**
```yaml
app: vjepa
eval_name: video_classification_frozen
val_only: true
resume_checkpoint: true
tag: ucmc-echojepa-lk-mr-severity
probe_checkpoint: checkpoints/probes/mr_severity/echojepa-l-k/best.pt

experiment:
  classifier:
    num_heads: 16
    num_probe_blocks: 1

  data:
    dataset_type: VideoDataset
    dataset_train: data/csv/ucmc_mr_severity_test.csv
    dataset_val: data/csv/ucmc_mr_severity_test.csv
    num_classes: 5
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

**`configs/inference/chicago/echojepa_lk_tr_severity.yaml`:**
Same as MR but change:
```yaml
tag: ucmc-echojepa-lk-tr-severity
probe_checkpoint: checkpoints/probes/tr_severity/echojepa-l-k/best.pt
dataset_train: data/csv/ucmc_tr_severity_test.csv
dataset_val: data/csv/ucmc_tr_severity_test.csv
num_classes: 5
```

**`configs/inference/chicago/echojepa_lk_ar_severity.yaml`:**
Same as MR but change:
```yaml
tag: ucmc-echojepa-lk-ar-severity
probe_checkpoint: checkpoints/probes/ar_severity/echojepa-l-k/best.pt
dataset_train: data/csv/ucmc_ar_severity_test.csv
dataset_val: data/csv/ucmc_ar_severity_test.csv
num_classes: 5
```

**`configs/inference/chicago/echojepa_lk_as_severity.yaml`:**
Same as MR but change:
```yaml
tag: ucmc-echojepa-lk-as-severity
probe_checkpoint: checkpoints/probes/as_severity/echojepa-l-k/best.pt
dataset_train: data/csv/ucmc_as_severity_test.csv
dataset_val: data/csv/ucmc_as_severity_test.csv
num_classes: 4                 # AS is 4-class (none/sclerosis merged)
```

### 6.3 EchoPrime configs

**`configs/inference/chicago/echoprime_mr_severity.yaml`:**
```yaml
app: vjepa
eval_name: video_classification_frozen
val_only: true
resume_checkpoint: true
tag: ucmc-echoprime-mr-severity
probe_checkpoint: checkpoints/probes/mr_severity/echoprime/best.pt

experiment:
  classifier:
    num_heads: 16
    num_probe_blocks: 1

  data:
    dataset_type: VideoDataset
    dataset_train: data/csv/ucmc_mr_severity_test.csv
    dataset_val: data/csv/ucmc_mr_severity_test.csv
    num_classes: 5
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

For TR, AR, AS: same pattern — change `tag`, `probe_checkpoint`, data paths, and `num_classes` (4 for AS).

### 6.4 PanEcho configs

**`configs/inference/chicago/panecho_mr_severity.yaml`:**
```yaml
app: vjepa
eval_name: video_classification_frozen
val_only: true
resume_checkpoint: true
tag: ucmc-panecho-mr-severity
probe_checkpoint: checkpoints/probes/mr_severity/panecho/best.pt

experiment:
  classifier:
    num_heads: 16
    num_probe_blocks: 1

  data:
    dataset_type: VideoDataset
    dataset_train: data/csv/ucmc_mr_severity_test.csv
    dataset_val: data/csv/ucmc_mr_severity_test.csv
    num_classes: 5
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

For TR, AR, AS: same pattern — change `tag`, `probe_checkpoint`, data paths, and `num_classes` (4 for AS).

---

## 7. Running Inference

### 7.1 Single run

```bash
cd /path/to/EchoJEPA

python -m evals.main \
  --fname configs/inference/chicago/echojepa_g_mr_severity.yaml \
  --devices cuda:0 \
  --val_only
```

**GPU memory requirements:**
- EchoJEPA-G: ~25-30 GB (needs A100/H100/A6000 or similar)
- EchoJEPA-L-K: ~12-16 GB (can run on V100/A100/RTX 3090)
- EchoPrime: ~8-10 GB (can run on V100/RTX 3090)
- PanEcho: ~6-8 GB (can run on V100/RTX 3090)

If you have multiple GPUs, you can parallelize within a single run:
```bash
python -m evals.main \
  --fname configs/inference/chicago/echojepa_g_mr_severity.yaml \
  --devices cuda:0 cuda:1 cuda:2 cuda:3 \
  --val_only
```

### 7.2 Run all 16 experiments

```bash
#!/bin/bash
# run_all_ucmc.sh

MODELS=("echojepa_g" "echojepa_lk" "echoprime" "panecho")
TASKS=("mr_severity" "tr_severity" "ar_severity" "as_severity")

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

If your institution uses SLURM:

```bash
#!/bin/bash
#SBATCH --job-name=ucmc-valve
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
  for task in mr_severity tr_severity ar_severity as_severity; do
    sbatch --export=MODEL=$model,TASK=$task run_ucmc.sbatch
  done
done
```

---

## 8. Output

### 8.1 What gets saved

Each run saves outputs to `<folder>/<tag>/` where `folder` is defined in the config (defaults to current directory if not set). You can add a `folder:` field to each config to control this:

```yaml
folder: results/ucmc
```

**Study-level predictions (main output):**
```
results/ucmc/ucmc-echojepa-g-mr-severity/study_predictions.csv
```

Contains:
```
study_id, label, predicted_class, n_clips, prob_class_0, prob_class_1, ..., prob_class_N
s00123, 2, 2, 4, 0.02, 0.05, 0.85, 0.06, 0.02
s00456, 0, 0, 3, 0.91, 0.04, 0.03, 0.01, 0.01
```

**Clip-level outputs:**
```
results/ucmc/ucmc-echojepa-g-mr-severity/clip_outputs.npz
```

Contains raw per-clip probabilities and features (useful for analysis).

### 8.2 Metrics

The pipeline automatically computes and prints:
- **AUROC** (one-vs-rest, macro-averaged)
- **Balanced accuracy**
- **Per-class accuracy**
- **Confusion matrix**

These are printed to stdout and saved in the log.

### 8.3 Collecting results

After all 16 runs, you should have:

```
results/ucmc/
├── ucmc-echojepa-g-mr-severity/study_predictions.csv
├── ucmc-echojepa-g-tr-severity/study_predictions.csv
├── ucmc-echojepa-g-ar-severity/study_predictions.csv
├── ucmc-echojepa-g-as-severity/study_predictions.csv
├── ucmc-echojepa-lk-mr-severity/study_predictions.csv
├── ucmc-echojepa-lk-tr-severity/study_predictions.csv
├── ucmc-echojepa-lk-ar-severity/study_predictions.csv
├── ucmc-echojepa-lk-as-severity/study_predictions.csv
├── ucmc-echoprime-mr-severity/study_predictions.csv
├── ucmc-echoprime-tr-severity/study_predictions.csv
├── ucmc-echoprime-ar-severity/study_predictions.csv
├── ucmc-echoprime-as-severity/study_predictions.csv
├── ucmc-panecho-mr-severity/study_predictions.csv
├── ucmc-panecho-tr-severity/study_predictions.csv
├── ucmc-panecho-ar-severity/study_predictions.csv
└── ucmc-panecho-as-severity/study_predictions.csv
```

Please share these CSVs (and logs) back via GDrive.

---

## 9. Troubleshooting

### "CUDA out of memory"

Reduce batch size in the config:
```yaml
batch_size: 2    # or even 1
```

For EchoJEPA-G, you need at least 24 GB VRAM. If you only have 16 GB GPUs, you can still run EchoPrime and PanEcho.

### "probe_checkpoint not found"

Ensure the path in the YAML is correct and the file exists. Use absolute paths if unsure:
```yaml
probe_checkpoint: /absolute/path/to/best.pt
```

### "No module named 'decord'"

```bash
pip install decord
```

If that fails on your system, try building from source or using conda.

### EchoPrime "model_data not found"

You need to download the EchoPrime model weights — see Section 3.3. The expected path is:
```
evals/video_classification_frozen/modelcustom/EchoPrime/model_data/weights/echo_prime_encoder.pt
```

### PanEcho "PanEcho source not found"

Ensure the PanEcho directory exists at:
```
evals/video_classification_frozen/modelcustom/PanEcho/
```

This should already be in the repo. If missing, the PanEcho GitHub repo needs to be cloned there.

### PanEcho weight download fails (no internet)

If your compute nodes have no internet access, pre-download PanEcho weights on a machine that does:

```python
import torch
model = torch.hub.load('echonet/panecho', 'PanEcho', pretrained=True)
# This caches to ~/.cache/torch/hub/
```

Then copy `~/.cache/torch/hub/` to the compute node.

### Video loading errors

- Ensure videos are valid MP4 files (not DICOM)
- Check that `decord` can read them: `python -c "import decord; vr = decord.VideoReader('path/to/video.mp4'); print(len(vr))"`
- Corrupt or truncated MP4s will cause errors — remove them from the CSV

### num_classes mismatch error

If the probe was trained with a different number of classes than what's in your config, you'll get a shape mismatch error. Double-check:
- MR, TR, AR: `num_classes: 5`
- AS: `num_classes: 4`

---

## 10. Preprocessing Reference

For best results, match the UHN preprocessing:

1. **DICOM to MP4:** Convert at native resolution, then resize to 224x224
2. **Sector masking:** Black out ECG trace, patient info, and non-sector regions
3. **Frame rate:** 8 fps (or keep native and let `frame_step` handle temporal sampling)
4. **B-mode only:** Exclude color/spectral/tissue Doppler clips
5. **View selection:** For best results, include only task-relevant views:
   - MR: A4C, A2C, A3C, PLAX
   - TR: A4C, subcostal/subxiphoid
   - AR: A4C, A2C, A3C, PLAX
   - AS: PLAX, PSAX-AV, A3C

If you include all views, the model will still work but may have slightly lower performance.

---

## 11. Expected Results

For reference, here are the **UHN internal validation** AUROC scores (not UCMC — these are from the Toronto training site's held-out test set):

| Task | EchoJEPA-G | EchoJEPA-L-K | EchoPrime | PanEcho |
|------|-----------|-------------|----------|---------|
| MR Severity | 0.882 | 0.836 | 0.818 | 0.789 |
| AS Severity | 0.932 | 0.868 | 0.868 | 0.813 |
| TR Severity | 0.854 | 0.817 | 0.780 | 0.778 |
| AR Severity | 0.765 | 0.680 | 0.701 | 0.692 |

Cross-institution performance may differ due to population differences, grading conventions, and ultrasound equipment. A 2-5 pp drop is typical for external validation.
