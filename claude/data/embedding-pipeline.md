# MIMIC Embedding Pipeline

Multi-model precomputed embedding pipeline for MIMIC-IV-Echo. Extracts clip-level embeddings from frozen encoders, then derives study-level pooled embeddings and patient-level train/val/test splits for 23 clinical tasks.

## Pipeline Overview

```
Source CSV (525K S3 paths)
    │
    ├─► evals.extract_embeddings ──► {model}_mimic_embeddings.npz   (clip-level, per model)
    │
    ├─► clip_index.npz              (shared: s3_paths, study_ids, patient_ids)
    ├─► patient_split.json          (shared: global 70/15/15 patient assignment)
    ├─► labels/*.npz                (shared: per-task indices + labels)
    │
    ├─► evals.pool_embeddings ──► {model}_study_level/{task}.npz    (mean-pooled per study)
    │
    └─► split script ──► {model}_splits/{task}/train|val|test.npz   (patient-level splits)
```

## Models

| Model | Architecture | Pretraining | Params | Embed dim | File |
|-------|-------------|-------------|--------|-----------|------|
| EchoJEPA-G | ViT-g/16 384px | JEPA on 18M echo clips | 1.1B | 1408 | `echojepa_g_mimic_embeddings.npz` |
| EchoJEPA-L | ViT-L/16 224px | JEPA on 18M echo clips | 304M | 1024 | `echojepa_l_mimic_embeddings.npz` |
| EchoJEPA-L Kinetics | ViT-L/16 224px | JEPA on Kinetics-400 | 304M | 1024 | `echojepa_l_kinetics_mimic_embeddings.npz` |
| EchoMAE | ViT-L/16 (VideoMAE) | MAE on 1.5M echo clips | 304M | 1024 | `echomae_mimic_embeddings.npz` |
| EchoFM | ViT-L/16 (MAE+triplet) | MAE on 290K echo clips | 304M | 1024 | `echofm_mimic_embeddings.npz` |
| PanEcho | ConvNeXt-T + Transformer | Multi-task supervised on 1.1M echo clips | 28M | 768 | `panecho_mimic_embeddings.npz` |
| EchoPrime | MViT-v2-S | CLIP-style contrastive on 12M echo clips + report text | 35M | 512 | `echoprime_mimic_embeddings.npz` |

Key controlled comparisons: EchoJEPA-L vs EchoMAE (same arch, JEPA vs MAE), EchoJEPA-L vs Kinetics (same arch, echo vs natural video data), EchoJEPA-G vs EchoJEPA-L (scale). EchoPrime is text-supervised (contrastive alignment with BiomedBERT report embeddings), not purely self-supervised.

All models are extracted from the same source CSV (`data/csv/nature_medicine/mimic/mortality_1yr.csv`, 525,312 clips) using `evals.extract_embeddings`, ensuring row-aligned outputs. This means `clip_index.npz`, `patient_split.json`, and all label NPZs are shared across models.

### Transform Pipeline & Normalization

The shared `make_transforms(training=False)` pipeline applies: Resize(short_side=256) → CenterCrop(224) → ClipToTensor → ImageNet Normalize. All encoder adapters receive this ImageNet-normalized input.

- **EchoJEPA-G/L, EchoMAE (VideoMAE):** trained with ImageNet normalization — use input directly.
- **PanEcho:** trained with ImageNet normalization — uses input directly (spatial resize to 224 if needed).
- **EchoPrime:** trained with custom normalization in 0-255 space — adapter undoes ImageNet norm, scales to 255, applies EchoPrime norm.
- **EchoFM:** trained on [0,1] range (no normalization) — adapter undoes ImageNet norm to recover [0,1].

**Note:** PanEcho, EchoPrime, and EchoFM normalization was fixed on 2026-03-06 (see `claude/dev/bugs/002-normalization-bugs.md`) and re-extracted on 2026-03-08. All 7 MIMIC models had a shuffle ordering bug (see `claude/dev/bugs/001-shuffle-bug.md`); original 4 models were fixed post-hoc, 3 re-extracted models have correct ordering. All 7 models are now probe-ready with 161/161 probe runs complete.

## Directory Layout

```
experiments/nature_medicine/mimic/
├── echojepa_g_mimic_embeddings.npz    # clip-level (525,312 × 1408)
├── echojepa_l_mimic_embeddings.npz    # clip-level (525,312 × 1024)
├── echojepa_l_kinetics_mimic_embeddings.npz  # clip-level (525,312 × 1024)
├── echomae_mimic_embeddings.npz       # clip-level (525,312 × 1024)
├── echofm_mimic_embeddings.npz        # clip-level (525,312 × 1024)
├── panecho_mimic_embeddings.npz       # clip-level (525,312 × 768)
├── echoprime_mimic_embeddings.npz     # clip-level (525,312 × 512)
├── clip_index.npz                     # shared: s3_paths, study_ids, patient_ids
├── patient_split.json                 # shared: {patient_id: "train"|"val"|"test"}
├── labels/                            # shared: 23 task label NPZs
│   ├── mortality_1yr.npz
│   └── ...
├── {model}_study_level/               # study-pooled per model (7 dirs)
│   ├── mortality_1yr.npz
│   └── ...
├── {model}_splits/                    # patient-level train/val/test per model (7 dirs)
│   ├── mortality_1yr/
│   │   ├── train.npz
│   │   ├── val.npz
│   │   └── test.npz
│   └── ...
└── {model}_mimic_all.zip              # self-contained distribution per model (7 zips)
```

## Shared Files

### clip_index.npz

Row-aligned with all master embedding NPZs. Created by parsing S3 paths from the source CSV.

| Array | Shape | Description |
|-------|-------|-------------|
| `s3_paths` | `(525312,)` | Full S3 URI per clip |
| `study_ids` | `(525312,)` | MIMIC study ID (int64) |
| `patient_ids` | `(525312,)` | MIMIC patient ID (int64) |

Path structure: `s3://echodata25/mimic-echo-224px/files/p{group}/p{patient_id}/s{study_id}/{study_id}_{clip}.mp4`

### patient_split.json

Global patient-level assignment, used consistently across all tasks and models to ensure no data leakage and comparable results. 3,205 / 686 / 688 patients (train / val / test), seed=42.

### labels/{task}.npz

Lightweight per-task files referencing the master NPZ by row index. Avoids duplicating 2.8GB+ of embeddings per task.

| Array | Description |
|-------|-------------|
| `indices` | Row positions into master NPZ (int64) |
| `labels` | Task-specific labels (float64) |

Created by `evals.remap_embeddings` from task CSVs + source CSV ordering.

## Mapping Embeddings to Studies and DICOMs

Each clip (MP4) corresponds to one DICOM file. The S3 path encodes the full hierarchy. Use `clip_index.npz` to trace any embedding row back to its source:

```python
import numpy as np

master = np.load("experiments/nature_medicine/mimic/echojepa_g_mimic_embeddings.npz")
index = np.load("experiments/nature_medicine/mimic/clip_index.npz", allow_pickle=True)

# Row 42 in the embedding matrix
print(index["s3_paths"][42])      # s3://echodata25/.../s90001295/90001295_0006.mp4
print(index["study_ids"][42])     # 90001295
print(index["patient_ids"][42])   # 15690862
print(master["embeddings"][42].shape)  # (1408,)

# All clips belonging to a specific study
study_mask = index["study_ids"] == 90001295
study_clips = master["embeddings"][study_mask]   # (N_clips, 1408)
study_paths = index["s3_paths"][study_mask]       # S3 URIs for each clip

# All studies belonging to a specific patient
patient_mask = index["patient_ids"] == 15690862
patient_studies = np.unique(index["study_ids"][patient_mask])
```

To map back to original MIMIC-IV DICOM files, strip the S3 prefix and clip suffix: `{study_id}_{clip_num}.mp4` corresponds to a single DICOM in `s{study_id}/` within MIMIC-IV-Echo.

## Connecting Embeddings to Task Labels

Label NPZs store row indices into the master embedding NPZ, not the embeddings themselves. To get labeled embeddings for a task:

```python
import numpy as np

# Load master embeddings (any model)
master = np.load("experiments/nature_medicine/mimic/echojepa_g_mimic_embeddings.npz")
index = np.load("experiments/nature_medicine/mimic/clip_index.npz", allow_pickle=True)

# Load a task's labels
task = np.load("experiments/nature_medicine/mimic/labels/mortality_1yr.npz")
indices = task["indices"]   # row positions into master
labels = task["labels"]     # task-specific label per clip

# Subset embeddings, study_ids, and patient_ids to this task
task_embeddings = master["embeddings"][indices]   # (N_task, 1408)
task_study_ids = index["study_ids"][indices]       # (N_task,)
task_patient_ids = index["patient_ids"][indices]   # (N_task,)

# Build study-level label lookup (labels are constant within a study)
study_label = {}
for sid, lab in zip(task_study_ids, labels):
    if sid not in study_label:
        study_label[sid] = lab

# Use with study-level pooled embeddings for UMAP, clustering, etc.
pooled = np.load("experiments/nature_medicine/mimic/echojepa_g_study_level/mortality_1yr.npz")
# pooled["embeddings"]: (N_studies, 1408)
# pooled["labels"]:     (N_studies,)
# pooled["study_ids"]:  (N_studies,)
# pooled["patient_ids"]: (N_studies,)
```

The same label indices work with any model's master NPZ (e.g., `panecho_mimic_embeddings.npz`) since all were extracted from the same source CSV in the same row order.

## UHN Embedding Pipeline

The UHN pipeline uses `evals/extract_uhn_embeddings.py` for chunked extraction (crash-safe resume) and `evals/regenerate_uhn_downstream.py` for per-task splits.

### Directory Layout

```
experiments/nature_medicine/uhn/
├── uhn_all_clips.csv              # 18.1M S3 paths (extraction source)
├── uhn_clip_index.npz             # shared: study_uids, patient_ids for 18.1M clips
├── patient_split.json             # shared: 138K patients, 70/10/20 temporal
├── labels/                        # shared: 53 task label NPZs
│   ├── lvef.npz                   # format: study_ids, patient_ids, labels, splits
│   ├── disease_hcm.npz
│   └── trajectory/                # paired-study format
│       ├── trajectory_lvef.npz    # format: study_id_1, study_id_2, label_1, label_2, delta, days_between, splits
│       └── trajectory_multi.npz   # format: study_id_1, study_id_2, days_between, splits (no labels)
├── {model}_embeddings/            # per-model extraction output
│   ├── chunks_rank*/              # intermediate chunks (crash-safe)
│   ├── clip_embeddings.npz        # merged clip-level (18.1M × dim)
│   └── study_embeddings.npz       # mean-pooled study-level (320K × dim)
└── {model}_splits/                # per-model per-task splits (generated by regenerate_uhn_downstream.py)
    ├── lvef/
    │   ├── train.npz              # embeddings, labels, study_ids, patient_ids
    │   ├── val.npz
    │   └── test.npz
    ├── disease_hcm/               # same format for all 47 standard tasks
    └── trajectory/trajectory_lvef/ # trajectory splits: embeddings_1, embeddings_2, study_id_1, study_id_2, etc.
```

### Key Differences from MIMIC Pipeline

| Aspect | UHN | MIMIC |
|--------|-----|-------|
| Scale | 18.1M clips, 320K studies | 525K clips, 7.2K studies |
| Extraction script | `extract_uhn_embeddings.py` (chunked, crash-safe) | `extract_embeddings.py` (single-pass) |
| Study pooling | Built into extraction script | Separate `pool_embeddings.py` step |
| Label format | Study-level with splits baked in | Clip-level indices into master NPZ |
| Split source | `splits` array in each label NPZ (from `patient_split.json`) | `patient_split.json` applied at split time |
| Split generation | `regenerate_uhn_downstream.py` (joins embeddings + labels on study_ids) | `regenerate_mimic_downstream.py` (indexes into clip array, pools, splits) |

### Split Pipeline

The same patient-level temporal split applies to all models. `regenerate_uhn_downstream.py` joins each model's `study_embeddings.npz` with label NPZs on `study_ids`, then splits by the `splits` field in each label file. Re-run for each model as its study-level embeddings become available:

```bash
# All available models
python -m evals.regenerate_uhn_downstream --uhn_dir experiments/nature_medicine/uhn

# Single model
python -m evals.regenerate_uhn_downstream --uhn_dir experiments/nature_medicine/uhn --model echojepa_g
```

A small number of studies (10-200 per task, <0.1%) are dropped per model when extraction is incomplete (e.g., zero-filled clips from `drop_last` gaps).

### UHN Extraction Commands

```bash
DEVICES="cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7"
DATA="experiments/nature_medicine/uhn/uhn_all_clips.csv"
INDEX="experiments/nature_medicine/uhn/uhn_clip_index.npz"

# EchoJEPA-G (ViT-g, 384px→224px, 1408-d) — ~25h
python -m evals.extract_uhn_embeddings --config configs/inference/vitg-384/extract_uhn.yaml \
    --data $DATA --clip_index $INDEX --output_dir experiments/nature_medicine/uhn/echojepa_g_embeddings \
    --devices $DEVICES --batch_size 32

# EchoJEPA-L (ViT-L, 224px, 1024-d) — ~12.5h
python -m evals.extract_uhn_embeddings --config configs/inference/vitl/extract_uhn.yaml \
    --data $DATA --clip_index $INDEX --output_dir experiments/nature_medicine/uhn/echojepa_l_embeddings \
    --devices $DEVICES --batch_size 64

# EchoMAE-L (VideoMAE ViT-L, 224px, 1024-d) — ~20h (manual attention, needs smaller bs)
python -m evals.extract_uhn_embeddings --config configs/inference/vitl/extract_uhn_echomae.yaml \
    --data $DATA --clip_index $INDEX --output_dir experiments/nature_medicine/uhn/echomae_embeddings \
    --devices $DEVICES --batch_size 16
```

## Scripts

| Script | Purpose |
|--------|---------|
| `evals/extract_embeddings.py` | Multi-GPU clip-level embedding extraction from frozen encoder (MIMIC) |
| `evals/extract_uhn_embeddings.py` | Chunked multi-GPU extraction for UHN 18M (crash-safe resume, built-in study pooling) |
| `evals/regenerate_mimic_downstream.py` | Generate MIMIC study-level + per-task splits for all models |
| `evals/regenerate_uhn_downstream.py` | Generate UHN per-task splits for all models (joins study embeddings with label NPZs) |
| `evals/fix_shuffle_order.py` | Post-hoc fix for extractions done with shuffle=True (reorders to CSV order) |
| `evals/remap_embeddings.py` | Create per-task label NPZs referencing master by index (single or batch mode) |
| `evals/pool_embeddings.py` | Mean-pool clip embeddings to study-level |
| `evals/train_probe.py` | Train sklearn linear probes on embeddings (supports `--labels` for label-only NPZs) |

For UHN extraction performance tuning, DataLoader settings, and operational details, see **`claude/dev/ops.md`**.

> **Important**: Both extraction scripts now set `DistributedSampler.shuffle=False` to maintain CSV-order alignment with `clip_index.npz`. Any existing embeddings extracted before 2026-03-07 have shuffled ordering and need either post-hoc reordering or re-extraction. See `claude/dev/bugs/001-shuffle-bug.md`.

## Extraction Commands

```bash
DEVICES="cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7"
DATA="data/csv/nature_medicine/mimic/mortality_1yr.csv"
OUT="experiments/nature_medicine/mimic"

# EchoJEPA-G (ViT-g, 384px, 1408-d)
python -m evals.extract_embeddings --config configs/inference/vitg-384/view/echojepa_224px.yaml \
    --data $DATA --output $OUT/echojepa_g_mimic_embeddings.npz --devices $DEVICES

# EchoJEPA-L (ViT-L, 224px, 1024-d)
python -m evals.extract_embeddings --config configs/inference/vitg-384/view/echojepa_large_224px.yaml \
    --data $DATA --output $OUT/echojepa_l_mimic_embeddings.npz --devices $DEVICES

# EchoJEPA-L Kinetics (ViT-L, 224px, 1024-d, Kinetics pretrained)
python -m evals.extract_embeddings --config configs/inference/vitg-384/view/echojepa_large_kinetics_224px.yaml \
    --data $DATA --output $OUT/echojepa_l_kinetics_mimic_embeddings.npz --devices $DEVICES

# EchoMAE (ViT-L VideoMAE, 224px, 1024-d)
python -m evals.extract_embeddings --config configs/inference/vitg-384/view/videomae_224px.yaml \
    --data $DATA --output $OUT/echomae_mimic_embeddings.npz --devices $DEVICES

# EchoFM (ViT-L MAE+triplet, 224px, 1024-d)
python -m evals.extract_embeddings --config configs/inference/vitg-384/view/echofm_224px.yaml \
    --data $DATA --output $OUT/echofm_mimic_embeddings.npz --devices $DEVICES

# PanEcho (ConvNeXt-T, 224px, 768-d)
python -m evals.extract_embeddings --config configs/inference/vitg-384/view/panecho_224px.yaml \
    --data $DATA --output $OUT/panecho_mimic_embeddings.npz --devices $DEVICES

# EchoPrime (MViT-v2-S, 224px, 512-d)
python -m evals.extract_embeddings --config configs/inference/vitg-384/view/echoprime_224px.yaml \
    --data $DATA --output $OUT/echoprime_mimic_embeddings.npz --devices $DEVICES
```

## Deriving Study-Level and Splits

Given a new model's master NPZ, the same shared files produce study-level and splits:

```bash
# 1. Remap labels (only needed once, shared across models)
python -m evals.remap_embeddings \
    --embeddings experiments/nature_medicine/mimic/{model}_mimic_embeddings.npz \
    --source_csv data/csv/nature_medicine/mimic/mortality_1yr.csv \
    --task_dir data/csv/nature_medicine/mimic/ \
    --output_dir experiments/nature_medicine/mimic/labels/

# 2. Pool to study level (per model)
for f in experiments/nature_medicine/mimic/labels/*.npz; do
    task=$(basename "$f" .npz)
    python -m evals.pool_embeddings \
        --embeddings experiments/nature_medicine/mimic/{model}_mimic_embeddings.npz \
        --clip_index experiments/nature_medicine/mimic/clip_index.npz \
        --labels "$f" \
        --output "experiments/nature_medicine/mimic/{model}_study_level/${task}.npz"
done

# 3. Create patient-level splits (Python, see README.md)
```

## Adding a New Model

1. Create an inference config in `configs/inference/` with the model's backbone adapter
2. Run `evals.extract_embeddings` with the same source CSV to produce `{model}_mimic_embeddings.npz`
3. Reuse existing `clip_index.npz`, `patient_split.json`, and `labels/` (all shared)
4. Pool to study level and create splits (same scripts, just change the model prefix)
5. Package into `{model}_mimic_all.zip`

The key invariant is that all models are extracted from the same source CSV in the same order, so row indices are interchangeable.
