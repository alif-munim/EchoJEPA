# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EchoJEPA — a latent predictive foundation model for echocardiography built on V-JEPA 2 (Video Joint-Embedding Predictive Architecture) from Meta. Trained on 18M echocardiograms via self-supervised latent prediction. The package installs as `vjepa2` (Python >=3.11, tested with 3.12).

### Nature Medicine Manuscript

The active research objective is a **Nature Medicine paper** demonstrating that EchoJEPA's frozen representations encode clinical information far beyond standard echocardiographic measurements. The paper establishes capabilities not previously shown from frozen echo representations: rare disease detection, clinical outcome/biomarker prediction, latent forward prediction, SAE interpretability, and fairness analysis. All downstream tasks use **frozen linear probes** (no fine-tuning) on mean-pooled study-level embeddings. The companion ICML preprint covers the method and standard benchmarks; Nature Medicine covers novel clinical findings. Datasets: UHN (18M echos, pretraining), MIMIC-IV-Echo linked to MIMIC-IV clinical data (outcomes, labs, ICD codes, notes).

### Reference Files

**Architecture** (`claude/architecture/`):
- `pretraining-and-cooldown.md` — two-phase training (pretrain vs cooldown), LR schedules, masking, kinetics vs echo config differences, resume/force-load behavior
- `probe-system.md` — frozen probe evaluation: attentive/linear/MLP heads, classification vs regression, multi-view fusion, hyperparameter grid search, inference mode, prediction output

**Data & Manuscript** (`claude/data/`):
- `nature-medicine-manuscript.md` — manuscript scope, ICML vs Nature Medicine delineation, models, evaluation protocol
- `uhn-database.md` — UHN echocardiography database (echo.db, Syngo/HeartLab schemas, rare disease cohorts)
- `mimic-database.md` — MIMIC-IV linked to echo (prediction targets, biomarker coverage, data engineering notes)

Full source references: `uhn_echo/nature_medicine/CLAUDE.md`, `uhn_echo/nature_medicine/data_exploration/CLAUDE.md`, `uhn_echo/nature_medicine/data_exploration/mimic/CLAUDE.md`

## Common Commands

### Setup
```bash
pip install -e .          # editable install (setup script is scripts/setup.py)
```

### Tests
```bash
pytest tests              # all tests
pytest tests/models/test_models.py   # single test file
```

### Linting (CI runs these on app/, evals/*.py, src/, tests/)
```bash
python -m black --check app evals/*.py src tests
python -m isort app evals/*.py src tests --check
python -m flake8 --config .flake8 --show-source --statistics app evals/*.py src tests
```

### Pretraining
```bash
python -m app.main --fname configs/train/vitl16/pretrain-mimic-224px-16f.yaml --devices cuda:0 cuda:1
python -m app.main_distributed --fname configs/train/vitl16/pretrain-mimic-224px-16f.yaml  # SLURM
```

### Probe Evaluation
```bash
python -m evals.main --fname configs/eval/vitg-384/lvef/enp_echojepa_lvef.yaml --devices cuda:0 cuda:1
python -m evals.main --fname configs/inference/vitl/lvef.yaml --devices cuda:0 --val_only  # inference only
```

### Embedding Extraction
```bash
python -m evals.extract_embeddings --config configs/inference/vitg-384/view/echojepa_224px.yaml \
    --data /path/to/test.csv --output embeddings/out.npz --devices cuda:0 cuda:1
```

## Code Style

- **black** (line-length 119), **isort** (profile=black), **flake8** (max-line-length 119, ignore E203/E701/W503)
- Config: `pyproject.toml` for black/isort, `.flake8` for flake8
- Dev lint deps: `pip install -r requirements-test.txt`

## Architecture

### Entry Points & Dispatch

Training and evaluation use a scaffold pattern:
- `app.main` / `app.main_distributed` → `app.scaffold.main(app_name)` → `app.<app_name>.train.main()`
- `evals.main` / `evals.main_distributed` → `evals.scaffold.main(eval_name)` → `evals.<eval_name>.eval.main()`

The `app_name` / `eval_name` comes from the YAML config (`app:` or `eval_name:` field) and maps directly to a Python subpackage.

### Core Library (`src/`)

- **`src/models/`**: VisionTransformer encoder (ViT-T through ViT-g with RoPE, SDPA, activation checkpointing), VisionTransformerPredictor (latent target prediction), AttentivePooler/Classifier/Regressor (cross-attention probe heads), LinearPooler variants
- **`src/datasets/`**: `VideoDataset` (single-view, decord-based, S3+local), `VideoGroupDataset` (multi-view studies), `data_manager.init_data()` factory
- **`src/masks/`**: `MaskCollator` for spatio-temporal block masking (context + target masks)
- **`src/utils/`**: distributed init (NCCL, supports both local multi-GPU and SLURM/submitit), LR schedulers (WarmupCosine, WSD), checkpoint loading with retry logic

### Training Objective

Encoder processes visible (context) tokens; predictor takes encoder output + mask token positions to predict target encoder's representations of masked regions. Target encoder updated via EMA.

### Eval Modules

- `evals/video_classification_frozen/` — single-view probe (classification or regression)
- `evals/video_classification_frozen_multi/` — multi-view probe (multiple echo views per study)
- Both support `probe_type: attentive | linear | mlp` in config, and contain `modelcustom/` backbone adapters for VJepa2, EchoPrime, PanEcho, VideoMAE

### Probe Types

Three probe architectures in `src/models/`. Set via `experiment.classifier.probe_type` in YAML config:
- **`attentive`** (default) — learned cross-attention pooling + self-attention blocks. Used in ICML paper. Config: `num_heads`, `num_probe_blocks`
- **`linear`** — mean-pool + LayerNorm + single linear layer. Used in Nature Medicine paper. Config: `use_layernorm`, `dropout`
- **`mlp`** — mean-pool + LayerNorm + 2-layer MLP. Middle ground

See `claude/architecture/probe-system.md` for full details including attentive-vs-linear comparison and hyperparameter guidance.

### Config System

All experiments driven by YAML configs in `configs/`. Key structure:
- `configs/train/` — pretraining (by model size: vitl16, vith16, vitg16)
- `configs/eval/` — probe training (by model: vitg-384, vitl; by task: lvef, rvsp, view, etc.)
- `configs/inference/` — inference-only configs (set `val_only: true`)

### Dataset CSV Formats

- **Pretraining**: `path/to/video.mp4 0` (space-delimited, dummy label)
- **Classification**: `path/to/video.mp4 <int_label>` (space-delimited)
- **Regression**: `path/to/video.mp4 <z_score_float>` (space-delimited, Z-score normalized)
- **Multi-view**: `path/to/view1.mp4 path/to/view2.mp4 <float_label>` (space-delimited)

### Distributed Training

- Local: `mp.Process` per GPU with `CUDA_VISIBLE_DEVICES`
- SLURM: `submitit` for cluster jobs
- Both use NCCL backend via `src/utils/distributed.init_distributed()`
- Debug mode: `--debugmode True` runs rank 0 in main process

### Pretrained Checkpoints

Available via torch.hub (`facebookresearch/vjepa2`): `vjepa2_vit_large`, `vjepa2_vit_huge`, `vjepa2_vit_giant`, `vjepa2_vit_giant_384`. Hub entry point is `scripts/hubconf.py`.

### Classifier Pipeline (`classifier/`)

ConvNeXt/Swin image classifiers for echo view/color/quality/zoom, plus distributed inference on the 18M dataset. Key files:
- `train_convnext.py` — distributed DDP training (S3, video-based)
- `cooldown.py` — low-LR fine-tuning (single-GPU, local, image-based)
- `inference_18m.py` — unified inference with `--task view|color|quality|zoom`
- `data_prep/` — sequential pipeline: patient splits → MP4 mapping → S3 URIs → verification → JEPA format
- `mappings/` — canonical label→int JSON maps (`views.json`, `color.json`, `quality.json`, `zoom.json`)

### Data Directory (`data/`)

- `csv/` — JEPA-format splits (153 files, referenced by eval configs — do not rename without updating configs)
- `scalers/` — sklearn StandardScaler pickles for Z-score normalization (ef, lvef, rvsp, tapse, pediatric)
- `labels/` — raw label CSVs (mimic_annotations, pacemaker, mvr, master lists)
- `parquet/` — large structured data exports (all_es_*.parquet)
- `notebooks/` — data exploration and split generation notebooks
- `scripts/` — processing and augmentation scripts (masking, depth attenuation, DICOM conversion)
- `training_logs/` — pretraining/cooldown loss curves
