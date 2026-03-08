# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EchoJEPA — a latent predictive foundation model for echocardiography built on V-JEPA 2 (Video Joint-Embedding Predictive Architecture) from Meta. Trained on 18M echocardiograms via self-supervised latent prediction. The package installs as `vjepa2` (Python >=3.11, tested with 3.12).

### Nature Medicine Manuscript

The active research objective is a **Nature Medicine paper** demonstrating that EchoJEPA's frozen representations encode clinical information far beyond standard echocardiographic measurements. The paper establishes capabilities not previously shown from frozen echo representations: rare disease detection, clinical outcome/biomarker prediction, latent forward prediction, SAE interpretability, and fairness analysis. All downstream tasks use **frozen linear probes** (no fine-tuning) on mean-pooled study-level embeddings. The companion ICML preprint covers the method and standard benchmarks; Nature Medicine covers novel clinical findings. Datasets: UHN (18M echos, pretraining), MIMIC-IV-Echo linked to MIMIC-IV clinical data (outcomes, labs, ICD codes, notes).

### Reference Documentation (`claude/`)

The `claude/` directory contains persistent reference docs organized by topic. See `claude/DIRECTORY.md` for the full index with file-level descriptions.

- **`claude/architecture/`** — codebase internals: pretraining pipeline, probe system (attentive/linear/MLP), classifier pipeline
- **`claude/data/`** — datasets and manuscript: `data/` directory layout, Nature Medicine scope, UHN database schemas, MIMIC-IV linkage
- **`claude/dev/`** — development log: bug tracker (6 issues), changelog, code review findings, UHN extraction ops guide. See `dev/README.md` for the bug index and planned fixes
- **`claude/preprint/`** — ICML preprint analysis: encoder fairness confounds, probe architecture mismatch (attentive vs linear inversion), claim validity assessment, hindsight recommendations for camera-ready
- **`claude/rebuttals/`** — ICML rebuttal preparation: TIER 1-4 vulnerability inventory, response templates, worst-case scenarios, competitive positioning, camera-ready action items

Additional source references: `uhn_echo/nature_medicine/CLAUDE.md`, `uhn_echo/nature_medicine/data_exploration/CLAUDE.md`, `uhn_echo/nature_medicine/data_exploration/mimic/CLAUDE.md`

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
    --data /path/to/test.csv --output experiments/out.npz --devices cuda:0 cuda:1
```

### Probe Training on Embeddings (sklearn, no GPU)
```bash
python -m evals.train_probe \
    --train experiments/nature_medicine/mimic/echojepa_g_splits/mortality_1yr/train.npz \
    --val experiments/nature_medicine/mimic/echojepa_g_splits/mortality_1yr/test.npz \
    --task classification --output_dir results/probes/nature_medicine/mortality_1yr
```

See `README.md` Quickstart and MIMIC sections for full examples (k-fold, label-only NPZs, model comparison, hyperparameter tuning).

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
- Both support `probe_type: attentive | linear | mlp` in config, and contain `modelcustom/` backbone adapters for VJepa2, EchoPrime, PanEcho, VideoMAE, EchoFM

### Probe Types

Three probe architectures in `src/models/`. Set via `experiment.classifier.probe_type` in YAML config:
- **`attentive`** (default) — learned cross-attention pooling + self-attention blocks. Used in ICML paper. Config: `num_heads`, `num_probe_blocks`
- **`linear`** — mean-pool + LayerNorm + single linear layer. Used in Nature Medicine paper. Config: `use_layernorm`, `dropout`
- **`mlp`** — mean-pool + LayerNorm + 2-layer MLP. Middle ground

See `claude/architecture/probe-system.md` for full details including attentive-vs-linear comparison and hyperparameter guidance.

### Embedding Pipeline (`evals/`)

Scripts for the MIMIC multi-model embedding pipeline (see `claude/data/embedding-pipeline.md`):
- `evals/extract_embeddings.py` — multi-GPU clip-level extraction from frozen encoder → master NPZ
- `evals/remap_embeddings.py` — create per-task label NPZs referencing master by row index (avoids duplicating embeddings)
- `evals/pool_embeddings.py` — mean-pool clip embeddings to study-level using `clip_index.npz`
- `evals/train_probe.py` — sklearn linear probes on embeddings; supports `--labels` for label-only NPZs and `--train`/`--val` for precomputed splits

All models share the same `clip_index.npz`, `patient_split.json`, and `labels/` directory. Per-model outputs use the naming convention `{model}_study_level/` and `{model}_splits/`.

### Config System

All experiments driven by YAML configs in `configs/`. Each subdirectory has a README.md with full details:
- `configs/train/` — pretraining and cooldown, by model size (vitl16, vith16, vitg16). Naming: `{phase}-{dataset}-{resolution}-{frames}.yaml`
- `configs/eval/` — probe training, by model (vitg-384, vitl) and task (lvef, rvsp, view, color, quality, zoom, tapse). `old/` contains archived configs. `multihead_kwargs` list trains multiple probes in parallel with different hyperparameters.
- `configs/inference/` — inference-only configs (set `val_only: true`, require `probe_checkpoint`). Specialized subdirs: `depth_attenuation/`, `echonet-dynamic/`, `echonet-pediatric/`

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

ConvNeXt/Swin image classifiers for echo view/color/quality/zoom, plus distributed inference on the 18M dataset. See `claude/architecture/classifier-pipeline.md` for full details.

### Data Directory (`data/`)

JEPA-format splits, raw labels, scalers, notebooks, and scripts. See `claude/data/data-directory.md` for full details. **Do not rename files in `csv/`** without updating corresponding eval configs.

`data/data/` contains the UHN deidentification key and mapping notebooks (deid_key.csv, identifiers.ipynb, ecs_mappings.ipynb, syngo.ipynb). Working copies of the key mapping files are at `experiments/nature_medicine/uhn/mapping/` (gitignored). See `claude/data/uhn-mapping.md` for the full mapping chain documentation.

### Other Root Directories

- `checkpoints/` — all model weights: pretrain, anneal, cooldown, eval probes, SSv2 probe
- `indices/` — S3 URI manifests for the 18M dataset (`master_index_18M.csv`, `master_index_18M_cleaned.csv`, `s3_pretrain.csv`, annotations)
- `experiments/` — precomputed frozen embeddings organized by paper: `icml/` (UHN benchmarks), `nature_medicine/mimic/` (7 models, 23 tasks), `nature_medicine/uhn/mapping/` (DICOM↔Syngo deid keys, gitignored). See `experiments/README.md`, `claude/data/embedding-pipeline.md`, `claude/data/uhn-mapping.md`
- `predictions/` — probe and classifier prediction CSVs (LVEF, RVSP, EchoNet, view, quality, zoom)
- `results/` — data efficiency experiment runs (epoch checkpoints + logs)
- `scripts/` — SBATCH scripts, Python utilities, demos, `run_details.md`
- `notebooks/` — project-level analysis and demo Jupyter notebooks
- `figures/` — publication-quality UMAP plots (PDF + PNG)
- `logs/` — training and evaluation logs (`.log` files, SLURM `.out` files, training plots)
- `uhn_echo/` — UHN research data, Nature Medicine analysis, MIMIC-IV linkage
