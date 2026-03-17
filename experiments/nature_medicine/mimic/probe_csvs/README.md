# MIMIC Structured Measurement Probe CSVs

Probe training data for echocardiographic structured measurements from MIMIC-IV-Echo. Each folder contains train/val/test CSVs ready for frozen probe training with EchoJEPA.

## Quick Start

```bash
# 1. Install the package (from repo root)
pip install -e .

# 2. Train a probe (single task, 8 GPUs)
python -m evals.main \
    --fname configs/eval/vitl/nature_medicine/echojepa_l_k_mitral_regurg.yaml \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7

# 3. Train with fewer GPUs (e.g. 2)
python -m evals.main \
    --fname configs/eval/vitl/nature_medicine/echojepa_l_k_septal_thickness.yaml \
    --devices cuda:0 cuda:1

# 4. Validation only (requires a trained probe checkpoint)
python -m evals.main \
    --fname configs/eval/vitl/nature_medicine/echojepa_l_k_mitral_regurg.yaml \
    --devices cuda:0 --val_only
```

Each config trains **20 probes in parallel** (5 learning rates x 4 weight decay values) via `multihead_kwargs`. The best is selected by validation metric.

## Requirements

- **Checkpoint**: `checkpoints/anneal/keep/vitl-kinetics-pt220-an55.pt` (EchoJEPA-L-K, ViT-Large, 304M params, Kinetics-initialized then pretrained on MIMIC)
- **Video data**: CSVs reference S3 paths (`s3://echodata25/mimic-echo-224px/...`). Ensure S3 access is configured.
- **GPUs**: 80GB VRAM per GPU recommended (A100/H100). `batch_size: 1` with `study_sampling: true` means 1 study per GPU per step.

## Output

Results are written to the `folder` path specified in each YAML config:

```
evals/vitl/nature_medicine/video_classification_frozen/<tag>/
  log_r0.csv        # Per-epoch metrics (acc, AUROC, balanced acc, kappa)
  epoch_001.pt      # Probe checkpoint per epoch
  best.pt           # Best epoch by validation metric
  latest.pt         # Most recent epoch
```

The `<tag>` matches the `tag:` field in the YAML (e.g., `echojepa-l-k-mitral-regurg`).

## Tasks

### Classification

| Task | Config | Classes | Train Studies | Views |
|------|--------|---------|---------------|-------|
| Mitral regurgitation | `echojepa_l_k_mitral_regurg.yaml` | 4 (None-Trivial / Mild / Moderate / Severe) | 4,713 | A4C+A2C+A3C+PLAX |
| Tricuspid regurgitation | `echojepa_l_k_tricuspid_regurg.yaml` | 4 (None-Trivial / Mild / Moderate / Severe) | 4,666 | A4C+A5C |
| LV wall thickness | `echojepa_l_k_lv_wall_thickness.yaml` | 4 (Normal / Mild LVH / Mod LVH / Severe LVH) | 4,676 | PLAX |
| WM inferior base | `echojepa_l_k_wm_inf_base.yaml` | 4 (Normal / Hypo / Akinetic / Dyskinetic) | 474 | A4C+A2C+A3C |
| WM inferior mid | `echojepa_l_k_wm_inf_mid.yaml` | 4 | 410 | A4C+A2C+A3C |
| WM apical cap | `echojepa_l_k_wm_apical_cap.yaml` | 4 | 330 | A4C+A2C+A3C |
| WM ant-sept mid | `echojepa_l_k_wm_ant_sept_mid.yaml` | 4 | 282 | A4C+A2C+A3C |

### Regression

| Task | Config | Unit | Train Studies | Views |
|------|--------|------|---------------|-------|
| LVEF (structured) | `echojepa_l_k_lvef_structured.yaml` | % | 3,159 | A4C+A2C |
| Septal thickness | `echojepa_l_k_septal_thickness.yaml` | cm | 4,718 | PLAX |

Regression CSVs store **raw values** (e.g., `1.30` cm). Z-score normalization is applied automatically at runtime using the `zscore_params.json` file in each task folder.

## CSV Format

Space-delimited, no header. Each row is one video clip:

```
s3://echodata25/mimic-echo-224px/files/p15/p15690862/s90001295/90001295_0001.mp4 1
s3://echodata25/mimic-echo-224px/files/p15/p15690862/s90001295/90001295_0002.mp4 1
```

- **Classification**: `<s3_path> <int_label>` (0-indexed class)
- **Regression**: `<s3_path> <float_value>` (raw units)

Multiple clips per study are expected. During training, `study_sampling: true` selects 1 random clip per study per epoch. At validation, all clips are scored and predictions are averaged per study.

## Folder Structure

```
probe_csvs/
  <task>/
    train.csv              # Training split
    val.csv                # Validation split
    test.csv               # Held-out test split
    task_meta.json         # Task metadata (type, views, label map, counts)
    zscore_params.json     # Regression only: {target_mean, target_std}
  misc/                    # Original 23 MIMIC tasks (mortality, biomarkers, diseases, etc.)
```

Patient-level splits (70/15/15) ensure no patient appears in multiple splits.

## Probe Architecture

All configs use the same frozen probe setup:

- **Encoder**: EchoJEPA-L-K (ViT-Large, 304M params) with RoPE, frozen (no gradient). Initialized from Kinetics V-JEPA 2, then pretrained + annealed on MIMIC-IV-Echo.
- **Probe**: depth=1 attentive (1 learnable query token, single cross-attention layer, no self-attention)
- **HP grid**: 20 combinations trained in parallel (`multihead_kwargs`)
  - LR: 1e-3, 5e-4, 1e-4, 5e-5, 1e-5
  - Weight decay: 0.001, 0.01, 0.1, 0.4
- **Epochs**: 35 (standard tasks), 50 (wall motion tasks with <500 studies)
- **Precision**: bfloat16

## Config YAML Reference

Key fields in each config:

```yaml
experiment:
  classifier:
    num_heads: 16              # Cross-attention heads
    num_probe_blocks: 1        # Probe depth (always 1)
    # For regression tasks only:
    task_type: regression
    num_targets: 1

  data:
    num_classes: 4             # Classification only
    resolution: 224
    frames_per_clip: 16
    frame_step: 2              # Sample every 2nd frame (32 frames coverage)
    num_segments: 2            # 2 temporal segments per clip
    study_sampling: true       # 1 random clip/study/epoch

  optimization:
    batch_size: 1              # Per-GPU batch size (1 study = 1 sample)
    num_epochs: 35
    multihead_kwargs: [...]    # 20 HP combos trained in parallel

model_kwargs:
  checkpoint: checkpoints/anneal/keep/vitl-kinetics-pt220-an55.pt
  module_name: evals.video_classification_frozen.modelcustom.vit_encoder_multiclip
  pretrain_kwargs:
    encoder:
      checkpoint_key: target_encoder
      model_name: vit_large
      patch_size: 16
      tubelet_size: 2
      uniform_power: true
      use_rope: true
```

## Resolution

All models use **224x224** input resolution. Videos are stored at native resolution on S3 and resized at runtime by the data loader (controlled by the `resolution: 224` field in each config). No pre-resizing is needed.

## Baseline Models

The same probe CSVs and training pipeline work with any supported encoder. To run a baseline model, create a config that swaps the `model_kwargs` section. The `experiment` block (classifier, data, optimization) stays the same.

### EchoPrime

Text-supervised contrastive model (MViT-v2-S, 34M params, 512-dim embeddings). Requires the [EchoPrime repository](https://github.com/echonet/echo_prime) cloned into `evals/video_classification_frozen/modelcustom/EchoPrime/`.

```bash
# Clone EchoPrime source and weights
cd evals/video_classification_frozen/modelcustom/
git clone https://github.com/echonet/echo_prime.git EchoPrime
# Unzip model_data.zip inside EchoPrime/ to get model_data/weights/echo_prime_encoder.pt
cd EchoPrime && unzip model_data.zip
```

Config `model_kwargs`:

```yaml
model_kwargs:
  checkpoint: null
  module_name: evals.video_classification_frozen.modelcustom.echo_prime_encoder
  pretrain_kwargs: {}
  wrapper_kwargs:
    echo_prime_root: evals/video_classification_frozen/modelcustom/EchoPrime
    force_fp32: true
    bin_size: 50
```

### PanEcho

Multi-task supervised model (ConvNeXt-T + Transformer, 768-dim embeddings). Weights download automatically from torch.hub on first run. Requires the [PanEcho repository](https://github.com/CarDS-Yale/PanEcho) cloned into `evals/video_classification_frozen/modelcustom/PanEcho/`.

```bash
cd evals/video_classification_frozen/modelcustom/
git clone https://github.com/CarDS-Yale/PanEcho.git PanEcho
```

Config `model_kwargs`:

```yaml
model_kwargs:
  checkpoint: null
  module_name: evals.video_classification_frozen.modelcustom.panecho_encoder
  pretrain_kwargs: {}
  wrapper_kwargs: {}
```

### EchoFM

Self-supervised MAE + triplet loss model (ViT-L, 1024-dim embeddings). Requires the [EchoFM repository](https://github.com/SekeunKim/EchoFM) cloned into `evals/video_classification_frozen/modelcustom/EchoFM/` and the pretrained checkpoint.

```bash
cd evals/video_classification_frozen/modelcustom/
git clone https://github.com/SekeunKim/EchoFM.git EchoFM
# Download EchoFM_latest.pth from the EchoFM repo's releases
# Place at: EchoFM/weights/EchoFM/EchoFM_latest.pth
```

Config `model_kwargs`:

```yaml
model_kwargs:
  checkpoint: evals/video_classification_frozen/modelcustom/EchoFM/weights/EchoFM/EchoFM_latest.pth
  module_name: evals.video_classification_frozen.modelcustom.echofm_encoder
  pretrain_kwargs:
    encoder:
      model_name: vit_large_patch16_224
      num_frames: 32
      t_patch_size: 4
  wrapper_kwargs: {}
```

### EchoMAE (VideoMAE)

Self-supervised MAE model (ViT-L, 1024-dim embeddings). This is our controlled comparison: same ViT-L architecture and MIMIC data as EchoJEPA-L-K, but trained with masked autoencoding instead of JEPA.

```bash
# Checkpoint is at checkpoints/videomae-ep163.pth (provided)
```

Config `model_kwargs`:

```yaml
model_kwargs:
  checkpoint: checkpoints/videomae-ep163.pth
  module_name: evals.video_classification_frozen.modelcustom.videomae_encoder
  pretrain_kwargs:
    encoder:
      model_name: vit_large_patch16_224
      tubelet_size: 2
  wrapper_kwargs: {}
```

## Adding a New Task

1. Build CSV files with `<s3_path> <label>` format, one per split
2. For regression: create `zscore_params.json` with `target_mean` and `target_std` from training set
3. Copy an existing YAML config and update:
   - `tag` (unique name for output folder)
   - `dataset_train` / `dataset_val` paths
   - `num_classes` (classification) or `task_type: regression` + `num_targets: 1`
4. Run: `python -m evals.main --fname <your_config.yaml> --devices cuda:0 cuda:1`
