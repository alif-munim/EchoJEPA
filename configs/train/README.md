# configs/train/

YAML configs for self-supervised pretraining and cooldown (annealing) phases. Organized by model size.

## Directory Structure

```
configs/train/
├── vitg16/                              # ViT-Giant (1B params)
│   ├── pretrain-256px-16f.yaml          #   Kinetics pretrain (template, /your_folder/)
│   ├── pretrain-336px-16f-echo.yaml     #   Echo 18M pretrain (336px, 16 frames)
│   ├── pretrain-echo-336px-16f-0820.yaml#   Echo pretrain (Aug 20 variant)
│   ├── cooldown-256px-64f.yaml          #   Kinetics cooldown (template)
│   ├── cooldown-336px-64f.yaml          #   Echo adapted cooldown (336px, 64 frames)
│   ├── cooldown-384px-64f.yaml          #   Kinetics cooldown 384px (template)
│   ├── cooldown-echo-336px-16f.yaml     #   Echo cooldown (336px, 16 frames)
│   ├── cooldown-echo-336px-16f-0820.yaml#   Echo cooldown (Aug 20 variant)
│   ├── cooldown-echo-336px-16f-0930.yaml#   Echo cooldown (Sep 30 variant)
│   ├── cooldown-echo-336px-32f-0828.yaml#   Echo cooldown (32 frames, Aug 28)
│   └── droid-256px-8f.yaml              #   DROID robotics (action-conditioned)
├── vitl16/                              # ViT-Large (300M params)
│   ├── pretrain-256px-16f.yaml          #   Kinetics pretrain (template)
│   ├── pretrain-mimic-224px-16f.yaml    #   MIMIC-IV-Echo JEPA pretrain (525K videos)
│   ├── pretrain-mimic-224px-16f-cont120.yaml # MIMIC continued pretrain (from ep 120)
│   ├── pretrain-byol-mimic-224px-16f.yaml #  MIMIC BYOL-Video pretrain (ICML rebuttal)
│   ├── cooldown-256px-64f.yaml          #   Kinetics cooldown (template)
│   └── cooldown-mimic-224px-16f.yaml    #   MIMIC-IV-Echo cooldown
└── vith16/                              # ViT-Huge (600M params)
    ├── pretrain-256px-16f.yaml          #   Kinetics pretrain (template)
    └── cooldown-256px-64f.yaml          #   Kinetics cooldown (template)
```

## Two-Phase Training

Training follows a two-phase schedule (see `claude/architecture/pretraining-and-cooldown.md`):

1. **Pretrain** — self-supervised latent prediction on video data with aggressive masking, moderate LR, and strong augmentation
2. **Cooldown (anneal)** — continued training with higher resolution and/or more frames, lower LR, reduced augmentation

Both phases use the same `app: vjepa` entry point and training loop. The key differences are in the data/optimization sections.

### BYOL-Video (ICML Rebuttal)

A separate `app: byol_video` training loop implements self-distillation with momentum encoder for the three-way controlled comparison (JEPA vs BYOL-Video vs MAE). See `app/byol_video/` for implementation. Uses `num_clips: 2` temporal clips per video, symmetric cosine loss, and no masking.

## Naming Convention

```
{phase}-{dataset}-{resolution}-{frames}{-date}.yaml
```

- **Phase**: `pretrain` or `cooldown`
- **Dataset**: `echo` (18M UHN), `mimic` (525K MIMIC-IV-Echo), or omitted (Kinetics/SSv2/HowTo)
- **Resolution**: `224px`, `256px`, `336px`, `384px`
- **Frames**: `8f`, `16f`, `32f`, `64f` (frames per clip)
- **Date**: optional MMDD suffix for experiment variants (e.g., `0820`, `0930`)

## Template vs Echo Configs

Configs with `/your_folder/` and `/your_data_path/` are **Kinetics templates** from the original V-JEPA 2 release. Replace paths before use.

Configs with actual paths (`/home/sagemaker-user/...`) are **echo-specific** configs used for EchoJEPA training.

## Usage

```bash
# Local pretraining
python -m app.main --fname configs/train/vitl16/pretrain-mimic-224px-16f.yaml \
    --devices cuda:0 cuda:1

# Distributed (SLURM)
python -m app.main_distributed --fname configs/train/vitl16/pretrain-mimic-224px-16f.yaml

# Cooldown (same command, different config)
python -m app.main --fname configs/train/vitl16/cooldown-mimic-224px-16f.yaml \
    --devices cuda:0 cuda:1
```

## Key Config Fields

```yaml
app: vjepa                    # Training app (vjepa, vjepa_droid, or byol_video)
folder: /path/to/checkpoints  # Output directory for checkpoints

data:
  datasets:
  - /path/to/train.csv         # Space-separated: <video_path> <dummy_label>
  batch_size: 80               # Per-GPU batch size
  crop_size: 336               # Spatial resolution
  dataset_fpcs: [16]           # Frames per clip
  fps: 24                      # Sampling frame rate
  tubelet_size: 2              # Temporal patch size

optimization:
  checkpoint: /path/to/init.pt # Resume from checkpoint (pretrained weights)
```

## Checkpoint Initialization Reference

The three-way controlled comparison (ICML rebuttal) uses **ImageNet ViT-L** as the common starting point:

| Model | Init Checkpoint | Config | Notes |
|-------|----------------|--------|-------|
| **EchoJEPA-L** | `vitl_in21k.pt` | `pretrain-mimic-224px-16f.yaml` | JEPA (local latent prediction) |
| **EchoBYOL-L** | `vitl_in21k.pt` | `pretrain-byol-mimic-224px-16f.yaml` | BYOL-Video (global self-distillation) |
| **EchoMAE-L** | `vitl_in21k.pt` | (external VideoMAE codebase) | MAE (pixel reconstruction) |
| **EchoJEPA-L-K** | `vitl.pt` | `pretrain-mimic-224px-16f.yaml` | JEPA from Kinetics V-JEPA2 (NOT in controlled comparison) |

**Checkpoint files** (`checkpoints/`):
- `vitl_in21k.pt` — ImageNet-21K ViT-L, heads stripped, flat state dict (296 keys). 2D patch_embed (inflated to 3D at load time).
- `vitl_raw.pth` — Raw ImageNet ViT-L with `module.head.*`/`module.fc_norm.*` keys. Source: S3 `vjepa2-bucket/pretrained_imagenet_vitl.pth`.
- `vitl.pt` — Kinetics V-JEPA2 ViT-L (epoch 40). Has `encoder`/`predictor`/`target_encoder` keys. Used for EchoJEPA-L-K.

## Related

- `configs/eval/` — frozen probe training configs
- `configs/inference/` — inference-only configs
