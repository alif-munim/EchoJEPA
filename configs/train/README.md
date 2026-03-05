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
│   ├── pretrain-mimic-224px-16f.yaml    #   MIMIC-IV-Echo pretrain (525K videos)
│   ├── pretrain-mimic-224px-16f-cont120.yaml # MIMIC continued pretrain (from ep 120)
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
app: vjepa                    # Training app (vjepa or vjepa_droid)
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

## Related

- `configs/eval/` — frozen probe training configs
- `configs/inference/` — inference-only configs
