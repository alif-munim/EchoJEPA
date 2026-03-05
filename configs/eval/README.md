# configs/eval/

YAML configs for frozen probe training. Each config defines a complete experiment: model backbone, probe architecture, dataset, and optimization hyperparameters.

## Directory Structure

```
configs/eval/
├── vitg-384/                    # ViT-Giant 384px backbone (primary)
│   ├── lvef/                    #   LVEF regression (25 configs)
│   │   └── mini_exp/            #     Checkpoint sweep experiments (10 configs)
│   ├── rvsp/                    #   RVSP regression (9 configs)
│   │   └── ablations/           #     Ablation studies (3 configs)
│   ├── view/                    #   Echo view classification, 13 classes (16 configs)
│   ├── color/                   #   Color Doppler classification, binary (2 configs)
│   ├── quality/                 #   Quality classification, binary (2 configs)
│   ├── zoom/                    #   Zoom level classification, 3 classes (2 configs)
│   ├── tapse/                   #   TAPSE regression (1 config)
│   └── old/                     #   Archived/deprecated configs (36 configs)
└── vitl/                        # ViT-Large backbone (12 configs)
    ├── lvef.yaml, rvsp.yaml, view.yaml         # Echo tasks
    ├── echonet_dynamic_lvef.yaml, enp_lvef.yaml # External datasets
    └── k400.yaml, ssv2.yaml, ...               # Video benchmarks
```

## Naming Conventions

**File naming pattern**: `{dataset_prefix}_{model}_{task}_{resolution}_{variant}.yaml`

**Dataset prefixes**:
- `end_` — EchoNet Dynamic
- `enp_` — EchoNet Pediatric
- (no prefix) — UHN internal dataset

**Model names**: `echojepa`, `echoprime`, `panecho`, `videomae`, `vjepa` (V-JEPA 2 kinetics)

**Suffixes**:
- `_224px`, `_336px` — input resolution
- `_multi` — multi-view (VideoGroupDataset)
- `_1p`, `_10p`, `_50p` — data efficiency subsets (% of training data)
- `_linear` — linear probe (vs default attentive)

## Usage

```bash
# Train a probe locally
python -m evals.main --fname configs/eval/vitg-384/lvef/enp_echojepa_lvef.yaml \
    --devices cuda:0 cuda:1

# Train a probe on SLURM
python -m evals.main_distributed \
    --fname configs/eval/vitg-384/view/echojepa_view_classification_336px.yaml
```

## Key Config Fields

```yaml
eval_name: video_classification_frozen        # Single-view eval module
# eval_name: video_classification_frozen_multi # Multi-view eval module

experiment:
  classifier:
    task_type: classification  # or "regression"
    probe_type: attentive      # "attentive", "linear", or "mlp"
    num_classes: 13            # classification only
    # num_targets: 1           # regression only

  data:
    dataset_type: VideoDataset       # or VideoGroupDataset for multi-view
    dataset_train: data/csv/train.csv
    dataset_val: data/csv/val.csv
    resolution: 336
    frames_per_clip: 16

  optimization:
    multihead_kwargs:          # Hyperparameter grid — trains one probe per entry
    - lr: 0.001
      weight_decay: 0.0
    - lr: 0.0005
      weight_decay: 0.001
```

The `multihead_kwargs` list trains multiple probes in parallel with different hyperparameters. The best is selected by validation performance.

## Model Backbones

Each model uses a different `model_kwargs.module_name`:

| Model | Module | Checkpoint |
|-------|--------|------------|
| EchoJEPA-G | `vit_encoder_multiclip` | `checkpoints/anneal/keep/pt-280-an81.pt` |
| EchoPrime | `echoprime_encoder_multiclip` | (built-in weights) |
| PanEcho | `panecho_encoder_multiclip` | (built-in weights) |
| VideoMAE | `videomae_encoder_multiclip` | `checkpoints/videomae-ep163.pt` |

## Archived Configs (`old/`)

Contains deprecated configs for tasks no longer actively evaluated:
- **RVFX** (RV function) — various checkpoint/hyperparameter variants
- **MVR/TVR** (mitral/tricuspid valve regurgitation) — early experiments
- **Pacemaker/AOV/LAD** — binary classification tasks
- **Video benchmarks** (K400, SSv2, Diving48, etc.) — moved here after vitl/ became canonical for benchmarks
- **Template configs** (`classification_pruned.yaml`, `regression_pruned.yaml`) — skeleton configs with placeholder paths

## Related

- `configs/inference/` — inference-only versions of these configs (set `val_only: true`, require `probe_checkpoint`)
- `configs/train/` — pretraining and cooldown configs
