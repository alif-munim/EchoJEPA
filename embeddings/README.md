# embeddings/

Precomputed frozen embeddings from EchoJEPA and baseline models, stored as `.npz` files.

## Directory Structure

```
embeddings/
├── lvef/                          # LVEF regression task embeddings
│   ├── echojepa_g_embeddings.npz  #   EchoJEPA ViT-Giant (305 MB)
│   ├── echojepa_l_embeddings.npz  #   EchoJEPA ViT-Large (222 MB)
│   ├── echomae_l_embeddings.npz   #   EchoMAE ViT-Large (222 MB)
│   ├── echoprime_embeddings.npz   #   EchoPrime (113 MB)
│   └── panecho_embeddings.npz     #   PanEcho (168 MB)
├── views/                         # View classification task embeddings
│   ├── echojepa_g_embeddings.npz
│   ├── echojepa_l_embeddings.npz
│   ├── echomae_l_embeddings.npz
│   ├── echoprime_embeddings.npz
│   └── panecho_embeddings.npz
├── test/                          # Test set embeddings (same models)
└── misc/                          # Miscellaneous (echojepa_g only)
```

## NPZ Format

Each `.npz` file contains:

| Key | Shape | Description |
|-----|-------|-------------|
| `embeddings` | `[N, D]` | Mean-pooled frozen representations (D = embed_dim) |
| `labels` | `[N]` | Integer class labels (classification) or float values (regression) |
| `paths` | `[N]` | Video file paths |

```python
import numpy as np

data = np.load("embeddings/views/echojepa_g_embeddings.npz", allow_pickle=True)
embeddings = data["embeddings"]  # [N, 1408] for ViT-Giant
labels = data["labels"]          # [N]
paths = data["paths"]            # [N]
```

## Generating Embeddings

Use `evals/extract_embeddings.py` to extract embeddings from any model. It supports multi-GPU extraction, both classification and regression CSVs, and any backbone with a config YAML.

```bash
# Single GPU
python -m evals.extract_embeddings \
    --config configs/inference/vitg-384/view/echojepa_224px.yaml \
    --data data/csv/uhn_views_22k_test_224px.csv \
    --output embeddings/views/echojepa_g_embeddings.npz \
    --devices cuda:0

# Multi-GPU (4x faster)
python -m evals.extract_embeddings \
    --config configs/inference/vitg-384/view/echojepa_224px.yaml \
    --data data/csv/uhn_views_22k_test_224px.csv \
    --output embeddings/views/echojepa_g_embeddings.npz \
    --devices cuda:0 cuda:1 cuda:2 cuda:3

# Regression (LVEF) — same script, different config and CSV
python -m evals.extract_embeddings \
    --config configs/inference/vitg-384/lvef/echojepa_336px.yaml \
    --data data/csv/a4c_b_lvef_test_224px.csv \
    --output embeddings/lvef/echojepa_g_embeddings.npz \
    --devices cuda:0 cuda:1 cuda:2 cuda:3

# EchoPrime baseline
python -m evals.extract_embeddings \
    --config configs/inference/vitg-384/view/echoprime_224px.yaml \
    --data data/csv/uhn_views_22k_test_224px.csv \
    --output embeddings/views/echoprime_embeddings.npz \
    --devices cuda:0 cuda:1 cuda:2 cuda:3
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | (required) | Inference config YAML (defines model, resolution, frame settings) |
| `--data` | (required) | Input CSV: `<video_path> <label>` (space-separated) |
| `--output` | (required) | Output `.npz` path |
| `--devices` | `cuda:0` | GPUs to use |
| `--batch_size` | `8` | Batch size per GPU |
| `--num_segments` | `1` | Clips per video (more = better but slower) |
| `--num_workers` | `8` | DataLoader workers per GPU |

### Input CSV Format

Standard JEPA-format (space-separated):
```
path/to/video1.mp4 0
path/to/video2.mp4 5
path/to/video3.mp4 -1.234
```

The label column is preserved in the output but not used during extraction. Both integer (classification) and float (regression) labels are supported.

## Probe Training

Use `evals/train_probe.py` to train sklearn linear probes directly on NPZ embeddings (no GPU needed).

```bash
# K-fold cross-validation on a single NPZ
python -m evals.train_probe \
    --data embeddings/views/echojepa_g_embeddings.npz \
    --cv 5 --output_dir results/probes/views/echojepa_g

# Train/val split (separate NPZ files)
python -m evals.train_probe \
    --train embeddings/views/echojepa_g_embeddings.npz \
    --val   embeddings/test/echojepa_g_embeddings.npz \
    --output_dir results/probes/views/echojepa_g

# Multi-model comparison
python -m evals.train_probe \
    --train embeddings/views/echojepa_g_embeddings.npz \
            embeddings/views/echoprime_embeddings.npz \
    --val   embeddings/test/echojepa_g_embeddings.npz \
            embeddings/test/echoprime_embeddings.npz \
    --model_names echojepa_g echoprime \
    --output_dir results/probes/views/comparison
```

Outputs `metrics.json`, `predictions.csv`, and `hp_search.json` per model. See `python -m evals.train_probe --help` for all options.

## Common Uses

- **UMAP visualization** — plot embedding space colored by view/LVEF/etc.
- **Linear probe comparison** — train sklearn classifiers on frozen embeddings
- **Nearest-neighbor retrieval** — find similar echos by cosine similarity
- **Downstream analysis** — clustering, fairness audits, distributional analysis
