# Probe System Reference

## Overview

Probes are lightweight heads trained on top of a **frozen** encoder to evaluate representation quality. The encoder is never fine-tuned — only probe weights are updated. This tests what information the self-supervised representations actually encode.

## Pipeline

```
Video (mp4) → DataLoader (sample clips, augment) → Frozen Encoder (ViT) → Probe Head → Prediction
```

### Entry Points

- **Single-view**: `evals/video_classification_frozen/eval.py`
- **Multi-view**: `evals/video_classification_frozen_multi/eval.py`

Both are launched via `evals.main` / `evals.main_distributed`, dispatched through `evals/scaffold.py` based on the `eval_name` field in the YAML config.

## Frozen Backbone Loading

`evals/video_classification_frozen/models.py:init_module()` dynamically imports the encoder module specified in config (`module_name`), builds the ViT, loads weights (typically from the `target_encoder` key of a pretraining checkpoint), then freezes all parameters:

```python
model.eval()
for p in model.parameters():
    p.requires_grad = False
```

The encoder module (e.g., `modelcustom/vit_encoder_multiclip.py`) wraps the ViT in a `ClipAggregation` layer that handles multi-clip batching: all clips are concatenated along the batch dimension, run through the encoder in one forward pass, then reshaped back.

Optional multi-layer extraction (`wrapper_kwargs.out_layers: [24, 29, 34, 39]`) fuses intermediate transformer block outputs instead of using only the final layer.

## Probe Head Architectures

All probes take encoder output `[B, N_tokens, D]` and produce `[B, num_classes]` (classification) or `[B, num_targets]` (regression). Set via `probe_type` in config: `"attentive"` (default), `"linear"`, or `"mlp"`. Supported in both single-view and multi-view eval.

### Attentive Probe (`src/models/attentive_pooler.py`)

Cross-attention pooling with a single learned query token that attends over all encoder tokens, followed by `depth-1` self-attention blocks, then a linear layer.

- `AttentiveClassifier` / `AttentiveRegressor`
- Key params: `num_heads` (attention heads), `depth` (num_probe_blocks), `embed_dim`
- Multi-view: supports `use_slot_embeddings` (learnable per-view/clip positional embeddings, factorized or joint)

### Linear Probe (`src/models/linear_pooler.py`)

Mean-pool all tokens → LayerNorm → optional dropout → single linear layer. Supports masked mean-pooling via `key_padding_mask` (for missing views in multi-view early fusion).

- `LinearClassifier` / `LinearRegressor`
- Key params: `use_layernorm` (default True), `dropout` (default 0.0)
- Accepts (and ignores) all AttentiveClassifier kwargs for interface compatibility — can be swapped in by just changing `probe_type`

### MLP Probe (`src/models/linear_pooler.py`)

Mean-pool → LayerNorm → Linear → GELU → Dropout → Linear. Two-layer alternative.

- `MLPClassifier` / `MLPRegressor`
- Key params: `hidden_dim` (default = embed_dim), `dropout` (default 0.1)

### Attentive vs Linear: When to Use Which

| | Attentive | Linear |
|---|---|---|
| **Pooling** | Learned cross-attention (query attends to all tokens) | Mean pool (equal weight to all tokens) |
| **Expressiveness** | High — learns which spatial/temporal tokens matter | Minimal — tests raw representation quality |
| **Trainable params** | ~1-10M (varies with depth/embed_dim) | ~1-2K (single linear layer) |
| **Multi-view slots** | Supports slot embeddings (view/clip identity) | No slot embeddings (tokens are anonymous) |
| **Use case** | ICML paper (depth=4, 16 heads); Nature Medicine primary (depth=1, 16 heads) | Nature Medicine sensitivity analysis (Extended Data) |
| **Interpretation** | d=1: lightweight cross-attention readout (contains linear as special case) | Shows what is linearly accessible in the representation |
| **Config** | `probe_type: attentive`, `num_heads: 16`, `num_probe_blocks: 1` | `probe_type: linear`, `use_layernorm: true`, `dropout: 0.0` |

**Nature Medicine evaluation protocol (Strategy E, adopted 2026-03-11):** Uniform d=1 attentive probes for ALL models. Verified non-harmful for all 4 tested models (+1.2 to +17.3pp over linear). d=1 attentive mathematically contains linear as a strict special case. Linear probes are Extended Data sensitivity analysis. See `uhn_echo/nature_medicine/context_files/decisions/01-probe-architecture.md`.

## How to Use Linear Probes

### Single-view classification

```yaml
# In config YAML
experiment:
  classifier:
    task_type: classification
    probe_type: linear
    use_layernorm: true
    dropout: 0.0
  data:
    num_classes: 13
    # ...
  optimization:
    multihead_kwargs:
      # Linear probes typically use higher LR, no warmup, light or no weight decay
      - lr: 0.001
        start_lr: 0.001
        final_lr: 0.0
        weight_decay: 0.0
        final_weight_decay: 0.0
        warmup: 0.0
      - lr: 0.001
        start_lr: 0.001
        final_lr: 0.0
        weight_decay: 0.001
        final_weight_decay: 0.001
        warmup: 0.0
      - lr: 0.0005
        start_lr: 0.0005
        final_lr: 0.0
        weight_decay: 0.0
        final_weight_decay: 0.0
        warmup: 0.0
```

### Single-view regression

```yaml
experiment:
  classifier:
    task_type: regression
    probe_type: linear
    num_targets: 1
    use_layernorm: true
    dropout: 0.0
  data:
    target_mean: 57.06    # for denormalization
    target_std: 11.33
    # ...
```

### Multi-view (works the same way)

```yaml
eval_name: video_classification_frozen_multi

experiment:
  classifier:
    task_type: regression
    probe_type: linear        # also works in multi-view eval
    num_targets: 1
    # slot embeddings are ignored for linear probes (no attention to inject them into)
    # key_padding_mask still works for masked mean-pooling of missing views
```

### Running

```bash
# Train linear probe
python -m evals.main --fname configs/eval/vitg-384/view/echojepa_view_linear.yaml --devices cuda:0 cuda:1

# Inference only with trained linear probe
python -m evals.main --fname configs/eval/vitg-384/view/echojepa_view_linear.yaml \
    --devices cuda:0 --val_only
```

### Example config

See `configs/eval/vitg-384/view/echojepa_view_linear.yaml` for a complete working example of linear probe classification.

### Linear probe hyperparameter grid

Linear probes are simpler so the search space differs from attentive probes:
- **Learning rates**: typically higher (1e-3, 5e-4, 1e-4) vs attentive (1e-4, 5e-5)
- **Weight decay**: typically lower or zero (0, 1e-3, 1e-2) vs attentive (1e-2, 1e-1, 4e-1)
- **Warmup**: typically zero vs attentive (often nonzero)
- **Epochs**: can be fewer since convergence is faster

## Classification vs Regression

| Aspect | Classification | Regression |
|--------|---------------|------------|
| Loss | `CrossEntropyLoss` (or `FocalLoss`) | `SmoothL1Loss` (Huber) |
| Metric | Top-1 accuracy (higher = better) | MAE (lower = better) |
| Output | `[B, num_classes]` logits | `[B, num_targets]` values |
| Config | `num_classes: N` | `num_targets: 1`, `target_mean`, `target_std` |
| Saved predictions | class, confidence | un-normalized real value, abs_error |

Regression labels are Z-score normalized in the CSV. The config provides `target_mean` and `target_std` for denormalization at inference time (e.g., LVEF: mean=57.06, std=11.33).

## Multi-Head Hyperparameter Grid Search

Multiple probe heads are trained **in parallel** with different hyperparameter combinations. Each head has its own optimizer. Typically 6-16 heads (e.g., 2 learning rates × 3 weight decays). The best-performing head on validation is selected.

```yaml
multihead_kwargs:
  - lr: 0.0001
    weight_decay: 0.01
  - lr: 0.0001
    weight_decay: 0.1
  - lr: 0.00005
    weight_decay: 0.01
  # ... etc
```

Per-head metrics are tracked independently. Checkpoint saves all heads; `best.pt` is based on the best single head.

## Single-View vs Multi-View Probing

### Single-View (`video_classification_frozen`)

Each sample is one video clip (or multiple temporal clips from the same video). Encoder tokens from all clips are concatenated along the token dimension before the probe.

Data format: `path/to/video.mp4 <label>`

### Multi-View (`video_classification_frozen_multi`)

Each sample is a **study** with multiple views (e.g., A4C + PSAX). Uses `VideoGroupDataset` which loads multiple videos per sample.

Data format: `path/to/view1.mp4 path/to/view2.mp4 <label>`

Key multi-view features:
- **Slot embeddings**: Learnable embeddings added to tokens to identify which view/clip they came from. Can be factorized (`view_embed + clip_embed`) or joint. Only used with attentive probes (linear/MLP probes ignore them since there's no attention layer to inject into, but `key_padding_mask` still correctly handles missing views via masked mean-pooling).
- **Miss augmentation**: During training, views are randomly dropped (`miss_augment_prob`) to handle missing views at inference. `slot_present` tensor tracks which views are available.
- **Early fusion** (default): All view tokens concatenated → single probe forward pass with `key_padding_mask` for missing views. Tokens from different views can attend to each other (attentive) or are uniformly averaged (linear/MLP).
- **Late fusion**: Each view processed independently by the probe → weighted average of per-view predictions. Better for interpretability and missing-view handling.

## Study-Level Evaluation with DistributedStudySampler

For study-level tasks (e.g., MIMIC clinical outcomes), each study has ~72 clips from different echo views. The `DistributedStudySampler` (`src/datasets/study_sampler.py`) provides per-epoch random clip selection for training, while prediction averaging is used at evaluation.

### Training: 1 random clip per study per epoch

Set `study_sampling: true` in the data config. The sampler:
1. Groups all CSV rows by study_id (extracted from S3 path: `/s(\d+)/\d+_\d+\.mp4`)
2. Each epoch (seeded by `seed + epoch`), selects 1 random clip per study
3. Shuffles and distributes across ranks (same interface as `DistributedSampler`)

### View-filtered training (Nature Medicine, adopted 2026-03-11)

For view-specific tasks (TAPSE, LVEF, IVSd, etc.), training and val CSVs are pre-filtered to contain only task-relevant views. DistributedStudySampler then picks 1 clip/study/epoch from the filtered set.

**Why filter training views:** Without filtering, ~81% of training clips for TAPSE are non-A4C (uninformative). The probe wastes gradient steps learning to predict the population mean from irrelevant views. With filtering, every gradient step provides useful supervision.

**Why still use 1 clip/study/epoch (not all filtered clips):** Within-study clips share the same label and highly correlated features. Training on all ~11 A4C clips per study in one epoch gives ~11 correlated gradient updates — less efficient than seeing 11 different studies. Study sampling maximizes patient diversity per gradient step.

**Three filter categories:**
- **View-specific tasks** (TAPSE→A4C, LVEF→A4C+A2C, IVSd→PLAX, etc.): view-filtered CSVs (`train_vf.csv`)
- **Hemodynamics tasks** (MR/AS/TR severity): view + modality filter (B-mode only, excludes Doppler)
- **Global tasks** (mortality, biomarkers, diseases): no filter — all clips informative

**CSV locations:** `experiments/nature_medicine/uhn/probe_csvs/{task}/train_vf.csv`, `val_vf.csv`, `test_vf.csv`
**Build script:** `experiments/nature_medicine/uhn/build_viewfiltered_csvs.py`
**Filter definitions:** `TASK_FILTERS` dict in the build script (maps task → allowed views + B-mode flag)
**Filter metadata:** `experiments/nature_medicine/uhn/probe_csvs/{task}/viewfilter_meta.json`

### Evaluation: prediction averaging

Val/test CSVs contain all clips per study. The probe scores each clip independently; predictions are averaged per study for the final study-level result. View-filtered inference (averaging only task-relevant clips) can also be applied post-hoc to any trained probe.

### Config example

```yaml
experiment:
  data:
    study_sampling: true  # Activates DistributedStudySampler
    # View-filtered CSVs for view-specific tasks:
    dataset_train: /path/to/train_vf.csv  # Only task-relevant views
    dataset_val: /path/to/val_vf.csv      # Only task-relevant views
    # Unfiltered CSVs for global tasks:
    # dataset_train: /path/to/train.csv   # All clips per study
    # dataset_val: /path/to/val.csv       # All clips per study
```

### Run scripts (Nature Medicine)

Phase 1 probe training is orchestrated by two shell scripts:
- **`scripts/run_uhn_probe.sh <task>`** — Generic single-task runner. Auto-detects task type (regression/classification), num_classes, Z-score params, view filtering (`train_vf.csv` vs `train.csv`), and study_sampling (disabled for `trajectory_*` tasks). Generates a YAML config on the fly and runs 5 models sequentially with a 20-head HP grid (5 LRs x 4 WDs).
- **`scripts/run_phase1.sh`** — Orchestrator for 18 Phase 1 tasks (RV mechanics, hemodynamics, standard benchmarks, disease detection). Supports `--group` and `--models` for partial runs.

### Epoch guidance

With view-filtered CSVs (~11 A4C clips per study for TAPSE), 20 epochs with 1 clip/study/epoch sees ~20 draws from ~11 unique clips (full coverage with replacement). Recommended:
- 20 epochs for large datasets (>10K studies)
- 35 epochs for medium datasets (~5K studies)
- 3-epoch warmup (proportional to schedule)

### Pipeline integration

`study_sampling` flows through: YAML config → `eval.py` (parsed from `args_data`) → `make_dataloader()` → `init_data()` → `make_videodataset()` → `DistributedStudySampler`. Applied to both train and val dataloaders when `study_sampling: true`.

## Inference-Only Mode

Set `val_only: true` in config (or `OVERRIDE_VAL_ONLY=true` env var). This:
1. Loads a trained probe checkpoint (`probe_checkpoint` config field)
2. Runs one validation pass with progress bar
3. Saves predictions to CSV (`predictions_save_path` config field)
4. Exits immediately (no training)

## Checkpoint Structure

Probe checkpoints (separate from backbone checkpoints) contain:
- `classifiers`: list of state_dicts for all probe heads
- `opt`, `scaler`: optimizer/scaler states (not needed for inference)
- `epoch`, `best_val_acc`, `val_acc_per_head`, `best_epoch_per_head`
- `opt_grid`: hyperparameter configuration used
- `task_type`: "classification" or "regression"

Saved to `{folder}/{eval_tag}/latest.pt` and `best.pt`.

## Prediction Output CSVs

**Classification**: `video_path, true_label, predicted_class, prediction_confidence`

**Regression**: `video_path, label_real, pred_real, abs_error` (values denormalized to real units)

## Competitor Baselines

The `modelcustom/` directories contain encoder adapters that make different backbones produce the same `[B, N, D]` token output format, enabling apples-to-apples probe comparison:
- `vit_encoder_multiclip.py` — VJepa2 (default)
- `echoprime_encoder.py` — EchoPrime
- `panecho_encoder.py` — PanEcho
- `videomae_encoder.py` — VideoMAE

Switch backbone by changing `module_name` in config. See `claude/preprint/encoder-fairness.md` for detailed analysis of encoder output differences and fairness considerations for model comparison.

## Key Config Fields

```yaml
eval_name: video_classification_frozen     # or video_classification_frozen_multi
val_only: true                             # inference mode
probe_checkpoint: /path/to/probe.pt        # trained probe weights
predictions_save_path: /path/to/preds.csv  # output predictions

experiment:
  classifier:
    task_type: regression                  # or classification
    probe_type: attentive                  # or linear, mlp
    num_targets: 1                         # regression outputs
    num_classes: N                         # classification outputs
    num_heads: 16                          # attention heads (attentive only)
    num_probe_blocks: 4                    # depth (attentive only)
    use_layernorm: true                    # linear/mlp only
    dropout: 0.0                           # linear/mlp only
  data:
    dataset_type: VideoDataset             # or VideoGroupDataset
    dataset_train: /path/to/train.csv
    dataset_val: /path/to/val.csv
    resolution: 224
    frames_per_clip: 16
    target_mean: 57.06                     # regression denormalization
    target_std: 11.33
  optimization:
    batch_size: 8
    num_epochs: 20
    use_bfloat16: true
    multihead_kwargs:                      # grid search configs
      - lr: 0.0001
        weight_decay: 0.01

model_kwargs:
  checkpoint: /path/to/backbone.pt
  module_name: evals.video_classification_frozen.modelcustom.vit_encoder_multiclip
  pretrain_kwargs:
    encoder:
      model_name: vit_giant_xformers
      use_rope: true
```
