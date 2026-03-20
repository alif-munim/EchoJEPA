# Pretraining and Cooldown Reference

## Overview

Training happens in two phases:
1. **Pretrain** — learn representations with warmup + cosine LR schedule on shorter clips
2. **Cooldown** — refine representations with linear LR decay on longer clips (or higher FPS)

Both phases run the same training loop (`app/vjepa/train.py`) with the same encoder-predictor-target architecture. The `is_anneal` flag in config switches between pretrain and cooldown behavior.

## Training Objective

The encoder processes visible (context) tokens from masked video. The predictor takes encoder output + mask token positions and predicts the target encoder's representations of the masked regions. Loss is L1 between predictor output and target encoder output. The target encoder is updated via EMA (exponential moving average) of the online encoder — always constant at 0.99925 in all configs.

## Entry Points

```bash
# Local multi-GPU
python -m app.main --fname configs/train/vitl16/pretrain-mimic-224px-16f.yaml --devices cuda:0 cuda:1

# SLURM (submitit)
python -m app.main_distributed --fname configs/train/vitl16/pretrain-mimic-224px-16f.yaml
```

Both dispatch via `app/scaffold.py` → `app.vjepa.train.main()`.

## Pretrain Phase (`is_anneal: false`)

Uses `WarmupCosineSchedule` from `src/utils/schedulers.py`:
- **Warmup**: linear ramp from `start_lr` → `ref_lr` over `warmup` epochs (typically 40)
- **Cosine decay**: `ref_lr` → `final_lr` over remaining epochs

In echo configs, `final_lr == ref_lr` (constant LR after warmup — no decay during pretrain).

Resume: checks for `latest.pt` in output folder; if found, resumes from stored epoch/iteration, burning through LR/WD/mask schedules to the correct step.

## Cooldown Phase (`is_anneal: true`)

Uses `LinearDecaySchedule` from `src/utils/schedulers.py`:
- **No warmup** (warmup=0 in all cooldown configs)
- **Linear decay**: `ref_lr` → `final_lr` over all steps
- Final LR is typically 500-5000x smaller than peak (e.g., 1.09e-4 → 2.08e-7)

Loading: always force-loads from a pretrain checkpoint (`anneal_ckpt` + `force_load_pretrain: true`). This resets EMA to step 0 and starts fresh epoch counting — any partial cooldown `latest.pt` is ignored on first run.

## Config Inventory

All configs live in `configs/train/{vitl16,vith16,vitg16}/`.

### Kinetics Configs (multi-dataset: K710 + SSv2 + HowTo100M)

| Config | Phase | Model | Frames | Crop | Epochs |
|--------|-------|-------|--------|------|--------|
| `vitl16/pretrain-256px-16f.yaml` | pretrain | ViT-L | 16 | 256 | 10 |
| `vitl16/cooldown-256px-64f.yaml` | cooldown | ViT-L | 64 | 256 | 40 |
| `vith16/pretrain-256px-16f.yaml` | pretrain | ViT-H | 16 | 256 | 10 |
| `vith16/cooldown-256px-64f.yaml` | cooldown | ViT-H | 64 | 256 | 40 |
| `vitg16/pretrain-256px-16f.yaml` | pretrain | ViT-g | 16 | 256 | 800 |
| `vitg16/cooldown-256px-64f.yaml` | cooldown | ViT-g | 64 | 256 | 40 |

### Echo/MIMIC Configs (single medical dataset)

| Config | Phase | App | Model | Frames | Crop | Batch | Epochs |
|--------|-------|-----|-------|--------|------|-------|--------|
| `vitl16/pretrain-mimic-224px-16f.yaml` | pretrain | vjepa | ViT-L | 16 | 224 | 128 | 240 |
| `vitl16/pretrain-mimic-224px-16f-cont120.yaml` | pretrain (cont.) | vjepa | ViT-L | 16 | 224 | 128 | 120 |
| `vitl16/cooldown-mimic-224px-16f.yaml` | cooldown | vjepa | ViT-L | 16 | 224 | 128 | 60 |
| `vitl16/pretrain-21-mimic-224px-16f.yaml` | pretrain | **vjepa_2_1** | ViT-L | 16 | 224 | 128 | 120 |
| `vitl16/cooldown-21-mimic-224px-32f.yaml` | cooldown | **vjepa_2_1** | ViT-L | **32** | 224 | 64 | 30 |
| `vitb16/pretrain-21-mimic-224px-16f.yaml` | pretrain | **vjepa_2_1** | ViT-B | 16 | 224 | 128 | 120 |
| `vitb16/cooldown-21-mimic-224px-32f.yaml` | cooldown | **vjepa_2_1** | ViT-B | **32** | 224 | 64 | 30 |
| `vitg16/pretrain-336px-16f-echo.yaml` | pretrain | vjepa | ViT-g | 16 | 336 | 80 | 12 |
| `vitg16/cooldown-336px-16f-echo.yaml` | cooldown | vjepa | ViT-g | 16 | 336 | 80 | 50 |

## Kinetics vs Echo: Key Differences

### Data Augmentation

| Parameter | Kinetics | Echo |
|-----------|----------|------|
| `random_resize_aspect_ratio` | [0.75, 1.35] | [0.9, 1.1] |
| `random_resize_scale` | [0.3, 1.0] | [0.5, 1.0] |

Echo uses much tighter augmentation ranges to preserve cardiac anatomy. Ultrasound images are nearly square and aggressive crops/distortions would destroy clinical features.

### Training Scale and LR

| Parameter | Kinetics | Echo |
|-----------|----------|------|
| Datasets | 3 (K710 33.5%, SSv2 10%, HowTo 56.5%) | 1 (MIMIC or UHN) |
| Global batch size | 3072 (multi-node) | 640-1024 (single node) |
| Peak LR | 5.25e-4 | 1.09e-4 to 1.75e-4 |
| FPS | 4 | 4-8 |

Echo LR is scaled from the kinetics recipe: `new_lr = 5.25e-4 × (actual_batch / 3072)`.

### Initialization

| Aspect | Kinetics | Echo |
|--------|----------|------|
| Start from | Random init | Force-load kinetics checkpoint |
| `force_load_pretrain` | false | true |

All echo configs force-load from a pretrained kinetics checkpoint — echo training is always a domain adaptation, never from scratch.

### Cooldown Differences

| Parameter | Kinetics Cooldown | Echo Cooldown |
|-----------|-------------------|---------------|
| Frames | 64 (4x pretrain) | 16-48 (same or modest increase) |
| FPS | 4 (same) | 24-30 (much higher) |
| Final LR | 1.0e-6 | 2.08e-7 (more aggressive) |
| Weight decay | 0.04 (same as pretrain) | 0.01 (reduced) |
| Epochs | 40 | 50-60 |

Kinetics cooldown extends temporal context by using 4x longer clips at the same FPS. Echo cooldown instead increases FPS (more temporal resolution within similar clip duration) and uses a more aggressive LR decay with reduced weight decay.

## Masking Strategy

Masking is configured identically across all pretrain and cooldown configs:

```yaml
mask:
  - spatial_scale: [0.15, 0.15]   # Small patches (15% spatial)
    temporal_scale: [1.0, 1.0]    # Full temporal span
    num_blocks: 8                  # 8 random blocks
    aspect_ratio: [0.75, 1.5]
    max_temporal_keep: 1.0

  - spatial_scale: [0.7, 0.7]    # Large patches (70% spatial)
    temporal_scale: [1.0, 1.0]
    num_blocks: 2                  # 2 random blocks
    aspect_ratio: [0.75, 1.5]
    max_temporal_keep: 1.0
```

Two mask generators per sample: one with many small blocks (8 × 15%), one with few large blocks (2 × 70%). Both span the full temporal extent. The encoder sees the context (unmasked) tokens; the predictor must reconstruct target (masked) token representations.

`MaskCollator` in `src/masks/multiseq_multiblock3d.py` generates these masks. `num_mask_tokens` (echo configs set to 10, kinetics use default None) controls how many learnable mask tokens the predictor uses.

## Resume and Checkpoint Handling

### Pretrain Resume
- Checks for `{folder}/latest.pt`
- If found: loads encoder, predictor, target_encoder, optimizer, scaler, epoch, iteration
- Burns through LR/WD/EMA schedules to reach the correct step
- Continues seamlessly

### Cooldown Force-Load
- `force_load_pretrain: true` + `anneal_ckpt` path
- Loads encoder and target_encoder weights from pretrain checkpoint
- Resets: epoch=0, iteration=0, EMA schedule from step 0, fresh optimizer
- Predictor weights are NOT loaded (reinitialized) unless present in checkpoint

### Checkpoint Contents
```python
{
    "encoder": encoder.state_dict(),
    "target_encoder": target_encoder.state_dict(),
    "predictor": predictor.state_dict(),
    "opt": optimizer.state_dict(),
    "scaler": grad_scaler.state_dict(),
    "epoch": int,
    "batch_size": int,
    "world_size": int,
    "loss": float,
}
```

Saved every epoch. `prune_local_checkpoints()` keeps only the last 4 epoch checkpoints. `robust_checkpoint_loader()` retries up to 3 times with exponential backoff for S3/NFS reliability.

## MIMIC ViT-L Config Analysis

Comparison of MIMIC pretrain/cooldown configs against the V-JEPA 2 paper recipe. See `vjepa2-paper-recipes.md` for full paper numbers.

### What matches the paper

| Parameter | MIMIC Config | Paper | Notes |
|---|---|---|---|
| LR scaling | 1.75e-4 | 5.25e-4 × (1024/3072) | Correct linear scaling |
| Warmup steps | 40ep × 300ipe = 12K | 12,000 | Exact match |
| LR shape | Warmup → constant | Warmup → constant | Correct |
| EMA | 0.99925 fixed | 0.99925 fixed | Match |
| Weight decay | 0.04 | 0.04 | Match |
| Masking | 8@0.15 + 2@0.7 | [0.15, 0.7] dual | Match |
| Loss | L1 | L1 | Match |

### Gaps identified

**1. Cooldown doesn't increase frames or resolution.**
Paper goes 16→64 frames and 256→384px during cooldown. MIMIC cooldown stays at 224px/16 frames. The paper shows +1.3% IN1K, +1.2% SSv2, +0.7% K400 from resolution increase alone. However, the V-JEPA 2 recipe (4fps × 64 frames = 16s) is impossible on MIMIC data (max video is 6s). See recommended alternative below.

**2. Training passes through data.**
Paper: ~35 passes (252K steps × 3072 batch / 22M samples). MIMIC: ~140 passes (72K × 1024 / 525K). 4× more passes through data, but rising loss during JEPA pretraining is expected (see below) and doesn't indicate overfitting.

**3. Resolution is 224px vs 256px.**
14×14 spatial grid vs 16×16 = 196 vs 256 tokens/frame. Acceptable for VideoMAE comparison but not optimal.

**4. FPS is 8 vs 4.**
Gives 2s vs 4s temporal window. Reasonable for echo (fast, repetitive motion), but narrower context.

### Recommended MIMIC cooldown (frame increase)

Based on video statistics (see `data/mimic-video-statistics.md`): MIMIC videos are 30fps native, median 74 frames (2.5s), ~42% single-frame stills.

Best feasible cooldown: **24fps, 32 frames**:
- 100% coverage (all videos have ≥32 native frames)
- 1.3s temporal window (1–2 cardiac cycles)
- 2× tokens (3136 vs 1568) — halve batch to 64/GPU
- No padding needed for actual videos

```yaml
# Suggested cooldown-mimic-224px-32f.yaml
data:
  dataset_fpcs: [32]
  fps: 24
  batch_size: 64       # halved for 2x tokens
optimization:
  epochs: 30            # ~9K steps
  lr: 1.75e-4
  final_lr: 1.0e-6
  is_anneal: true
  warmup: 0
```

## Loss Curve Behavior in JEPA Pretraining

**Rising pretraining loss is expected and normal** for V-JEPA / joint-embedding methods. The target encoder is updated via EMA of the online encoder. As the encoder learns better representations, prediction targets become harder (more informative), so L1 loss naturally rises — even as downstream feature quality improves.

From Meta maintainer David Fan ([issue #56](https://github.com/facebookresearch/vjepa2/issues/56)):
> "The pretraining loss is not a 'traditional' loss...the target is moving (we use EMA)...if the prediction target becomes easier, the loss will look nice, but maybe the model didn't learn anything useful."

Typical JEPA loss curve: sharp drop in early epochs → gradual rise for the remainder of training. **Later checkpoints with higher loss produce better downstream probe results.** Multiple users independently confirmed this pattern across datasets (ImageNet, Kinetics, custom data).

**What to watch for:**
- **Gradual loss rise**: Normal, indicates improving representations
- **Sudden loss drop**: Potential partial collapse — may need restart with different seed or hyperparameter tuning ([issue #74](https://github.com/facebookresearch/vjepa2/issues/74))
- **Downstream probe evaluation**: The only reliable way to assess feature quality. The paper evaluates every 60K steps.

V-JEPA 2.0 ViT-L on MIMIC showed this exact pattern: 0.53 → 0.47 (epoch 9) → 0.57 (epoch 224). This is healthy training, not overfitting.

## V-JEPA 2.1 Pretraining

The V-JEPA 2.1 training loop lives in `app/vjepa_2_1/train.py` (set `app: vjepa_2_1` in config). See `vjepa21-code-diff.md` for architecture differences and `vjepa2-paper-recipes.md` for hyperparameters.

Key config additions for 2.1:
```yaml
app: vjepa_2_1
loss:
  predict_all: true           # context loss
  weight_distance_loss: false  # distance-weighted context loss
  offset_context_loss: false
model:
  lambda_value_vid: 0.5       # context loss weight (video)
  lambda_value_img: 0.7       # context loss weight (image)
  lambda_progressive: false    # IMPORTANT: hardcoded warmup (15K-30K steps) assumes 135K+ total steps.
                               # For short runs (<40K steps), set false for constant lambda from step 0.
  levels_predictor: 4          # hierarchical 4-layer output (1 for distilled checkpoints)
  n_output_distillation: 4     # must match levels_predictor (1 for distilled checkpoints)
  normalize_predictor: false
  modality_embedding: true     # required for distilled checkpoints (have img/video mod embeds)
  img_temporal_dim_size: 1     # required for distilled checkpoints (have patch_embed_img)
```

**Distilled vs full 2.1 configs**: Public ViT-B/L checkpoints are distilled from ViT-G and use `n_output_distillation: 1`, `levels_predictor: 1`, `pred_depth: 12`. Full 2.1 recipe (ViT-g/G) uses `4`, `4`, `24`. See `vjepa21-code-diff.md` for details.

**Operational notes**: ViT-B MIMIC training verified at 3.4s/iter on 8× A100 80GB (batch=128, num_workers=4). batch=256 caused OOM; num_workers=8 exhausted /dev/shm. See `vjepa21-code-diff.md` operational notes for full issue tracker.

## Distributed Training

- **Local**: `app/main.py` spawns `mp.Process` per GPU, each sets `CUDA_VISIBLE_DEVICES`
- **SLURM**: `app/main_distributed.py` uses `submitit` to submit array jobs
- Backend: NCCL via `torch.distributed.init_process_group` (timeout 1800s)
- Debug mode: `--debugmode True` runs rank 0 in main process (no spawn)
- FSDP not used during pretraining (DDP only)
