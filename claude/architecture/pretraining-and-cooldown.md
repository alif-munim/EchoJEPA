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

| Config | Phase | Model | Frames | Crop | Epochs |
|--------|-------|-------|--------|------|--------|
| `vitl16/pretrain-mimic-224px-16f.yaml` | pretrain | ViT-L | 16 | 224 | 240 |
| `vitl16/pretrain-mimic-224px-16f-cont120.yaml` | pretrain (continuation) | ViT-L | 16 | 224 | 120 |
| `vitg16/pretrain-336px-16f-echo.yaml` | pretrain | ViT-g | 16 | 336 | 12 |
| `vitg16/cooldown-336px-16f-echo.yaml` | cooldown | ViT-g | 16 | 336 | 50 |

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

## Distributed Training

- **Local**: `app/main.py` spawns `mp.Process` per GPU, each sets `CUDA_VISIBLE_DEVICES`
- **SLURM**: `app/main_distributed.py` uses `submitit` to submit array jobs
- Backend: NCCL via `torch.distributed.init_process_group` (timeout 1800s)
- Debug mode: `--debugmode True` runs rank 0 in main process (no spawn)
- FSDP not used during pretraining (DDP only)
