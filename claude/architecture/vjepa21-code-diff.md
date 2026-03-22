# V-JEPA 2.1 Code: Architecture Differences from V-JEPA 2.0

Reference code copied from `refs/vjepa2/app/vjepa_2_1/` into `app/vjepa_2_1/` (2026-03-20), with EchoJEPA infrastructure ported on top.

## Encoder (`app/vjepa_2_1/models/vision_transformer.py`)

### Hierarchical multi-layer output
V-JEPA 2.0 has a single `self.norm` and returns `self.norm(x)` from the final block. V-JEPA 2.1 replaces this with `self.norms_block` — a `ModuleList` of per-layer LayerNorms at 4 "hierarchical layers" (e.g., `[9, 19, 29, 39]` for depth=40). During training (`training_mode=True`), outputs are concatenated along the embedding dim: `[B, N, 4*embed_dim]`. At inference, only the last layer is returned.

### No sincos positional embedding
V-JEPA 2.0 has `self.pos_embed` (frozen sincos). V-JEPA 2.1 removes it entirely — RoPE is the only position encoding. If RoPE is off, there is no position information.

### Register tokens
`n_registers` extra learnable tokens appended to the sequence. RoPE is not applied to them. They absorb global information without polluting patch representations.

### Modality embeddings
Learned `img_mod_embed` / `video_mod_embed` added to tokens after patch embedding. Separate `patch_embed_img` (3D conv, tubelet_size=1) handles image inputs.

### CLS token support
`has_cls_first` flag excludes the first token from RoPE rotation.

### RoPE interpolation
`interpolate_rope=True` rescales spatial positions to a pretrained grid size for resolution transfer.

### Initialization options
`init_type`: `"default"` (trunc_normal), `"xavier_uniform"`, `"xavier_normal"`.

## Predictor (`app/vjepa_2_1/models/predictor.py`)

### Input projection
V-JEPA 2.0: single linear `embed_dim → pred_dim`. V-JEPA 2.1: 2-layer MLP `4*embed_dim → embed_dim → pred_dim` (when using hierarchical layers).

### Output projection
V-JEPA 2.0: `pred_dim → embed_dim`. V-JEPA 2.1: `pred_dim → 4*embed_dim` to match all hierarchical layers.

### Dual output (context + prediction)
When `return_all_tokens=True`, separate projection head for context tokens: `predictor_proj_context`. Returns `(x_pred, x_context)` tuple instead of single tensor.

### No positional embedding
Like encoder, removed sincos; RoPE only.

### Modality embedding
Learned image/video tokens applied after token sorting, before transformer blocks.

## Wrappers (`app/vjepa_2_1/wrappers.py`)

### MultiSeqWrapper (encoder)
- Adds `training_mode` flag → controls hierarchical vs final-layer output
- Adds `gram_mode` → 2× spatial upsample → backbone → 2× downsample (super-res path)
- Exposes `self.embed_dim = backbone.embed_dim`

### PredictorMultiSeqWrapper
- Returns `(outs_pred, outs_context)` instead of single list
- Passes `mod="video"|"image"` instead of `has_cls`

## Training Loop (`app/vjepa_2_1/train.py`)

### Context loss
Predictor reconstructs both masked and visible tokens. Total loss:
```
loss = loss_pred + λ * loss_context
```
λ warmed up from 0 via `Lambda_LinearWarmupHold` (0 until step 15K, ramp to λ by 30K, then constant). Separate λ for image (`lambda_value_img`) and video (`lambda_value_vid`).

### Distance-weighted context loss
When `weight_distance_loss=True`, context loss weighted by spatial/temporal distance to nearest masked token via `compute_mask_distance`.

### Joint image+video training
A fraction of GPU ranks (`img_rank_ratio`, default 25%) process image batches. Remaining ranks process video. Separate batch sizes, crop sizes, mask configs per modality.

### Causal attention
`is_causal` / `pred_is_causal` enable causal masking in SDPA.

### Loss regulation
`loss_reg_std_mult` tracks trailing loss window and skips optimizer steps when current loss exceeds `mean + k*std`.

### Optimizer options
Supports RAdamW (`use_radamw`). Uses `LinearDecaySchedule` for annealing. Zero weight decay for bias/1D parameters.

## EchoJEPA Ports (applied to 2.1 code)

The following EchoJEPA infrastructure was ported into `app/vjepa_2_1/train.py`:
- S3 checkpoint upload via `boto3` (multipart for >5GB)
- Checkpoint pruning (`prune_local_checkpoints`)
- `dist.barrier()` at epoch boundaries
- Robust DataLoader recovery (rebuild after exceeding retries)
- Mid-epoch resume (saves `itr` in checkpoint)
- `persistent_workers` config plumbing
- `mp.set_sharing_strategy("file_system")`
- `try/finally` with `dist.destroy_process_group()`
- `.pt` extension (consistent with rest of codebase)

## Checkpoint Compatibility: `force_load_pretrain` + `ema_encoder`

The publicly released V-JEPA 2.1 checkpoints (e.g., `vjepa2_1_vitl_dist_vitG_384.pt`) differ from the save format used by the 2.1 training loop:

| Field | Training loop saves | Public checkpoint has |
|---|---|---|
| Target encoder key | `target_encoder` | `ema_encoder` |
| Key prefix | `module.backbone.` (DDP + wrapper) | `module.backbone.` |
| Predictor depth (ViT-L distilled) | 24 (full recipe) | **12** (distilled) |
| Predictor embed | Sequential (4-layer input) | **Single Linear** (1-layer input) |
| Hierarchical layers | 4 | **1** (distilled) |

### Changes made to handle this

**`app/vjepa_2_1/utils.py`**:

1. **`load_checkpoint`**: Falls back from `target_encoder` to `ema_encoder` key:
   ```python
   target_key = "target_encoder" if "target_encoder" in checkpoint else "ema_encoder"
   ```

2. **`init_video_model`**: Added `n_output_distillation` parameter, forwarded to **both** the encoder and predictor constructors. Controls how many hierarchical layers are output/predicted:
   - `n_output_distillation=4` (default) → full 2.1 recipe, 4-layer hierarchical, Sequential predictor_embed
   - `n_output_distillation=1` → distilled architecture, 1-layer, single Linear predictor_embed
   - **Critical**: must be passed to the encoder too (controls `out_layers_distillation`), not just the predictor. Without this, the encoder outputs 4× embed_dim but the predictor expects 1× embed_dim → shape mismatch at `predictor_embed`.

**`app/vjepa_2_1/train.py`**:

3. **`force_load_pretrain` path**: New checkpoint loading mode with `_load_with_shape_filter` that strips `module.backbone.` prefixes, skips shape-mismatched keys (e.g., distilled predictor_proj outputs teacher dim 1664 instead of student dim 768), and loads remaining weights. Handles `ema_encoder` → `target_encoder` mapping. Resets epoch/iteration/optimizer (fresh start).

4. **`n_output_distillation` config field**: Read from `model.n_output_distillation` in YAML, passed through to `init_video_model`. When set to `1`, both encoder and predictor architecture match the distilled checkpoint.

5. **`levels_predictor` config field**: Controls `forward_target()` normalization. When `1`, applies single `layer_norm` to encoder output (matching distilled model's single-layer prediction). When `4`, splits into 4 chunks and normalizes each separately.

6. **`embed_dim_encoder` lookup**: Expanded from 3 model names to a full dictionary covering all ViT sizes (tiny through gigantic). The original code only handled vit_large/vit_giant_xformers/vit_gigantic_xformers with a silent fallback on unknown names.

7. **Robust `mask_meters` update**: Wrapped in try/except since the collated batch structure varies depending on sample tuple length (3 vs 4 elements from VideoDataset). This is logging-only code.

### Config fields for distilled checkpoints

```yaml
model:
  pred_depth: 12               # match distilled predictor depth
  n_output_distillation: 1     # single-layer prediction (not hierarchical 4)
  levels_predictor: 1          # single layer_norm in forward_target
  modality_embedding: true     # checkpoint has img/video mod embeds
  img_temporal_dim_size: 1     # checkpoint has patch_embed_img
  interpolate_rope: true       # checkpoint uses RoPE interpolation
optimization:
  force_load_pretrain: true
  anneal_ckpt: checkpoints/vjepa2_1_vitl.pt
```

### Resume after force_load_pretrain

After initial loading, switch to standard resume:
```yaml
optimization:
  force_load_pretrain: false    # disable force-load
meta:
  load_checkpoint: true         # resume from latest.pt
```

## Image-Aware Mask Collator

`src/masks/multiseq_multiblock3d.py` updated twice:

1. Handle both video samples (with `clip_indices`) and image samples (without). Image samples get `fpc=1`. Unknown FPCs not in `mask_generators` are safely skipped with `if fpc in filtered_batches` guard.

2. **EchoJEPA 4-element sample fix**: The EchoJEPA `VideoDataset` returns `(buffer, label, clip_indices, sample_uri)` — 4 elements, not 3. The original collator used `sample[-1]` to find clip_indices, which now returns the URI string. Fixed to use `sample[2]` (clip_indices is always the 3rd element).

## Distilled Checkpoint Shape Mismatches

The public distilled checkpoints (ViT-B, ViT-L) were trained to predict the **teacher's** (ViT-G, 1664-dim) feature space. The predictor projection heads have teacher dimensions:
- `predictor_proj`: `[1664, 384]` (teacher) vs `[768, 384]` (ViT-B student)
- `predictor_proj_context`: same mismatch

The `force_load_pretrain` path handles this via `_load_with_shape_filter`: keys with shape mismatches are logged and skipped, leaving those layers randomly initialized. The encoder and predictor blocks (which have matching shapes) load successfully.

## MIMIC Training Configs

### ViT-L (`configs/train/vitl16/`)

**`pretrain-21-mimic-224px-16f.yaml`** — Domain adaptation from V-JEPA 2.1 ViT-L:
- `app: vjepa_2_1`, `force_load_pretrain: true` from `vjepa2_1_vitl.pt` (4.8GB)
- Distilled architecture: `pred_depth: 12`, `n_output_distillation: 1`, `levels_predictor: 1`
- Context loss: `predict_all: true`, `lambda_value_vid: 0.5`, `lambda_progressive: false`
- batch=128/GPU, 120 epochs, 36K steps, ~70 passes
- Warmup → constant LR at 1.75e-4

**`cooldown-21-mimic-224px-32f.yaml`** — Cooldown with frame increase:
- **32 frames at 24fps** (vs 16 @ 8fps in pretrain): 2× tokens, 1.3s window
- Batch 64/GPU; linear decay 1.75e-4 → 1e-6, 30 epochs (9K steps)

### ViT-B (`configs/train/vitb16/`)

**`pretrain-21-mimic-224px-16f.yaml`** — Same pattern, from `vjepa2_1_vitb.pt` (1.6GB):
- batch=128/GPU, LR=1.75e-4, 120 epochs, `num_workers: 4`
- Same as ViT-L except model_name=vit_base

**`cooldown-21-mimic-224px-32f.yaml`** — Same cooldown pattern, batch=64/GPU.

### Training flow
```
vjepa2_1_vit{b,l}.pt (distilled from ViT-G-384)
  → pretrain-21-mimic-224px-16f.yaml (36K steps, 16f@8fps)
    → cooldown-21-mimic-224px-32f.yaml (9K steps, 32f@24fps)
      → frozen probes (configs/eval/)
```

## Operational Notes from First Training Run (ViT-B on MIMIC)

### Issues encountered and fixes

| Issue | Symptom | Fix |
|---|---|---|
| `embed_dim_encoder` not set for vit_base | `NameError` in `forward_target` | Expanded model→dim lookup to cover all ViT sizes |
| Encoder outputs 4×embed_dim, predictor expects 1× | `mat1 and mat2 shapes cannot be multiplied (71680x3072 and 768x384)` | Pass `n_output_distillation` to **both** encoder and predictor in `init_video_model` |
| Distilled predictor_proj has teacher dim (1664) not student (768) | `RuntimeError: size mismatch` on `load_state_dict` | Added `_load_with_shape_filter` to skip mismatched keys |
| Mask collator reads `sample[-1]` but EchoJEPA returns 4-element tuple | All samples get `fpc=1`, empty batches → `IndexError: list index out of range` | Changed collator to use `sample[2]` for clip_indices |
| Mask meter reads collated batch last element | `AttributeError: 'str' object has no attribute 'size'` | Wrapped in try/except (logging only) |
| `batch_size: 256` per GPU | OOM on 3 of 8 GPUs, DDP hangs | Reduced to `batch_size: 128` |
| `num_workers: 8` × 8 GPUs = 64 persistent workers | `/dev/shm` exhaustion (72GB/144GB) → `unable to allocate shared memory` → data pipeline deadlock. GPUs spin at 100% waiting for batches, **appears** as slow GPU compute but is actually a data stall. This was the critical fix: 44s→16s→3.4s/iter. | Reduced to `num_workers: 4` (32 workers, 38% shm) |

### Verified training performance (ViT-B, 8× A100 80GB)

| Metric | Value |
|---|---|
| Batch size | 128/GPU, 1024 global |
| GPU memory | ~22 GB / 80 GB |
| GPU utilization | 93-99% all 8 GPUs |
| Iter time (steady state) | **3.4s** |
| Per epoch (300 iters) | ~17 minutes |
| Total (120 epochs, 36K iters) | **~34 hours (1.4 days)** |
| /dev/shm usage | 38% of 144GB |
| Initial loss | 0.66 |
| Loss at iter 65 | 0.47 |

### Speed comparison: V-JEPA 2.1 ViT-B vs V-JEPA 2.0 ViT-L

V-JEPA 2.1 ViT-B runs at **3.4s/iter** — essentially identical to V-JEPA 2.0 ViT-L's **3.5s/iter** at the same batch size. This is because the 4× smaller encoder cancels out the 2.1 training overhead.

Note: earlier attempts showed 16s/iter and 44s/iter, but these were from failing runs (OOM and /dev/shm exhaustion causing DDP hangs, not real compute cost).

**Why ViT-B 2.1 ≈ ViT-L 2.0 in wall-clock time:**

The predictor processes the **same number of tokens** in both versions — it always sees the full sequence (context + masked tokens, concatenated and sorted). The transformer blocks are identical. The only extra work in 2.1 is after the predictor blocks:

- V-JEPA 2.0: one `predictor_proj` on masked tokens → one `loss_fn` call
- V-JEPA 2.1: `predictor_proj` on masked tokens + `predictor_proj_context` on context tokens → two `loss_fn` calls + `compute_mask_distance` for distance weighting

This adds ~10-20% predictor overhead. Combined with ViT-B's 4× cheaper encoder (80M vs 300M params), the total roughly balances:

| Component | V-JEPA 2.0 ViT-L | V-JEPA 2.1 ViT-B | Relative cost |
|---|---|---|---|
| Encoder (context tokens) | 300M, depth=24 | 80M, depth=12 | **~4× cheaper** |
| Target encoder (all tokens, no grad) | 300M, depth=24 | 80M, depth=12 | **~4× cheaper** |
| Predictor (all tokens) | 22M, depth=12 | 22M, depth=12 | Same |
| Predictor output | 1 projection, 1 loss | 2 projections, 2 losses + distance | ~1.2× more |
| **Net** | **3.5s/iter** | **3.4s/iter** | **~1×** |

### Estimated ViT-L 2.1 training time

The predictor (22M params, depth=12, dim=384) is a **fixed cost** (~1.5s/iter) regardless of encoder size — it's the same architecture in both ViT-B and ViT-L configs. The encoder cost scales with model size:

| Component | ViT-B 2.1 | ViT-L 2.1 (estimated) |
|---|---|---|
| Encoder + target encoder | ~1.5s | ~6s (4× bigger: 300M vs 80M, depth 24 vs 12) |
| Predictor (identical) | ~1.5s | ~1.5s |
| Loss + overhead | ~0.4s | ~0.4s |
| **Total** | **3.4s/iter** | **~8s/iter** |
| Per epoch (300 iters) | 17 min | ~40 min |
| Total (120 epochs, 36K iters) | **34 hours (1.4 days)** | **~80 hours (3.3 days)** |

ViT-L 2.1 is ~2.3× slower than ViT-B 2.1 (not 4×) because the predictor's fixed cost doesn't scale with encoder size.

### Monitoring commands

```bash
# Loss curve (CSV, every iteration)
tail -f checkpoints/pretrain_21/mimic/vjepa2_1_vitb_224px_16f/log_r0.csv

# Console log (every 10 iterations)
tail -f $(ls -t logs/pretrain-21-vitb-mimic-*.log | head -1)

# GPU utilization
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv

# Shared memory health
df -h /dev/shm
```

### /dev/shm exhaustion after crashes

PyTorch DataLoader workers use `/dev/shm` (shared memory) for IPC — each worker `mmap`s segments to pass prefetched batches to the main process. When processes crash or are killed ungracefully (SIGKILL, OOM, DDP hang), these anonymous mmap'd segments become **orphaned** and persist in `/dev/shm` until a reboot or manual cleanup.

Multiple crash-restart cycles accumulate orphaned segments. With 8 GPUs × 4 workers × large video batches, each crash cycle leaks several GB. After a few restarts, `/dev/shm` fills up (143GB/144GB), causing `unable to allocate shared memory` errors that prevent any new training from starting.

**Recovery:**
```bash
# Check usage
df -h /dev/shm

# If full, clear orphaned segments (safe if no other training runs active)
rm -rf /dev/shm/*

# Or reboot the instance / restart the kernel to reclaim
```

**Prevention:**
- Use `num_workers: 4` (not 8) to halve per-crash leak
- Clean up /dev/shm between crash-restart cycles
- If a run crashes, check `/dev/shm` before relaunching

### Partial collapse and restart (epoch 153-154)

At epoch 153-154 (original seed=231), loss suddenly dropped from ~0.62 to ~0.45 — a partial collapse (sudden drops are bad in JEPA training; rising loss is normal). Recovery steps:
1. Stopped the run
2. Deleted `latest.pt` (points to collapsed checkpoint) and `e154.pt`
3. Set `read_checkpoint: e144.pt` and `seed: 571` in config
4. Cleared `/dev/shm` and relaunched
5. After successful resume, set `read_checkpoint: null` so future restarts use `latest.pt`

### Checkpoint outputs

- `latest.pt` — overwritten every epoch
- `e{N}.pt` — every 5 epochs, pruned to keep 3 most recent, uploaded to S3
- S3 URI: `s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts/checkpoints/vjepa-2.1-b/`
- CSV log: `log_r0.csv` in the checkpoint folder (all iterations, rank 0)
