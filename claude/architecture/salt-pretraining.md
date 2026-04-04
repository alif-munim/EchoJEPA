# SALT Pretraining

SALT (Static-teacher Asymmetric Latent Training) is a two-stage alternative to V-JEPA's EMA self-distillation, based on [arXiv:2509.24317](https://arxiv.org/abs/2509.24317). It replaces the co-evolving momentum target encoder with a frozen teacher trained via pixel reconstruction.

## Two-Stage Design

### Stage 1: V-Pixel (Pixel Reconstruction)

Train an encoder + lightweight pixel decoder using MSE loss on masked patches. Uses the same multi-block masking as V-JEPA (NOT VideoMAE's random-tube masking — the SALT paper found multi-block masking works better).

- Encoder processes only visible (unmasked) patches via `MultiSeqWrapper`
- Decoder reconstructs raw pixel values of masked patches
- Loss: MSE on per-patch normalized pixel targets (VideoMAE-style normalization)
- No target encoder, no EMA — just standard supervised reconstruction
- Config: `model.stage: 1`

### Stage 2: Frozen-Teacher Student Training

Freeze the Stage 1 encoder as teacher. Train a new student encoder + predictor to predict the teacher's latent representations at masked positions.

- Teacher processes FULL unmasked video (all patches) → hierarchical features
- Student processes only visible patches (same masking as V-JEPA)
- Predictor maps student features → teacher latent space
- Loss: L1 on patch tokens (same as V-JEPA 2)
- Teacher can be smaller than student (e.g., ViT-L teacher → ViT-g student)
- Config: `model.stage: 2`

Key advantage: student training loss correlates with downstream accuracy (R²=0.95), unlike V-JEPA's uninformative loss.

## File Layout

```
app/salt/
├── __init__.py
├── models/
│   ├── __init__.py
│   └── pixel_decoder.py      # PixelDecoder — ViT decoder for Stage 1
├── utils.py                   # init_vpixel_model, init_salt_student_model, init_opt, load_checkpoint
└── train.py                   # Main training loop (Stage 1 + 2, selected by config)

configs/train/vitl16/
├── pretrain-salt-s1-mimic-224px-16f.yaml   # Stage 1 config
└── pretrain-salt-s2-mimic-224px-16f.yaml   # Stage 2 config

scripts/
├── salt_vpixel_pretrain_h100.sbatch        # Stage 1 SBATCH
└── salt_student_pretrain_h100.sbatch       # Stage 2 SBATCH
```

## How to Run

### Stage 1 (V-Pixel)

```bash
# Single-GPU test
python -m app.main --fname configs/train/vitl16/pretrain-salt-s1-mimic-224px-16f.yaml --devices cuda:0

# H100 cluster (8 GPUs)
~/deploy.sh  # Push code to compute nodes
sbatch scripts/salt_vpixel_pretrain_h100.sbatch
```

### Stage 2 (Student)

Update `model.teacher_checkpoint` in the Stage 2 config to point to the Stage 1 `latest.pt`, then:

```bash
# Single-GPU test
python -m app.main --fname configs/train/vitl16/pretrain-salt-s2-mimic-224px-16f.yaml --devices cuda:0

# H100 cluster
~/deploy.sh
sbatch scripts/salt_student_pretrain_h100.sbatch
```

## Config Reference

### Stage 1 (`model` section)

| Key | Default | Description |
|-----|---------|-------------|
| `stage` | — | Must be `1` |
| `model_name` | `vit_large` | Encoder architecture |
| `decoder_embed_dim` | `512` | Decoder hidden dimension |
| `decoder_depth` | `8` | Decoder transformer blocks |
| `decoder_num_heads` | `8` | Decoder attention heads |

### Stage 2 (`model` section)

| Key | Default | Description |
|-----|---------|-------------|
| `stage` | — | Must be `2` |
| `model_name` | `vit_large` | Student encoder architecture |
| `teacher_model_name` | same as `model_name` | Teacher encoder architecture (can differ) |
| `teacher_checkpoint` | — | Path to Stage 1 `latest.pt` |
| `pred_depth` | `12` | Predictor transformer blocks |
| `pred_embed_dim` | `384` | Predictor hidden dimension |

### Optimization (shared)

| Key | Default | Description |
|-----|---------|-------------|
| `grad_clip` | `0.02` | Max gradient norm (new vs V-JEPA 2) |
| `betas` | `[0.9, 0.95]` | AdamW β₁, β₂ (β₂=0.95 per SALT paper) |

## Checkpoint Compatibility

SALT checkpoints use the same encoder format as V-JEPA 2 (`MultiSeqWrapper` + `DistributedDataParallel`). The existing eval pipeline loads encoder weights via the `"encoder"` key after stripping `module.backbone.` prefixes, so SALT checkpoints are directly compatible with `evals.main`.

## Key Design Decisions

1. **No EMA**: The frozen teacher provides stable targets without momentum updates.
2. **Gradient clipping**: SALT uses `grad_clip=0.02`; V-JEPA 2 does not clip gradients.
3. **Single-level encoder output for Stage 1**: The pixel decoder only needs the final encoder layer (`training_mode=False`), not hierarchical features.
4. **Hierarchical features for Stage 2**: Both teacher and student produce 4-level hierarchical features (`training_mode=True`), same as V-JEPA 2.
5. **Predictor `teacher_embed_dim`**: The existing `VisionTransformerPredictor` already handles student→teacher dimension mismatch via its `teacher_embed_dim` parameter.
6. **Per-patch pixel normalization**: Stage 1 targets are per-patch normalized (subtract mean, divide by std) following VideoMAE convention.

## Reference

Paper: "Rethinking JEPA: Compute-Efficient Video SSL with Frozen Teachers" (Li et al., Apple, 2025). LaTeX source at `claude/papers/vjepa-salt/arxiv.tex`.
