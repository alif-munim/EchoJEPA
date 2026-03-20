# V-JEPA 2 / 2.1 Official Training Recipes

Extracted from the V-JEPA 2 paper (Meta, 2025) and V-JEPA 2.1 paper. All numbers from the papers directly.

## V-JEPA 2 Recipe

### Hyperparameters (shared across ViT-L/H/g)

| Parameter | Primary Phase | Cooldown Phase |
|---|---|---|
| **Frames** | 16 | **64** |
| **FPS** | 4.0 | 4.0 |
| **Crop size** | 256×256 | 256 / 384 / 512 |
| **Global batch size** | 3,072 | 3,072 |
| **LR schedule** | Warmup → constant | Linear decay |
| **Start LR** | 1e-4 | 5.25e-4 |
| **Peak / ref LR** | 5.25e-4 | 5.25e-4 |
| **Final LR** | 5.25e-4 (constant) | 1e-6 |
| **Warmup steps** | 12,000 | 0 |
| **Total steps** | ~252K (12K warmup + 228K constant + 12K) | 12,000 |
| **Weight decay** | 0.04 | 0.04 |
| **EMA momentum** | 0.99925 (fixed) | 0.99925 (fixed) |
| **Loss** | L1 (loss_exp=1.0) | L1 |
| **Optimizer** | AdamW, betas=(0.9, 0.999), eps=1e-8 | same |
| **Precision** | bfloat16 mixed | bfloat16 mixed |

### Abbreviated Recipe (for ablations)

90K steps total: 12K warmup + 66K constant + 12K cosine decay. Start LR 2e-4, peak 6.25e-4, final 1e-6. WD ramps 0.04 → 0.4, EMA ramps 0.999 → 1.0.

### Masking Strategy

Two mask generators per sample (identical across pretrain and cooldown):
- **8 small blocks**: spatial_scale=[0.15, 0.15], temporal_scale=[1.0, 1.0], aspect_ratio=[0.75, 1.5]
- **2 large blocks**: spatial_scale=[0.7, 0.7], temporal_scale=[1.0, 1.0], aspect_ratio=[0.75, 1.5]
- Tubelet size: 2×16×16

### Data: VideoMix22M (VM22M)

| Dataset | Samples | Hours | Weight |
|---|---|---|---|
| SSv2 | 168K | 168 | 0.056 |
| Kinetics (400/600/700) | 733K | 614 | 0.188 |
| HowTo100M | 1.1M | 134K | 0.318 |
| YT-Temporal-1B (curated) | 19M | 1.6M | 0.188 |
| ImageNet | 1M images | — | 0.250 |
| **Total** | **~22M** | **~1M+ hours** | |

### Model Sizes

| Model | Params | Embed dim | Depth | Heads | MLP dim |
|---|---|---|---|---|---|
| ViT-L | 300M | 1024 | 24 | 16 | 4096 |
| ViT-H | 600M | 1280 | 32 | 16 | 5120 |
| ViT-g | 1B | 1408 | 40 | 22 | 6144 |

Predictor is fixed at ViT-s: 22M params, dim=384, depth=12, heads=12.

### Progressive Resolution Results

| Phase | IN1K | COIN | SSv2 | K400 |
|---|---|---|---|---|
| Pretrain only (epoch 800) | 83.8% | 89.1% | 75.1% | 85.8% |
| + Cooldown 256px | 84.6% | 90.7% | 75.3% | 86.6% |
| + Cooldown 384px | **85.1%** | 90.2% | **76.5%** | **87.3%** |

Progressive training provides **8.4× speedup** vs. training at high resolution from scratch.

---

## V-JEPA 2.1 Recipe

### Key Changes from V-JEPA 2

1. **Context loss**: predictor also reconstructs visible (context) tokens, not just masked
2. **Deep self-supervision**: hierarchical loss at 4 intermediate encoder layers
3. **Joint image+video training**: separate image/video tokenizers, modality embeddings
4. **Predictor depth doubled**: 12 → 24 blocks
5. **Data scaling**: 142M images (VisionMix-163M) vs 1M
6. **Model scaling**: up to ViT-G (2B params)

### Hyperparameters (shared across ViT-L/g/G)

| Parameter | Primary Phase | Cooldown Phase |
|---|---|---|
| **Frames** | 16 | **64** |
| **FPS** | 4.0 | 4.0 |
| **Crop size (video)** | 256×256 | **384×384** |
| **Crop size (image)** | 256×256 | **512×512** |
| **Video batch (global)** | 128 | 128 |
| **Image batch (global)** | 2,304 | 2,304 |
| **LR schedule** | Warmup → constant | Decay |
| **Start LR** | 1e-4 | 6e-4 |
| **Peak LR** | 5.25e-4 | 6e-4 |
| **Final LR** | 5.25e-4 (constant) | 1e-6 |
| **Warmup steps** | 12,000 | 0 |
| **Total steps** | 135,000 | 12,000 |
| **Weight decay** | 0.04 | 0.04 |
| **EMA momentum** | 0.99925 | 0.99925 |
| **Context loss λ (video)** | 0.5 | 0.5 |
| **Context loss λ (image)** | 0.7 | 0.7 |
| **Lambda warmup** | 0→λ over steps 15K–30K | same |
| **Predictor depth** | 24 | 24 |
| **Hierarchical layers** | 4 (equally spaced) | 4 |

### V-JEPA 2.1 Model Sizes

| Model | Params | Embed dim | Depth | Heads |
|---|---|---|---|---|
| ViT-L | 300M | 1024 | 24 | 16 |
| ViT-g | 1B | 1408 | 40 | 22 |
| ViT-G | 2B | 1664 | 48 | 26 |

### Deep Self-Supervision Ablation

| Setting | ADE20k (last) | ADE20k (4-layer) | NYU Depth (last) | Diving-48 (last) |
|---|---|---|---|---|
| Without | 34.9 mIoU | 39.1 mIoU | 0.513 RMSE | 85.8% |
| **With** | **42.0** | **43.9** | **0.381** | **87.2%** |

### Resolution Ablation (ViT-g)

| Cooldown res | SSv2 | K400 | IN1K | EK100 |
|---|---|---|---|---|
| 256×256 | 76.7% | 86.2% | 84.1% | 35.7% |
| **384×384** | **76.9%** | **87.0%** | **84.8%** | **38.4%** |

---

## LR Scaling Rule

Both papers use linear scaling from the 3072-batch reference:

```
new_lr = 5.25e-4 × (global_batch_size / 3072)
```

Examples:
- BS=3072 → LR=5.25e-4 (paper default)
- BS=1024 → LR=1.75e-4 (MIMIC ViT-L)
- BS=640 → LR=1.09e-4 (Echo ViT-g)
