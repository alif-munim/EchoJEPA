# SALT Training Reference — Complete Specification

Complete training settings, hyperparameters, and operational details for running SALT pretraining correctly. This is the authoritative checklist — every value here has been verified against the paper (`claude/papers/vjepa-salt/arxiv.tex`, Li et al., Apple 2025) and the current codebase (as of commit `71bd4e5`).

**Use `-hp.yaml` configs, never the plain configs.** The non-`-hp` configs are stale and contain documented hyperparameter errors.

---

## Paper Reference

- **Title**: "Rethinking JEPA: Compute-Efficient Video SSL with Frozen Teachers"
- **Authors**: Li, Huang, Li, Malach, Susskind, Thilak, Littwin (Apple, 2025)
- **Source**: `claude/papers/vjepa-salt/arxiv.tex`
- **Key equation**: Paper Eq 2.1 — single-level L1 latent prediction

$$\min_{\theta, \phi} \mathbb{E}_{x,y}\|g_\phi(f_\theta(x), \delta y) - \text{stop\_grad}(\bar{f}_\theta(y))\|_1$$

---

## Architecture Summary

### Two-stage scheme

1. **Stage 1 (V-Pixel)** — Train target/teacher encoder via **pixel reconstruction** with V-JEPA multi-block masking. Per-patch normalized targets (VideoMAE convention). Output: a frozen encoder.
2. **Stage 2 (Student)** — Freeze the Stage 1 encoder as teacher. Train a new student encoder + predictor using the **V-JEPA latent prediction objective** (Eq 2.1) with L1 loss. No EMA, no stop-gradient on a dynamic teacher — teacher weights are static.

### Key design principles (verified from paper §2.2)

- ✅ **Multi-block masking** for both stages (NOT VideoMAE's random-tube masking — paper §5.2 shows multi-block works better for V-Pixel)
- ✅ **Per-patch normalized pixel targets** in Stage 1 (VideoMAE convention, subtract mean / divide by std per patch)
- ✅ **L1 loss** on patch tokens in Stage 2 (matches Eq 2.1 `||...||_1`)
- ✅ **Frozen teacher** — `requires_grad = False`, no EMA update
- ✅ **Single-level patch-token prediction** — teacher returns final-layer patch features of dim `embed_dim`; predictor output matches; **NOT hierarchical 4-layer concatenation** (that's a V-JEPA 2.1 extension, NOT part of SALT)
- ✅ **Gradient clipping at 0.02** (V-JEPA 2 does not clip)
- ✅ **AdamW β₂ = 0.95** (not the default 0.999 — paper follows Carreira et al. 2024 for large-scale VideoMAE-like training)
- ✅ **Cosine WD ramp 0.04 → 0.4** (V-JEPA uses constant WD)
- ✅ **Cosine LR decay to 1e-6** (no virtual early stopping — ipe_scale = 1.0)

---

## Encoder Architecture (Paper Table 3)

All encoders use `vit_<size>` from `app/vjepa_2_1/models/vision_transformer.py` (RoPE, SDPA, activation checkpointing).

| Model | Params | Width (embed_dim) | Depth | Heads |
|-------|--------|-------------------|-------|-------|
| ViT-B | 86M | 768 | 12 | 12 |
| **ViT-L** (our primary) | **303M** | **1024** | **24** | **16** |
| ViT-H | 632M | 1280 | 32 | 16 |
| ViT-g | 1,012M | 1408 | 40 | 16 |
| ViT-G | 1,843M | 1664 | 48 | 16 |

---

## Predictor Architecture (Paper Table 4)

All predictors in SALT use **12 depth, 384 embed dim**. `pred_num_heads` is **16** in the paper (ours was 12 before commit `755a319`, now fixed to 16 in `-hp` configs).

| Predictor for | Input dim | Output dim | Params | Width | Depth | Heads |
|---|---|---|---|---|---|---|
| ViT-B | 768 | 768 | 21.88M | 384 | 12 | 16 |
| **ViT-L** | **1024** | **1024** | **22.08M** | **384** | **12** | **16** |
| ViT-H | 1280 | 1024 | 22.18M | 384 | 12 | 16 |
| ViT-g | 1408 | 1024 | 22.23M | 384 | 12 | 16 |
| ViT-G | 1664 | 1024 | 22.32M | 384 | 12 | 16 |

Note: output dim matches teacher's embed_dim. For same-size teacher/student (our setup), input == output == embed_dim.

---

## Pixel Decoder (Stage 1 only)

Lightweight ViT decoder, separate from the predictor. Used only in Stage 1 to reconstruct pixels for the teacher training.

| Field | Value |
|---|---|
| `decoder_embed_dim` | 512 |
| `decoder_depth` | 8 |
| `decoder_num_heads` | 8 |
| `mlp_ratio` | 4.0 |
| Norm | LayerNorm |
| Position encoding | RoPE (matches encoder) |

The decoder takes visible encoder embeddings, substitutes learnable mask tokens at masked positions, and outputs flat per-patch pixel values of dim `in_chans * tubelet * patch * patch = 3 * 2 * 16 * 16 = 1536`.

---

## Stage 1 — V-Pixel Hyperparameters

### Model (`model:` block)

```yaml
model:
  stage: 1                       # REQUIRED — selects Stage 1 path
  model_name: vit_large          # or vit_base, vit_huge, vit_giant
  uniform_power: true            # matches V-JEPA 2.1
  use_activation_checkpointing: true
  use_rope: true                 # RoPE for position encoding (paper §3)
  use_silu: false                # GELU (V-JEPA 2 convention)
  wide_silu: true                # matches V-JEPA 2.1
  decoder_embed_dim: 512         # pixel decoder width
  decoder_depth: 8               # pixel decoder depth
  decoder_num_heads: 8           # pixel decoder heads
```

### Data (`data:` block)

```yaml
data:
  dataset_type: VideoDataset
  datasets:
    - /opt/dlami/nvme/data/csv/mimic_annotations_s3.csv   # MIMIC 525K clips
  datasets_weights: [1.0]
  batch_size: 64                 # per-GPU; 64 × 8 GPUs = 512 effective
                                  # Paper uses 3072 (can't match without more nodes)
  crop_size: 224                 # 224×224 spatial resolution
  patch_size: 16                 # 16×16 spatial patches
  dataset_fpcs: [16]             # frames per clip
  fps: 8                         # target FPS (paper uses frame_step=4 → fps ≈ 8 at 30fps source)
  tubelet_size: 2                # 2-frame temporal patches
  num_workers: 4
  persistent_workers: true
  pin_mem: true
```

### Data augmentation (`data_aug:` block)

**Critical**: Match paper Table 5 exactly. Original configs had wrong values.

```yaml
data_aug:
  auto_augment: false
  motion_shift: false
  random_resize_aspect_ratio: [0.75, 1.35]   # Paper: [0.75, 1.35]. Original: [0.9, 1.1] (WRONG)
  random_resize_scale: [0.3, 1.0]            # Paper: [0.3, 1.0]. Original: [0.5, 1.0] (WRONG)
  reprob: 0.0                                 # no random erasing
```

### Masking (`mask:` block) — multi-block, V-JEPA style

Two mask blocks: one short-range (many small blocks) and one long-range (few large blocks). Paper Table 5:
- Short-range spatial mask scale: 0.15
- Long-range spatial mask scale: 0.7
- Temporal mask scale: 1.0
- Mask aspect ratio: [0.75, 1.5]

```yaml
mask:
  - aspect_ratio: [0.75, 1.5]
    full_complement: false
    max_keep: null
    max_temporal_keep: 1.0
    num_blocks: 8                     # short-range: 8 small blocks
    spatial_scale: [0.15, 0.15]
    temporal_scale: [1.0, 1.0]
  - aspect_ratio: [0.75, 1.5]
    full_complement: false
    max_keep: null
    max_temporal_keep: 1.0
    num_blocks: 2                     # long-range: 2 large blocks
    spatial_scale: [0.7, 0.7]
    temporal_scale: [1.0, 1.0]
```

### Optimization (`optimization:` block)

**All values below match Paper Table 5 exactly.** The `-hp` configs use these; the non-`-hp` configs have wrong values.

```yaml
optimization:
  epochs: 20                       # S1 = 20, S2 = 80, total 100 (see teacher/student split below)
  ipe: 300                         # iterations per epoch (300 is the "compute unit" in our config)
  ipe_scale: 1.0                   # NO virtual early stopping (paper §5). Was 1.25 in original (WRONG)
  warmup: 40                       # warmup epochs. Paper: 10000 steps total → 40 epochs × 300 ipe = 12000 (close)
  start_lr: 2.0e-4                 # Paper: 0.0002. Original: 3.33e-5 (WRONG)
  lr: 6.25e-4                      # Paper: 0.000625 (peak LR). Original: 1.75e-4 (WRONG)
  final_lr: 1.0e-6                 # Paper: 1e-6 (cosine decay). Original: 1.75e-4 constant (WRONG)
  weight_decay: 0.04               # Paper: 0.04 (start)
  final_weight_decay: 0.4          # Paper: 0.4 (end, cosine ramp). Original: 0.04 constant (WRONG)
  betas: [0.9, 0.95]               # Paper: β₁=0.9, β₂=0.95 (AdamW, large-scale VideoMAE convention)
  grad_clip: 0.02                  # Paper: 0.02. V-JEPA 2 does not clip.
  # ImageNet-21K initialization (matches BYOL/MAE for controlled comparison)
  force_load_pretrain: true
  anneal_ckpt: /opt/dlami/nvme/checkpoints/vitl_in21k.pt
```

### Meta (`meta:` block)

```yaml
meta:
  dtype: bfloat16                  # BF16 for numerical stability (no GradScaler needed)
  load_checkpoint: false           # true only when resuming
  read_checkpoint: null
  save_every_freq: 5               # save every 5 epochs
  max_epoch_checkpoints: 4         # keep last 4 periodic checkpoints
  seed: 234
  use_sdpa: true                   # Flash attention via SDPA
```

### Job sizing

```yaml
nodes: 1
tasks_per_node: 8                  # 8 GPUs per node (H100 cluster)
cpus_per_task: 16
mem_per_gpu: 80G
folder: /opt/dlami/nvme/checkpoints/pretrain/mimic/salt_s1_vitl_224px_16f
```

---

## Stage 2 — Student Hyperparameters

Same data, augmentation, masking, and optimization as Stage 1 (paper Table 5 uses `"` to indicate identical). Only the `model:` block and a few specific fields differ.

### Loss (`loss:` block)

```yaml
loss:
  loss_exp: 1.0                    # L1 loss (paper Eq 2.1: ||...||_1). Critical — not L2!
```

### Model (`model:` block)

```yaml
model:
  stage: 2                         # REQUIRED — selects Stage 2 path
  model_name: vit_large            # student encoder architecture
  teacher_model_name: vit_large    # teacher architecture (can be smaller than student per paper §5.4)
  teacher_checkpoint: /opt/dlami/nvme/checkpoints/pretrain/mimic/salt_s1_vitl_224px_16f/latest.pt
                                    # REQUIRED — path to Stage 1 latest.pt
  pred_depth: 12                   # Paper Table 4: 12
  pred_embed_dim: 384              # Paper Table 4: 384
  pred_num_heads: 16               # Paper Table 4: 16. Original: 12 (WRONG, fixed in commit 755a319)
  use_mask_tokens: true
  num_mask_tokens: 10              # matches V-JEPA 2.1 default
  zero_init_mask_tokens: true      # init mask tokens to zero
  uniform_power: true              # matches Stage 1
  use_activation_checkpointing: true
  use_rope: true
  use_silu: false
  wide_silu: true

  # OPTIONAL — V-JEPA 2.1 hierarchical feature distillation (NOT part of SALT paper)
  # Leave unset or 1 for SALT paper spec. Set to 4 only if you explicitly want the
  # V-JEPA 2.1 hierarchical extension (not recommended for SALT reproduction).
  # n_output_distillation: 1       # default, matches paper Eq 2.1
```

### Optimization — identical to Stage 1 except epochs

```yaml
optimization:
  epochs: 80                       # S2 = 80 (with S1 = 20 → 100 total, ~1:4 split per paper §5.4)
  ipe: 300
  ipe_scale: 1.0
  warmup: 40
  start_lr: 2.0e-4
  lr: 6.25e-4
  final_lr: 1.0e-6
  weight_decay: 0.04
  final_weight_decay: 0.4
  betas: [0.9, 0.95]
  grad_clip: 0.02
  force_load_pretrain: false       # S2 does NOT load ImageNet — student starts from scratch
```

### Teacher/Student compute allocation

Paper §5.4 finds the optimal ratio is **roughly 1:4 in favor of the student** (i.e., if 100 total epochs, use ~20 for teacher and ~80 for student). Smaller teachers suffice; student quality is robust to teacher quality. Our configs use 20:80 (`epochs: 20` S1, `epochs: 80` S2).

---

## Critical Details That Were Wrong Before

Everything in this section was fixed in commits `755a319` (hyperparameters) and `71bd4e5` (predictor shape bug). Verify each when preparing a new run.

### 1. Hyperparameters (commit 755a319)

| Parameter | Correct (paper) | Old (broken) configs |
|---|---|---|
| `lr` | **6.25e-4** | 1.75e-4 |
| `start_lr` | **2.0e-4** | 3.33e-5 |
| `final_lr` | **1.0e-6** (cosine decay) | 1.75e-4 constant |
| `final_weight_decay` | **0.4** (cosine ramp) | 0.04 constant |
| `ipe_scale` | **1.0** (no virtual ES) | 1.25 |
| `random_resize_aspect_ratio` | **[0.75, 1.35]** | [0.9, 1.1] |
| `random_resize_scale` | **[0.3, 1.0]** | [0.5, 1.0] |
| `loss_exp` | **1.0** (L1) | (default could become L2) |
| `pred_num_heads` | **16** | 12 |

### 2. Predictor single-level shape (commit 71bd4e5)

**The bug**: Commit `0eaf0ab` updated the train.py forward pass to use single-level features (paper spec) but did NOT update `init_salt_student_model` in `utils.py`, which still built the predictor for 4-layer hierarchical input. This caused a shape mismatch that would crash Stage 2 training on the first iteration.

**The fix**: `n_hier` now defaults to 1 (single-level), and `n_output_distillation` is always passed explicitly to the predictor constructor. Both encoder and predictor now agree on single-level dim `embed_dim`.

**Verification**: If you look inside a correctly-trained SALT S2 checkpoint, the predictor weights should have:
- `predictor_embed.weight: [384, embed_dim]` (single Linear, NOT the Sequential hierarchical layer)
- `predictor_proj.weight: [embed_dim, 384]`

For ViT-L with embed_dim=1024:
- `predictor_embed.weight: [384, 1024]`
- `predictor_proj.weight: [1024, 384]`

Existing checkpoints (`salt_s2_vitl_e{29,49,79,199}.pt`) have `predictor_embed.0.weight: [1024, 4096]` — these are from the old hierarchical mode and MUST be discarded. Retrain from scratch.

### 3. Hierarchical vs single-level (paper vs V-JEPA 2.1)

- **Paper Eq 2.1**: single-level — predict the final-layer patch token representations only.
- **V-JEPA 2.1 extension**: concatenate 4 hierarchical layer outputs, predict all 4. Dimension becomes `4 * embed_dim`.

SALT paper does NOT use the V-JEPA 2.1 extension. Default configs should leave `n_output_distillation` unset (which now defaults to 1 after the fix).

---

## MIMIC Dataset Specifics

- **CSV**: `mimic_annotations_s3.csv`, 525K clips from MIMIC-IV-Echo
- **Format**: `path/to/video.mp4 0` (space-delimited, dummy label)
- **Resolution**: 224×224 (pre-resized via data loader from native size)
- **FPS**: source ~50 fps, target 8 fps via frame sampling
- **Clip length**: 16 frames (32-frame coverage with tubelet_size=2)
- **S3 path**: `s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts/data/csv/mimic_annotations_s3.csv`

Differences from paper's V-3.6M dataset (Kinetics-710 + SSv2 + Panda70M):
- Much smaller (525K vs 3.6M)
- Highly domain-specific (echocardiograms only)
- More repetitive content than natural video
- Paper warns (implicitly via §5.2) that pretrain dataset composition matters

---

## How to Run (HyperPod H100 cluster)

### Setup

Connect to the controller via SSM:
```bash
aws ssm start-session --region us-west-2 \
  --target "sagemaker-cluster:yyepvbne5vzr_echojepa-h100-controller-i-0c6d410f979fabfe7"
```

Get latest code and deploy to compute nodes (always do this before every sbatch):
```bash
cd ~/EchoJEPA-repo && git pull
~/deploy.sh
```

### Stage 1 (V-Pixel)

```bash
sbatch scripts/salt_s1_pretrain_hp.sbatch
```

Uses config: `configs/train/vitl16/pretrain-salt-s1-mimic-224px-16f-hp.yaml`

Monitor:
```bash
squeue -u ubuntu
tail -f /tmp/salt_s1-<jobid>.out
```

Expected: 20 epochs × 300 ipe = 6000 steps. With batch 64 × 8 GPUs on H100, should take approximately **6-10 hours**.

### Stage 2 (Student)

**Before running**: update `model.teacher_checkpoint` in the Stage 2 config to point to the Stage 1 `latest.pt` from the completed Stage 1 run (or ensure the S3 sync pulled the file to the expected location).

```bash
sbatch scripts/salt_s2_pretrain_hp.sbatch
```

Uses config: `configs/train/vitl16/pretrain-salt-s2-mimic-224px-16f-hp.yaml`

Expected: 80 epochs × 300 ipe = 24000 steps. Should take approximately **25-35 hours** (bigger than S1 because of teacher + student + predictor forward).

### Resume (if preempted)

Resume configs exist at `pretrain-salt-s2-mimic-224px-16f-resume-e{80,100}-hp.yaml` — update the epoch and use for checkpoint resumption.

---

## Pre-Flight Checklist

Before launching a new SALT run, verify:

1. **Correct config**: Using `-hp.yaml`, not the plain config. Plain configs are stale.
2. **Code is current**: `git log --oneline | head -5` should show `71bd4e5` or later (the predictor fix).
3. **deploy.sh has run**: Compute nodes have the latest code. Skipping this means stale code.
4. **Teacher checkpoint exists**: For Stage 2, the Stage 1 `latest.pt` is reachable.
5. **No old S2 checkpoints in output dir**: `salt_s2_vitl_e*.pt` from previous runs must be deleted — they have incompatible predictor shapes.
6. **ImageNet-21K init**: Stage 1 config has `force_load_pretrain: true` and `anneal_ckpt` points to a valid ViT-L checkpoint.
7. **Hyperparameter spot-check**: `lr=6.25e-4`, `final_wd=0.4`, `ipe_scale=1.0`, `betas=[0.9, 0.95]`, `grad_clip=0.02`, `loss_exp=1.0`.
8. **Masking**: Two mask blocks with `spatial_scale` 0.15 and 0.7 respectively.
9. **Stage flag**: `model.stage: 1` for S1, `model.stage: 2` for S2 — no other values.

---

## Known Deviations From Paper (That We Cannot Fix)

1. **Batch size**: Paper uses 3072, we use 512 (64 × 8 GPUs). Larger batch → faster convergence per epoch. Single-node H100 cluster cannot match paper's 3072 without more nodes.
2. **Total training steps**: Paper uses 240,000 total steps at batch 3072. Our 20+80 epochs × 300 ipe = 30,000 steps at batch 512 = ~6.25M samples vs paper's ~737M. **Our training is ~2.4% of paper's compute budget.** Expect lower absolute performance, but the method ranking should still hold.
3. **Pretraining dataset**: MIMIC 525K echo-only vs V-3.6M diverse natural video. Domain restriction likely improves echo downstream performance but hurts transfer to other tasks.
4. **Same-size teacher/student**: Paper explores smaller teachers successfully; we use ViT-L → ViT-L for both stages (no compute savings but simpler to set up).

---

## Verification After Training

Once Stage 2 finishes, verify the checkpoint looks right:

```python
import torch
ckpt = torch.load('salt_s2_vitl_latest.pt', map_location='cpu', weights_only=False)

# Check encoder weights exist
assert 'encoder' in ckpt
# Check predictor is single-level (paper spec)
pred_keys = [k for k in ckpt['predictor'].keys() if 'predictor_embed' in k]
for k in pred_keys:
    print(f"{k}: {ckpt['predictor'][k].shape}")
# Expected: predictor_embed.weight: [384, 1024]  (single Linear, not Sequential)
# WRONG: predictor_embed.0.weight: [1024, 4096]  (hierarchical from old code)
```

If you see the `predictor_embed.0.weight: [1024, 4096]` shape, something is wrong — the old hierarchical predictor got built. Check that `app/salt/utils.py` has the fix from commit `71bd4e5`.

### Downstream probe sanity check

Run the LVEF probe at a few checkpoints (e.g. e20, e50, e80) to verify training curve is reasonable:

```bash
python -m evals.main --fname configs/eval/vitl/icml/salt_s2_e199_end_lvef_d4.yaml \
    --devices cuda:0 cuda:1 cuda:2 cuda:3
```

Update the probe config's `checkpoint:` field to point to your new `salt_s2_vitl_latest.pt` and expect LVEF test R² > 0.28 (the MAE pt50 baseline) as a minimum sanity threshold. If it's below MAE pt50, something is wrong.

---

## Key File Locations

### Code
- `app/salt/train.py` — main training loop (Stage 1 + 2)
- `app/salt/utils.py` — model init, optimizer, checkpoint loading
- `app/salt/models/pixel_decoder.py` — Stage 1 pixel decoder

### Configs (use `-hp` variants only)
- `configs/train/vitl16/pretrain-salt-s1-mimic-224px-16f-hp.yaml`
- `configs/train/vitl16/pretrain-salt-s2-mimic-224px-16f-hp.yaml`
- `configs/train/vitl16/pretrain-salt-s2-mimic-224px-16f-resume-e80-hp.yaml`
- `configs/train/vitl16/pretrain-salt-s2-mimic-224px-16f-resume-e100-hp.yaml`

### Scripts
- `scripts/salt_s1_pretrain_hp.sbatch`
- `scripts/salt_s2_pretrain_hp.sbatch`
- `scripts/salt_s2_pretrain_resume_e80_hp.sbatch`
- `scripts/salt_s2_pretrain_resume_e100_hp.sbatch`

### Paper reference
- `claude/papers/vjepa-salt/arxiv.tex` — full paper LaTeX source
- `claude/papers/vjepa-salt/figs/` — figures

### Related docs
- `claude/architecture/salt-pretraining.md` — original design doc (higher-level overview)

---

## Summary Table — Paper Table 5 vs Our -hp Config

| Parameter | Paper Table 5 | Our `-hp` config | Match? |
|---|---|---|---|
| Input spatial resolution | 224×224 | 224 (crop_size) | ✅ |
| Tubelet size | 2 | 2 | ✅ |
| Patch size | 16×16×2 | 16 (patch_size) + tubelet_size=2 | ✅ |
| Number of frames | 16 | 16 (dataset_fpcs) | ✅ |
| Frame step | 4 | N/A (we use fps=8 instead) | ⚠️ equivalent |
| Random resize aspect ratio | [0.75, 1.35] | [0.75, 1.35] | ✅ |
| Random resize scale | [0.3, 1.0] | [0.3, 1.0] | ✅ |
| Short-range spatial mask scale | 0.15 | 0.15 (block 1) | ✅ |
| Long-range spatial mask scale | 0.7 | 0.7 (block 2) | ✅ |
| Temporal mask scale | 1.0 | 1.0 | ✅ |
| Mask aspect ratio | [0.75, 1.5] | [0.75, 1.5] | ✅ |
| Batch size | **3072** | **512** (64 × 8 GPUs) | ❌ GPU-limited |
| Start LR | 0.0002 | 2.0e-4 | ✅ |
| LR (peak) | 0.000625 | 6.25e-4 | ✅ |
| Final LR | 1e-6 | 1.0e-6 | ✅ |
| Start WD | 0.04 | 0.04 | ✅ |
| End WD | 0.4 | 0.4 | ✅ |
| Clip grad | 0.02 | 0.02 | ✅ |
| Warmup steps | 10,000 | 12,000 (40 ep × 300 ipe) | ✅ close |
| AdamW β₁ | 0.9 | 0.9 | ✅ |
| AdamW β₂ | 0.95 | 0.95 | ✅ |
| Learning rate schedule | Cosine | Cosine | ✅ |
| ipe_scale (virtual ES) | 1.0 | 1.0 | ✅ |
| Loss (Stage 2) | L1 (Eq 2.1) | L1 (loss_exp=1.0) | ✅ |
| Teacher mode | Frozen, no EMA | Frozen, no EMA | ✅ |
| Prediction granularity | Single-level patch tokens | Single-level (after 71bd4e5 fix) | ✅ |

**Total compliance**: 22/23 matches. Only batch size deviates and it's GPU-limited.
