# ICML Rebuttal Scripts

Scripts for the mechanistic evidence experiments (P0) in the ICML rebuttal.
See `claude/rebuttals/11-rebuttal-task-tracker.md` for the full task list.

## Scripts

| Script | P0 Task | Purpose |
|--------|---------|---------|
| `echo_perturbations.py` | — | Core module: echo-specific perturbations (depth attenuation, acoustic shadow, haze) as PyTorch tensor transforms. Based on USAugment (Ostvik 2021, Smistad 2018). |
| `generate_perturbed_videos.py` | P0.2 | Generate perturbed tensor cache (200 videos × 3 types × 3 severities) for CKA and noise probe. |
| `cka_speckle.py` | P0.3 | CKA between clean and perturbed representations. Measures representation stability. |
| `noise_level_probe.py` | P0.4 | Linear probe to predict perturbation severity from frozen features. Tests what information the representation encodes. |
| `frame_shuffling.py` | P0.1 | Frame shuffling temporal ablation. Measures whether encoder relies on temporal order. |
| `noised_inference.py` | P0.6 | On-the-fly noised test inference for regression/classification probes (LVEF, view, etc.). |
| `noised_segmentation.py` | P0.6 | On-the-fly noised CAMUS segmentation inference. Runs trained decoder on perturbed test videos, reports Dice degradation per structure. |

## Quick Start

All commands assume you're in the repo root. Prefix with `TMPDIR=/tmp LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH` on the A100 node.

### 1. Generate perturbed cache (P0.2) — runs on CPU, ~2h

```bash
python scripts/rebuttal/generate_perturbed_videos.py \
    --csv data/csv/uhn_views_22k_val.csv \
    --n_videos 200 \
    --output scripts/rebuttal/perturbed_cache.pt
```

### 2. CKA analysis (P0.3) — single GPU, ~4h

```bash
python scripts/rebuttal/cka_speckle.py \
    --cache scripts/rebuttal/perturbed_cache.pt \
    --device cuda:0
```

### 3. Noise-level probe (P0.4) — single GPU, ~4h

```bash
python scripts/rebuttal/noise_level_probe.py \
    --cache scripts/rebuttal/perturbed_cache.pt \
    --device cuda:0
```

### 4. Frame shuffling (P0.1) — single GPU, ~4h

```bash
python scripts/rebuttal/frame_shuffling.py \
    --csv data/csv/uhn_views_22k_val.csv \
    --n_videos 200 \
    --device cuda:0
```

### 5. Noised test inference (P0.6) — single GPU per run

Run a trained probe on clean + 9 perturbation conditions (3 types × 3 severities).
Loads model once, iterates all conditions. One run per model.

**EchoJEPA-L pt50 LVEF:**
```bash
python scripts/rebuttal/noised_inference.py \
    --encoder_type vjepa \
    --encoder_checkpoint checkpoints/echojepa-l-pt50.pt \
    --encoder_model_name vit_large \
    --encoder_key target_encoder \
    --probe_checkpoint evals/vitb/icml/echojepa_l_pt50_lvef/video_classification_frozen/icml-echojepa-l-pt50-lvef-d4/best.pt \
    --task_type regression \
    --test_csv data/csv/rebuttal/lvef/lvef_val_1k.csv \
    --device cuda:0 \
    --label echojepa_l_pt50
```

**EchoBYOL-L pt50 LVEF:**
```bash
python scripts/rebuttal/noised_inference.py \
    --encoder_type vjepa \
    --encoder_checkpoint checkpoints/byol_vitl_imagenet_v2_e50.pt \
    --encoder_model_name vit_large \
    --encoder_key target_encoder \
    --probe_checkpoint evals/vitb/icml/byol_pt50_lvef/video_classification_frozen/icml-echobyol-l-pt50-lvef-d4/best.pt \
    --task_type regression \
    --test_csv data/csv/rebuttal/lvef/lvef_val_1k.csv \
    --device cuda:0 \
    --label echobyol_l_pt50
```

**EchoMAE-L pt50 LVEF:**
```bash
python scripts/rebuttal/noised_inference.py \
    --encoder_type videomae \
    --encoder_checkpoint checkpoints/videomae_l_mimic_ep50.pth \
    --probe_checkpoint path/to/mae_pt50_lvef_retrained/best.pt \
    --task_type regression \
    --test_csv data/csv/rebuttal/lvef/lvef_val_1k.csv \
    --device cuda:0 \
    --label echomae_l_pt50
```

**Run all 3 models in parallel** (one GPU each):
```bash
# GPU 0: JEPA
CUDA_VISIBLE_DEVICES=0 python scripts/rebuttal/noised_inference.py \
    --encoder_type vjepa --encoder_checkpoint checkpoints/echojepa-l-pt50.pt \
    --encoder_model_name vit_large --encoder_key target_encoder \
    --probe_checkpoint evals/vitb/icml/echojepa_l_pt50_lvef/.../best.pt \
    --task_type regression --test_csv data/csv/rebuttal/lvef/lvef_val_1k.csv \
    --device cuda:0 --label echojepa_l_pt50 &

# GPU 1: BYOL
CUDA_VISIBLE_DEVICES=1 python scripts/rebuttal/noised_inference.py \
    --encoder_type vjepa --encoder_checkpoint checkpoints/byol_vitl_imagenet_v2_e50.pt \
    --encoder_model_name vit_large --encoder_key target_encoder \
    --probe_checkpoint evals/vitb/icml/byol_pt50_lvef/.../best.pt \
    --task_type regression --test_csv data/csv/rebuttal/lvef/lvef_val_1k.csv \
    --device cuda:0 --label echobyol_l_pt50 &

# GPU 2: MAE
CUDA_VISIBLE_DEVICES=2 python scripts/rebuttal/noised_inference.py \
    --encoder_type videomae --encoder_checkpoint checkpoints/videomae_l_mimic_ep50.pth \
    --probe_checkpoint path/to/mae_probe/best.pt \
    --task_type regression --test_csv data/csv/rebuttal/lvef/lvef_val_1k.csv \
    --device cuda:0 --label echomae_l_pt50 &

wait
```

### 6. Noised CAMUS segmentation (P0.6) — single GPU per run

Same pattern as noised_inference.py but for segmentation (Dice score degradation).

**EchoJEPA-L pt50:**
```bash
python scripts/rebuttal/noised_segmentation.py \
    --encoder_type vjepa \
    --encoder_checkpoint checkpoints/echojepa-l-pt50.pt \
    --encoder_model_name vit_large \
    --decoder_checkpoint results/segmentation/echojepa_l_pt50/lr5e-02_wd1e-04/best_decoder.pt \
    --device cuda:0 \
    --label echojepa_l_pt50
```

**EchoBYOL-L pt50:**
```bash
python scripts/rebuttal/noised_segmentation.py \
    --encoder_type byol \
    --encoder_checkpoint checkpoints/byol_vitl_imagenet_v2_e50.pt \
    --encoder_model_name vit_large \
    --decoder_checkpoint results/segmentation/echobyol_l_pt50/lr5e-02_wd1e-04/best_decoder.pt \
    --device cuda:0 \
    --label echobyol_l_pt50
```

**EchoMAE-L pt50:**
```bash
python scripts/rebuttal/noised_segmentation.py \
    --encoder_type videomae \
    --encoder_checkpoint checkpoints/videomae_l_mimic_ep50.pth \
    --decoder_checkpoint results/segmentation/echomae_l_pt50/lr1e-02_wd1e-04/best_decoder.pt \
    --device cuda:0 \
    --label echomae_l_pt50
```

Supports same `--perturbation_types`, `--severity_levels`, `--skip_clean` flags as noised_inference.py.

## Perturbation Types

All perturbations are based on USAugment (refs/usaugment) — echo-specific, physics-based degradation modes that occur during clinical acquisition:

| Type | Clinical Scenario | Effect |
|------|------------------|--------|
| **Depth attenuation** | Obese patient, poor acoustic window | Signal falls off with depth — deep structures become invisible |
| **Acoustic shadow** | Rib or calcification blocks beam | Localized dark patch — structures behind obstruction lost |
| **Haze artifact** | Reverberation from chest wall | Contrast reduction + brightness wash — fine detail lost |

Three severity levels per type (mild / moderate / severe) with fixed parameters.
Perturbations are applied per-frame identically (temporally static), matching real physics.

## Perturbation Module API

For use in other scripts or on-the-fly inference:

```python
from scripts.rebuttal.echo_perturbations import apply_perturbation

clip = load_clip(...)  # [C, T, H, W] float32 [0, 1]
perturbed = apply_perturbation(
    clip,
    perturbation_type="depth_attenuation",  # or "gaussian_shadow", "haze_artifact"
    severity="severe",                       # or "mild", "moderate"
    seed=42,                                 # deterministic per-video
)
```

## Output

- `noised_inference.py` prints a results table and saves a CSV per model
- `cka_speckle.py` prints CKA tables (per perturbation type + mean)
- `noise_level_probe.py` prints accuracy tables (per perturbation type + mean)
- `frame_shuffling.py` prints cosine similarity degradation table

## Samples

`samples/` contains visual samples for manual inspection:
- `perturbation_grid_labeled.png` — labeled grid (rows=severity, cols=perturbation type)
- `perturbation_grid.mp4` — animated grid video
- Individual videos: `{perturbation}_{severity}.mp4`
