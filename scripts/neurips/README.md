# NeurIPS Experiment Scripts

Scripts for the NeurIPS 2026 paper: *"The Temporal Shortcut in Self-Supervised Video Learning."*

Previously `scripts/rebuttal/` (from the ICML rebuttal). Renamed April 2026; all internal references updated.

## Checkpoints

**`model_registry.py`** — Central registry of all checkpoint paths and model configs. Scripts use `--model` or `--group` to select models.

All four models use the same ViT-L (304M params), ImageNet-21K init, trained on MIMIC-IV-Echo (525K clips). Checkpoints at e25/e50/e75/e100 are used for training dynamics analysis.

**GDrive (primary):** `echo_foundation/nature_medicine/neurips/`

| Model | Checkpoints | GDrive path |
|-------|------------|-------------|
| **JEPA** | e25, e50, e75, e100 | `neurips/encoders/jepa_in21k_vitl_e{25,50,75,100}.pt` |
| **BYOL** | e24, e50, e75, e100 | `neurips/encoders/byol_vitl_e{24,50,75,100}.pt` |
| **MAE** | e24, e50, e74, e99 | `neurips/encoders/mae_vitl_e{24,50,74,99}.pth` |
| **SALT S1** (teacher) | e19 | `neurips/encoders/salt_s1_vitl_e19.pt` |
| **SALT S2** (student) | e4, e29, e49, e79 | `neurips/encoders/salt_s2_vitl_e{4,29,49,79}.pt` |

**S3 mirror:** `s3://echodata25/neurips/encoders/` (same files)

**Baselines:** EchoPrime (`checkpoints/echo_prime_encoder.pt`), PanEcho (`checkpoints/panecho.pt`)

**Init:** `checkpoints/vitl_in21k.pt` (ImageNet-21K supervised ViT-L, 2D, inflated to 3D at load). Same init for all four models (SALT uses it for Stage 1).

**Probes:** `s3://echodata25/neurips/probes/end_lvef_e100/{model}/best.pt`

## Datasets

**GDrive:** `echo_foundation/nature_medicine/neurips/datasets/`

| Dataset | File | Size | Use |
|---------|------|------|-----|
| EchoNet-Dynamic | `echonet_data.zip` | 6.6 GB | LVEF regression (1,277 test), primary benchmark |
| EchoNet-Pediatric | `echonet_pediatric.tar.gz` | 2.1 GB | Zero-shot pediatric transfer (368 test) |
| CAMUS | `camus.zip` | 3.6 GB | Cardiac segmentation (50 test patients), ranking inversion |

**EFS:** `data/camus/CAMUS_public/` (unzipped), `data/EchoNet-Dynamic/`, `data/echonetpediatric/`

**S3:** `s3://echodata25/neurips/datasets/` (same files mirrored)

> **Warning:** For JEPA, always use the canonical run 376 encoder (`jepa_in21k_vitl_e100.pt`). The older run 125 encoder has a different md5 and produces near-random predictions with the canonical probes. See `claude/neurips/canonical-checkpoints.md`.

## Frame Shuffling

Core experiment. Randomly permute video frames before inference to measure temporal dependence.

| Script | Description |
|--------|-------------|
| `frame_shuffling.py` | Cosine similarity degradation (200 videos, 3 seeds) |
| `frame_shuffle_task.py` | LVEF R² degradation with RoPE position remapping |
| `frame_shuffle_6cond.py` | All 6 conditions: clean, tubelet, reverse, matched, shuffle, matched_frame |
| `frame_shuffle_severity.py` | Severity gradient: shuffle 0/25/50/75/100% of frames |
| `frame_shuffle_segmentation.py` | CAMUS segmentation Dice under shuffle |
| `frame_shuffle_segmentation_extended.py` | Larger sample set |
| `frame_shuffle_segmentation_persample.py` | Per-sample tracking |
| `frame_shuffle_segmentation_tracked.py` | Maintains sample identity across conditions |

## EchoBench (Physics-Based Perturbations)

Three ultrasound degradations (depth attenuation, acoustic shadow, haze) at three severity levels. Probes are never retrained.

| Script | Description |
|--------|-------------|
| `echo_perturbations.py` | Core perturbation module (import this) |
| `generate_perturbed_videos.py` | Pre-compute perturbed tensor cache (200 videos x 9 conditions) |
| `noised_inference.py` | Run LVEF/view probes on clean + 9 perturbation conditions |
| `noised_inference_persample.py` | Per-sample version |
| `noised_segmentation.py` | CAMUS Dice under perturbations |
| `noised_segmentation_persample.py` | Per-sample version |
| `cka_speckle.py` | Linear CKA between clean and perturbed features |
| `noise_level_probe.py` | Probe to predict perturbation severity from frozen features |
| `noise_autocorrelation_sweep.py` | Vary temporal correlation of perturbations |

## Bootstrap Confidence Intervals

Paired bootstrap (10K samples) for all primary comparisons.

| Script | Description |
|--------|-------------|
| `camus_shuffle_bootstrap_ci.py` | CIs for CAMUS Dice under frame shuffling |
| `camus_bootstrap_persample.py` | Per-sample Dice bootstrap |
| `camus_noised_bootstrap.py` | CIs for CAMUS under EchoBench perturbations |
| `lvef_noised_bootstrap.py` | CIs for LVEF R² under perturbations |
| `severity_stratification_bootstrap.py` | Per-severity-bin CIs (reduced/mildly reduced/normal EF) |
| `enp_zeroshot_bootstrap.py` | EchoNet-Pediatric zero-shot CIs |

## Speckle Probing

How much speckle noise does each model encode?

| Script | Description |
|--------|-------------|
| `layerwise_speckle_probing.py` | Ridge regression for speckle energy at layers 1/6/12/18/24 |
| `token_speckle_probing.py` | Per-spatial-token speckle sensitivity heatmap |

## Representation Analysis

| Script | Description |
|--------|-------------|
| `information_probing.py` | Ridge probes for nuisance (speckle, intensity) vs target (EF, ESV, EDV) |
| `temporal_consistency.py` | Frame-to-frame cosine similarity |
| `cardiac_phase_probe.py` | Probe for cardiac phase (end-systole vs end-diastole) |
| `cardiac_trajectory.py` | 2D/3D embedding trajectory across cardiac cycle |
| `temporal_attention_trial.py` | Cross-temporal vs within-frame attention ratio per layer |
| `umap_clean_vs_perturbed.py` | UMAP of clean vs perturbed representations |
| `compute_rankme.py` | Effective dimensionality (RankMe) |
| `rankme.py` | Full RankMe analysis |
| `mae_reconstruction_temporal.py` | VideoMAE reconstruction temporal consistency |

## SALT Debugging

| Script | Description |
|--------|-------------|
| `debug_salt_compare_ckpts.py` | Compare SALT checkpoint weights |
| `debug_salt_lvef.py` | Debug SALT LVEF probe evaluation |
| `debug_salt_lvef_batch.py` | Batch version |
| `debug_salt_lvef_v2.py` | V2 with additional diagnostics |
| `diagnose_p0.py` | P0 task verification |

## Other

| Script | Description |
|--------|-------------|
| `run_rvsp_noise_grid.py` | RVSP evaluation across perturbation parameter sweep |
| `run_rvsp_noise_grid.sh` | Shell wrapper |

## Probe Training and Inference

All probe training uses the `evals.main` entry point with YAML configs. Configs for the NeurIPS controlled comparison are in `configs/eval/vitl/neurips/`.

### EchoNet-Dynamic LVEF (primary benchmark)

Train a frozen attentive probe for LVEF regression. Probes are trained on the EchoNet-Dynamic training split (7,465 videos) and evaluated on the test split (1,277 videos). LVEF labels are z-score normalized at runtime (mean=55.78, std=12.41).

```bash
# JEPA e100 — d=4 attentive probe, 16-head HP grid
python -m evals.main \
    --fname configs/eval/vitl/neurips/echobyol_l_e100_end_lvef_d4.yaml \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7

# Run all 4 models in parallel (one config per model):
#   echojepa_l_e100_end_lvef_d4.yaml   (JEPA, not yet created — use enp as template)
#   echobyol_l_e100_end_lvef_d4.yaml   (BYOL)
#   echomae_l_e100_end_lvef_d4.yaml    (MAE — note: uses videomae module)
#   salt_s2v1_echonet_lvef_d4.yaml     (SALT)
```

Key config fields:
- `experiment.classifier.num_probe_blocks: 4` (d=4) or `1` (d=1)
- `experiment.classifier.num_heads: 16` (HP grid size; best head selected by val loss)
- `experiment.data.dataset_train/val`: paths to space-separated CSVs (`<s3_path> <raw_float_label>`)
- `model_kwargs.checkpoint`: path to frozen encoder

Output: `{folder}/best.pt` (probe checkpoint), `{folder}/log_r0.csv` (per-epoch metrics).

### EchoNet-Pediatric (zero-shot transfer)

No training needed — apply adult-trained probes to pediatric data. Uses d=1 attentive probes trained on EchoNet-Dynamic, evaluated on EchoNet-Pediatric (368 test videos).

```bash
# JEPA e100 → pediatric zero-shot
python -m evals.main \
    --fname configs/eval/vitl/neurips/echojepa_l_e100_enp_lvef_d1.yaml \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7

# Similarly for BYOL, MAE, SALT:
#   echobyol_l_e100_enp_lvef_d1.yaml
#   echomae_l_e99_enp_lvef_d1.yaml
#   salt_s2v1_e79_enp_lvef_d1.yaml
```

These configs set `val_only: true` and load a pre-trained probe via `probe_checkpoint`. The probe was trained on adult EchoNet-Dynamic data and is evaluated directly on pediatric data without retraining.

### CAMUS Segmentation

CAMUS uses a separate segmentation pipeline (`evals/segmentation_frozen/`) with a UNetR-style linear decoder head on frozen encoder features. The dataset is loaded from NIfTI files via `evals/segmentation_frozen/camus_dataset.py`.

```bash
# Train segmentation decoder (single GPU, ~1 hour per model)
python evals/segmentation_frozen/train.py \
    --encoder_type vjepa \
    --encoder_checkpoint checkpoints/jepa_in21k_vitl_e95.pt \
    --encoder_model_name vit_large \
    --lr 5e-2 --weight_decay 1e-4 \
    --epochs 100 --batch_size 4 \
    --output_dir results/segmentation/echojepa_l_e100/ \
    --device cuda:0

# Evaluate (reports per-structure Dice: LV, MYO, LA)
python evals/segmentation_frozen/eval.py \
    --encoder_type vjepa \
    --encoder_checkpoint checkpoints/jepa_in21k_vitl_e95.pt \
    --encoder_model_name vit_large \
    --decoder_checkpoint results/segmentation/echojepa_l_e100/lr5e-02_wd1e-04/best_decoder.pt \
    --device cuda:0

# Frame shuffling on CAMUS (uses the trained decoder)
python scripts/neurips/frame_shuffle_segmentation.py \
    --encoder_type vjepa \
    --encoder_checkpoint checkpoints/jepa_in21k_vitl_e95.pt \
    --encoder_model_name vit_large \
    --decoder_checkpoint results/segmentation/echojepa_l_e100/lr5e-02_wd1e-04/best_decoder.pt \
    --device cuda:0 --label echojepa_l_e100

# Noised segmentation (perturbation robustness)
python scripts/neurips/noised_segmentation.py \
    --encoder_type vjepa \
    --encoder_checkpoint checkpoints/jepa_in21k_vitl_e95.pt \
    --encoder_model_name vit_large \
    --decoder_checkpoint results/segmentation/echojepa_l_e100/lr5e-02_wd1e-04/best_decoder.pt \
    --device cuda:0 --label echojepa_l_e100
```

For VideoMAE, use `--encoder_type videomae` (no `--encoder_model_name` or `--encoder_key` needed). For BYOL, use `--encoder_type byol`. See the docstrings in each script for full examples with all four models.

CAMUS data is expected at `data/camus/CAMUS_public/` (unzip `camus.zip` from GDrive).


## Perturbation Module API

```python
from scripts.neurips.echo_perturbations import apply_perturbation

clip = load_clip(...)  # [C, T, H, W] float32 [0, 1]
perturbed = apply_perturbation(
    clip,
    perturbation_type="depth_attenuation",  # or "gaussian_shadow", "haze_artifact"
    severity="severe",                       # or "mild", "moderate"
    seed=42,
)
```

| Type | Clinical scenario | Effect |
|------|------------------|--------|
| Depth attenuation | Obese patient, poor acoustic window | Signal falls off with depth |
| Acoustic shadow | Rib or calcification blocks beam | Localized dark patch |
| Haze artifact | Reverberation from chest wall | Contrast reduction + brightness wash |

## Usage

```bash
# Single model
python scripts/neurips/frame_shuffle_task.py --model jepa_in21k_e100

# All models in a group
python scripts/neurips/noised_inference.py --group e100

# With output directory
python scripts/neurips/noised_inference.py --group e100 --output results/neurips/
```
