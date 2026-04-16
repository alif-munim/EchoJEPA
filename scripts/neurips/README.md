# NeurIPS Experiment Scripts

Scripts for the NeurIPS 2026 paper: *"The Temporal Shortcut in Self-Supervised Video Learning."*

Previously `scripts/rebuttal/` (from the ICML rebuttal). Renamed April 2026; all internal references updated.

## Model Registry

**`model_registry.py`** — Central registry of all checkpoints used across experiments. Select models by name or group:

| Group | Models | Use |
|-------|--------|-----|
| `e100` | JEPA-IN21K-e100, BYOL-e100, MAE-e99, SALT-S2v1-e79 | NeurIPS primary (init-matched) |
| `pt50` | JEPA-L-pt50, BYOL-L-pt50, MAE-L-pt50 | ICML rebuttal (50-epoch) |
| `salt` | SALT-S2v1-e79, SALT-S2v3-e79 | SALT variants |
| `baselines` | EchoPrime, PanEcho | System-level baselines |

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
