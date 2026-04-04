# Speckle Probing (Information Probing)

**Date:** 2026-03-31
**Status:** Complete.
**NeurIPS section:** §4 (Mechanistic Evidence)

---

## Overview

Linear probes trained to predict ultrasound acquisition properties (speckle energy, mean intensity, texture variance) from frozen encoder embeddings. Measures how much stochastic noise each SSL objective retains in its representations. Partial R² controlling for intensity confound isolates texture-specific (speckle) information.

## Setup

**Data:** EchoNet-Dynamic training set, 2,554 clips
**Features:** Mean-pooled frozen embeddings (1024-dim for all ViT-L models)
**Probing:** Ridge regression, 5-fold cross-validation
**Speckle measure:** Mean high-frequency power (FFT magnitude above Nyquist/2)
**Confound control:** Partial R² after conditioning on mean intensity (speckle-intensity correlation r=0.530)

| Model | Encoder Checkpoint |
|-------|-------------------|
| JEPA-L pt50 | `checkpoints/echojepa-l-pt50.pt` |
| BYOL-L pt50 | `checkpoints/byol_vitl_imagenet_v2_e50.pt` |
| MAE-L pt50 | `checkpoints/videomae_l_mimic_ep50.pth` |

## Results

### Nuisance Variable Probing (R²)

| Variable | JEPA | BYOL | MAE |
|----------|------|------|-----|
| Speckle energy (raw) | 0.764 | 0.835 | 0.910 |
| Mean intensity | 0.998 | 0.984 | 0.995 |
| Texture variance | 0.956 | 0.970 | 0.975 |
| **Speckle energy (partial R², controlling for intensity)** | **0.674** | 0.775 | **0.875** |

### Target Variable Probing (R²)

| Variable | JEPA | BYOL | MAE |
|----------|------|------|-----|
| EF | -0.093 | -0.430 | 0.075 |
| ESV | 0.100 | -0.112 | 0.291 |
| EDV | 0.073 | -0.155 | 0.172 |

EF not linearly decodable from mean-pooled embeddings — requires spatial attention (attentive probe). ESV/EDV favor MAE (spatial volume = anatomical information).

## Key Finding

**JEPA encodes 23% less speckle than MAE** (partial R²=0.674 vs 0.875). The ordering is monotonic and matches the noise-filtering prediction: JEPA (0.674) < BYOL (0.775) < MAE (0.875). 

**Mechanism:** The EMA target encoder produces prediction targets that average over stochastic frame-to-frame speckle variation. Features tracking speckle are unrewarded during JEPA/BYOL training and progressively suppressed. MAE must reconstruct pixels including speckle, so it retains this information.

## References

**Source:** `claude/rebuttals/10-rebuttal-experiment-results.md` §6e
**Script:** `scripts/rebuttal/information_probing.py`
**Data:** `scripts/rebuttal/samples/information_probing_{JEPA,BYOL,MAE}-L-pt50.npz`
