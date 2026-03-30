# EchoBench: Acoustic Robustness Benchmark for Echocardiography Foundation Models

**Status**: v0.1 implemented (echobench branch)
**Created**: 2026-03-29
**Updated**: 2026-03-29
**Goal**: Package the noise-based robustness evaluations into a reproducible, open-source benchmark that the community can use to stress-test echo foundation models under realistic acoustic degradation.

---

## Motivation

- EchoJEPA introduces compute-matched comparisons AND a synthetic noise benchmark — the first systematic robustness evaluation for echo foundation models
- US-JEPA has already adopted this approach, confirming it is becoming a community need
- Different models degrade differently under noise (EchoJEPA degrades gracefully, MAE is already bad so looks flat, EchoPrime degrades steeply)
- Currently, these evaluations use internal data and ad-hoc scripts — packaging as a benchmark makes them reproducible and citable

## Core Design

### Datasets (all publicly available)

| Dataset | Task | Metric | Notes |
|---------|------|--------|-------|
| **EchoNet-Dynamic** | LVEF regression | MAE, R² | Stanford, 10k+ videos, standard benchmark |
| **EchoNet-Pediatric** | LVEF regression (pediatric) | MAE, R² | Tests distribution shift (pediatric anatomy) |
| **CAMUS** | LV/MYO/LA segmentation | Dice, Hausdorff | 500 patients, ED/ES, 2CH/4CH |

### Noise Types (from USAugment)

All perturbations model real clinical acoustic degradation — not generic image corruptions.

| Noise Type | Physics | Source |
|------------|---------|--------|
| **Depth Attenuation** | Exponential signal loss with depth from transducer | Ostvik et al. 2021 |
| **Gaussian Shadow** | Localized dark region from beam-blocking structures (ribs, calcification) | Smistad et al. 2018 |
| **Haze Artifact** | Reverberation: brightens dark regions + washes out contrast | Ostvik et al. 2021 |
| **Speckle Reduction** | Variable bilateral-filter despeckling (simulates scanner post-processing) | Smistad et al. 2018 |

### Severity Levels

Three levels per noise type: **Low**, **Medium**, **High**. Parameters TBD — calibrate so that:
- Low: barely perceptible, minimal metric impact on strong models
- Medium: clearly visible artifact, moderate degradation
- High: severe but still clinically plausible

### Proposed severity parameters

#### Depth Attenuation
| Level | `attenuation_rate` | Deepest-point intensity retained |
|-------|-------------------|----------------------------------|
| Low   | 0.75              | ~47%                             |
| Med   | 1.50              | ~22%                             |
| High  | 2.15              | ~12%                             |

#### Gaussian Shadow
| Level | `strength` | `sigma_x`, `sigma_y` |
|-------|-----------|----------------------|
| Low   | 0.4       | 0.15                 |
| Med   | 0.6       | 0.20                 |
| High  | 0.8       | 0.25                 |

#### Haze Artifact
| Level | `haze_intensity` | `sigma` |
|-------|-----------------|---------|
| Low   | 0.15            | 0.25    |
| Med   | 0.30            | 0.30    |
| High  | 0.50            | 0.35    |

#### Speckle Reduction (NEW — not yet implemented)
| Level | `sigma_spatial` | `sigma_color` |
|-------|----------------|---------------|
| Low   | 0.5            | 0.2           |
| Med   | 1.0            | 0.5           |
| High  | 2.0            | 1.0           |

**Note**: Speckle reduction severity parameters need empirical calibration. USAugment defaults: `sigma_spatial=(0.1, 2.0)`, `sigma_color=(0.0, 1.0)`, `window_size=5`.

### Output Table Format

The benchmark produces tables like:

```
                    Depth Attenuation    Gaussian Shadow      Haze Artifact        Speckle Reduction
Model         Clean Low  Med  High  Low  Med  High  Low  Med  High  Low  Med  High  Avg Deg ↓
─────────────────────────────────────────────────────────────────────────────────────────────────
EchoPrime     4.87  5.58 5.71 5.91  5.55 5.61 5.78  ...  ...  ...   ...  ...  ...   +16.8%
PanEcho       5.10  5.10 5.39 5.46  5.19 5.21 5.38  ...  ...  ...   ...  ...  ...   +3.7%
EchoMAE-L     8.52  8.51 8.57 8.58  8.56 8.57 8.57  ...  ...  ...   ...  ...  ...   +0.5%†
EchoJEPA-L    5.76  5.72 5.91 6.10  5.79 5.87 5.97  ...  ...  ...   ...  ...  ...   +2.3%
EchoJEPA-G    3.97  4.01 4.07 4.17  4.02 4.04 4.07  ...  ...  ...   ...  ...  ...   +2.3%
```

Plus degradation curves (severity on x-axis, metric on y-axis, one line per model).

†Models with poor baselines show misleadingly low relative degradation — report absolute numbers too.

---

## Architecture

### Inspiration: ChestAgentBench

Adopt the following patterns from ChestAgentBench (Wang Lab, ICML 2025):
- **HuggingFace distribution**: Single download command gets everything
- **Single-script evaluation**: `evaluate.py` with CLI args for model, task, noise config
- **Model-agnostic adapters**: Any model that outputs embeddings can be plugged in
- **Comprehensive JSON logging**: Per-sample results with full metadata
- **Minimal dependencies**: No Docker, no complex framework
- **`--max-cases` for rapid iteration**: Test on subset before full run

### Proposed structure

```
echobench/
├── README.md                     # Quick start, installation, citation
├── pyproject.toml                # pip install echobench
├── echobench/
│   ├── __init__.py
│   ├── evaluate.py               # Main entry point
│   ├── noise/
│   │   ├── __init__.py
│   │   ├── depth_attenuation.py
│   │   ├── gaussian_shadow.py
│   │   ├── haze_artifact.py
│   │   ├── speckle_reduction.py
│   │   └── severity.py           # Severity level definitions (LOW/MED/HIGH)
│   ├── tasks/
│   │   ├── __init__.py
│   │   ├── lvef_regression.py    # EchoNet-Dynamic / Pediatric
│   │   └── segmentation.py       # CAMUS
│   ├── adapters/
│   │   ├── __init__.py
│   │   ├── base.py               # Abstract model adapter
│   │   ├── echojepa.py
│   │   ├── echoprime.py
│   │   ├── panecho.py
│   │   ├── videomae.py
│   │   └── echofm.py
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── regression.py         # MAE, R², Pearson r
│   │   └── segmentation.py       # Dice, Hausdorff
│   └── reporting/
│       ├── __init__.py
│       ├── tables.py             # LaTeX / markdown table generation
│       └── plots.py              # Degradation curves
├── configs/
│   ├── echonet_dynamic.yaml
│   ├── echonet_pediatric.yaml
│   └── camus.yaml
└── scripts/
    ├── download_data.sh          # Helper to download public datasets
    └── generate_figures.py       # Reproduce paper figures
```

### Evaluation flow

```
1. User provides: model adapter + probe checkpoint + dataset path
2. EchoBench loads clean test set
3. For each (noise_type × severity_level) + clean:
   a. Apply perturbation (deterministic seed per video)
   b. Run model → predictions
   c. Compute metrics
4. Output: JSON log + summary table + degradation curves
```

### Key design decisions

- **Perturbations applied on-the-fly** (not pre-generated videos) — saves storage, ensures reproducibility via deterministic seeding
- **Scan mask auto-detection** from first frame (threshold at 10/255) — perturbations only inside ultrasound cone
- **Temporally consistent** perturbations — same noise map across all frames of a video (matches real physics)
- **Deterministic seeding** from video path hash — same video gets identical perturbation across models

---

## What exists vs. what needs to be built

### Already implemented (in `scripts/rebuttal/`)

| Component | File | Status |
|-----------|------|--------|
| Depth attenuation | `echo_perturbations.py` | ✅ Done |
| Gaussian shadow | `echo_perturbations.py` | ✅ Done |
| Haze artifact | `echo_perturbations.py` | ✅ Done |
| Scan mask detection | `echo_perturbations.py` | ✅ Done |
| Noised inference pipeline | `noised_inference.py` | ✅ Done |
| Perturbed video cache generation | `generate_perturbed_videos.py` | ✅ Done |
| CKA representational stability | `cka_speckle.py` | ✅ Done |
| Severity classification probe | `noise_level_probe.py` | ✅ Done |
| Frame shuffling (temporal) | `frame_shuffling.py` | ✅ Done |
| Batch depth attenuation | `data/scripts/batch_depth_attenuation.py` | ✅ Done |
| Batch gaussian shadow | `data/scripts/batch_gaussian_shadow.py` | ✅ Done |
| EchoNet-Dynamic inference configs | `configs/inference/*/echonet-dynamic/` | ✅ Done |
| EchoNet-Pediatric inference configs | `configs/inference/*/echonet-pediatric/` | ✅ Done |
| Model adapters (5 models) | `evals/*/modelcustom/` | ✅ Done |

### Implemented (in `echobench/` package — v0.1)

| Component | File | Status |
|-----------|------|--------|
| SpeckleReduction noise type | `echobench/noise/perturbations.py` | ✅ Done |
| Unified evaluation script | `echobench/evaluate.py` | ✅ Done |
| CLI entry point | `echobench/cli.py` | ✅ Done |
| Adapter interface (Protocol) | `echobench/adapters/base.py` | ✅ Done |
| EchoJEPA adapter | `echobench/adapters/echojepa.py` | ✅ Done |
| VideoMAE adapter | `echobench/adapters/videomae.py` | ✅ Done |
| LVEF regression task | `echobench/tasks/lvef.py` | ✅ Done |
| Regression metrics | `echobench/metrics/regression.py` | ✅ Done |
| Segmentation metrics | `echobench/metrics/segmentation.py` | ✅ Done |
| Markdown + LaTeX tables | `echobench/reporting/tables.py` | ✅ Done |
| Degradation curve plots | `echobench/reporting/plots.py` | ✅ Done |
| pip-installable package | `echobench/pyproject.toml` | ✅ Done |
| Tests (48 passing) | `tests/echobench/` | ✅ Done |

### Still needs to be built

| Component | Priority | Notes |
|-----------|----------|-------|
| **CAMUS segmentation task** | P1 | Segmentation eval under noise (Dice/Hausdorff degradation) |
| **EchoPrime/PanEcho/EchoFM adapters** | P1 | Additional model adapters |
| **HuggingFace dataset card** | P2 | Distribution + documentation |
| **`download_data.sh` helper** | P2 | One-command dataset setup |
| **Per-sample JSON logging** | P2 | `--log-samples` flag for detailed output |
| **Paper figures script** | P2 | Reproduce all benchmark figures from JSON logs |

---

## Additional noise types to consider

Beyond the 4 USAugment transforms, potential additions for comprehensiveness:

| Noise Type | Source | Rationale |
|------------|--------|-----------|
| **Gaussian noise** | Standard | Baseline comparison to generic corruption |
| **Resolution downsampling** | Standard | Tests resolution sensitivity |
| **Frame dropping** | Custom | Temporal robustness (missing frames) |
| **Frame rate change** | Custom | FPS sensitivity |
| **Sector rotation** | Custom | View angle variation |
| **Combined perturbations** | Custom | Real scans have multiple simultaneous artifacts |

These are lower priority — the 4 USAugment types are physics-grounded and form the core. Generic corruptions can be added later to show that domain-specific noise matters more than ImageNet-C style corruption.

---

## Open questions

1. **Severity calibration**: Should severity levels be perceptual (look the same across noise types) or physics-based (match real clinical variation)?
2. **Segmentation benchmark**: CAMUS is the obvious choice, but only has 2D still frames (ED/ES). Should we look for a video segmentation dataset instead?
3. **Number of test samples**: How many videos per dataset? Full test sets or a curated subset?
4. **Prediction averaging**: EchoBench is clip-level (one prediction per video), not study-level. Correct?
5. **Scope for paper**: Do we include CKA + severity probes + frame shuffling in EchoBench, or keep it focused on task degradation only?
6. **Name**: EchoBench — final name or placeholder?

---

## References

- USAugment: https://github.com/adamtupper/usaugment (Tupper & Gagné, TMLR 2025)
- ChestAgentBench: https://huggingface.co/datasets/wanglab/chest-agent-bench (Wang Lab, ICML 2025)
- EchoNet-Dynamic: https://echonet.github.io/dynamic/
- EchoNet-Pediatric: https://echonet.github.io/pediatric/
- CAMUS: https://www.creatis.insa-lyon.fr/Challenge/camus/
- Ostvik et al. 2021 — Depth attenuation + haze physics
- Smistad et al. 2018 — Gaussian shadow + speckle reduction physics
