# EchoBench: Acoustic Robustness Benchmark for Echocardiography Foundation Models

**Status**: v0.2 implemented (echobench branch), 62/62 tests passing
**Created**: 2026-03-29
**Updated**: 2026-03-30
**Goal**: Package the noise-based robustness evaluations into a reproducible, open-source benchmark that the community can use to stress-test echo foundation models under realistic acoustic degradation.

---

## Motivation

- EchoJEPA introduces compute-matched comparisons AND a synthetic noise benchmark — the first systematic robustness evaluation for echo foundation models
- US-JEPA has already adopted this approach, confirming it is becoming a community need
- Different models degrade differently under noise (EchoJEPA degrades gracefully, MAE is already bad so looks flat, EchoPrime degrades steeply)
- Packaging as a benchmark makes these evaluations reproducible and citable

## Package Location

`echobench/` directory in the EchoJEPA repo (branch: `echobench`). pip-installable via `pip install -e echobench/`.

---

## Implemented Components

### Noise Module (`echobench/noise/perturbations.py`)

4 physics-grounded acoustic perturbation types at 3 severity levels (Low/Medium/High):

| Noise Type | Physics | Reference |
|------------|---------|-----------|
| **Depth Attenuation** | Exponential signal loss with depth from transducer | Ostvik et al. 2021 |
| **Gaussian Shadow** | Localized dark region from beam-blocking structures | Smistad et al. 2018 |
| **Haze Artifact** | Reverberation: brightens dark regions + washes out contrast | Ostvik et al. 2021 |
| **Speckle Reduction** | Variable bilateral-filter despeckling (scanner post-processing) | Smistad et al. 2018 |

Severity parameters:

| Type | Low | Medium | High |
|------|-----|--------|------|
| Depth Atten. | rate=0.75 (47% retained) | rate=1.50 (22%) | rate=2.15 (12%) |
| Gauss. Shadow | strength=0.4, sigma=0.15 | strength=0.6, sigma=0.20 | strength=0.8, sigma=0.25 |
| Haze Artifact | intensity=0.15, sigma=0.25 | intensity=0.30, sigma=0.30 | intensity=0.50, sigma=0.35 |
| Speckle Red. | sigma_s=0.5, sigma_c=0.2 | sigma_s=1.0, sigma_c=0.5 | sigma_s=2.0, sigma_c=1.0 |

Key properties: temporally consistent, deterministic from seed, scan-mask-aware, on-the-fly (no pre-generation).

### Tasks

| Task | File | Dataset | Metrics |
|------|------|---------|---------|
| **LVEF Regression** | `tasks/lvef.py` | EchoNet-Dynamic / EchoNet-Pediatric | MAE, R², Pearson r |
| **Cardiac Segmentation** | `tasks/segmentation.py` | CAMUS (500 patients, ED/ES, 4CH/2CH) | Dice (LV/MYO/LA), Hausdorff-95 |

CAMUS task: loads NIfTI with `normalize=False`, injects noise before ImageNet norm, extracts spatial features at ED/ES temporal tokens via model-type-aware reshaping, runs decoder, computes per-structure metrics.

### Adapters

| Adapter | Architecture | embed_dim | Output | Status |
|---------|-------------|-----------|--------|--------|
| `echojepa` | ViT (V-JEPA 2) | 1024/1536 | [B, 1568, D] | Ready |
| `videomae` | ViT-L (VideoMAE) | 1024 | [B, N, D] | Ready |
| `echoprime` | MViT-v2-S | 512 | [B, 1, 512] | Needs checkpoint |
| `panecho` | ConvNeXt-Tiny | 768 | [B, 1, 768] | Needs source repo |
| `echofm` | ViT-L (MAE) | 1024 | [B, 1568, 1024] | Needs source repo |

### Other Components

| Component | File | Notes |
|-----------|------|-------|
| Evaluation orchestrator | `evaluate.py` | Runs 13 conditions (clean + 4 noise x 3 severity), computes Avg Degradation |
| CLI | `cli.py` | `echobench evaluate`, `echobench report`, `echobench list-noise` |
| Regression metrics | `metrics/regression.py` | MAE, R², Pearson r |
| Segmentation metrics | `metrics/segmentation.py` | Dice, Hausdorff-95 |
| Decoders | `tasks/decoders.py` | LinearSegDecoder (1x1 conv), SmallConvSegDecoder (4-stage transposed conv) |
| Markdown/LaTeX tables | `reporting/tables.py` | Multi-model comparison tables with Avg Deg column |
| Degradation curves | `reporting/plots.py` | Matplotlib plots (optional dep) |
| CAMUS dataset loader | `tasks/camus_dataset.py` | NIfTI loading with normalize flag for noise injection |
| Tests (62 passing) | `tests/echobench/` | Noise, metrics, decoders, spatial extraction, adapter imports |

---

## File Tree

```
echobench/
├── pyproject.toml
├── README.md
├── echobench/
│   ├── __init__.py
│   ├── cli.py
│   ├── evaluate.py
│   ├── noise/
│   │   ├── __init__.py
│   │   └── perturbations.py       # 4 noise types + severity configs
│   ├── adapters/
│   │   ├── __init__.py
│   │   ├── base.py                 # EncoderAdapter Protocol + BaseAdapter
│   │   ├── echojepa.py
│   │   ├── videomae.py
│   │   ├── echoprime.py
│   │   ├── panecho.py
│   │   └── echofm.py
│   ├── tasks/
│   │   ├── __init__.py
│   │   ├── lvef.py                 # LVEF regression (EchoNet-Dynamic/Pediatric)
│   │   ├── segmentation.py         # CAMUS segmentation
│   │   ├── camus_dataset.py        # NIfTI dataset loader
│   │   └── decoders.py             # LinearSegDecoder + SmallConvSegDecoder
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── regression.py           # MAE, R², Pearson
│   │   └── segmentation.py         # Dice, Hausdorff-95
│   └── reporting/
│       ├── __init__.py
│       ├── tables.py               # Markdown + LaTeX tables
│       └── plots.py                # Degradation curves
tests/
└── echobench/
    ├── __init__.py
    ├── test_noise.py               # 48 noise tests
    ├── test_metrics.py             # 12 metric tests
    └── test_decoders.py            # 14 decoder/adapter tests
```

---

## Output Table Format

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

†Models with poor baselines show misleadingly low relative degradation — report absolute numbers too.

---

## Remaining Work

### P1 — Run the benchmark
- [ ] Run all 5 models on EchoNet-Dynamic LVEF with all noise conditions to populate the paper table
- [ ] Run CAMUS segmentation under noise for all models
- [ ] Calibrate speckle reduction severity on real echos (current params from USAugment defaults)

### P2 — Polish for release
- [ ] HuggingFace dataset card for community distribution
- [ ] `download_data.sh` helper for public datasets
- [ ] Per-sample JSON logging (`--log-samples` flag)
- [ ] EchoNet-Pediatric as explicit second LVEF variant
- [ ] Batch/multi-GPU evaluation for speed

### P3 — Potential extensions
- Additional noise types: generic Gaussian, frame dropping, combined perturbations
- Resolution downsampling, frame rate variation
- These are lower priority — the 4 USAugment types are physics-grounded and form the core

---

## References

- USAugment: https://github.com/adamtupper/usaugment (Tupper & Gagne, TMLR 2025)
- ChestAgentBench: https://huggingface.co/datasets/wanglab/chest-agent-bench (Wang Lab, ICML 2025)
- EchoNet-Dynamic: https://echonet.github.io/dynamic/
- EchoNet-Pediatric: https://echonet.github.io/pediatric/
- CAMUS: https://www.creatis.insa-lyon.fr/Challenge/camus/
- Ostvik et al. 2021 — Depth attenuation + haze physics
- Smistad et al. 2018 — Gaussian shadow + speckle reduction physics
