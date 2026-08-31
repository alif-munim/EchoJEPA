# EchoBench

Acoustic robustness benchmark for echocardiography foundation models.

EchoBench evaluates how echo foundation models degrade under realistic, physics-grounded ultrasound noise. It applies four types of acoustic perturbation at three severity levels and measures task performance degradation across LVEF regression and cardiac segmentation.

## Noise Types

All perturbations model real clinical degradation modes in echocardiography (not generic image corruptions). Based on [USAugment](https://github.com/adamtupper/usaugment) (Tupper & Gagne, TMLR 2025).

| Noise Type | Physics | Reference |
|------------|---------|-----------|
| **Depth Attenuation** | Exponential signal loss with depth from transducer | Ostvik et al. 2021 |
| **Gaussian Shadow** | Localized dark region from beam-blocking structures | Smistad et al. 2018 |
| **Haze Artifact** | Reverberation: brightens darks + washes out contrast | Ostvik et al. 2021 |
| **Speckle Reduction** | Variable bilateral-filter despeckling (scanner post-processing) | Smistad et al. 2018 |

Each noise type has **Low**, **Medium**, and **High** severity levels. Perturbations are temporally consistent (same map across all frames), deterministic from a path-based seed, and applied only within the auto-detected ultrasound scan region.

## Tasks

| Task | Dataset | Metrics | Format |
|------|---------|---------|--------|
| **LVEF Regression** | EchoNet-Dynamic / EchoNet-Pediatric | MAE, R², Pearson r | Space-delimited CSV: `path label` |
| **Cardiac Segmentation** | CAMUS | Dice (LV, MYO, LA), Hausdorff-95 | NIfTI sequences + GT masks |

## Installation

```bash
# Core (LVEF task only)
pip install -e .

# With CAMUS segmentation support
pip install -e ".[camus]"

# With reporting (matplotlib)
pip install -e ".[reporting]"

# Everything
pip install -e ".[all]"
```

**Note:** EchoBench must be installed within the [EchoJEPA](https://github.com/alif-munim/EchoJEPA) repository for the built-in adapters and probe loading to work. The EchoJEPA `src/` package must be importable.

```bash
# From the EchoJEPA repo root:
pip install -e .          # install EchoJEPA (vjepa2 package)
pip install -e echobench/ # install EchoBench
```

## Quick Start

### LVEF Regression (EchoNet-Dynamic)

```bash
echobench evaluate \
    --adapter echojepa \
    --checkpoint checkpoints/echojepa-l.pt \
    --probe path/to/lvef_probe/best.pt \
    --task lvef \
    --data-csv data/csv/echonet_dynamic_test.csv \
    --model-name vit_large \
    --output results_echojepa_lvef.json
```

### CAMUS Segmentation

```bash
echobench evaluate \
    --adapter echojepa \
    --checkpoint checkpoints/echojepa-l.pt \
    --task camus \
    --camus-root data/camus/CAMUS_public \
    --decoder-checkpoint path/to/decoder/best.pt \
    --decoder-type linear \
    --model-type vjepa \
    --output results_echojepa_camus.json
```

### Generate Tables from Results

```bash
# Markdown table comparing multiple models
echobench report results_echojepa.json results_echomae.json --format markdown

# LaTeX table for paper
echobench report results_*.json --format latex --metric mae --output table.tex
```

### List Available Noise Types

```bash
echobench list-noise
```

## Adapters

EchoBench supports multiple encoder architectures via adapters:

| Adapter | Architecture | embed_dim | Output Tokens | Status |
|---------|-------------|-----------|---------------|--------|
| `echojepa` | ViT (V-JEPA 2) | 1024/1536 | 1568 | Ready |
| `videomae` | ViT-L (VideoMAE) | 1024 | 196 | Ready |
| `echoprime` | MViT-v2-S | 512 | 1 (pooled) | Requires checkpoint |
| `panecho` | ConvNeXt-Tiny | 768 | 1 (pooled) | Requires source repo |
| `echofm` | ViT-L (MAE) | 1024 | 1568 | Requires source repo |

### Adapter-Specific Requirements

**EchoPrime**: Requires the EchoPrime encoder checkpoint (`echo_prime_encoder.pt`). The checkpoint is not publicly available — contact the EchoPrime authors.

**PanEcho**: Requires the PanEcho source repository cloned locally:
```bash
git clone https://github.com/echonet/panecho.git PanEcho
export PANECHO_ROOT=./PanEcho
```

**EchoFM**: Requires the EchoFM source repository and checkpoint:
```bash
# Clone EchoFM and set environment variable
export ECHOFM_ROOT=./EchoFM
```

### Adding Your Own Adapter

Implement the `EncoderAdapter` protocol:

```python
import torch
from echobench.adapters.base import BaseAdapter

class MyAdapter(BaseAdapter):
    def __init__(self, checkpoint, device="cuda", **kwargs):
        super().__init__()
        self.embed_dim = 1024  # your encoder's output dimension
        self.encoder = load_your_model(checkpoint)
        self.to(device)
        self.freeze()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, C, T, H, W] -> [B, N, D] token embeddings."""
        return self.encoder(x)
```

Then use it directly:

```python
from echobench.evaluate import evaluate
from echobench.tasks import LVEFTask

adapter = MyAdapter(checkpoint="path/to/weights.pt", device="cuda")
task = LVEFTask(data_csv="test.csv", probe_checkpoint="probe.pt")
results = evaluate(adapter, task, output_path="results.json")
```

## CLI Reference

```
echobench evaluate
    --adapter NAME          Encoder adapter (echojepa, videomae, echoprime, panecho, echofm)
    --checkpoint PATH       Encoder checkpoint path
    --task TASK             Task: lvef or camus (default: lvef)

    # LVEF task options:
    --probe PATH            Probe checkpoint (required for lvef)
    --data-csv PATH         Test CSV (required for lvef)
    --task-type TYPE        regression or classification (default: regression)

    # CAMUS task options:
    --camus-root PATH       CAMUS_public directory (required for camus)
    --decoder-checkpoint P  Decoder checkpoint (required for camus)
    --decoder-type TYPE     linear or conv (default: linear)
    --model-type TYPE       vjepa/videomae/echoprime/panecho/echofm (default: vjepa)
    --camus-split SPLIT     testing or training (default: testing)
    --camus-views VIEWS     4CH 2CH (default: 4CH)

    # Common options:
    --noise-types TYPES     Noise types to test (default: all 4)
    --severity-levels LEVS  Severity levels (default: all 3)
    --max-cases N           Limit test videos per condition
    --device DEVICE         cuda or cpu (default: cuda)
    --output PATH           Output JSON path
    --resolution N          Input resolution (default: 224)
    --frames N              Frames per clip (default: 16)
    --frame-step N          Frame stride (default: 2)

    # EchoJEPA-specific:
    --model-name NAME       ViT variant (default: vit_large)
    --checkpoint-key KEY    Checkpoint dict key (default: target_encoder)

echobench report RESULTS...
    --format FMT            markdown, latex, or csv (default: markdown)
    --metric NAME           Primary metric for table (default: mae)
    --output PATH           Save to file

echobench list-noise        List noise types and severity parameters
```

## Output Format

EchoBench outputs a JSON file with three sections:

```json
{
    "meta": {
        "timestamp": "2026-03-30T...",
        "num_conditions": 13,
        "noise_types": ["depth_attenuation", "gaussian_shadow", "haze_artifact", "speckle_reduction"],
        "severity_levels": ["low", "medium", "high"],
        "total_seconds": 1234.5
    },
    "conditions": [
        {"condition": "clean", "metrics": {"mae": 3.97, "r2": 0.82, "pearson_r": 0.91}, ...},
        {"condition": "depth_attenuation/low", "metrics": {"mae": 4.01, ...}, ...},
        ...
    ],
    "summary": {
        "primary_metric": "mae",
        "clean_value": 3.97,
        "avg_degradation_pct": 2.3,
        "per_noise_degradation_pct": {
            "depth_attenuation": 2.1,
            "gaussian_shadow": 1.8,
            ...
        }
    }
}
```

## TODOs

- [ ] EchoNet-Pediatric as explicit second LVEF dataset
- [ ] Per-sample JSON logging (`--log-samples` flag)
- [ ] HuggingFace dataset card for distribution
- [ ] `download_data.sh` helper for public datasets
- [ ] Batch/multi-GPU evaluation
- [ ] Additional noise types: generic Gaussian, frame dropping, combined perturbations
- [ ] CAMUS decoder training script (currently requires pre-trained decoder)

## Citation

If you use EchoBench in your research, please cite:

```bibtex
@article{echojepa2026,
    title={EchoJEPA: Towards a Cardiac World Model},
    author={...},
    journal={Nature Medicine},
    year={2026}
}
```

## License

Apache 2.0
