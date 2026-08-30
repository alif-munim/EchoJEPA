# UCMC Data Efficiency External Validation — Inference Guide

Instructions for running frozen data-efficiency probes on UCMC echocardiography data. Same protocol as the other UCMC guides, but for probes trained at UHN on **fractions** of the full training data (50%, 25%, 12.5%, 6.25%, 3.125%, 1.5625%, and a task-specific 10%). Inference only — no fine-tuning.

**Contact:** Alif Munim (alif.munim@uhn.ca)

---

## 1. Overview

The data efficiency experiment retrains each probe on nested stratified subsets of the UHN training set (each smaller fraction is a strict subset of the next larger, seed=42, class/quintile-stratified) and evaluates on the same held-out test set. Applying these same probes to UCMC gives us the external-validation curve — how well cross-institution transfer holds as the training-set budget shrinks.

You will run **4 frozen encoder models** across **4 tasks** at **up to 7 fractions each**:

| Model | Architecture | Params | Encoder File | Embed Dim |
|-------|-------------|--------|-------------|-----------|
| **EchoJEPA-G** | ViT-Giant | 1,012M | `pt-280-an81.pt` (15.3 GB) | 1408 |
| **EchoJEPA-L-K** | ViT-Large | 304M | `vitl-kinetics-pt220-an55.pt` (4.8 GB) | 1024 |
| **EchoPrime** | MViT-v2-S | ~35M | `echo_prime_encoder.pt` (~200 MB) | 512 |
| **PanEcho** | ConvNeXt-T + Transformer | ~30M | Auto-downloads from hub | 768 |

| Task | Type | Views | Full-data guide |
|------|------|-------|-----------------|
| **LVEF** | Regression (%) | A4C, A2C | `lv_function.md` |
| **RV Basal Diam** | Regression (cm, B-mode only) | A4C | (not in a task-specific guide — same protocol as `lv_function.md`) |
| **Disease VSD** | Binary classification | PLAX, PSAX-AV, A4C | `disease_detection.md` |
| **AS Severity (color)** | 4-class classification | PLAX, PSAX-AV, A3C, A5C (B-mode + color) | `valve_severity.md` §7.7 |

**Fractions actually available on GDrive:**

| Task | 50% | 25% | 12.5% | 10% | 6.25% | 3.125% | 1.5625% |
|------|:---:|:---:|:-----:|:---:|:-----:|:------:|:-------:|
| LVEF | ✓ (all 4) | ✓ (all 4) | ✓ (all 4) | ✓ (all 4) | ✓ (all 4) | ✓ (all 4) | ✓ (all 4) |
| RV Basal Diam | ✓ (all 4) | ✓ (all 4) | ✓ (all 4) | — | ✓ (all 4) | ✓ (all 4) | ✓ (all 4) |
| Disease VSD | ✓ (all 4) | ✓ (all 4) | ✓ (all 4) | — | ✓ (all 4) | ✓ (all 4) | ✓ (all 4) |
| AS Severity | in progress | in progress | ✓ (all 4) | — | ✓ (all 4) | ✓ (all 4) | ✓ (all 4) |

- LVEF has a 7th fraction (`10pct`) unique to that task.
- AS 50% and 25% probe weights are being restaged and will be added shortly (predictions exist at UHN, weights are being re-uploaded).
- Total inference runs available today: **LVEF 28 + RV Basal 24 + VSD 24 + AS 16 = 92 runs**. AS grows to 24 once 25%/50% land.

---

## 2. Shared Google Drive

All probe weights use one flat directory, named `{model}_{task}_{fraction}/best.pt`. Fractions are written `1p5pct`, `3pct`, `6pct`, `10pct`, `12pct`, `25pct`, `50pct`.

```
gdrive:echo_foundation/nature_medicine/chicago/probes/data_efficiency/
├── echojepa-g_lvef_50pct/best.pt              # 3.0 GB each
├── echojepa-g_lvef_25pct/best.pt
├── echojepa-g_lvef_12pct/best.pt
├── echojepa-g_lvef_10pct/best.pt
├── echojepa-g_lvef_6pct/best.pt
├── echojepa-g_lvef_3pct/best.pt
├── echojepa-g_lvef_1p5pct/best.pt
├── echojepa-l-k_lvef_{50pct,25pct,12pct,10pct,6pct,3pct,1p5pct}/best.pt   # 1.6 GB each
├── echoprime_lvef_{...}/best.pt                                            # 398 MB each
├── panecho_lvef_{...}/best.pt                                              # 893 MB each
│
├── {model}_rv_basal_diam_{50pct,25pct,12pct,6pct,3pct,1p5pct}/best.pt
├── {model}_disease_vsd_{50pct,25pct,12pct,6pct,3pct,1p5pct}/best.pt
└── {model}_as_severity_color_{12pct,6pct,3pct,1p5pct}/best.pt              # 25/50pct pending
```

Encoders are the **same files** you already have from the other UCMC guides — no re-download:
```
gdrive:echo_foundation/nature_medicine/chicago/checkpoints/echojepa_g_uhn_pt280_an81/pt-280-an81.pt
gdrive:echo_foundation/nature_medicine/checkpoints/vitl-kinetics-pt220-an55.pt
```

---

## 3. Environment Setup

Same as `valve_severity.md` Sections 3.1–3.4. If you have already run any of the other UCMC guides, no additional setup is required.

---

## 4. Organize Checkpoints

The clean pattern is `checkpoints/probes/data_efficiency/{task}/{model}/{fraction}/best.pt`. Loop over what you have:

```bash
mkdir -p checkpoints/encoders

# Encoders (skip if already downloaded)
cp <gdrive>/chicago/checkpoints/echojepa_g_uhn_pt280_an81/pt-280-an81.pt checkpoints/encoders/
cp <gdrive>/checkpoints/vitl-kinetics-pt220-an55.pt checkpoints/encoders/

# Probes: download all of chicago/probes/data_efficiency/ into a mirror, then reorganize.
# Or use rclone directly:
rclone copy gdrive:echo_foundation/nature_medicine/chicago/probes/data_efficiency \
  ./raw_de_probes --transfers 8 --progress

# Reorganize
for d in raw_de_probes/*/; do
  name=$(basename "$d")
  # name format: {model}_{task}_{fraction}
  # models: echojepa-g, echojepa-l-k, echoprime, panecho
  # extract fraction (last token after final _)
  frac=${name##*_}
  # extract model (longest known prefix)
  for m in echojepa-l-k echojepa-g echoprime panecho; do
    if [[ "$name" == "${m}_"* ]]; then model=$m; task=${name#${m}_}; task=${task%_${frac}}; break; fi
  done
  dest="checkpoints/probes/data_efficiency/${task}/${model}/${frac}"
  mkdir -p "$dest"
  cp "$d/best.pt" "$dest/best.pt"
done
```

After this you should have:
```
checkpoints/probes/data_efficiency/
├── lvef/{echojepa-g,echojepa-l-k,echoprime,panecho}/{1p5,3,6,10,12,25,50}pct/best.pt
├── rv_basal_diam/{echojepa-g,echojepa-l-k,echoprime,panecho}/{1p5,3,6,12,25,50}pct/best.pt
├── disease_vsd/{echojepa-g,echojepa-l-k,echoprime,panecho}/{1p5,3,6,12,25,50}pct/best.pt
└── as_severity_color/{echojepa-g,echojepa-l-k,echoprime,panecho}/{1p5,3,6,12}pct/best.pt
```

---

## 5. Prepare Your Data

You should already have all four task CSVs from the other UCMC guides — the data-efficiency probes use the **same** test data as their full-data counterparts. Nothing new to prepare.

| Task | CSV | View filter | B-mode only? | Existing guide |
|------|-----|-------------|:------------:|----------------|
| LVEF | `data/csv/ucmc_lvef_test.csv` | A4C, A2C | no | `lv_function.md` §5 |
| RV Basal Diam | `data/csv/ucmc_rv_basal_diam_test.csv` (new; see below) | A4C | **yes** | this doc |
| Disease VSD | `data/csv/ucmc_disease_vsd_test.csv` | PLAX, PSAX-AV, A4C | no | `disease_detection.md` §5 |
| AS Severity | `data/csv/ucmc_as_severity_test.csv` | PLAX, PSAX-AV, A3C, A5C | no (B-mode + color) | `valve_severity.md` §5 |

### 5.1 New CSV — RV Basal Diameter

```
/data/ucmc/rv/study001_a4c_clip01.mp4 3.9
/data/ucmc/rv/study001_a4c_clip02.mp4 3.9
/data/ucmc/rv/study002_a4c_clip01.mp4 4.2
```

- Space-delimited, no header. Two columns: video path and raw label.
- Label = right ventricular basal diameter in **cm**, raw value (Z-score is applied at runtime).
- Views: **A4C only** (this probe was trained on A4C only).
- **B-mode only** (no color / spectral / tissue Doppler).
- Valid range: 1.0 to 7.0 cm; drop outside-range clips.

Z-score parameters (already in `zscore_params.json`, listed here for reference):

| task | target_mean | target_std | unit |
|---|---|---|---|
| `rv_basal_diam` | 3.8145958425 | 0.6663278617 | cm |

---

## 6. Inference Configs

Configs are identical to the full-data guides except (a) `probe_checkpoint` points to a fraction-specific probe, and (b) `tag` includes the fraction. Reuse the encoder + wrapper blocks verbatim from the appropriate guide:

- LVEF → copy from `lv_function.md` §6
- RV Basal Diam → copy from `lv_function.md` §6, change task name and dims
- Disease VSD → copy from `disease_detection.md` §6
- AS Severity → copy from `valve_severity.md` §6 (color-trained variant)

The only per-fraction changes are the two fields below. Example for EchoJEPA-G LVEF at 25%:

```yaml
tag: ucmc-echojepa-g-lvef-de-25pct
probe_checkpoint: checkpoints/probes/data_efficiency/lvef/echojepa-g/25pct/best.pt
```

Everything else in the config — encoder path, `num_heads: 16`, `num_probe_blocks: 1`, batch size, `study_sampling: true`, etc. — is unchanged from the full-data version.

**Generate the ~92 configs programmatically** rather than by hand:

```bash
mkdir -p configs/inference/chicago/data_efficiency

python3 - <<'EOF'
import os, glob, yaml

# Load the four "full-data" configs you already have as templates
TEMPLATES = {
    "lvef":              "configs/inference/chicago/echojepa_g_lvef.yaml",
    "rv_basal_diam":     "configs/inference/chicago/echojepa_g_lvef.yaml",  # same shape; edit task-specific bits
    "disease_vsd":       "configs/inference/chicago/echojepa_g_disease_vsd.yaml",
    "as_severity_color": "configs/inference/chicago/echojepa_g_as_severity_color.yaml",
}

FRACS_BY_TASK = {
    "lvef":              ["1p5pct","3pct","6pct","10pct","12pct","25pct","50pct"],
    "rv_basal_diam":     ["1p5pct","3pct","6pct","12pct","25pct","50pct"],
    "disease_vsd":       ["1p5pct","3pct","6pct","12pct","25pct","50pct"],
    "as_severity_color": ["1p5pct","3pct","6pct","12pct"],  # 25/50pct arriving later
}

MODELS = ["echojepa-g", "echojepa-l-k", "echoprime", "panecho"]

for task, tpl_path in TEMPLATES.items():
    for model in MODELS:
        for frac in FRACS_BY_TASK[task]:
            # (Load your per-model template and swap tag + probe_checkpoint)
            # tag           = f"ucmc-{model.replace('_','-')}-{task}-de-{frac}"
            # probe_ckpt    = f"checkpoints/probes/data_efficiency/{task}/{model}/{frac}/best.pt"
            # write to configs/inference/chicago/data_efficiency/{model}_{task}_{frac}.yaml
            pass
EOF
```

If you'd rather have me pre-generate all 92 YAMLs, let me know and I'll drop them in GDrive.

---

## 7. Running Inference

### 7.1 Loop over all runs

```bash
#!/bin/bash
# run_all_ucmc_data_efficiency.sh
set -eu
cd /path/to/EchoJEPA
mkdir -p logs/ucmc_de

for cfg in configs/inference/chicago/data_efficiency/*.yaml; do
  tag=$(basename "$cfg" .yaml)
  echo "=== $tag ==="
  python -m evals.main \
    --fname "$cfg" \
    --devices cuda:0 \
    --val_only 2>&1 | tee "logs/ucmc_de/${tag}.log"
done
```

### 7.2 SLURM (one array job per task)

```bash
#!/bin/bash
#SBATCH --job-name=ucmc-de
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=logs/ucmc_de/%x_%A_%a.out
#SBATCH --array=0-27         # one index per (model, fraction) for a single task

source activate echojepa
cd /path/to/EchoJEPA

CFGS=(configs/inference/chicago/data_efficiency/*_lvef_*.yaml)
python -m evals.main --fname "${CFGS[$SLURM_ARRAY_TASK_ID]}" --devices cuda:0 --val_only
```

### 7.3 GPU memory & wall-clock

Per-run VRAM matches the full-data version of each task (probes are the same size as their non-DE counterparts — same architecture, just trained on less data):

- EchoJEPA-G: ~25–30 GB (A100/H100/A6000)
- EchoJEPA-L-K: ~12–16 GB (V100/A100/RTX 3090)
- EchoPrime: ~8–10 GB
- PanEcho: ~6–8 GB

Wall-clock per run scales with your UCMC test-set size. Expect the same time per run as the full-data guides — the "data efficiency" refers to *training* data used at UHN, not to inference cost.

---

## 8. Output

Each run drops a `study_predictions.csv` under `<folder>/<tag>/`. Set `folder: results/ucmc/data_efficiency` in the configs to keep them separate. Directory layout after all runs:

```
results/ucmc/data_efficiency/
├── ucmc-echojepa-g-lvef-de-50pct/study_predictions.csv
├── ucmc-echojepa-g-lvef-de-25pct/study_predictions.csv
├── ...
├── ucmc-echojepa-g-lvef-de-1p5pct/study_predictions.csv
├── ucmc-echojepa-l-k-lvef-de-50pct/study_predictions.csv
├── ...
├── ucmc-echojepa-g-rv-basal-diam-de-50pct/study_predictions.csv
├── ...
├── ucmc-echojepa-g-disease-vsd-de-50pct/study_predictions.csv
├── ...
└── ucmc-echojepa-g-as-severity-color-de-12pct/study_predictions.csv
```

Please share the whole `results/ucmc/data_efficiency/` tree back via GDrive when done.

---

## 9. Troubleshooting

See `valve_severity.md` Section 9 and `lv_function.md` Section 9. Nothing DE-specific.

**One thing that trips people up on this experiment:** the DE probes are named by *training-data fraction*, not by anything about your test data. All fractions test on the same UCMC set. If you see identical AUROCs for two fractions of the same model, it can mean the probe learned essentially nothing new by increasing data — that's a real finding, not a bug.

---

## 10. Expected Results

For reference, UHN internal test results at each fraction (study-level predavg, N=15,097 LVEF / 14,703 RV basal / 1,657 VSD / 23,233 AS):

**LVEF Pearson r — UHN test:**

| Model | 100% | 50% | 25% | 12.5% | 6.25% | 3.125% |
|-------|------|------|------|-------|-------|--------|
| EchoJEPA-G | 0.889 | 0.887 | 0.883 | 0.878 | 0.874 | 0.869 |
| EchoJEPA-L-K | 0.852 | 0.850 | 0.838 | 0.815 | 0.765 | 0.629 |
| EchoPrime | 0.844 | 0.840 | 0.838 | 0.824 | 0.826 | 0.819 |
| PanEcho | 0.840 | 0.839 | 0.831 | 0.824 | 0.828 | 0.808 |

**AS Severity AUROC (≥Moderate) — UHN test:**

| Model | 100% | 50% | 25% | 12.5% | 6.25% | 3.125% |
|-------|------|------|------|-------|-------|--------|
| EchoJEPA-G | 0.972 | 0.968 | 0.962 | 0.960 | 0.953 | 0.941 |
| EchoJEPA-L-K | 0.940 | 0.929 | 0.920 | 0.917 | 0.902 | 0.888 |
| EchoPrime | 0.946 | 0.948 | 0.941 | 0.934 | 0.936 | 0.918 |
| PanEcho | 0.937 | 0.936 | 0.932 | 0.931 | 0.930 | 0.925 |

VSD and RV basal diameter internal numbers are in the manuscript-facing tracker; ask if you want them for reference.

Cross-site performance will be lower than the internal numbers above — how much lower is exactly the finding this experiment is designed to measure. A ~0.03–0.10 AUROC / Pearson r drop is typical for external validation on similar UHN→other-site transfers.
