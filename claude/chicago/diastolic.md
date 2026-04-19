# UCMC Diastolic Function External Validation — Inference Guide

Instructions for running frozen diastolic function regression probes on UCMC echocardiography data. All probes were trained at UHN using frozen depth=1 attentive cross-attention heads. No model fine-tuning occurs — inference only.

**Contact:** Alif Munim (alif.munim@uhn.ca)

---

## 1. Overview

You will run **4 frozen encoder models** on **1 diastolic function regression task**:

| Model | Architecture | Params | Encoder File | Embed Dim |
|-------|-------------|--------|-------------|-----------|
| **EchoJEPA-G** | ViT-Giant | 1,012M | `pt-280-an81.pt` (15.3 GB) | 1408 |
| **EchoJEPA-L-K** | ViT-Large | 304M | `vitl-kinetics-pt220-an55.pt` (4.8 GB) | 1024 |
| **EchoPrime** | MViT-v2-S | ~35M | `echo_prime_encoder.pt` (~200 MB) | 512 |
| **PanEcho** | ConvNeXt-T + Transformer | ~30M | Auto-downloads from hub | 768 |

| Task | Type | Unit | Description |
|------|------|------|-------------|
| **MV e' (medial)** | Regression | cm/s | Medial mitral annular early diastolic velocity (tissue Doppler-derived, predicted from B-mode) |

That's **4 inference runs** total (4 models x 1 task).

**Why this task matters:** MV e' is normally measured by tissue Doppler imaging. The probes predict it from B-mode video alone, testing whether the model has learned the structural correlates of diastolic relaxation.

---

## 2. Shared Google Drive

Encoder checkpoints: same as other guides.

Probes:

```
gdrive:echo_foundation/nature_medicine/chicago/probes/diastolic/
├── echojepa-g_mv_ee_medial/best.pt               # 3.0 GB
├── echojepa-l-k_mv_ee_medial/best.pt             # 1.6 GB
├── echoprime_mv_ee_medial/best.pt                # 398 MB
└── panecho_mv_ee_medial/best.pt                  # 893 MB
```

---

## 3. Environment Setup

Same as `valve_severity.md` Sections 3.1–3.4.

---

## 4. Organize Checkpoints

```bash
mkdir -p checkpoints/encoders
mkdir -p checkpoints/probes/mv_ee_medial/{echojepa-g,echojepa-l-k,echoprime,panecho}

# Encoders (skip if already downloaded)
cp <gdrive>/chicago/checkpoints/echojepa_g_uhn_pt280_an81/pt-280-an81.pt checkpoints/encoders/
cp <gdrive>/checkpoints/vitl-kinetics-pt220-an55.pt checkpoints/encoders/

# Probes (from GDrive chicago/probes/diastolic/)
cp <gdrive>/probes/diastolic/echojepa-g_mv_ee_medial/best.pt    checkpoints/probes/mv_ee_medial/echojepa-g/
cp <gdrive>/probes/diastolic/echojepa-l-k_mv_ee_medial/best.pt  checkpoints/probes/mv_ee_medial/echojepa-l-k/
cp <gdrive>/probes/diastolic/echoprime_mv_ee_medial/best.pt     checkpoints/probes/mv_ee_medial/echoprime/
cp <gdrive>/probes/diastolic/panecho_mv_ee_medial/best.pt       checkpoints/probes/mv_ee_medial/panecho/
```

---

## 5. Prepare Your Data

### 5.1 Video format

Same as `valve_severity.md` Section 5.1.

### 5.2 CSV format

Space-delimited, no header. Two columns: video path and float label.

- MV e' medial in cm/s (raw float values, not normalized)

### 5.3 View filtering

These probes were trained on **all B-mode views** per study. Include all available B-mode clips.

**Most informative views:** A4C is the primary view for mitral annular assessment. A2C also contributes.

**Exclude:** Color Doppler, spectral Doppler, and tissue Doppler clips.

### 5.4 Create CSV

```
data/csv/ucmc_mv_ee_medial_test.csv
```

---

## 6. Inference Configs

Same structure as `lv_function.md` Section 6. Key settings:

- `task_type: regression`
- `num_classes: 1`

Example for EchoJEPA-G:
```yaml
tag: ucmc-echojepa-g-mv-ee-medial
probe_checkpoint: checkpoints/probes/mv_ee_medial/echojepa-g/best.pt
dataset_train: data/csv/ucmc_mv_ee_medial_test.csv
dataset_val: data/csv/ucmc_mv_ee_medial_test.csv
```

For the full YAML template per model, see `lv_function.md` Sections 6.1–6.4.

---

## 7. Running Inference

```bash
MODELS=("echojepa_g" "echojepa_lk" "echoprime" "panecho")

for model in "${MODELS[@]}"; do
  echo "Running ${model} on mv_ee_medial..."
  python -m evals.main \
    --fname "configs/inference/chicago/${model}_mv_ee_medial.yaml" \
    --devices cuda:0 \
    --val_only \
    2>&1 | tee "logs/ucmc_${model}_mv_ee_medial.log"
done
```

---

## 8. Output

### 8.1 Expected output files

```
results/ucmc/
├── ucmc-echojepa-g-mv-ee-medial/study_predictions.csv
├── ucmc-echojepa-lk-mv-ee-medial/study_predictions.csv
├── ucmc-echoprime-mv-ee-medial/study_predictions.csv
└── ucmc-panecho-mv-ee-medial/study_predictions.csv
```

---

## 9. Troubleshooting

See `valve_severity.md` Section 9.

---

## 10. Expected Results

For reference, here are the **UHN internal validation** results (not UCMC — these are from the Toronto training site's held-out test set):

| Task | Metric | EchoJEPA-G | EchoJEPA-L-K | EchoPrime | PanEcho |
|------|--------|-----------|-------------|----------|---------|
| MV e' (medial) | R² | 0.598 | 0.492 | 0.422 | 0.454 |
| MV e' (medial) | MAE (cm/s) | 2.79 | 3.20 | 3.29 | 3.21 |

Cross-institution performance may differ due to population differences, measurement conventions, and ultrasound equipment.
