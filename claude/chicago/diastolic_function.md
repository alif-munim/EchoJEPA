# UCMC Diastolic Function Grading External Validation — Inference Guide

Instructions for running frozen diastolic function **classification** probes on UCMC echocardiography data. This is a 4-class grading task (distinct from the MV e' regression task in `diastolic.md`). All probes were trained at UHN using frozen depth=1 attentive cross-attention heads. No model fine-tuning occurs — inference only.

**Contact:** Alif Munim (alif.munim@uhn.ca)

---

## 1. Overview

You will run **4 frozen encoder models** on **1 diastolic function grading task**:

| Model | Architecture | Params | Encoder File | Embed Dim |
|-------|-------------|--------|-------------|-----------|
| **EchoJEPA-G** | ViT-Giant | 1,012M | `pt-280-an81.pt` (15.3 GB) | 1408 |
| **EchoJEPA-L-K** | ViT-Large | 304M | `vitl-kinetics-pt220-an55.pt` (4.8 GB) | 1024 |
| **EchoPrime** | MViT-v2-S | ~35M | `echo_prime_encoder.pt` (~200 MB) | 512 |
| **PanEcho** | ConvNeXt-T + Transformer | ~30M | Auto-downloads from hub | 768 |

| Task | Classes | Class Labels |
|------|---------|-------------|
| **Diastolic Function** | 4 | 0=normal, 1=Grade I (impaired relaxation), 2=Grade II (pseudonormal), 3=Grade III/IV (restrictive) |

That's **4 inference runs** total (4 models x 1 task).

**Why this task matters:** Diastolic function grading integrates multiple echocardiographic parameters (E/A ratio, e' velocity, E/e' ratio, TR velocity, LA volume index) into a single clinical grade. Predicting the composite grade from B-mode video alone tests whether the model has learned the structural correlates of diastolic dysfunction.

---

## 2. Shared Google Drive

Encoder checkpoints: same as other guides.

Probes:

```
gdrive:echo_foundation/nature_medicine/chicago/probes/diastolic_function/
├── echojepa-g_diastolic_function/best.pt         # 3.0 GB
├── echojepa-l-k_diastolic_function/best.pt       # 1.6 GB
├── echoprime_diastolic_function/best.pt          # 398 MB
└── panecho_diastolic_function/best.pt            # 893 MB
```

---

## 3. Environment Setup

Same as `valve_severity.md` Sections 3.1–3.4.

---

## 4. Organize Checkpoints

```bash
mkdir -p checkpoints/encoders
mkdir -p checkpoints/probes/diastolic_function/{echojepa-g,echojepa-l-k,echoprime,panecho}

# Encoders (skip if already downloaded)
cp <gdrive>/chicago/checkpoints/echojepa_g_uhn_pt280_an81/pt-280-an81.pt checkpoints/encoders/
cp <gdrive>/checkpoints/vitl-kinetics-pt220-an55.pt checkpoints/encoders/

# Probes (from GDrive chicago/probes/diastolic_function/)
cp <gdrive>/probes/diastolic_function/echojepa-g_diastolic_function/best.pt    checkpoints/probes/diastolic_function/echojepa-g/
cp <gdrive>/probes/diastolic_function/echojepa-l-k_diastolic_function/best.pt  checkpoints/probes/diastolic_function/echojepa-l-k/
cp <gdrive>/probes/diastolic_function/echoprime_diastolic_function/best.pt     checkpoints/probes/diastolic_function/echoprime/
cp <gdrive>/probes/diastolic_function/panecho_diastolic_function/best.pt       checkpoints/probes/diastolic_function/panecho/
```

---

## 5. Prepare Your Data

### 5.1 Video format

Same as `valve_severity.md` Section 5.1.

### 5.2 CSV format

Space-delimited, no header. Two columns: video path and integer label.

```
/data/ucmc/diastolic/study001_clip01.mp4 0
/data/ucmc/diastolic/study001_clip02.mp4 0
/data/ucmc/diastolic/study002_clip01.mp4 2
```

**Class mapping:**
- **0** = Normal diastolic function
- **1** = Grade I (impaired relaxation)
- **2** = Grade II (pseudonormal filling)
- **3** = Grade III/IV (restrictive filling)

### 5.3 View filtering (CRITICAL)

These probes were trained on **view-filtered** clips. Include only:

| Allowed Views | Exclude |
|--------------|---------|
| A4C, A2C, A3C, PLAX | Color Doppler, spectral Doppler, tissue Doppler, all other views |

Use the view/color classifiers from `valve_severity.md` Section 5.4 to filter.

### 5.4 Create CSV

```
data/csv/ucmc_diastolic_function_test.csv
```

---

## 6. Inference Configs

Same structure as `lv_function.md` Section 6. Key settings:

- `num_classes: 4`

Example for EchoJEPA-G:
```yaml
tag: ucmc-echojepa-g-diastolic-function
probe_checkpoint: checkpoints/probes/diastolic_function/echojepa-g/best.pt
dataset_train: data/csv/ucmc_diastolic_function_test.csv
dataset_val: data/csv/ucmc_diastolic_function_test.csv
num_classes: 4
```

For the full YAML template per model, see `lv_function.md` Sections 6.1–6.4. Set `num_classes: 4` and remove `task_type: regression` (classification is the default).

---

## 7. Running Inference

```bash
MODELS=("echojepa_g" "echojepa_lk" "echoprime" "panecho")

for model in "${MODELS[@]}"; do
  echo "Running ${model} on diastolic_function..."
  python -m evals.main \
    --fname "configs/inference/chicago/${model}_diastolic_function.yaml" \
    --devices cuda:0 \
    --val_only \
    2>&1 | tee "logs/ucmc_${model}_diastolic_function.log"
done
```

---

## 8. Output

### 8.1 Expected output files

```
results/ucmc/
├── ucmc-echojepa-g-diastolic-function/study_predictions.csv
├── ucmc-echojepa-lk-diastolic-function/study_predictions.csv
├── ucmc-echoprime-diastolic-function/study_predictions.csv
└── ucmc-panecho-diastolic-function/study_predictions.csv
```

---

## 9. Troubleshooting

See `valve_severity.md` Section 9.

---

## 10. Expected Results

For reference, here are the **UHN internal validation** results (not UCMC — these are from the Toronto training site's held-out test set):

| Task | Metric | EchoJEPA-G | EchoJEPA-L-K | EchoPrime | PanEcho |
|------|--------|-----------|-------------|----------|---------|
| Diastolic Function | AUROC (macro) | 0.903 | 0.855 | 0.846 | 0.830 |

Cross-institution performance may differ due to population differences, grading conventions, and ultrasound equipment. Grade definitions may vary across institutions.
