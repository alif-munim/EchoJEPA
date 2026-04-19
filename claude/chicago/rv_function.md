# UCMC RV Function External Validation — Inference Guide

Instructions for running frozen RV function regression probes on UCMC echocardiography data. All probes were trained at UHN using frozen depth=1 attentive cross-attention heads. No model fine-tuning occurs — inference only.

**Contact:** Alif Munim (alif.munim@uhn.ca)

---

## 1. Overview

You will run **4 frozen encoder models** on **4 RV function regression tasks**:

| Model | Architecture | Params | Encoder File | Embed Dim |
|-------|-------------|--------|-------------|-----------|
| **EchoJEPA-G** | ViT-Giant | 1,012M | `pt-280-an81.pt` (15.3 GB) | 1408 |
| **EchoJEPA-L-K** | ViT-Large | 304M | `vitl-kinetics-pt220-an55.pt` (4.8 GB) | 1024 |
| **EchoPrime** | MViT-v2-S | ~35M | `echo_prime_encoder.pt` (~200 MB) | 512 |
| **PanEcho** | ConvNeXt-T + Transformer | ~30M | Auto-downloads from hub | 768 |

| Task | Type | Unit | Description |
|------|------|------|-------------|
| **TAPSE** | Regression | cm | Tricuspid annular plane systolic excursion |
| **RVSP** | Regression | mmHg | Right ventricular systolic pressure |
| **RV FAC** | Regression | % | Right ventricular fractional area change |
| **RV S'** | Regression | m/s | Tricuspid annular systolic velocity (tissue Doppler-derived, predicted from B-mode) |

That's **16 inference runs** total (4 models x 4 tasks).

---

## 2. Shared Google Drive

Encoder checkpoints: same as other guides.

Probes:

```
gdrive:echo_foundation/nature_medicine/chicago/probes/rv_function/
├── echojepa-g_tapse/best.pt                      # 3.0 GB
├── echojepa-g_rvsp/best.pt                       # 3.0 GB
├── echojepa-g_rv_fac/best.pt                     # 3.0 GB
├── echojepa-g_rv_sp/best.pt                      # 3.0 GB
├── echojepa-l-k_tapse/best.pt                    # 1.6 GB
├── echojepa-l-k_rvsp/best.pt                     # 1.6 GB
├── echojepa-l-k_rv_fac/best.pt                   # 1.6 GB
├── echojepa-l-k_rv_sp/best.pt                    # 1.6 GB
├── echoprime_tapse/best.pt                       # 398 MB
├── echoprime_rvsp/best.pt                        # 398 MB
├── echoprime_rv_fac/best.pt                      # 398 MB
├── echoprime_rv_sp/best.pt                       # 398 MB
├── panecho_tapse/best.pt                         # 893 MB
├── panecho_rvsp/best.pt                          # 893 MB
├── panecho_rv_fac/best.pt                        # 893 MB
└── panecho_rv_sp/best.pt                         # 893 MB
```

---

## 3. Environment Setup

Same as `valve_severity.md` Sections 3.1–3.4.

---

## 4. Organize Checkpoints

```bash
mkdir -p checkpoints/encoders
mkdir -p checkpoints/probes/{tapse,rvsp,rv_fac,rv_sp}/{echojepa-g,echojepa-l-k,echoprime,panecho}

# Encoders (skip if already downloaded)
cp <gdrive>/chicago/checkpoints/echojepa_g_uhn_pt280_an81/pt-280-an81.pt checkpoints/encoders/
cp <gdrive>/checkpoints/vitl-kinetics-pt220-an55.pt checkpoints/encoders/

# Probes (download all 16 from GDrive chicago/probes/rv_function/)
cp <gdrive>/probes/rv_function/echojepa-g_tapse/best.pt        checkpoints/probes/tapse/echojepa-g/
cp <gdrive>/probes/rv_function/echojepa-g_rvsp/best.pt         checkpoints/probes/rvsp/echojepa-g/
cp <gdrive>/probes/rv_function/echojepa-g_rv_fac/best.pt       checkpoints/probes/rv_fac/echojepa-g/
cp <gdrive>/probes/rv_function/echojepa-g_rv_sp/best.pt        checkpoints/probes/rv_sp/echojepa-g/
cp <gdrive>/probes/rv_function/echojepa-l-k_tapse/best.pt      checkpoints/probes/tapse/echojepa-l-k/
cp <gdrive>/probes/rv_function/echojepa-l-k_rvsp/best.pt       checkpoints/probes/rvsp/echojepa-l-k/
cp <gdrive>/probes/rv_function/echojepa-l-k_rv_fac/best.pt     checkpoints/probes/rv_fac/echojepa-l-k/
cp <gdrive>/probes/rv_function/echojepa-l-k_rv_sp/best.pt      checkpoints/probes/rv_sp/echojepa-l-k/
cp <gdrive>/probes/rv_function/echoprime_tapse/best.pt         checkpoints/probes/tapse/echoprime/
cp <gdrive>/probes/rv_function/echoprime_rvsp/best.pt          checkpoints/probes/rvsp/echoprime/
cp <gdrive>/probes/rv_function/echoprime_rv_fac/best.pt        checkpoints/probes/rv_fac/echoprime/
cp <gdrive>/probes/rv_function/echoprime_rv_sp/best.pt         checkpoints/probes/rv_sp/echoprime/
cp <gdrive>/probes/rv_function/panecho_tapse/best.pt           checkpoints/probes/tapse/panecho/
cp <gdrive>/probes/rv_function/panecho_rvsp/best.pt            checkpoints/probes/rvsp/panecho/
cp <gdrive>/probes/rv_function/panecho_rv_fac/best.pt          checkpoints/probes/rv_fac/panecho/
cp <gdrive>/probes/rv_function/panecho_rv_sp/best.pt           checkpoints/probes/rv_sp/panecho/
```

---

## 5. Prepare Your Data

### 5.1 Video format

Same as `valve_severity.md` Section 5.1.

### 5.2 CSV format

Space-delimited, no header. Two columns: video path and float label.

- TAPSE in cm, RVSP in mmHg, RV FAC in %, RV S' in m/s
- Raw float values (not normalized)

### 5.3 View filtering

These probes were trained on **all B-mode views** per study. Include all available B-mode clips.

**Most informative views:** A4C is the primary view for RV assessment (TAPSE, RV FAC, RV S'). Subcostal views also contribute. RVSP is estimated from tricuspid regurgitation velocity, but the probes predict from B-mode structure alone.

**Exclude:** Color Doppler, spectral Doppler, and tissue Doppler clips.

### 5.4 Create CSVs

```
data/csv/ucmc_tapse_test.csv
data/csv/ucmc_rvsp_test.csv
data/csv/ucmc_rv_fac_test.csv
data/csv/ucmc_rv_sp_test.csv
```

---

## 6. Inference Configs

Same structure as `lv_function.md` Section 6. Key differences:

- `task_type: regression`
- `num_classes: 1`
- Change `tag`, `probe_checkpoint`, and data paths per task

Example for EchoJEPA-G TAPSE:
```yaml
tag: ucmc-echojepa-g-tapse
probe_checkpoint: checkpoints/probes/tapse/echojepa-g/best.pt
dataset_train: data/csv/ucmc_tapse_test.csv
dataset_val: data/csv/ucmc_tapse_test.csv
```

For the full YAML template, see `lv_function.md` Section 6.1 (EchoJEPA-G), 6.2 (L-K), 6.3 (EchoPrime), 6.4 (PanEcho) — only `tag`, `probe_checkpoint`, and data paths change.

---

## 7. Running Inference

```bash
MODELS=("echojepa_g" "echojepa_lk" "echoprime" "panecho")
TASKS=("tapse" "rvsp" "rv_fac" "rv_sp")

for model in "${MODELS[@]}"; do
  for task in "${TASKS[@]}"; do
    echo "Running ${model} on ${task}..."
    python -m evals.main \
      --fname "configs/inference/chicago/${model}_${task}.yaml" \
      --devices cuda:0 \
      --val_only \
      2>&1 | tee "logs/ucmc_${model}_${task}.log"
  done
done
```

---

## 8. Output

### 8.1 Expected output files

```
results/ucmc/
├── ucmc-echojepa-g-tapse/study_predictions.csv
├── ucmc-echojepa-g-rvsp/study_predictions.csv
├── ucmc-echojepa-g-rv-fac/study_predictions.csv
├── ucmc-echojepa-g-rv-sp/study_predictions.csv
├── ucmc-echojepa-lk-tapse/study_predictions.csv
├── ucmc-echojepa-lk-rvsp/study_predictions.csv
├── ucmc-echojepa-lk-rv-fac/study_predictions.csv
├── ucmc-echojepa-lk-rv-sp/study_predictions.csv
├── ucmc-echoprime-tapse/study_predictions.csv
├── ucmc-echoprime-rvsp/study_predictions.csv
├── ucmc-echoprime-rv-fac/study_predictions.csv
├── ucmc-echoprime-rv-sp/study_predictions.csv
├── ucmc-panecho-tapse/study_predictions.csv
├── ucmc-panecho-rvsp/study_predictions.csv
├── ucmc-panecho-rv-fac/study_predictions.csv
└── ucmc-panecho-rv-sp/study_predictions.csv
```

---

## 9. Troubleshooting

See `valve_severity.md` Section 9.

---

## 10. Expected Results

For reference, here are the **UHN internal validation** results (not UCMC — these are from the Toronto training site's held-out test set):

| Task | Metric | EchoJEPA-G | EchoJEPA-L-K | EchoPrime | PanEcho |
|------|--------|-----------|-------------|----------|---------|
| TAPSE | R² | 0.633 | 0.555 | 0.430 | 0.385 |
| TAPSE | MAE (cm) | 0.26 | 0.29 | 0.32 | 0.34 |
| RVSP | R² | 0.504 | 0.317 | 0.169 | 0.274 |
| RVSP | MAE (mmHg) | 8.01 | 9.35 | 9.50 | 9.33 |
| RV FAC | R² | 0.539 | 0.444 | 0.278 | 0.301 |
| RV FAC | MAE (%) | 5.77 | 6.51 | 7.14 | 7.10 |
| RV S' | R² | 0.591 | 0.473 | 0.353 | 0.301 |
| RV S' | MAE (m/s) | 0.016 | 0.019 | 0.020 | 0.021 |

Cross-institution performance may differ due to population differences, measurement conventions, and ultrasound equipment.
