# UCMC Disease Detection External Validation — Inference Guide

Instructions for running frozen disease detection probes on UCMC echocardiography data. These are **binary classification** tasks (disease present vs. hard-negative control). All probes were trained at UHN using frozen depth=1 attentive cross-attention heads. No model fine-tuning occurs — inference only.

**Contact:** Alif Munim (alif.munim@uhn.ca)

---

## 1. Overview

You will run **4 frozen encoder models** on **7 disease detection tasks**:

| Model | Architecture | Params | Encoder File | Embed Dim |
|-------|-------------|--------|-------------|-----------|
| **EchoJEPA-G** | ViT-Giant | 1,012M | `pt-280-an81.pt` (15.3 GB) | 1408 |
| **EchoJEPA-L-K** | ViT-Large | 304M | `vitl-kinetics-pt220-an55.pt` (4.8 GB) | 1024 |
| **EchoPrime** | MViT-v2-S | ~35M | `echo_prime_encoder.pt` (~200 MB) | 512 |
| **PanEcho** | ConvNeXt-T + Transformer | ~30M | Auto-downloads from hub | 768 |

| Task | Classes | Positive | Negative (Hard Control) | View Filter |
|------|---------|----------|------------------------|-------------|
| **HCM** | 2 | Hypertrophic cardiomyopathy | Concentric LVH | PLAX, PSAX-PM, PSAX-MV, A4C |
| **DCM** | 2 | Dilated cardiomyopathy | HF without DCM | All B-mode views |
| **Amyloidosis** | 2 | Cardiac amyloidosis | HCM | PLAX, A4C, PSAX-AV/MV/PM/AP |
| **Bicuspid AV** | 2 | Bicuspid aortic valve | Tricuspid AV | PLAX, PSAX-AV, A3C |
| **Myxomatous MV** | 2 | Myxomatous mitral valve | Non-myxomatous MR | A4C, A2C, PLAX |
| **Rheumatic MV** | 2 | Rheumatic mitral valve | Non-rheumatic MS | A4C, A2C, PLAX |
| **STEMI** | 2 | ST-elevation MI | NSTEMI | All B-mode views |

That's **28 inference runs** total (4 models x 7 tasks).

**Important:** All negative cohorts are **hard negatives** (clinically similar conditions that must be distinguished from the target disease), not random controls. This makes the task clinically meaningful but also more challenging.

**Prerequisite:** UCMC must have institutional diagnostic labels for these conditions. If labels are not available for a disease, skip that task.

---

## 2. Shared Google Drive

Encoder checkpoints: same as other guides.

Probes:

```
gdrive:echo_foundation/nature_medicine/chicago/probes/disease_detection/
├── echojepa-g_disease_hcm/best.pt                # 3.0 GB
├── echojepa-g_disease_dcm/best.pt                # 3.0 GB
├── echojepa-g_disease_amyloidosis/best.pt        # 3.0 GB
├── echojepa-g_disease_bicuspid_av/best.pt        # 3.0 GB
├── echojepa-g_disease_myxomatous_mv/best.pt      # 3.0 GB
├── echojepa-g_disease_rheumatic_mv/best.pt       # 3.0 GB
├── echojepa-g_disease_stemi/best.pt              # 3.0 GB
├── echojepa-l-k_disease_hcm/best.pt              # 1.6 GB
├── echojepa-l-k_disease_dcm/best.pt              # 1.6 GB
├── echojepa-l-k_disease_amyloidosis/best.pt      # 1.6 GB
├── echojepa-l-k_disease_bicuspid_av/best.pt      # 1.6 GB
├── echojepa-l-k_disease_myxomatous_mv/best.pt    # 1.6 GB
├── echojepa-l-k_disease_rheumatic_mv/best.pt     # 1.6 GB
├── echojepa-l-k_disease_stemi/best.pt            # 1.6 GB
├── echoprime_disease_hcm/best.pt                 # 398 MB
├── echoprime_disease_dcm/best.pt                 # 398 MB
├── echoprime_disease_amyloidosis/best.pt         # 398 MB
├── echoprime_disease_bicuspid_av/best.pt         # 398 MB
├── echoprime_disease_myxomatous_mv/best.pt       # 398 MB
├── echoprime_disease_rheumatic_mv/best.pt        # 398 MB
├── echoprime_disease_stemi/best.pt               # 398 MB
├── panecho_disease_hcm/best.pt                   # 893 MB
├── panecho_disease_dcm/best.pt                   # 893 MB
├── panecho_disease_amyloidosis/best.pt           # 893 MB
├── panecho_disease_bicuspid_av/best.pt           # 893 MB
├── panecho_disease_myxomatous_mv/best.pt         # 893 MB
├── panecho_disease_rheumatic_mv/best.pt          # 893 MB
└── panecho_disease_stemi/best.pt                 # 893 MB
```

---

## 3. Environment Setup

Same as `valve_severity.md` Sections 3.1–3.4.

---

## 4. Organize Checkpoints

```bash
mkdir -p checkpoints/encoders
mkdir -p checkpoints/probes/{disease_hcm,disease_dcm,disease_amyloidosis,disease_bicuspid_av,disease_myxomatous_mv,disease_rheumatic_mv,disease_stemi}/{echojepa-g,echojepa-l-k,echoprime,panecho}

# Encoders (skip if already downloaded)
cp <gdrive>/chicago/checkpoints/echojepa_g_uhn_pt280_an81/pt-280-an81.pt checkpoints/encoders/
cp <gdrive>/checkpoints/vitl-kinetics-pt220-an55.pt checkpoints/encoders/

# Probes (download from GDrive chicago/probes/disease_detection/)
# HCM
cp <gdrive>/probes/disease_detection/echojepa-g_disease_hcm/best.pt      checkpoints/probes/disease_hcm/echojepa-g/
cp <gdrive>/probes/disease_detection/echojepa-l-k_disease_hcm/best.pt    checkpoints/probes/disease_hcm/echojepa-l-k/
cp <gdrive>/probes/disease_detection/echoprime_disease_hcm/best.pt       checkpoints/probes/disease_hcm/echoprime/
cp <gdrive>/probes/disease_detection/panecho_disease_hcm/best.pt         checkpoints/probes/disease_hcm/panecho/
# DCM
cp <gdrive>/probes/disease_detection/echojepa-g_disease_dcm/best.pt      checkpoints/probes/disease_dcm/echojepa-g/
cp <gdrive>/probes/disease_detection/echojepa-l-k_disease_dcm/best.pt    checkpoints/probes/disease_dcm/echojepa-l-k/
cp <gdrive>/probes/disease_detection/echoprime_disease_dcm/best.pt       checkpoints/probes/disease_dcm/echoprime/
cp <gdrive>/probes/disease_detection/panecho_disease_dcm/best.pt         checkpoints/probes/disease_dcm/panecho/
# Amyloidosis
cp <gdrive>/probes/disease_detection/echojepa-g_disease_amyloidosis/best.pt    checkpoints/probes/disease_amyloidosis/echojepa-g/
cp <gdrive>/probes/disease_detection/echojepa-l-k_disease_amyloidosis/best.pt  checkpoints/probes/disease_amyloidosis/echojepa-l-k/
cp <gdrive>/probes/disease_detection/echoprime_disease_amyloidosis/best.pt     checkpoints/probes/disease_amyloidosis/echoprime/
cp <gdrive>/probes/disease_detection/panecho_disease_amyloidosis/best.pt       checkpoints/probes/disease_amyloidosis/panecho/
# Bicuspid AV
cp <gdrive>/probes/disease_detection/echojepa-g_disease_bicuspid_av/best.pt    checkpoints/probes/disease_bicuspid_av/echojepa-g/
cp <gdrive>/probes/disease_detection/echojepa-l-k_disease_bicuspid_av/best.pt  checkpoints/probes/disease_bicuspid_av/echojepa-l-k/
cp <gdrive>/probes/disease_detection/echoprime_disease_bicuspid_av/best.pt     checkpoints/probes/disease_bicuspid_av/echoprime/
cp <gdrive>/probes/disease_detection/panecho_disease_bicuspid_av/best.pt       checkpoints/probes/disease_bicuspid_av/panecho/
# Myxomatous MV
cp <gdrive>/probes/disease_detection/echojepa-g_disease_myxomatous_mv/best.pt    checkpoints/probes/disease_myxomatous_mv/echojepa-g/
cp <gdrive>/probes/disease_detection/echojepa-l-k_disease_myxomatous_mv/best.pt  checkpoints/probes/disease_myxomatous_mv/echojepa-l-k/
cp <gdrive>/probes/disease_detection/echoprime_disease_myxomatous_mv/best.pt     checkpoints/probes/disease_myxomatous_mv/echoprime/
cp <gdrive>/probes/disease_detection/panecho_disease_myxomatous_mv/best.pt       checkpoints/probes/disease_myxomatous_mv/panecho/
# Rheumatic MV
cp <gdrive>/probes/disease_detection/echojepa-g_disease_rheumatic_mv/best.pt    checkpoints/probes/disease_rheumatic_mv/echojepa-g/
cp <gdrive>/probes/disease_detection/echojepa-l-k_disease_rheumatic_mv/best.pt  checkpoints/probes/disease_rheumatic_mv/echojepa-l-k/
cp <gdrive>/probes/disease_detection/echoprime_disease_rheumatic_mv/best.pt     checkpoints/probes/disease_rheumatic_mv/echoprime/
cp <gdrive>/probes/disease_detection/panecho_disease_rheumatic_mv/best.pt       checkpoints/probes/disease_rheumatic_mv/panecho/
# STEMI
cp <gdrive>/probes/disease_detection/echojepa-g_disease_stemi/best.pt    checkpoints/probes/disease_stemi/echojepa-g/
cp <gdrive>/probes/disease_detection/echojepa-l-k_disease_stemi/best.pt  checkpoints/probes/disease_stemi/echojepa-l-k/
cp <gdrive>/probes/disease_detection/echoprime_disease_stemi/best.pt     checkpoints/probes/disease_stemi/echoprime/
cp <gdrive>/probes/disease_detection/panecho_disease_stemi/best.pt       checkpoints/probes/disease_stemi/panecho/
```

---

## 5. Prepare Your Data

### 5.1 Video format

Same as `valve_severity.md` Section 5.1.

### 5.2 CSV format

Space-delimited, no header. Two columns: video path and integer label.

- **0** = negative (hard control)
- **1** = positive (disease present)

```
/data/ucmc/hcm/study001_clip01.mp4 1
/data/ucmc/hcm/study001_clip02.mp4 1
/data/ucmc/hcm/study002_clip01.mp4 0
```

### 5.3 Label source requirements

Each task requires institutional diagnostic labels (not just echo report mentions):

| Task | Positive Definition | Recommended Negative Control |
|------|--------------------|-----------------------------|
| HCM | Confirmed HCM diagnosis | Concentric LVH (hypertensive heart disease) |
| DCM | Confirmed DCM diagnosis | Heart failure without DCM |
| Amyloidosis | Confirmed cardiac amyloidosis | HCM (similar wall thickening) |
| Bicuspid AV | Confirmed bicuspid aortic valve | Normal (tricuspid) aortic valve |
| Myxomatous MV | Confirmed myxomatous mitral valve | Non-myxomatous mitral regurgitation |
| Rheumatic MV | Confirmed rheumatic mitral valve | Non-rheumatic mitral stenosis |
| STEMI | Confirmed STEMI | NSTEMI |

If hard-negative controls are not available, random age/sex-matched controls without the target disease are acceptable but will inflate AUROC relative to the UHN results.

### 5.4 View filtering (CRITICAL)

These probes were trained on **view-filtered** clips. You must filter your CSVs to include only task-relevant views.

**Required view filters per task:**

| Task | Allowed Views |
|------|--------------|
| HCM | PLAX, PSAX-PM, PSAX-MV, A4C |
| DCM | All B-mode views |
| Amyloidosis | PLAX, A4C, PSAX-AV, PSAX-MV, PSAX-PM, PSAX-AP |
| Bicuspid AV | PLAX, PSAX-AV, A3C |
| Myxomatous MV | A4C, A2C, PLAX |
| Rheumatic MV | A4C, A2C, PLAX |
| STEMI | All B-mode views |

Use the view/color classifiers from `valve_severity.md` Section 5.4 to filter clips by view. **Exclude** all Doppler clips (color, spectral, tissue Doppler).

### 5.5 Create CSVs

```
data/csv/ucmc_disease_hcm_test.csv
data/csv/ucmc_disease_dcm_test.csv
data/csv/ucmc_disease_amyloidosis_test.csv
data/csv/ucmc_disease_bicuspid_av_test.csv
data/csv/ucmc_disease_myxomatous_mv_test.csv
data/csv/ucmc_disease_rheumatic_mv_test.csv
data/csv/ucmc_disease_stemi_test.csv
```

Only create CSVs for diseases where you have institutional labels.

---

## 6. Inference Configs

Same structure as `lv_function.md` Section 6. Key settings:

- `task_type: classification` (this is the default; omit or set explicitly)
- `num_classes: 2`

Example for EchoJEPA-G HCM:
```yaml
tag: ucmc-echojepa-g-disease-hcm
probe_checkpoint: checkpoints/probes/disease_hcm/echojepa-g/best.pt
dataset_train: data/csv/ucmc_disease_hcm_test.csv
dataset_val: data/csv/ucmc_disease_hcm_test.csv
num_classes: 2
```

For the full YAML template per model, see `lv_function.md` Sections 6.1–6.4. Change `task_type` to classification (or remove it, since classification is the default), set `num_classes: 2`, and update `tag`, `probe_checkpoint`, and data paths.

---

## 7. Running Inference

```bash
MODELS=("echojepa_g" "echojepa_lk" "echoprime" "panecho")
TASKS=("disease_hcm" "disease_dcm" "disease_amyloidosis" "disease_bicuspid_av" "disease_myxomatous_mv" "disease_rheumatic_mv" "disease_stemi")

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
├── ucmc-echojepa-g-disease-hcm/study_predictions.csv
├── ucmc-echojepa-g-disease-dcm/study_predictions.csv
├── ucmc-echojepa-g-disease-amyloidosis/study_predictions.csv
├── ucmc-echojepa-g-disease-bicuspid-av/study_predictions.csv
├── ucmc-echojepa-g-disease-myxomatous-mv/study_predictions.csv
├── ucmc-echojepa-g-disease-rheumatic-mv/study_predictions.csv
├── ucmc-echojepa-g-disease-stemi/study_predictions.csv
├── ucmc-echojepa-lk-disease-hcm/study_predictions.csv
│   ... (same pattern for L-K, EchoPrime, PanEcho)
└── ucmc-panecho-disease-stemi/study_predictions.csv
```

### 8.2 Metrics

For binary classification, the pipeline prints:
- **AUROC** (area under the ROC curve)
- **AUPRC** (area under the precision-recall curve)
- **Balanced accuracy**
- **Accuracy**

---

## 9. Troubleshooting

See `valve_severity.md` Section 9.

---

## 10. Expected Results

For reference, here are the **UHN internal validation** results (not UCMC — these are from the Toronto training site's held-out test set):

| Task | N | Prevalence | EchoJEPA-G | EchoJEPA-L-K | EchoPrime | PanEcho |
|------|---|-----------|-----------|-------------|----------|---------|
| HCM | 7,299 | 25.5% | 0.960 | 0.903 | 0.806 | 0.866 |
| DCM | 2,152 | 23.9% | 0.837 | 0.772 | 0.785 | 0.768 |
| Amyloidosis | 1,918 | 3.0% | 0.927 | 0.754 | 0.771 | 0.826 |
| Bicuspid AV | 23,990 | 6.1% | 0.975 | 0.881 | 0.901 | 0.876 |
| Myxomatous MV | 20,509 | 3.6% | 0.946 | 0.912 | 0.859 | 0.835 |
| Rheumatic MV | 360 | 72.2% | 0.846 | 0.785 | 0.739 | 0.745 |
| STEMI | 156 | 32.1% | 0.826 | 0.623 | 0.810 | 0.788 |

All values are AUROC. Cross-institution performance may differ due to population differences, label definitions, and ultrasound equipment. Performance is especially sensitive to how negative controls are defined (hard negatives vs. random controls).
