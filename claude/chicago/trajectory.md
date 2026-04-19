# UCMC Trajectory Prediction External Validation — Inference Guide

Instructions for running frozen trajectory prediction probes on UCMC echocardiography data. These tasks predict **future clinical change from a single baseline echocardiogram**. The probes were trained at UHN on longitudinal echo pairs (baseline + follow-up), but inference requires only the **baseline study**. All probes use frozen depth=1 attentive cross-attention heads. No model fine-tuning occurs — inference only.

**Contact:** Alif Munim (alif.munim@uhn.ca)

---

## 1. Overview

You will run **4 frozen encoder models** on **5 trajectory prediction tasks**:

| Model | Architecture | Params | Encoder File | Embed Dim |
|-------|-------------|--------|-------------|-----------|
| **EchoJEPA-G** | ViT-Giant | 1,012M | `pt-280-an81.pt` (15.3 GB) | 1408 |
| **EchoJEPA-L-K** | ViT-Large | 304M | `vitl-kinetics-pt220-an55.pt` (4.8 GB) | 1024 |
| **EchoPrime** | MViT-v2-S | ~35M | `echo_prime_encoder.pt` (~200 MB) | 512 |
| **PanEcho** | ConvNeXt-T + Transformer | ~30M | Auto-downloads from hub | 768 |

| Task | Type | Classes | Definition | View Filter |
|------|------|---------|------------|-------------|
| **LVEF Trajectory** | 3-class | 3 | 0=declined (delta <= -8), 1=stable, 2=improved (delta >= +8) | A4C, A2C |
| **LVEF Onset** | Binary | 2 | Baseline EF >= 50% who later drop below 50% (new-onset cardiomyopathy) | A4C, A2C |
| **MR Severity Onset** | Binary | 2 | Baseline MR <= mild who progress to >= moderate (30-365 day window) | A4C, A2C, A3C, PLAX |
| **TAPSE Onset** | Binary | 2 | Baseline TAPSE >= 1.7 cm who later drop below 1.7 cm (new RV dysfunction) | A4C |
| **TR Severity Onset** | Binary | 2 | Baseline TR <= mild who progress to >= moderate | A4C, Subcostal, PLAX |

**Note on model availability:** TAPSE Onset and TR Severity Onset have probes for **EchoJEPA-G and EchoJEPA-L-K only** (EchoPrime and PanEcho probes are not available for these two tasks). The other three tasks have all 4 models.

That's **18 inference runs** total (4 models x 3 tasks + 2 models x 2 tasks).

**Why these tasks matter:** Trajectory prediction is the most demanding test of representation quality. The model sees only a single baseline echocardiogram and must predict whether the patient will deteriorate on a future study. This requires the model to encode not just current cardiac state but subtle structural signatures that presage future decline.

---

## 2. Shared Google Drive

Encoder checkpoints: same as other guides.

Probes:

```
gdrive:echo_foundation/nature_medicine/chicago/probes/trajectory/
├── echojepa-g_trajectory_lvef/best.pt                    # 3.0 GB
├── echojepa-l-k_trajectory_lvef/best.pt                  # 1.6 GB
├── echoprime_trajectory_lvef/best.pt                     # 398 MB
├── panecho_trajectory_lvef/best.pt                       # 893 MB
├── echojepa-g_trajectory_lvef_onset/best.pt              # 3.0 GB
├── echojepa-l-k_trajectory_lvef_onset/best.pt            # 1.6 GB
├── echoprime_trajectory_lvef_onset/best.pt               # 398 MB
├── panecho_trajectory_lvef_onset/best.pt                 # 893 MB
├── echojepa-g_trajectory_mr_severity_onset/best.pt       # 3.0 GB
├── echojepa-l-k_trajectory_mr_severity_onset/best.pt     # 1.6 GB
├── echoprime_trajectory_mr_severity_onset/best.pt        # 398 MB
├── panecho_trajectory_mr_severity_onset/best.pt          # 893 MB
├── echojepa-g_trajectory_tapse_onset/best.pt             # 3.0 GB
├── echojepa-l-k_trajectory_tapse_onset/best.pt           # 1.6 GB
├── echojepa-g_trajectory_tr_severity_onset/best.pt       # 3.0 GB
└── echojepa-l-k_trajectory_tr_severity_onset/best.pt     # 1.6 GB
```

Note: TAPSE Onset and TR Severity Onset only have EchoJEPA-G and EchoJEPA-L-K probes (no EchoPrime or PanEcho).

---

## 3. Environment Setup

Same as `valve_severity.md` Sections 3.1-3.4.

---

## 4. Organize Checkpoints

```bash
mkdir -p checkpoints/encoders
mkdir -p checkpoints/probes/trajectory_lvef/{echojepa-g,echojepa-l-k,echoprime,panecho}
mkdir -p checkpoints/probes/trajectory_lvef_onset/{echojepa-g,echojepa-l-k,echoprime,panecho}
mkdir -p checkpoints/probes/trajectory_mr_severity_onset/{echojepa-g,echojepa-l-k,echoprime,panecho}
mkdir -p checkpoints/probes/trajectory_tapse_onset/{echojepa-g,echojepa-l-k}
mkdir -p checkpoints/probes/trajectory_tr_severity_onset/{echojepa-g,echojepa-l-k}

# Encoders (skip if already downloaded)
cp <gdrive>/chicago/checkpoints/echojepa_g_uhn_pt280_an81/pt-280-an81.pt checkpoints/encoders/
cp <gdrive>/checkpoints/vitl-kinetics-pt220-an55.pt checkpoints/encoders/

# Probes — LVEF Trajectory (4 models)
cp <gdrive>/probes/trajectory/echojepa-g_trajectory_lvef/best.pt      checkpoints/probes/trajectory_lvef/echojepa-g/
cp <gdrive>/probes/trajectory/echojepa-l-k_trajectory_lvef/best.pt    checkpoints/probes/trajectory_lvef/echojepa-l-k/
cp <gdrive>/probes/trajectory/echoprime_trajectory_lvef/best.pt       checkpoints/probes/trajectory_lvef/echoprime/
cp <gdrive>/probes/trajectory/panecho_trajectory_lvef/best.pt         checkpoints/probes/trajectory_lvef/panecho/

# Probes — LVEF Onset (4 models)
cp <gdrive>/probes/trajectory/echojepa-g_trajectory_lvef_onset/best.pt      checkpoints/probes/trajectory_lvef_onset/echojepa-g/
cp <gdrive>/probes/trajectory/echojepa-l-k_trajectory_lvef_onset/best.pt    checkpoints/probes/trajectory_lvef_onset/echojepa-l-k/
cp <gdrive>/probes/trajectory/echoprime_trajectory_lvef_onset/best.pt       checkpoints/probes/trajectory_lvef_onset/echoprime/
cp <gdrive>/probes/trajectory/panecho_trajectory_lvef_onset/best.pt         checkpoints/probes/trajectory_lvef_onset/panecho/

# Probes — MR Severity Onset (4 models)
cp <gdrive>/probes/trajectory/echojepa-g_trajectory_mr_severity_onset/best.pt      checkpoints/probes/trajectory_mr_severity_onset/echojepa-g/
cp <gdrive>/probes/trajectory/echojepa-l-k_trajectory_mr_severity_onset/best.pt    checkpoints/probes/trajectory_mr_severity_onset/echojepa-l-k/
cp <gdrive>/probes/trajectory/echoprime_trajectory_mr_severity_onset/best.pt       checkpoints/probes/trajectory_mr_severity_onset/echoprime/
cp <gdrive>/probes/trajectory/panecho_trajectory_mr_severity_onset/best.pt         checkpoints/probes/trajectory_mr_severity_onset/panecho/

# Probes — TAPSE Onset (2 models only)
cp <gdrive>/probes/trajectory/echojepa-g_trajectory_tapse_onset/best.pt      checkpoints/probes/trajectory_tapse_onset/echojepa-g/
cp <gdrive>/probes/trajectory/echojepa-l-k_trajectory_tapse_onset/best.pt    checkpoints/probes/trajectory_tapse_onset/echojepa-l-k/

# Probes — TR Severity Onset (2 models only)
cp <gdrive>/probes/trajectory/echojepa-g_trajectory_tr_severity_onset/best.pt      checkpoints/probes/trajectory_tr_severity_onset/echojepa-g/
cp <gdrive>/probes/trajectory/echojepa-l-k_trajectory_tr_severity_onset/best.pt    checkpoints/probes/trajectory_tr_severity_onset/echojepa-l-k/
```

---

## 5. Prepare Your Data

### 5.1 Video format

Same as `valve_severity.md` Section 5.1.

### 5.2 Longitudinal cohort requirements

These tasks require **longitudinal data**: patients with at least two echocardiograms separated in time. However, inference uses only the **baseline (earlier) study**. The follow-up study provides the ground truth label but is never fed to the model.

For each task, you need:
1. **Baseline echo** (the study whose video clips are input to the model)
2. **Follow-up measurement** (the clinical value that determines the label)
3. **Sufficient time gap** between studies (see per-task details below)

### 5.3 Task-specific cohort definitions

#### LVEF Trajectory (3-class)

Patients with at least two LVEF measurements. Compute delta = LVEF_followup - LVEF_baseline:
- **0 = Declined**: delta <= -8 percentage points
- **1 = Stable**: -8 < delta < +8
- **2 = Improved**: delta >= +8 percentage points

No baseline value restriction. No minimum time gap (any follow-up interval).

#### LVEF Onset (binary)

Patients with **normal baseline LVEF (>= 50%)** who either maintain normal function or develop new cardiomyopathy:
- **0 = Stable**: follow-up LVEF >= 50%
- **1 = Onset**: follow-up LVEF < 50% (new-onset reduced EF)

This is the Nature Medicine "new-onset cardiomyopathy" task. No minimum time gap.

#### MR Severity Onset (binary)

Patients with **baseline MR severity <= mild (grade <= 2)** within a **30-365 day follow-up window**:
- **0 = Stable**: follow-up MR severity <= mild
- **1 = Worsened**: follow-up MR severity >= moderate (grade >= 3)

MR severity grades: 0=none/trivial, 1=mild, 2=mild-moderate, 3=moderate, 4=moderate-severe, 5=severe. Time window: 30-365 days between baseline and follow-up.

#### TAPSE Onset (binary)

Patients with **baseline TAPSE >= 1.7 cm** (normal RV function):
- **0 = Stable**: follow-up TAPSE >= 1.7 cm
- **1 = Decline**: follow-up TAPSE < 1.7 cm (new RV dysfunction)

No minimum time gap.

#### TR Severity Onset (binary)

Patients with **baseline TR severity <= mild (grade <= 2)**:
- **0 = Stable**: follow-up TR severity <= mild
- **1 = Worsened**: follow-up TR severity >= moderate (grade >= 3)

TR severity grades: same scale as MR (0-5). No minimum time gap.

### 5.4 CSV format

Space-delimited, no header. Two columns: video path and integer label.

For 3-class tasks (LVEF Trajectory):
```
/data/ucmc/trajectory/study001_clip01.mp4 0
/data/ucmc/trajectory/study001_clip02.mp4 0
/data/ucmc/trajectory/study002_clip01.mp4 1
/data/ucmc/trajectory/study003_clip01.mp4 2
```

For binary tasks (all onset tasks):
```
/data/ucmc/trajectory/study001_clip01.mp4 0
/data/ucmc/trajectory/study001_clip02.mp4 0
/data/ucmc/trajectory/study002_clip01.mp4 1
```

Include **all clips per baseline study** in the CSV (the pipeline averages predictions across clips).

### 5.5 View filtering (CRITICAL)

These probes were trained on **view-filtered** clips. Include only the allowed views for each task:

| Task | Allowed Views |
|------|--------------|
| LVEF Trajectory | A4C, A2C |
| LVEF Onset | A4C, A2C |
| MR Severity Onset | A4C, A2C, A3C, PLAX |
| TAPSE Onset | A4C |
| TR Severity Onset | A4C, Subcostal, PLAX |

Use the view/color classifiers from `valve_severity.md` Section 5.4 to filter clips by view. **Exclude** all Doppler clips (color, spectral, tissue Doppler).

### 5.6 Create CSVs

```
data/csv/ucmc_trajectory_lvef_test.csv
data/csv/ucmc_trajectory_lvef_onset_test.csv
data/csv/ucmc_trajectory_mr_severity_onset_test.csv
data/csv/ucmc_trajectory_tapse_onset_test.csv
data/csv/ucmc_trajectory_tr_severity_onset_test.csv
```

Only create CSVs for tasks where you have longitudinal data with the required measurements.

---

## 6. Inference Configs

Same structure as `lv_function.md` Section 6. Key settings:

- `task_type: classification` (default; can omit)
- `num_classes: 3` for LVEF Trajectory
- `num_classes: 2` for all onset tasks

Example for EchoJEPA-G LVEF Onset:
```yaml
tag: ucmc-echojepa-g-trajectory-lvef-onset
probe_checkpoint: checkpoints/probes/trajectory_lvef_onset/echojepa-g/best.pt
dataset_train: data/csv/ucmc_trajectory_lvef_onset_test.csv
dataset_val: data/csv/ucmc_trajectory_lvef_onset_test.csv
num_classes: 2
```

Example for EchoJEPA-G LVEF Trajectory (3-class):
```yaml
tag: ucmc-echojepa-g-trajectory-lvef
probe_checkpoint: checkpoints/probes/trajectory_lvef/echojepa-g/best.pt
dataset_train: data/csv/ucmc_trajectory_lvef_test.csv
dataset_val: data/csv/ucmc_trajectory_lvef_test.csv
num_classes: 3
```

For the full YAML template per model, see `lv_function.md` Sections 6.1-6.4. Set the appropriate `num_classes` and remove `task_type: regression` (classification is the default).

---

## 7. Running Inference

```bash
MODELS_4=("echojepa_g" "echojepa_lk" "echoprime" "panecho")
MODELS_2=("echojepa_g" "echojepa_lk")

# Tasks with all 4 models
TASKS_4=("trajectory_lvef" "trajectory_lvef_onset" "trajectory_mr_severity_onset")

# Tasks with only G and L-K
TASKS_2=("trajectory_tapse_onset" "trajectory_tr_severity_onset")

# Run 4-model tasks
for model in "${MODELS_4[@]}"; do
  for task in "${TASKS_4[@]}"; do
    echo "Running ${model} on ${task}..."
    python -m evals.main \
      --fname "configs/inference/chicago/${model}_${task}.yaml" \
      --devices cuda:0 \
      --val_only \
      2>&1 | tee "logs/ucmc_${model}_${task}.log"
  done
done

# Run 2-model tasks
for model in "${MODELS_2[@]}"; do
  for task in "${TASKS_2[@]}"; do
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
├── ucmc-echojepa-g-trajectory-lvef/study_predictions.csv
├── ucmc-echojepa-lk-trajectory-lvef/study_predictions.csv
├── ucmc-echoprime-trajectory-lvef/study_predictions.csv
├── ucmc-panecho-trajectory-lvef/study_predictions.csv
├── ucmc-echojepa-g-trajectory-lvef-onset/study_predictions.csv
├── ucmc-echojepa-lk-trajectory-lvef-onset/study_predictions.csv
├── ucmc-echoprime-trajectory-lvef-onset/study_predictions.csv
├── ucmc-panecho-trajectory-lvef-onset/study_predictions.csv
├── ucmc-echojepa-g-trajectory-mr-severity-onset/study_predictions.csv
├── ucmc-echojepa-lk-trajectory-mr-severity-onset/study_predictions.csv
├── ucmc-echoprime-trajectory-mr-severity-onset/study_predictions.csv
├── ucmc-panecho-trajectory-mr-severity-onset/study_predictions.csv
├── ucmc-echojepa-g-trajectory-tapse-onset/study_predictions.csv
├── ucmc-echojepa-lk-trajectory-tapse-onset/study_predictions.csv
├── ucmc-echojepa-g-trajectory-tr-severity-onset/study_predictions.csv
└── ucmc-echojepa-lk-trajectory-tr-severity-onset/study_predictions.csv
```

---

## 9. Troubleshooting

See `valve_severity.md` Section 9.

**Trajectory-specific issues:**

- **Small positive class**: Onset tasks have low event rates (e.g., LVEF Onset has ~5.5% positive rate). Ensure you have enough positive cases for meaningful AUROC. Below ~20 positives, confidence intervals will be very wide.
- **Time window**: MR Severity Onset requires 30-365 day follow-up window. Studies outside this window should be excluded.
- **Baseline filter**: Onset tasks restrict the baseline population (e.g., only EF >= 50% for LVEF Onset). Patients who already have the condition at baseline should not be included.

---

## 10. Expected Results

For reference, here are the **UHN internal validation** results (not UCMC — these are from the Toronto training site's held-out test set):

### Binary Onset Tasks (AUROC)

| Task | N | Positive Rate | EchoJEPA-G | EchoJEPA-L-K | EchoPrime | PanEcho |
|------|---|--------------|-----------|-------------|----------|---------|
| LVEF Onset | 3,516 | 5.5% | 0.794 | 0.683 | 0.782 | 0.781 |
| MR Severity Onset | 5,932 | 4.4% | 0.732 | 0.605 | 0.666 | 0.688 |
| TAPSE Onset | 2,809 | 26.6% | 0.689 | 0.638 | — | — |
| TR Severity Onset | 2,976 | 6.7% | 0.674 | 0.650 | 0.612 | 0.632 |

### 3-Class Trajectory Task (AUROC macro)

| Task | N | Class Distribution | EchoJEPA-G | EchoJEPA-L-K | EchoPrime | PanEcho |
|------|---|-------------------|-----------|-------------|----------|---------|
| LVEF Trajectory | 3,863 | 13.2% / 72.6% / 14.2% | 0.613 | 0.531 | 0.628 | 0.633 |

LVEF Trajectory class distribution: 13.2% declined, 72.6% stable, 14.2% improved.

Cross-institution performance may differ due to population differences, measurement conventions, follow-up patterns, and ultrasound equipment. Trajectory tasks are particularly sensitive to institutional differences in follow-up timing and measurement practice.
