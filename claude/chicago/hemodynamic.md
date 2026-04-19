# UCMC Hemodynamic / Cross-Modal External Validation — Inference Guide

Instructions for running frozen hemodynamic regression probes on UCMC echocardiography data. These tasks predict aortic valve hemodynamic quantities from B-mode video alone (cross-modal inference — no Doppler input). All probes were trained at UHN using frozen depth=1 attentive cross-attention heads. No model fine-tuning occurs — inference only.

**Contact:** Alif Munim (alif.munim@uhn.ca)

---

## 1. Overview

You will run **4 frozen encoder models** on **2 hemodynamic regression tasks**:

| Model | Architecture | Params | Encoder File | Embed Dim |
|-------|-------------|--------|-------------|-----------|
| **EchoJEPA-G** | ViT-Giant | 1,012M | `pt-280-an81.pt` (15.3 GB) | 1408 |
| **EchoJEPA-L-K** | ViT-Large | 304M | `vitl-kinetics-pt220-an55.pt` (4.8 GB) | 1024 |
| **EchoPrime** | MViT-v2-S | ~35M | `echo_prime_encoder.pt` (~200 MB) | 512 |
| **PanEcho** | ConvNeXt-T + Transformer | ~30M | Auto-downloads from hub | 768 |

| Task | Type | Unit | Description |
|------|------|------|-------------|
| **AOV Mean Gradient** | Regression | mmHg | Aortic valve mean pressure gradient |
| **AOV Vmax** | Regression | m/s | Aortic valve peak velocity |

That's **8 inference runs** total (4 models x 2 tasks).

**Why these tasks matter:** The models predict hemodynamic quantities (normally measured by Doppler) from B-mode structural video alone. This is a cross-modal inference test — the model must have learned enough about the relationship between cardiac structure and blood flow dynamics to infer one from the other.

---

## 2. Shared Google Drive

Encoder checkpoints: same as the valve severity and LV function guides.

Probes:

```
gdrive:echo_foundation/nature_medicine/chicago/probes/hemodynamic/
├── echojepa-g_aov_mean_grad/best.pt              # 3.0 GB
├── echojepa-g_aov_vmax/best.pt                   # 3.0 GB
├── echojepa-l-k_aov_mean_grad/best.pt            # 1.6 GB
├── echojepa-l-k_aov_vmax/best.pt                 # 1.6 GB
├── echoprime_aov_mean_grad/best.pt               # 398 MB
├── echoprime_aov_vmax/best.pt                    # 398 MB
├── panecho_aov_mean_grad/best.pt                 # 893 MB
└── panecho_aov_vmax/best.pt                      # 893 MB
```

---

## 3. Environment Setup

Same as the valve severity guide — see `valve_severity.md` Sections 3.1–3.4.

---

## 4. Organize Checkpoints

```bash
mkdir -p checkpoints/encoders
mkdir -p checkpoints/probes/{aov_mean_grad,aov_vmax}/{echojepa-g,echojepa-l-k,echoprime,panecho}

# Encoders (skip if already downloaded for valve severity or LV function)
cp <gdrive>/chicago/checkpoints/echojepa_g_uhn_pt280_an81/pt-280-an81.pt checkpoints/encoders/
cp <gdrive>/checkpoints/vitl-kinetics-pt220-an55.pt checkpoints/encoders/

# Probes (download all 8 from GDrive chicago/probes/hemodynamic/)
cp <gdrive>/probes/hemodynamic/echojepa-g_aov_mean_grad/best.pt    checkpoints/probes/aov_mean_grad/echojepa-g/
cp <gdrive>/probes/hemodynamic/echojepa-g_aov_vmax/best.pt         checkpoints/probes/aov_vmax/echojepa-g/
cp <gdrive>/probes/hemodynamic/echojepa-l-k_aov_mean_grad/best.pt  checkpoints/probes/aov_mean_grad/echojepa-l-k/
cp <gdrive>/probes/hemodynamic/echojepa-l-k_aov_vmax/best.pt       checkpoints/probes/aov_vmax/echojepa-l-k/
cp <gdrive>/probes/hemodynamic/echoprime_aov_mean_grad/best.pt     checkpoints/probes/aov_mean_grad/echoprime/
cp <gdrive>/probes/hemodynamic/echoprime_aov_vmax/best.pt          checkpoints/probes/aov_vmax/echoprime/
cp <gdrive>/probes/hemodynamic/panecho_aov_mean_grad/best.pt       checkpoints/probes/aov_mean_grad/panecho/
cp <gdrive>/probes/hemodynamic/panecho_aov_vmax/best.pt            checkpoints/probes/aov_vmax/panecho/
```

---

## 5. Prepare Your Data

### 5.1 Video format

Same as `valve_severity.md` Section 5.1. MP4, 224x224, 8 fps preferred, B-mode only, sector-masked.

### 5.2 CSV format

Space-delimited, no header. Two columns: video path and float label.

```
/data/ucmc/hemo/study001_clip01.mp4 12.3
/data/ucmc/hemo/study001_clip02.mp4 12.3
```

- Labels are raw float values: AOV mean gradient in mmHg, AOV Vmax in m/s
- Multiple clips per study is fine (prediction averaging pools them)

### 5.3 View filtering

These probes were trained on **all B-mode views** per study (not view-filtered). Include all available B-mode clips.

**Most informative views:** PLAX and A3C/A5C provide the best structural view of the aortic valve. Apical views (A4C, A2C) also contribute.

**Exclude:** Color Doppler, spectral Doppler, and tissue Doppler clips.

### 5.4 Create CSVs

```
data/csv/ucmc_aov_mean_grad_test.csv
data/csv/ucmc_aov_vmax_test.csv
```

### 5.5 Study-level prediction averaging

Same as `valve_severity.md` Section 5.6.

---

## 6. Inference Configs

### 6.1 EchoJEPA-G configs

**`configs/inference/chicago/echojepa_g_aov_mean_grad.yaml`:**
```yaml
app: vjepa
eval_name: video_classification_frozen
val_only: true
resume_checkpoint: true
tag: ucmc-echojepa-g-aov-mean-grad
probe_checkpoint: checkpoints/probes/aov_mean_grad/echojepa-g/best.pt

experiment:
  classifier:
    num_heads: 16
    num_probe_blocks: 1

  data:
    dataset_type: VideoDataset
    task_type: regression
    dataset_train: data/csv/ucmc_aov_mean_grad_test.csv
    dataset_val: data/csv/ucmc_aov_mean_grad_test.csv
    num_classes: 1
    resolution: 224
    frames_per_clip: 16
    frame_step: 2
    num_segments: 2
    num_views_per_segment: 1
    study_sampling: true

  optimization:
    batch_size: 4
    num_epochs: 1
    use_bfloat16: true
    multihead_kwargs:
    - {lr: 0.0, start_lr: 0.0, final_lr: 0.0, warmup: 0.0, weight_decay: 0.0, final_weight_decay: 0.0}

model_kwargs:
  checkpoint: checkpoints/encoders/pt-280-an81.pt
  module_name: evals.video_classification_frozen.modelcustom.vit_encoder_multiclip
  pretrain_kwargs:
    encoder:
      checkpoint_key: target_encoder
      model_name: vit_giant_xformers
      patch_size: 16
      tubelet_size: 2
      uniform_power: true
      use_rope: true
  wrapper_kwargs:
    max_frames: 128
    use_pos_embed: false
```

For AOV Vmax: change `tag`, `probe_checkpoint`, and data paths.

### 6.2 EchoJEPA-L-K, EchoPrime, PanEcho

Same pattern as `lv_function.md` Sections 6.2–6.4 — change `model_name`/`module_name`/`wrapper_kwargs` per model, plus `tag`, `probe_checkpoint`, and data paths per task.

---

## 7. Running Inference

Same as `lv_function.md` Section 7. Adjust the task list:

```bash
MODELS=("echojepa_g" "echojepa_lk" "echoprime" "panecho")
TASKS=("aov_mean_grad" "aov_vmax")
```

---

## 8. Output

### 8.1 Expected output files

```
results/ucmc/
├── ucmc-echojepa-g-aov-mean-grad/study_predictions.csv
├── ucmc-echojepa-g-aov-vmax/study_predictions.csv
├── ucmc-echojepa-lk-aov-mean-grad/study_predictions.csv
├── ucmc-echojepa-lk-aov-vmax/study_predictions.csv
├── ucmc-echoprime-aov-mean-grad/study_predictions.csv
├── ucmc-echoprime-aov-vmax/study_predictions.csv
├── ucmc-panecho-aov-mean-grad/study_predictions.csv
└── ucmc-panecho-aov-vmax/study_predictions.csv
```

### 8.2 Metrics

MAE, R², Pearson r (printed to stdout and saved in log).

---

## 9. Troubleshooting

See `valve_severity.md` Section 9.

---

## 10. Expected Results

For reference, here are the **UHN internal validation** results (not UCMC — these are from the Toronto training site's held-out test set):

| Task | Metric | EchoJEPA-G | EchoJEPA-L-K | EchoPrime | PanEcho |
|------|--------|-----------|-------------|----------|---------|
| AOV Mean Grad | R² | 0.579 | 0.328 | 0.462 | 0.378 |
| AOV Mean Grad | MAE (mmHg) | 5.50 | 6.72 | 6.02 | 6.46 |
| AOV Vmax | R² | 0.679 | 0.492 | 0.574 | 0.479 |
| AOV Vmax | MAE (m/s) | 0.33 | 0.41 | 0.37 | 0.41 |

Cross-institution performance may differ due to population differences, measurement conventions, and ultrasound equipment.
