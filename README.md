<h1 align="center"><b>EchoJEPA</b></h1>
<h3 align="center">A Latent Predictive Foundation Model for Echocardiography</h3>

<p align="center">
    <a href="https://arxiv.org/abs/2602.02603" target="_blank"><img src="https://img.shields.io/badge/arXiv-Paper-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv"></a>
    <a href="https://github.com/bowang-lab/EchoJEPA"><img src="https://img.shields.io/badge/GitHub-Code-4A90E2?style=for-the-badge&logo=github&logoColor=white" alt="GitHub"></a>
    <a href="https://echojepa.com/"><img src="https://img.shields.io/badge/Website-Online-00B89E?style=for-the-badge&logo=internet-explorer&logoColor=white" alt="Website"></a>
</p>


## Abstract

Foundation models for echocardiography often struggle to disentangle anatomical signal from the stochastic speckle and acquisition artifacts inherent to ultrasound. We present EchoJEPA, a foundation model trained on 18 million echocardiograms across 300K patients, representing the largest pretraining corpus for this modality to date. By leveraging a latent predictive objective, EchoJEPA learns robust anatomical representations that ignore speckle noise. We validate this using a novel multi-view probing framework with frozen backbones, where EchoJEPA outperforms state-of-the-art baselines by approximately 20% in left ventricular ejection fraction (LVEF) estimation and 17% in right ventricular systolic pressure (RVSP) estimation. The model also exhibits remarkable sample efficiency, reaching 79% view classification accuracy with only 1% of labeled data versus 42% for the best baseline trained on 100%. Crucially, EchoJEPA demonstrates superior generalization, degrading by only 2% under physics-informed acoustic perturbations compared to 17% for competitors. Most remarkably, its zero-shot performance on pediatric patients surpasses fully fine-tuned baselines, establishing latent prediction as a superior paradigm for robust, generalizable medical AI.

<p align="center">
	<img src="assets/echo_fig1a.png" width=100%>
</p>

EchoJEPA models trained on just 1% of labeled data outperform baselines trained on 100%. This efficiency implies that latent prediction yields dense representations capable of defining the view manifold with minimal supervision, as evidenced by the distinct anatomical clustering in the figure below.

<p align="center">
	<img src="assets/umap_views.png" width=100%>
</p>

EchoJEPA demonstrates anatomical localization, focusing on the mitral valve leaflets, ventricular walls, and annulus while ignoring sector background. Received attention clusters at Doppler jet edges while given attention localizes on valve structures generating flow. Across the cardiac cycle, focus shifts from valve tips during opening to chamber walls during relaxation, indicating it interprets the echocardiogram as a functional biological system.

<p align="center">
	<img src="assets/echo_attention.png" width=100%>
</p>

---


## Nature Medicine Paper

This repository supports two companion papers. The **ICML preprint** establishes the method (JEPA training, multi-view probing, robustness, sample efficiency, pediatric transfer). The **Nature Medicine paper** (active) demonstrates that EchoJEPA's frozen representations encode clinical information far beyond standard echocardiographic measurements.

### Objective

Clinical echocardiography reduces a rich spatiotemporal recording to a handful of standardized measurements, primarily ejection fraction. The Nature Medicine paper shows that self-supervised representations from EchoJEPA capture substantially more, enabling capabilities not previously demonstrated from frozen echocardiographic representations:

1. **Rare disease detection** — HCM, amyloidosis, endocarditis, takotsubo, constrictive pericarditis
2. **Clinical outcome prediction** — mortality, readmission, discharge disposition
3. **Biomarker inference** — creatinine, troponin T, NT-proBNP, lactate from echo alone
4. **Fairness analysis** — equitable performance across demographic subgroups
5. **Latent forward prediction** — forecasting future cardiac state from past echoes
6. **SAE interpretability** — sparse autoencoder decomposition of learned features

All downstream tasks use **frozen linear probes** on mean-pooled study-level embeddings (no fine-tuning), testing representation quality rather than task-specific adaptation.

### Datasets

**UHN Echocardiography Database** (pretraining):
- 18M echocardiograms across ~300K patients (2002-2019)
- Two reporting systems: Syngo (390K studies, 2005-2019) and HeartLab (432K studies, 2002-2014)
- 26M structured measurements, 6.6M clinical observations
- Rare disease cohorts (deduped): HCM 12,291 / endocarditis 5,236 / amyloidosis 1,174 / constrictive 376 / takotsubo 186
- No mortality or outcome data (outcome prediction is MIMIC-only)

**MIMIC-IV-Echo** (evaluation, linked to MIMIC-IV clinical data):
- 7,243 echo studies from 4,579 patients, ~525K DICOM files
- Linked to labs, vitals, medications, diagnoses, discharge notes, and mortality

| Prediction Target | Coverage | Prevalence |
|-------------------|----------|------------|
| 30-day mortality | 7,243 studies | 5.5% |
| 90-day mortality | 7,243 studies | 8.9% |
| 1-year mortality | 7,243 studies | 16.8% |
| 30-day readmission | 3,492 inpatient | 20.9% |
| 1-year readmission | 3,492 inpatient | 49.3% |
| EF from discharge notes | 2,743 studies | HFrEF 18.2%, HFmrEF 13.1%, HFpEF 68.7% |
| Creatinine (±24h) | 3,883 studies (53.6%) | continuous |
| Troponin T (±24h) | 1,686 studies (23.3%) | 48% undetectable |
| NT-proBNP (±24h) | 867 studies (11.8%) | continuous |
| Lactate (±24h) | 1,226 studies (16.9%) | continuous |

MIMIC rare disease cohorts: STEMI 237 / HCM 196 / tamponade 134 / DCM 97 / endocarditis 84 / amyloidosis 71 / takotsubo 43 patients.

Repeat echoes: 1,408 patients (30.7%) with 2+ studies; 1,504 pairs within 30-365 day windows (used for forward prediction).

### Models

| Model | Backbone | Pretraining Data | Method |
|-------|----------|-----------------|--------|
| EchoJEPA-G | ViT-G (1B) | 18M UHN echos | JEPA |
| EchoJEPA-L | ViT-L (300M) | MIMIC-IV-Echo | JEPA |
| EchoMAE-L | ViT-L (300M) | MIMIC-IV-Echo | MAE (controlled) |
| EchoJEPA-L-K | ViT-L (300M) | Kinetics → MIMIC | JEPA (domain transfer) |
| Echo-Vision-FM | — | Published | Various |
| PanEcho | — | Published | Various |
| EchoPrime | — | Published | Various |
| EchoFM | — | Published | Various |
| Random Init | ViT-L (300M) | None | Baseline |

### Where things live

| What | Location |
|------|----------|
| Manuscript (LaTeX) | `uhn_echo/nature_medicine/sn-article.tex` |
| UHN database (sqlite3) | `uhn_echo/nature_medicine/data_exploration/echo.db` |
| MIMIC analysis | `uhn_echo/nature_medicine/data_exploration/mimic/` |
| Database docs & query guides | `uhn_echo/nature_medicine/data_exploration/docs/` |
| Precomputed embeddings | `embeddings/` |
| Probe predictions | `predictions/` |
| Nature Medicine reference docs | `claude/data/nature-medicine-manuscript.md` |

---


## Setup

```bash
conda create -n vjepa2-312 python=3.12
conda activate vjepa2-312
pip install .  # or `pip install -e .` for development mode
```

---


## Quickstart: Working with Embeddings

If you've been given precomputed embedding files (`.npz`), you can start training probes immediately — no GPU or video data needed.

### 1. Inspect an embedding file

```python
import numpy as np

data = np.load("embeddings/views/echojepa_g_embeddings.npz", allow_pickle=True)
print(f"Embeddings: {data['embeddings'].shape}")  # e.g. [27216, 1408]
print(f"Labels:     {data['labels'].shape}")       # [27216] — int for classification, float for regression
print(f"Paths:      {data['paths'].shape}")        # [27216] — video file paths or sample IDs
```

### 2. Train a probe (k-fold cross-validation)

The script auto-detects classification vs regression from the label type.

```bash
# Classification (e.g. echo view classification — integer labels)
python -m evals.train_probe \
    --data embeddings/views/echojepa_g_embeddings.npz \
    --cv 5 --output_dir results/probes/views/echojepa_g

# Regression (e.g. LVEF estimation — float labels)
python -m evals.train_probe \
    --data embeddings/lvef/echojepa_g_embeddings.npz \
    --task regression --cv 5 \
    --output_dir results/probes/lvef/echojepa_g
```

### 3. Check results

```bash
cat results/probes/views/echojepa_g/echojepa_g_embeddings/metrics.json
# → accuracy, balanced_accuracy, F1, AUC-ROC, confusion matrix, ...

head results/probes/views/echojepa_g/echojepa_g_embeddings/predictions.csv
# → video_path, true_label, predicted_class, prediction_confidence, fold
```

### 4. Compare multiple models

If you have embeddings from different models, compare them side by side:

```bash
python -m evals.train_probe \
    --data embeddings/views/echojepa_g_embeddings.npz \
           embeddings/views/echoprime_embeddings.npz \
           embeddings/views/panecho_embeddings.npz \
    --model_names EchoJEPA-G EchoPrime PanEcho \
    --cv 5 --output_dir results/probes/views/comparison
```

This prints a comparison table and saves `comparison.json`.

### 5. Tune hyperparameters

```bash
# Custom regularization grid for classification
python -m evals.train_probe \
    --data embeddings/views/echojepa_g_embeddings.npz \
    --C 0.001 0.01 0.1 1.0 10.0 100.0 \
    --cv 5 --output_dir results/probes/views/tuned

# Regression with Lasso instead of Ridge
python -m evals.train_probe \
    --data embeddings/lvef/echojepa_g_embeddings.npz \
    --task regression --regression_model lasso \
    --alpha 0.0001 0.001 0.01 0.1 1.0 \
    --cv 5 --output_dir results/probes/lvef/lasso

# Regression with denormalization (if labels are z-scored)
python -m evals.train_probe \
    --data embeddings/lvef/echojepa_g_embeddings.npz \
    --task regression --labels_are_zscored \
    --target_mean 57.06 --target_std 11.33 \
    --cv 5 --output_dir results/probes/lvef/denormed
```

### 6. Train/val mode (separate files)

If you have separate train and validation embeddings:

```bash
python -m evals.train_probe \
    --train embeddings/views/echojepa_g_embeddings.npz \
    --val   embeddings/test/echojepa_g_embeddings.npz \
    --save_model --output_dir results/probes/views/echojepa_g
```

Use `--save_model` to save the trained sklearn model as `probe.joblib` for later inference. Run `python -m evals.train_probe --help` for all options.

For the full pipeline (extracting your own embeddings from videos, training GPU-based probes, pretraining), continue reading below.

---


## Working with Video Files (MP4)

EchoJEPA expects echocardiogram videos as MP4 files. This section covers how to prepare your videos and organize them into dataset CSVs for training, evaluation, and embedding extraction.

### Video requirements

| Property | Requirement |
|----------|-------------|
| Format | `.mp4` (H.264 codec recommended) |
| Resolution | Any — the pipeline resizes to `crop_size` (typically 224px or 336px) |
| Frame rate | Any — the pipeline samples at the configured `fps` (typically 8 fps) |
| Duration | At least `frames_per_clip / fps` seconds (e.g. 16 frames / 8 fps = 2s minimum) |
| Color | Grayscale or RGB (both work; grayscale is converted to 3-channel internally) |

Videos are decoded on-the-fly using [decord](https://github.com/dmlc/decord). No preprocessing is needed — just point to the raw MP4 files.

### Converting DICOM to MP4

If your echocardiograms are in DICOM format, convert them to MP4 first:

```python
import pydicom
import cv2
import numpy as np

ds = pydicom.dcmread("echo.dcm")
frames = ds.pixel_array  # [T, H, W] or [T, H, W, 3]

# Write to MP4
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = float(getattr(ds, "RecommendedDisplayFrameRate", 30))
h, w = frames.shape[1], frames.shape[2]
writer = cv2.VideoWriter("echo.mp4", fourcc, fps, (w, h), isColor=len(frames.shape) == 4)

for frame in frames:
    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    writer.write(frame)

writer.release()
```

### Preparing a dataset CSV

All dataset CSVs are **space-delimited text files** with no header. Each line is a video path followed by its label. The path can be absolute or relative to the working directory.

**Classification** — integer labels (0-indexed):
```
data/echo_views_22k/19068955.mp4 5
data/echo_views_22k/19076133.mp4 7
data/echo_views_22k/19083831.mp4 2
data/echo_views_22k/19086809.mp4 2
data/echo_views_22k/19089161.mp4 5
```

**Regression** — Z-score normalized float labels (see [below](#z-score-normalization-for-regression)):
```
data/echo_a4c_lvef/2230801.mp4 -3.4486802913030026
data/echo_a4c_lvef/3260170.mp4 -0.16931876118450664
data/echo_a4c_lvef/2758271.mp4 0.7278549632852218
```

**Pretraining** — dummy label (self-supervised, no labels needed):
```
mimic-echo-224px/files/p10/p10002221/s94106955/94106955_0001.mp4 0
mimic-echo-224px/files/p10/p10002221/s94106955/94106955_0006.mp4 0
```

**Multi-view** — multiple video paths per line, label last:
```
studies/patient1/a4c.mp4 studies/patient1/psax.mp4 0.82
studies/patient2/a4c.mp4 studies/patient2/psax.mp4 -1.05
```

### Z-score normalization for regression

For regression tasks, labels must be Z-score normalized. Fit the scaler on the training set only to prevent data leakage:

```python
from sklearn.preprocessing import StandardScaler
import pickle

scaler = StandardScaler()
train_values = train_df['Value'].values.reshape(-1, 1)
scaler.fit(train_values)

print(f"Mean: {scaler.mean_[0]:.4f}, Std: {scaler.scale_[0]:.4f}")

# Transform all splits
train_df['norm_value'] = scaler.transform(train_df['Value'].values.reshape(-1, 1))
val_df['norm_value']   = scaler.transform(val_df['Value'].values.reshape(-1, 1))
test_df['norm_value']  = scaler.transform(test_df['Value'].values.reshape(-1, 1))

# Save the scaler — you need this to convert predictions back to real units
with open('lvef_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
```

Save the mean and standard deviation (or the scaler pickle) — you will need these for inference to convert predictions back to real units.

### Verifying your videos load correctly

Quick sanity check that your MP4 files are readable by the pipeline:

```python
import decord
decord.bridge.set_bridge("torch")

vr = decord.VideoReader("path/to/your/video.mp4", num_threads=1)
print(f"Frames: {len(vr)}, Resolution: {vr[0].shape}")  # e.g. Frames: 120, Resolution: (600, 800, 3)
```

If this works, the video is ready. If it fails, re-encode with ffmpeg:

```bash
ffmpeg -i input.avi -c:v libx264 -pix_fmt yuv420p -r 30 output.mp4
```

---


## Pretraining

Pretraining and cooldown phases use the same command with different configs. These examples launch training of a ViT-L model on [MIMIC-IV-ECHO](https://physionet.org/content/mimic-iv-echo/0.1/), a dataset of 525K echocardiograms accessible through PhysioNet.

### Local

```bash
python -m app.main --fname configs/train/vitl16/pretrain-mimic-224px-16f.yaml \
  --devices cuda:0
```

### Distributed (SLURM)

```bash
python -m app.main_distributed \
  --fname configs/train/vitl16/pretrain-mimic-224px-16f.yaml
  --time 6000
  --account my_account --qos=my_qos
```

### Pretrained checkpoints

Since we are doing self-supervised pre-training, all the video labels are set to zero. You can begin pretraining from any of the pre-trained V-JEPA models below:

<table>
  <tr>
    <th colspan="1">Model</th>
    <th colspan="1">#Parameters</th>
    <th colspan="1">Resolution</th>
    <th colspan="1">Download Link</th>
    <th colspan="1">Pretraining Config</th>
  </tr>
  <tr>
    <td>ViT-L/16</td>
    <td>300M</td>
    <td>256</td>
    <td><a href="https://dl.fbaipublicfiles.com/vjepa2/vitl.pt">checkpoint</a></td>
    <td><a href="configs/train/vitl16">configs</a></td>
  </tr>
  <tr>
    <td>ViT-H/16</td>
    <td>600M</td>
    <td>256</td>
    <td><a href="https://dl.fbaipublicfiles.com/vjepa2/vith.pt">checkpoint</a></td>
    <td><a href="configs/train/vith16/">configs</a></td>
  </tr>
  <tr>
    <td>ViT-g/16</td>
    <td>1B</td>
    <td>256</td>
    <td><a href="https://dl.fbaipublicfiles.com/vjepa2/vitg.pt">checkpoint</a></td>
    <td><a href="configs/train/vitg16">configs</a></td>
  </tr>
  <tr>
    <td>ViT-g/16<sub>384</sub></td>
    <td>1B</td>
    <td>384</td>
    <td><a href="https://dl.fbaipublicfiles.com/vjepa2/vitg-384.pt">checkpoint</a></td>
    <td><a href="configs/train/vitg16">configs</a></td>
  </tr>
</table>

### Pretraining configuration

We keep the configuration mostly the same as V-JEPA 2, but adjust sampling and augmentation for echocardiography:

```yaml
app: vjepa
nodes: 1
tasks_per_node: 8
cpus_per_task: 16
mem_per_gpu: 220G
folder: checkpoints/pretrain/mimic/vjepa2_vitl_224px_16f
data:
  dataset_type: VideoDataset
  datasets:
  - /path/to/your/pretrain_videos.csv       # space-delimited: <video_path> 0
  datasets_weights:
  - 1.0
  batch_size: 128
  crop_size: 224              # flexible thanks to RoPE scaling, but 224 matches other models
  patch_size: 16
  dataset_fpcs:
  - 16                        # frames per clip — 16 works well in practice
  fps: 8                      # lower = greater temporal coverage, higher = greater fidelity
  tubelet_size: 2
  num_workers: 8
  persistent_workers: true
  pin_mem: true
data_aug:
  auto_augment: false
  motion_shift: false
  random_resize_aspect_ratio: # narrowed from [0.75, 1.35] for echo
  - 0.9
  - 1.1
  random_resize_scale:        # narrowed from [0.3, 1.0] for echo
  - 0.5
  - 1.0
```

If you are not training from scratch, set `optimization.checkpoint` to your downloaded checkpoint path. Make sure to scale your learning rates!

---


## Probe Evaluation

Probe evaluation trains a lightweight head on top of frozen EchoJEPA features. We support three probe types:

| Probe Type | Architecture | Pooling | Use Case |
|-----------|-------------|---------|----------|
| `attentive` (default) | Cross-attention + self-attention blocks | Learned query token | Best downstream performance (ICML paper) |
| `linear` | Single linear layer | Mean pooling | Tests linear separability of representations (Nature Medicine paper) |
| `mlp` | Two-layer MLP with GELU | Mean pooling | Middle ground between linear and attentive |

We provide training scripts for training your own probes, and checkpoints to run inference directly.

<p align="center">
	<img src="assets/echo_fig2.png" width=100%>
</p>

### Training config

Here is an example config for regression. The settings are the same for classification, except you change `task_type: regression` to `task_type: classification` and replace `num_targets: 1` with `num_classes: N`.

```yaml
app: vjepa
cpus_per_task: 32
folder: /path/to/experiments/lvef_regression
mem_per_gpu: 80G
nodes: 1
tasks_per_node: 8
num_workers: 8

eval_name: video_classification_frozen  # for multi-view: video_classification_frozen_multi
resume_checkpoint: true
tag: lvef-echonet-dynamic-112px

experiment:
  classifier:
    task_type: regression         # "classification" or "regression"
    num_targets: 1                # for classification, use num_classes: N instead
    probe_type: attentive         # "attentive" (default), "linear", or "mlp"
    num_heads: 16                 # attentive probe only (ignored for linear/mlp)
    num_probe_blocks: 4           # attentive probe only (ignored for linear/mlp)

    # --- Multi-view only ---
    # num_views: 2               # number of views (e.g. A4C + PSAX-AV)
    # clips_per_view: 2          # clips sampled from each view
    # use_slot_embeddings: true
    # use_factorized: true

  data:
    dataset_type: VideoDataset
    dataset_train: data/csv/echonet_dynamic_train.csv
    dataset_val:   data/csv/echonet_dynamic_val.csv

    resolution: 112               # flexible thanks to RoPE — can differ from pretraining
    frames_per_clip: 16
    frame_step: 2
    num_segments: 2
    num_views_per_segment: 1

    target_mean: 57.06           # regression only — for denormalization
    target_std: 11.33            # regression only — for denormalization

    # --- Multi-view only ---
    # num_clips_per_video: 2
    # miss_augment_prob: 0.10
    # min_present: 1
```

For multi-view, set `model_kwargs.module_name` to `evals.video_classification_frozen_multi.modelcustom.vit_encoder_multiclip`. See `configs/eval/vitg-384` for more examples.

### Using linear probes

To use a linear probe, set `probe_type: linear`. Linear probes use mean pooling instead of learned cross-attention, making them a stricter test of representation quality. The only trainable parameters are a LayerNorm and a single linear layer.

```yaml
experiment:
  classifier:
    task_type: classification     # or regression
    probe_type: linear            # switch to linear probe
    use_layernorm: true
    dropout: 0.0
    num_classes: 13               # classification only
    # num_targets: 1              # regression only
```

Linear probes typically use higher learning rates and less regularization than attentive probes:

```yaml
  optimization:
    batch_size: 1
    num_epochs: 10
    use_bfloat16: true
    multihead_kwargs:
    - lr: 0.001
      start_lr: 0.001
      final_lr: 0.0
      weight_decay: 0.0
      final_weight_decay: 0.0
      warmup: 0.0
    - lr: 0.0005
      start_lr: 0.0005
      final_lr: 0.0
      weight_decay: 0.001
      final_weight_decay: 0.001
      warmup: 0.0
    - lr: 0.0001
      start_lr: 0.0001
      final_lr: 0.0
      weight_decay: 0.01
      final_weight_decay: 0.01
      warmup: 0.0
```

See `configs/eval/vitg-384/view/echojepa_view_linear.yaml` for a complete example. Linear probes are supported in both single-view and multi-view evaluation.

### Running probe training

Evaluations can be run locally or distributed via SLURM. Use provided configs under `configs/eval/`. These configs support training multiple probes in parallel with different hyperparameters via `multihead_kwargs`. Change filepaths (e.g. `folder`, `checkpoint`, `dataset_train`, `dataset_val`) to match your local filesystem.

**Local:**

```bash
python -m evals.main --fname configs/eval/vitg-384/lvef/echojepa_lvef.yaml \
  --devices cuda:0 cuda:1
```

**Distributed (SLURM):**

```bash
python -m evals.main_distributed \
  --fname configs/eval/vitg-384/lvef/echojepa_lvef.yaml  \
  --time 8600 \
  --account my_account --qos=my_qos
```

Additional configs can be found for other tasks under `configs/eval/vitg-384/view` and `configs/eval/vitg-384/rvsp`. Each directory has ready-made configs for PanEcho, EchoPrime, VideoMAE, and EchoJEPA. There are also configs for EchoJEPA-L under `configs/eval/vitl`.

---


## Probe Inference

To run inference with a trained probe, use configs under `configs/inference/vitg-384` or `configs/inference/vitl`.

```bash
python -m evals.main --fname configs/inference/vitl/lvef.yaml --devices cuda:0 cuda:1 cuda:2
```

The key settings for inference:

```yaml
# --- Critical inference settings ---
val_only: true                     # do NOT train — inference only
predictions_save_path: /path/to/predictions.csv
probe_checkpoint: /path/to/probe.pt

experiment:
  classifier:                      # must match your trained probe config exactly
    task_type: regression
    num_targets: 1
    probe_type: attentive
    num_heads: 16
    num_probe_blocks: 4

  data:
    dataset_type: VideoDataset
    dataset_val:   /path/to/test.csv    # point to your test CSV
    dataset_train: /path/to/test.csv    # can be the same as val for inference

    resolution: 336                # must match training config
    frames_per_clip: 16
    frame_step: 2
    num_segments: 2
    num_views_per_segment: 1

    target_mean: 57.06             # regression only — for denormalization
    target_std: 11.33
```

Zero out the optimization parameters and set `num_epochs` to 1. Batch size can be increased since gradients are off:

```yaml
  optimization:
    batch_size: 4
    multihead_kwargs:
    - final_lr: 0.0
      final_weight_decay: 0.0
      lr: 0.0
      start_lr: 0.0
      warmup: 0.0
      weight_decay: 0.0
    num_epochs: 1
    use_bfloat16: true
    use_pos_embed: false
```

See configs under `configs/inference/vitg-384` or `configs/inference/vitl` for more examples.

---


## Embedding Extraction

Extract frozen embeddings from any model and save as `.npz` files using `evals/extract_embeddings.py`. Supports multi-GPU extraction, both classification and regression CSVs, and any backbone with a config YAML.

```bash
# Single GPU
python -m evals.extract_embeddings \
    --config configs/inference/vitg-384/view/echojepa_224px.yaml \
    --data data/csv/uhn_views_22k_test_224px.csv \
    --output embeddings/views/echojepa_g_embeddings.npz \
    --devices cuda:0

# Multi-GPU (4x faster)
python -m evals.extract_embeddings \
    --config configs/inference/vitg-384/view/echojepa_224px.yaml \
    --data data/csv/uhn_views_22k_test_224px.csv \
    --output embeddings/views/echojepa_g_embeddings.npz \
    --devices cuda:0 cuda:1 cuda:2 cuda:3

# Regression (LVEF)
python -m evals.extract_embeddings \
    --config configs/inference/vitg-384/lvef/echojepa_336px.yaml \
    --data data/csv/a4c_b_lvef_test_224px.csv \
    --output embeddings/lvef/echojepa_g_embeddings.npz \
    --devices cuda:0 cuda:1 cuda:2 cuda:3
```

Output `.npz` files contain `embeddings` (`[N, D]`), `labels` (`[N]`), and `paths` (`[N]`). See `embeddings/README.md` for full format details and available options.

To train sklearn probes on extracted embeddings, see the [Quickstart](#quickstart-working-with-embeddings) section above.

---


## Code Structure

```
.
├── app/                                   # training loops
│   ├── vjepa/                             #   video JEPA pre-training
│   ├── main_distributed.py                #   entrypoint for launch app on slurm cluster
│   └── main.py                            #   entrypoint for launch app locally on your machine
├── configs/                               # YAML experiment configs (see README.md in each subdir)
│   ├── train/                             #   pretraining and cooldown, by model size (vitl16, vith16, vitg16)
│   ├── eval/                              #   frozen probe training, by model and task (lvef, rvsp, view, ...)
│   └── inference/                         #   inference-only configs (val_only: true)
├── evals/                                 # evaluation loops training an attentive probe with frozen backbone...
│   ├── video_classification_frozen/       #   single-view echocardiogram probes
│   ├── video_classification_frozen_multi/ #   multi-view echocardiogram probes
│   ├── extract_embeddings.py              #   embedding extraction script
│   ├── train_probe.py                     #   sklearn probe training on NPZ embeddings
│   ├── main_distributed.py                #   entrypoint for distributed evaluations
│   └── main.py                            #   entrypoint for locally-run evaluations
├── src/                                   # the package
│   ├── datasets/                          #   datasets, data loaders, ...
│   ├── models/                            #   model definitions (ViT, attentive pooler, linear pooler, ...)
│   ├── masks/                             #   mask collators, masking utilities, ...
│   └── utils/                             #   shared utilities
├── data/                                  # data assets (splits, labels, scalers, notebooks)
│   ├── csv/                               #   JEPA-format splits (referenced by eval configs)
│   ├── scalers/                           #   sklearn scalers for Z-score normalization
│   ├── labels/                            #   raw label CSVs
│   ├── notebooks/                         #   data exploration and split generation
│   └── scripts/                           #   processing and augmentation scripts
├── classifier/                            # ConvNeXt/Swin echo classifiers
│   ├── train_convnext.py                  #   distributed DDP training
│   ├── inference_18m.py                   #   unified inference (view/color/quality/zoom)
│   ├── data_prep/                         #   data preparation pipeline
│   ├── mappings/                          #   canonical label-to-int JSON maps
│   └── utils/                             #   checkpoint fixing, resizing, format conversion
├── checkpoints/                           #   model weights (pretrain, anneal, cooldown, eval probes)
├── indices/                               #   S3 URI manifests for 18M dataset
├── embeddings/                            #   precomputed embeddings (NPZ by task and model)
├── predictions/                           #   probe and classifier prediction CSVs
├── results/                               #   data efficiency experiment runs
├── scripts/                               #   SBATCH scripts, Python utilities, demos
├── notebooks/                             #   analysis and demo Jupyter notebooks
├── figures/                               #   publication-quality plots (UMAP, etc.)
├── logs/                                  #   training/evaluation logs and SLURM outputs
├── tests/                                 #   unit tests for some modules in `src`
├── uhn_echo/                              #   UHN research data and Nature Medicine analysis
└── claude/                                #   architecture docs and project reference files
```

---


## License

The majority of V-JEPA 2 is licensed under MIT, however portions of the project are available under separate license terms:

[src/datasets/utils/video/randaugment.py](src/datasets/utils/video/randaugment.py)<br>
[src/datasets/utils/video/randerase.py](src/datasets/utils/video/randerase.py)<br>
[src/datasets/utils/worker_init_fn.py](src/datasets/utils/worker_init_fn.py)<br>

are licensed under the Apache 2.0 license.

---


## Citation

**Alif Munim, Adibvafa Fallahpour, Teodora Szasz, Ahmadreza Attarpour**, River Jiang, Brana Sooriyakanthan, Maala Sooriyakanthan, Heather Whitney, Jeremy Slivnick, Barry Rubin, Wendy Tsang, Bo Wang

If you find this repository useful in your research, please consider giving a star :star: and a citation
```bibtex
@misc{munim2026echojepalatentpredictivefoundation,
      title={EchoJEPA: A Latent Predictive Foundation Model for Echocardiography},
      author={Alif Munim and Adibvafa Fallahpour and Teodora Szasz and Ahmadreza Attarpour and River Jiang and Brana Sooriyakanthan and Maala Sooriyakanthan and Heather Whitney and Jeremy Slivnick and Barry Rubin and Wendy Tsang and Bo Wang},
      year={2026},
      eprint={2602.02603},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2602.02603},
}
```
