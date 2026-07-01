# Faraz — EchoNet-Dynamic & EchoNet-Pediatric Benchmarks (LVEF)

**Affiliation:** Meta Superintelligence Labs
**Contact:** Alif Munim (alif.munim@uhn.ca)
**Scope:** Public-benchmark LVEF regression on EchoNet-Dynamic and EchoNet-Pediatric for 3 models — EchoJEPA-L-K, EchoPrime, PanEcho. Two phases: (1) inference-only using the UHN-trained probes we already have on Google Drive, (2) train fresh Strategy E depth=1 attentive probes on each dataset's train split, then run inference on the test split. Frozen-encoder throughout — no encoder fine-tuning.

**Framing.** These are companion-preprint / Extended Data numbers, not the primary Nature Medicine claims. The manuscript's headline LVEF numbers are on UHN + UCMC + MIMIC. EchoNet-Dynamic and EchoNet-Pediatric are the public benchmarks that let external readers reproduce and compare against prior art (EchoNet, PanEcho, EchoPrime, EchoCardioMAE, EchoFM).

---

## 1. Overview

You will produce **12 study-level prediction CSVs** total:

| Phase | Dataset | Models | Runs |
|---|---|---|---|
| 1. Inference only (UHN-trained probes) | EchoNet-Dynamic test | L-K, EchoPrime, PanEcho | 3 |
| 1. Inference only (UHN-trained probes) | EchoNet-Pediatric test | L-K, EchoPrime, PanEcho | 3 |
| 2. Train Strategy E d=1 probe, then infer | EchoNet-Dynamic (train → test) | L-K, EchoPrime, PanEcho | 3 |
| 2. Train Strategy E d=1 probe, then infer | EchoNet-Pediatric (train → test) | L-K, EchoPrime, PanEcho | 3 |

**Frozen encoders throughout** — Strategy E means only the depth=1 attentive cross-attention head is trained. No encoder fine-tuning at any point. Both phases use frozen probes at inference and prediction averaging at the study level.

| Model | Architecture | Params | Encoder file | Embed dim |
|---|---|---|---|---|
| **EchoJEPA-L-K** | ViT-Large + Kinetics init | 304M | `vitl-kinetics-pt220-an55.pt` (4.8 GB) | 1024 |
| **EchoPrime** | MViT-v2-S | ~35M | `echo_prime_encoder.pt` (bundled with the EchoPrime repo, ~200 MB) | 512 |
| **PanEcho** | ConvNeXt-T + 4L Transformer | ~30M | auto-downloads from torchvision hub | 768 |

**Datasets** (both are public):
- **EchoNet-Dynamic** — 10,030 apical-four-chamber videos, LVEF labels. Splits already prepared: train 7,465 / val 1,288 / test 1,277. Download: https://echonet.github.io/dynamic/
- **EchoNet-Pediatric** — 7,643 videos (A4C + PSAX subsets), LVEF labels. Splits already prepared: train 2,580 / val 336 / test 368. Download: https://echonet.github.io/pediatric/

**Bottom line for both datasets:** for phase 1 we want to see how well the UHN-trained LVEF probe transfers to a totally different institution and (in the pediatric case) a totally different population. For phase 2 we want the fair EchoNet-only number: a Strategy E d=1 probe trained on that dataset's train split, evaluated on its test split. Phase 2 is the number that goes into the paper's benchmark table.

---

## 2. Repo Setup

```bash
git clone git@github.com:AlifMunim/vjepa2.git EchoJEPA
cd EchoJEPA
git checkout master

# Python 3.12 conda env (name it whatever you like)
conda create -n echojepa python=3.12 -y
conda activate echojepa
pip install -e .
pip install -r requirements.txt  # if present; otherwise setup.py handles it
```

EchoPrime and PanEcho backbones require an extra one-time setup — see `uhn_echo/nature_medicine/context_files/roles/teodora.md` (renamed `chicago/valve_severity.md` in this repo) sections 3.3–3.4 for EchoPrime weight setup and PanEcho hub cache. Both scripts have run successfully many times inside this repo, so if you hit an obscure error, ping Alif rather than trying to debug from scratch.

---

## 3. Google Drive Layout (rclone)

**Configure rclone remote** named `gdrive` pointing at the shared drive (Alif can share the credentials). Then verify:

```bash
rclone lsf gdrive:echo_foundation/nature_medicine/
```

Expected output includes `chicago/` and `checkpoints/`. The specific paths you'll need:

**Encoder checkpoints** (2 files, ~5 GB total):
```
gdrive:echo_foundation/nature_medicine/checkpoints/
└── vitl-kinetics-pt220-an55.pt          # EchoJEPA-L-K encoder (4.8 GB)
```

EchoPrime and PanEcho encoders are NOT on GDrive — EchoPrime bundles its own weights inside its released repo (see setup guide), and PanEcho auto-downloads from torchvision.

**UHN-trained LVEF probes** (Phase 1 only; 3 files, ~2.9 GB total):
```
gdrive:echo_foundation/nature_medicine/chicago/probes/lv_function/
├── echojepa-l-k_lvef/best.pt            # 1.6 GB
├── echoprime_lvef/best.pt               # 398 MB
└── panecho_lvef/best.pt                 # 893 MB
```

(There is also an `echojepa-g_lvef/best.pt` for EchoJEPA-G — you can skip it; we're only running the 3 smaller models in this sweep.)

**Copy them locally:**

```bash
# Encoder
mkdir -p checkpoints/encoders
rclone copy gdrive:echo_foundation/nature_medicine/checkpoints/vitl-kinetics-pt220-an55.pt \
  checkpoints/encoders/ --progress

# UHN-trained probes for phase 1
mkdir -p checkpoints/probes/uhn_lvef/{echojepa-l-k,echoprime,panecho}
rclone copy gdrive:echo_foundation/nature_medicine/chicago/probes/lv_function/echojepa-l-k_lvef/best.pt \
  checkpoints/probes/uhn_lvef/echojepa-l-k/ --progress
rclone copy gdrive:echo_foundation/nature_medicine/chicago/probes/lv_function/echoprime_lvef/best.pt \
  checkpoints/probes/uhn_lvef/echoprime/ --progress
rclone copy gdrive:echo_foundation/nature_medicine/chicago/probes/lv_function/panecho_lvef/best.pt \
  checkpoints/probes/uhn_lvef/panecho/ --progress
```

---

## 4. Data CSVs

The repo already ships label CSVs for both datasets. Format is space-delimited, no header, two columns: `video_path label` (label is raw LVEF as a float; the pipeline Z-score normalizes internally when the config points to a raw CSV, or you can point to the pre-normalized CSV directly).

**EchoNet-Dynamic**:
- `data/csv/echonet_dynamic_train.csv` — 7,464 clips, Z-scored labels
- `data/csv/echonet_dynamic_val.csv` — 1,287 clips
- `data/csv/echonet_dynamic_test.csv` — 1,276 clips
- Raw-label variants: `..._train_local_raw.csv`, `..._val_local_raw.csv` — use these for training a fresh probe. The pipeline handles Z-scoring at runtime.

**EchoNet-Pediatric**:
- `data/csv/echonet_pediatric_train.csv` — 2,579 clips (A4C + PSAX combined)
- `data/csv/echonet_pediatric_val.csv` — 335 clips
- `data/csv/echonet_pediatric_test.csv` — 367 clips
- Raw-label variants: `..._{train,val}_s3_raw.csv`

If your video paths differ from the ones baked into these CSVs, re-generate them with the same two-column format and point the config at your file. **The label column must be raw LVEF, not pre-normalized** — the training pipeline computes its own Z-score from the train split.

---

## 5. Phase 1 — Inference-only with UHN-trained probes

Six runs (2 datasets × 3 models). This measures how well a probe trained on the University Health Network's 18M-echo dataset transfers to a completely different institution (Stanford, in EchoNet-Dynamic's case) and to a completely different population (pediatric).

### 5.1 Configs

Create `configs/inference/faraz/`. One config per (dataset, model). Below is the L-K + EchoNet-Dynamic example; the other five are identical modulo the four fields marked `# CHANGE`.

**`configs/inference/faraz/uhn_probe_echonet_dynamic_lk.yaml`:**

```yaml
app: vjepa
eval_name: video_classification_frozen
val_only: true
resume_checkpoint: true
tag: uhn-probe-lk-echonet-dynamic-test   # CHANGE per (model, dataset)
folder: results/faraz/uhn_probe            # CHANGE per phase if desired

# UHN-trained probe — inference only
probe_checkpoint: checkpoints/probes/uhn_lvef/echojepa-l-k/best.pt   # CHANGE per model

experiment:
  classifier:
    task_type: regression
    num_heads: 16
    num_probe_blocks: 1
    num_targets: 1

  data:
    dataset_type: VideoDataset
    dataset_train: data/csv/echonet_dynamic_test.csv    # placeholder for parser
    dataset_val:   data/csv/echonet_dynamic_test.csv    # CHANGE per dataset
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
  checkpoint: checkpoints/encoders/vitl-kinetics-pt220-an55.pt   # CHANGE per model
  module_name: evals.video_classification_frozen.modelcustom.vit_encoder_multiclip
  pretrain_kwargs:
    encoder:
      checkpoint_key: target_encoder
      model_name: vit_large
      patch_size: 16
      tubelet_size: 2
      uniform_power: true
      use_rope: true
  wrapper_kwargs:
    max_frames: 128
    use_pos_embed: false
```

**Model-specific `model_kwargs`.** For EchoPrime and PanEcho replace the whole `model_kwargs:` block with:

```yaml
# EchoPrime
model_kwargs:
  checkpoint: null
  module_name: evals.video_classification_frozen.modelcustom.echo_prime_encoder
  pretrain_kwargs: {}
  wrapper_kwargs:
    echo_prime_root: evals/video_classification_frozen/modelcustom/EchoPrime
    force_fp32: true
    bin_size: 50

# PanEcho
model_kwargs:
  checkpoint: null
  module_name: evals.video_classification_frozen.modelcustom.panecho_encoder
  pretrain_kwargs: {}
  wrapper_kwargs: {}
```

**Pediatric.** Change `dataset_val` to `data/csv/echonet_pediatric_test.csv` and update `tag` (e.g., `uhn-probe-lk-echonet-pediatric-test`). Everything else stays the same.

### 5.2 Run

```bash
mkdir -p logs results/faraz
python -m evals.main \
  --fname configs/inference/faraz/uhn_probe_echonet_dynamic_lk.yaml \
  --devices cuda:0 \
  --val_only \
  2>&1 | tee logs/uhn_probe_echonet_dynamic_lk.log
```

Repeat for all six configs. Each run takes ~5–15 min on a single A100/H100.

### 5.3 Output

Each run writes:

```
results/faraz/uhn_probe/uhn-probe-lk-echonet-dynamic-test/
├── study_predictions.csv    # study_id, label, prediction, n_clips
└── log_r0.csv               # per-epoch metrics (MAE, R², Pearson r)
```

For EchoNet-Dynamic each "study" is one video (1 clip per study), so `n_clips` will typically be 2 (`num_segments: 2`). Prediction averaging still applies over those two clip-level scores.

---

## 6. Phase 2 — Train fresh Strategy E d=1 probes, then infer

**Six runs**: for each (dataset, model), train a fresh depth=1 attentive probe on the dataset's train split (encoder frozen), pick the best epoch on the val split, then run inference on the test split.

### 6.1 Training configs

Create `configs/eval/faraz/`. One config per (dataset, model). L-K + EchoNet-Dynamic example:

**`configs/eval/faraz/train_probe_echonet_dynamic_lk.yaml`:**

```yaml
app: vjepa
cpus_per_task: 32
folder: results/faraz/probe_train    # CHANGE per phase
mem_per_gpu: 80G
nodes: 1
tasks_per_node: 8
num_workers: 8

eval_name: video_classification_frozen
resume_checkpoint: true
tag: probe-lk-echonet-dynamic-train    # CHANGE per (model, dataset)

experiment:
  classifier:
    task_type: regression
    num_heads: 16
    num_probe_blocks: 1              # Strategy E: depth=1 attentive probe
    num_targets: 1

  data:
    dataset_type: VideoDataset
    dataset_train: data/csv/echonet_dynamic_train_local_raw.csv  # CHANGE per dataset
    dataset_val:   data/csv/echonet_dynamic_val_local_raw.csv    # CHANGE per dataset
    resolution: 224
    frames_per_clip: 16
    frame_step: 2
    num_segments: 2
    num_views_per_segment: 1

  optimization:
    batch_size: 4
    num_epochs: 20
    use_bfloat16: true
    use_pos_embed: false
    # 6-way LR × WD grid — same one Alif used for UHN. Each of the 6 rows is a
    # parallel probe head; the pipeline picks the best on val Pearson r.
    multihead_kwargs:
      - {lr: 0.0001, start_lr: 0.0001, final_lr: 0.0, warmup: 0.0, weight_decay: 0.01,  final_weight_decay: 0.01}
      - {lr: 0.0001, start_lr: 0.0001, final_lr: 0.0, warmup: 0.0, weight_decay: 0.1,   final_weight_decay: 0.1}
      - {lr: 0.0001, start_lr: 0.0001, final_lr: 0.0, warmup: 0.0, weight_decay: 0.4,   final_weight_decay: 0.4}
      - {lr: 0.00005, start_lr: 0.00005, final_lr: 0.0, warmup: 0.0, weight_decay: 0.01, final_weight_decay: 0.01}
      - {lr: 0.00005, start_lr: 0.00005, final_lr: 0.0, warmup: 0.0, weight_decay: 0.1,  final_weight_decay: 0.1}
      - {lr: 0.00005, start_lr: 0.00005, final_lr: 0.0, warmup: 0.0, weight_decay: 0.4,  final_weight_decay: 0.4}

model_kwargs:
  checkpoint: checkpoints/encoders/vitl-kinetics-pt220-an55.pt   # CHANGE per model
  module_name: evals.video_classification_frozen.modelcustom.vit_encoder_multiclip
  pretrain_kwargs:
    encoder:
      checkpoint_key: target_encoder
      model_name: vit_large
      patch_size: 16
      tubelet_size: 2
      uniform_power: true
      use_rope: true
  wrapper_kwargs:
    max_frames: 128
    use_pos_embed: false
```

Reuse the EchoPrime and PanEcho `model_kwargs:` blocks from §5.1.

**Pediatric variant.** Swap the two `dataset_*` paths to `echonet_pediatric_{train,val}_s3_raw.csv` (or the local-path variants if the videos are on-disk) and update `tag`.

**Grid.** Six LR × WD combos is what we used for the UHN LVEF probes and it's typically enough. If a run's best-val Pearson r is clearly still climbing at epoch 20, extend `num_epochs` to 40 and re-run just that model.

### 6.2 Run training

```bash
# Single-GPU run
python -m evals.main \
  --fname configs/eval/faraz/train_probe_echonet_dynamic_lk.yaml \
  --devices cuda:0 \
  2>&1 | tee logs/train_probe_lk_echonet_dynamic.log
```

Wall time on a single A100 80GB:
- EchoJEPA-L-K: ~4–6 hours per (dataset, model) for the full 6-head grid × 20 epochs on EchoNet-Dynamic
- EchoPrime, PanEcho: ~2–4 hours

If you have >1 GPU, pass `--devices cuda:0 cuda:1 ... cuda:7`; the pipeline shards clips across ranks.

### 6.3 Pick the best head

At the end of training, `results/faraz/probe_train/probe-lk-echonet-dynamic-train/` will contain one `head_<n>_best.pt` per multihead entry and a `log_r0.csv` with per-epoch, per-head val metrics.

Pick the head with the highest val Pearson r (equivalently, lowest val MAE — they usually agree). Note the head index and its LR/WD; that's the head to plug into the inference config below.

### 6.4 Test-set inference with the freshly-trained probe

Reuse the phase-1 inference config from §5.1 with three changes:
- `probe_checkpoint:` → `results/faraz/probe_train/probe-lk-echonet-dynamic-train/head_<best>_best.pt`
- `tag:` → `probe-lk-echonet-dynamic-test` (or similar; different from the training tag)
- `dataset_val:` → the test CSV (raw or normalized — either works for inference)

Run with `--val_only`, same command as §5.2.

---

## 7. Deliverables

Please share back via GDrive (`gdrive:echo_foundation/nature_medicine/faraz_lvef_bench/` — create it if it doesn't exist):

**Phase 1** (6 files):
```
uhn_probe/
├── uhn-probe-lk-echonet-dynamic-test/study_predictions.csv
├── uhn-probe-echoprime-echonet-dynamic-test/study_predictions.csv
├── uhn-probe-panecho-echonet-dynamic-test/study_predictions.csv
├── uhn-probe-lk-echonet-pediatric-test/study_predictions.csv
├── uhn-probe-echoprime-echonet-pediatric-test/study_predictions.csv
└── uhn-probe-panecho-echonet-pediatric-test/study_predictions.csv
```

**Phase 2** (6 test predictions + 6 training logs):
```
probe_train/
├── probe-lk-echonet-dynamic-{train,test}/    # log_r0.csv + best head + study_predictions.csv
├── probe-echoprime-echonet-dynamic-{train,test}/
├── probe-panecho-echonet-dynamic-{train,test}/
├── probe-lk-echonet-pediatric-{train,test}/
├── probe-echoprime-echonet-pediatric-{train,test}/
└── probe-panecho-echonet-pediatric-{train,test}/
```

Include the training `log_r0.csv` files — I need them to document which head was picked and the val-vs-test gap.

Also please share the final trained probe checkpoints (the six `head_<best>_best.pt` files, one per model×dataset) so I can re-run inference downstream if needed.

---

## 8. Expected results (for sanity checking)

**UHN-trained probes on their own site (frozen d=1 attentive, prediction averaging):**

| Model | UHN LVEF Pearson r | UHN LVEF MAE (%) |
|---|---:|---:|
| EchoJEPA-L-K | 0.852 | 4.90 |
| EchoPrime | 0.844 | 5.03 |
| PanEcho | 0.840 | 5.16 |

**Published benchmarks on EchoNet-Dynamic test (for context):**
- Original EchoNet-Dynamic paper (Ouyang et al. 2020): MAE ≈ 4.05
- PanEcho paper (Vukadinovic 2024): MAE ≈ 3.80
- EchoPrime paper (Vukadinovic 2024): MAE ≈ 4.0
- Rough expectation for our probes:
  - Phase 1 (UHN-trained, transfer): expect ~2–5 pp MAE degradation vs on-site, so **MAE ≈ 6–8** for L-K and EchoPrime, closer to 7–9 for PanEcho.
  - Phase 2 (probe trained on EchoNet-Dynamic train): should match or beat the published numbers, i.e. **MAE ≈ 4.0–5.0**. If a phase-2 number is significantly worse than the corresponding UHN number, something is wrong (check the train CSV, the label column, and whether the pipeline is Z-scoring).

**Published benchmarks on EchoNet-Pediatric (A4C):**
- PanEcho: MAE ≈ 3.9
- EchoNet-Pediatric original: MAE ≈ 4.15

Pediatric numbers matter less for our story (different anatomy, smaller cohort). We're mostly interested in showing that the probe can adapt with just 2,500 training clips.

---

## 9. Troubleshooting

- **CUDA OOM** — L-K at 224×224 with 16 frames × 2 segments should fit in 40 GB, but if you're on a smaller GPU drop `batch_size` to 2 or 1.
- **Predictions all ≈ same value or ≈ 0** — the pipeline expects raw-float labels and Z-scores internally. If your CSV is *already* Z-scored, the probe outputs will be Z-scored too and the metrics will look wrong. Use the `*_raw.csv` variants for training, or manually un-Z-score the predictions using the training-set mean/std.
- **`FileNotFoundError` on video paths** — the shipped CSVs use absolute paths on Alif's SageMaker EFS. Regenerate with your own paths. Format is: `space-delimited, no header, {abs_video_path} {raw_float_label}`.
- **EchoPrime fails to load** — Make sure you cloned the EchoPrime repo into `evals/video_classification_frozen/modelcustom/EchoPrime` and downloaded the weight zips. See the LV function guide (`vjepa2/claude/chicago/lv_function.md`) → `valve_severity.md` §3.3 for the full recipe.
- **PanEcho fails to load** — Set `TORCH_HOME` to a writable dir. First run downloads a ConvNeXt-T checkpoint from torchvision.
- **Bootstrap CIs** — you don't need to compute these; I'll do it downstream from the `study_predictions.csv` you send. Just make sure the columns are `study_id, label, prediction, n_clips` (or `study_id, label, predicted_class, n_clips, prob_class_*` for classification, which is not this task).

Ping Alif on Slack for anything else. Faster to ask than to debug pipeline internals from scratch.

---

## 10. Related references in this repo

- `vjepa2/claude/chicago/lv_function.md` — the UCMC external validation guide Teodora is running. Same probes, same procedure, different data. Copy conventions from here.
- `vjepa2/uhn_echo/nature_medicine/CLAUDE.md` — Nature Medicine manuscript context. §"Evaluation Protocol" explains Strategy E.
- `vjepa2/uhn_echo/nature_medicine/context_files/decisions/evaluation_protocol_decision.md` — the full scoring matrix that led us to depth=1 attentive probes + prediction averaging.
- `vjepa2/CLAUDE.md` §"Probe Evaluation" — command-line reference for running probe training and inference.
