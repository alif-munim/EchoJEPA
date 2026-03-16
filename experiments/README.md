# experiments/

Precomputed frozen embeddings and experiment data for EchoJEPA and baseline models. Organized by paper.

## Directory Structure

```
experiments/
├── icml/                              # ICML preprint experiments (UHN data)
│   ├── lvef/                          #   LVEF regression (5 models)
│   ├── views/                         #   View classification (5 models)
│   ├── test/                          #   Test set embeddings
│   └── misc/                          #   Miscellaneous (echojepa_g only)
│
└── nature_medicine/                   # Nature Medicine experiments
    ├── uhn/                           #   UHN experiments (18M echos, 153K patients)
    │   ├── probe_csvs/                #   Task-specific CSVs (47 tasks x 3 splits)
    │   ├── build_probe_csvs.py        #   All-clip CSV builder
    │   ├── build_viewfiltered_csvs.py #   View-filtered CSV builder
    │   ├── build_trajectory_csvs.py   #   Trajectory pair CSV builder
    │   ├── build_labels.py            #   Raw label extraction from databases
    │   └── mapping/                   #   DICOM↔Syngo deid key files (gitignored CSVs)
    ├── mimic/                         #   MIMIC-IV-Echo multi-model pipeline
    │   ├── {model}_mimic_embeddings.npz   # clip-level master NPZs (7 models)
    │   ├── clip_index.npz                 # shared: s3_paths, study_ids, patient_ids
    │   ├── patient_split.json             # shared: 70/15/15 patient assignment
    │   ├── labels/                        # shared: 23 task label NPZs
    │   ├── {model}_study_level/           # study-pooled embeddings (23 tasks per model)
    │   ├── {model}_splits/                # patient-level train/val/test (23 tasks per model)
    │   ├── {model}_mimic_all.zip          # self-contained distribution packages
    │   ├── covariates/                    # covariate CSVs (demographics, comorbidity, acuity, EHR)
    │   ├── build_probe_csvs.py            # MIMIC task CSV builder
    │   └── mimic_covariates.zip           # covariates zip (unzips into covariates/)
    └── physiological_ranges.json      # plausible value ranges per task (shared by all build scripts)
```

## ICML (`icml/`)

UHN benchmark embeddings from the ICML preprint: LVEF regression, view classification. 5 models: EchoJEPA-G, EchoJEPA-L, EchoMAE-L, EchoPrime, PanEcho.

## Nature Medicine — UHN (`nature_medicine/uhn/`)

UHN experiments for 18.1M echocardiographic clips across 320K studies. The primary mapping source is `data/aws/aws_syngo.csv` (320K studies, 2002-2019) which maps STUDY_REF ↔ DeidentifiedStudyID ↔ s3_key ↔ PATIENT_ID. See `claude/data/uhn-mapping.md` for full details.

### Task CSVs

Task-specific CSVs live under `probe_csvs/<task>/` with `train.csv`, `val.csv`, `test.csv` per task. These are JEPA-format (space-delimited: `<s3_path> <label>`). Build scripts:

- `build_probe_csvs.py` — all-clip CSVs (47 tasks)
- `build_viewfiltered_csvs.py` — view-filtered variants (41 tasks, `train_vf.csv`)
- `build_trajectory_csvs.py` — temporal pair CSVs (5 tasks)
- `build_labels.py` — raw label extraction from UHN databases

Regression CSVs store **raw values** (not Z-scored). Z-score normalization happens at runtime via `zscore_params.json` files alongside each task's CSVs.

### Pipeline Status

| Stage | Status | Output |
|-------|--------|--------|
| P0 Mapping | **DONE** | `data/aws/aws_syngo.csv` (320K studies) |
| P1 Clip Index | **DONE** | `uhn_clip_index.npz` (18.1M clips, 320K studies) |
| P2 Splits | **DONE** | `patient_split.json` (138K patients, 70/10/20 temporal) |
| P3 Labels | **DONE** | `labels/` — 53 NPZs |
| P4 Probe CSVs | **DONE** | `probe_csvs/` — 47 tasks + 5 trajectory + view-filtered variants |

## Nature Medicine — MIMIC (`nature_medicine/mimic/`)

Multi-model embedding pipeline for MIMIC-IV-Echo (7,243 studies, 4,579 patients, 525K clips). 7 frozen encoders, 23 clinical tasks, patient-level splits. See `claude/data/embedding-pipeline.md` for the full pipeline reference.

### Covariate CSVs (`covariates/`)

Companion data for baselines, fairness, and confound analysis. Joined to probe results by `study_id`. All 7,243 MIMIC-IV-Echo studies. Built by scripts in `uhn_echo/nature_medicine/data_exploration/mimic/csv/`.

```bash
# From the repository root
unzip experiments/nature_medicine/mimic/mimic_covariates.zip
```

This restores 4 CSVs into `experiments/nature_medicine/mimic/covariates/`:

| CSV | Columns | Purpose |
|-----|---------|---------|
| `charlson_elixhauser.csv` | 58 | Charlson CCI + Elixhauser comorbidity scores (Table 3 baseline) |
| `demographics_fairness.csv` | 9 | Sex, race (5 groups), age, insurance, intersectional subgroups (Table 4) |
| `acuity_covariates.csv` | 10 | DRG severity, ICU flags, triage acuity, admission type (H_confound) |
| `ehr_features.csv` | 56 | 54-feature EHR baseline: demographics, labs, vitals, medications, comorbidity |

---

## Embedding Pipeline (Legacy)

The NPZ-based embedding pipeline has been **superseded by Strategy E** (d=1 attentive probes from video + prediction averaging) for the Nature Medicine evaluation. These tools remain available for quick prototyping or when GPU video decoding is not available.

Scripts live in `evals/`:
- `evals/extract_embeddings.py` — clip-level extraction → master NPZ
- `evals/remap_embeddings.py` — per-task label NPZs from master embeddings
- `evals/pool_embeddings.py` — mean-pool clip → study-level embeddings
- `evals/train_probe.py` — sklearn linear probes on NPZ embeddings

### Quickstart: Working with Embeddings

If you have precomputed embedding files (`.npz`), you can train sklearn probes immediately — no GPU or video data needed.

**Inspect an embedding file:**

```python
import numpy as np

data = np.load("experiments/icml/views/echojepa_g_embeddings.npz", allow_pickle=True)
print(f"Embeddings: {data['embeddings'].shape}")  # e.g. [27216, 1408]
print(f"Labels:     {data['labels'].shape}")       # [27216]
print(f"Paths:      {data['paths'].shape}")        # [27216]
```

**Train a probe (k-fold cross-validation):**

```bash
# Classification (e.g. echo view classification — integer labels)
python -m evals.train_probe \
    --data experiments/icml/views/echojepa_g_embeddings.npz \
    --cv 5 --output_dir results/probes/views/echojepa_g

# Regression (e.g. LVEF estimation — float labels)
python -m evals.train_probe \
    --data experiments/icml/lvef/echojepa_g_embeddings.npz \
    --task regression --cv 5 \
    --output_dir results/probes/lvef/echojepa_g
```

**Compare multiple models:**

```bash
python -m evals.train_probe \
    --data experiments/icml/views/echojepa_g_embeddings.npz \
           experiments/icml/views/echoprime_embeddings.npz \
           experiments/icml/views/panecho_embeddings.npz \
    --model_names EchoJEPA-G EchoPrime PanEcho \
    --cv 5 --output_dir results/probes/views/comparison
```

**Train/val mode (separate files):**

```bash
python -m evals.train_probe \
    --train experiments/nature_medicine/mimic/echojepa_g_splits/mortality_1yr/train.npz \
    --val experiments/nature_medicine/mimic/echojepa_g_splits/mortality_1yr/test.npz \
    --task classification \
    --output_dir results/probes/nature_medicine/echojepa_g/mortality_1yr
```

Run `python -m evals.train_probe --help` for all options (hyperparameter grids, regression models, denormalization, etc.).

### Embedding Extraction

Extract frozen embeddings from any model and save as `.npz`:

```bash
python -m evals.extract_embeddings \
    --config configs/inference/vitg-384/view/echojepa_224px.yaml \
    --data data/csv/uhn_views_22k_test_224px.csv \
    --output experiments/icml/views/echojepa_g_embeddings.npz \
    --devices cuda:0 cuda:1 cuda:2 cuda:3
```

Output `.npz` files contain `embeddings` (`[N, D]`), `labels` (`[N]`), and `paths` (`[N]`).

Supported model backends:

| Model | `module_name` | Embed dim | Config examples |
|-------|---------------|-----------|-----------------|
| EchoJEPA (V-JEPA 2) | `...modelcustom.vit_encoder_multiclip` | 1408 (ViT-g) | `configs/inference/vitg-384/view/echojepa_224px.yaml` |
| EchoPrime | `...modelcustom.echo_prime_encoder` | 768 | `configs/inference/vitg-384/view/echoprime_224px.yaml` |
| PanEcho | `...modelcustom.panecho_encoder` | 768 | `configs/inference/vitg-384/view/panecho_224px.yaml` |
| VideoMAE | `...modelcustom.videomae_encoder` | 1024 | `configs/inference/vitg-384/view/videomae_224px.yaml` |

Embeddings are label-independent (frozen encoder output), so datasets sharing the same videos produce identical embeddings. Extract once on the superset, then use `evals/remap_embeddings.py` to create per-task label subsets.

### MIMIC-IV-Echo Precomputed Embeddings

Precomputed embeddings for all 525K MIMIC-IV-Echo clips from 7 frozen encoders, with label mappings for 23 clinical tasks.

**Available models:**

| Model | Architecture | Pretraining | Embed dim | Zip |
|-------|-------------|-------------|-----------|-----|
| EchoJEPA-G | ViT-g/16 384px | JEPA on 18M echo clips | 1408 | `echojepa_g_mimic_all.zip` |
| EchoJEPA-L | ViT-L/16 224px | JEPA on 18M echo clips | 1024 | `echojepa_l_mimic_all.zip` |
| EchoJEPA-L Kinetics | ViT-L/16 224px | JEPA on Kinetics-400 | 1024 | `echojepa_l_kinetics_mimic_all.zip` |
| EchoMAE | ViT-L/16 (VideoMAE) | MAE on 1.5M echo clips | 1024 | `echomae_mimic_all.zip` |
| EchoFM | ViT-L/16 (MAE+triplet) | MAE on 290K echo clips | 1024 | `echofm_mimic_all.zip` |
| PanEcho | ConvNeXt-T + Transformer | Supervised on 1.1M echo clips | 768 | `panecho_mimic_all.zip` |
| EchoPrime | MViT-v2-S | CLIP-style contrastive on 12M echo clips | 512 | `echoprime_mimic_all.zip` |

**Downloading:**

Embedding zips are hosted on S3 at `s3://echodata25/nature_medicine/mimic/`:

```bash
# Download a single model (e.g. EchoJEPA-G, 5.0 GiB)
aws s3 cp s3://echodata25/nature_medicine/mimic/echojepa_g_mimic_all.zip .

# Download all models + covariates (~26.3 GiB total)
aws s3 sync s3://echodata25/nature_medicine/mimic/ . --exclude "*" \
    --include "*.zip"

# Unzip from repository root (restores to experiments/nature_medicine/mimic/)
unzip echojepa_g_mimic_all.zip
```

**Contents of a zip:**

```
experiments/nature_medicine/mimic/
├── echojepa_g_mimic_embeddings.npz    # 2.8GB — clip-level embeddings (525,312 × 1408)
├── clip_index.npz                     # shared: maps each row to S3 path, study_id, patient_id
├── patient_split.json                 # shared: global patient-level train/val/test (70/15/15)
├── labels/                            # shared: per-task label NPZs (23 files)
│   ├── mortality_1yr.npz
│   └── ...
├── echojepa_g_study_level/            # study-level pooled embeddings (23 tasks)
│   ├── mortality_1yr.npz
│   └── ...
├── echojepa_g_splits/                 # patient-level train/val/test splits (23 tasks)
│   ├── mortality_1yr/
│   │   ├── train.npz
│   │   ├── val.npz
│   │   └── test.npz
│   └── ...

data/csv/nature_medicine/mimic/        # source label CSVs (23 files, VJEPA format: "s3_path label")
├── mortality_1yr.csv
└── ...
```

**NPZ formats:**

`{model}_mimic_embeddings.npz` — master clip-level embeddings:

| Array | Shape | Dtype | Description |
|-------|-------|-------|-------------|
| `embeddings` | `(525312, D)` | float32 | Mean-pooled encoder output per clip |
| `labels` | `(525312,)` | int64 | Labels from extraction source CSV (use task-specific labels instead) |
| `paths` | `(525312,)` | str | Placeholder indices — use `clip_index.npz` for real paths |

`clip_index.npz` — maps row index to identifiers:

| Array | Shape | Description |
|-------|-------|-------------|
| `s3_paths` | `(525312,)` | Full S3 URI for each MP4 clip |
| `study_ids` | `(525312,)` | MIMIC study ID (groups clips into studies) |
| `patient_ids` | `(525312,)` | MIMIC patient ID (groups studies into patients) |

`labels/<task>.npz` — per-task label files (lightweight, no embedding duplication):

| Array | Shape | Description |
|-------|-------|-------------|
| `indices` | `(N_task,)` | Row positions into the master NPZ |
| `labels` | `(N_task,)` | Task-specific label per clip |

**Available tasks (23):**

| Category | Task | Studies | Type | Notes |
|----------|------|---------|------|-------|
| Mortality | `mortality_30d`, `mortality_90d`, `mortality_1yr` | 7,243 | Binary | |
| Mortality | `in_hospital_mortality` | 3,437 | Binary | Inpatient only |
| Outcomes | `readmission_30d` | 3,145 | Binary | Inpatient only |
| Outcomes | `icu_transfer` | 2,496 | Binary | |
| Outcomes | `discharge_destination` | 2,758 | Binary | Institutional vs home |
| Outcomes | `los_remaining` | 3,156 | Regression | Days remaining |
| Diseases | `disease_afib`, `disease_hf`, `disease_hcm`, `disease_dcm` | 7,243 | Binary | ICD-coded |
| Diseases | `disease_stemi`, `disease_amyloidosis`, `disease_takotsubo`, `disease_tamponade` | 7,243 | Binary | Rare |
| Biomarkers | `creatinine`, `troponin_t`, `nt_probnp`, `lactate` | 852–3,883 | Regression | Lab within ±24h |
| Clinical | `ef_note_extracted` | 2,580 | Regression | EF from discharge notes |
| Clinical | `drg_severity`, `triage_acuity` | 2,614–3,103 | Ordinal (1–4) | |

**Working with embeddings:**

```python
import numpy as np

# Load master embeddings + index
master = np.load("experiments/nature_medicine/mimic/echojepa_g_mimic_embeddings.npz")
index = np.load("experiments/nature_medicine/mimic/clip_index.npz", allow_pickle=True)

# Load a task's labels and subset
task = np.load("experiments/nature_medicine/mimic/labels/mortality_1yr.npz")
task_embeddings = master["embeddings"][task["indices"]]
task_study_ids = index["study_ids"][task["indices"]]

# Pool to study level
unique_ids, inverse = np.unique(task_study_ids, return_inverse=True)
study_embeddings = np.array([task_embeddings[inverse == i].mean(axis=0) for i in range(len(unique_ids))])
```

**Regenerating derived files:**

```bash
MODEL=echojepa_g  # or any model prefix
for f in experiments/nature_medicine/mimic/labels/*.npz; do
    task=$(basename "$f" .npz)
    python -m evals.pool_embeddings \
        --embeddings "experiments/nature_medicine/mimic/${MODEL}_mimic_embeddings.npz" \
        --clip_index experiments/nature_medicine/mimic/clip_index.npz \
        --labels "$f" \
        --output "experiments/nature_medicine/mimic/${MODEL}_study_level/${task}.npz"
done
```

**Adding a new model:**

Extract with the same source CSV to ensure row alignment, then reuse shared files (`clip_index.npz`, `patient_split.json`, `labels/`):

```bash
python -m evals.extract_embeddings \
    --config configs/inference/your_model_config.yaml \
    --data data/csv/nature_medicine/mimic/mortality_1yr.csv \
    --output experiments/nature_medicine/mimic/newmodel_mimic_embeddings.npz \
    --devices cuda:0 cuda:1 cuda:2 cuda:3
```
