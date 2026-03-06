# experiments/

Precomputed frozen embeddings and experiment data for EchoJEPA and baseline models. Organized by paper.

See the [root README](../README.md) for full documentation: NPZ formats, probe training commands, model details, and worked examples.

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
    └── mimic/                         #   MIMIC-IV-Echo multi-model pipeline
        ├── {model}_mimic_embeddings.npz   # clip-level master NPZs (7 models)
        ├── clip_index.npz                 # shared: s3_paths, study_ids, patient_ids
        ├── patient_split.json             # shared: 70/15/15 patient assignment
        ├── labels/                        # shared: 23 task label NPZs
        ├── {model}_study_level/           # study-pooled embeddings (23 tasks per model)
        ├── {model}_splits/                # patient-level train/val/test (23 tasks per model)
        ├── {model}_mimic_all.zip          # self-contained distribution packages
        ├── covariates/                    # covariate CSVs (demographics, comorbidity, acuity, EHR)
        └── mimic_covariates.zip           # covariates zip (unzips into covariates/)
```

## ICML (`icml/`)

UHN benchmark embeddings from the ICML preprint: LVEF regression, view classification. 5 models: EchoJEPA-G, EchoJEPA-L, EchoMAE-L, EchoPrime, PanEcho.

## Nature Medicine (`nature_medicine/mimic/`)

Multi-model embedding pipeline for MIMIC-IV-Echo (7,243 studies, 4,579 patients, 525K clips). 7 frozen encoders, 23 clinical tasks, patient-level splits. See `claude/data/embedding-pipeline.md` for the full pipeline reference.

Models: EchoJEPA-G (1408-dim), EchoJEPA-L, EchoJEPA-L-K, EchoMAE (1024-dim), EchoFM (1024-dim), PanEcho (768-dim), EchoPrime (512-dim). See the [root README](../README.md#available-models) for the full model table and task list.

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
| `ehr_features.csv` | 56 | 54-feature EHR baseline: demographics, labs, vitals, medications, comorbidity (Table 3) |

```python
import pandas as pd
import numpy as np

# Load probe predictions + covariates for fairness analysis
preds = np.load("experiments/nature_medicine/mimic/echojepa_g_splits/mortality_1yr/test.npz")
demo = pd.read_csv("experiments/nature_medicine/mimic/covariates/demographics_fairness.csv")

# Join on study_id for subgroup evaluation
pred_df = pd.DataFrame({"study_id": preds["study_ids"], "label": preds["labels"]})
merged = pred_df.merge(demo, on="study_id")
```
