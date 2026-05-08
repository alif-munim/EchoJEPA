# UCMC Regression Probes — Z-score Parameters

The regression probes we shared via GDrive were uploaded from an earlier training-run directory and **do not have z-score normalization parameters embedded**. Without these, the inference pipeline will error out at startup on any regression task. This file contains the parameters you need and two ways to plug them in.

**Contact:** Alif Munim (alif.munim@uhn.ca)

---

## 1. Why you need this

The EchoJEPA regression probes were trained against Z-score–normalized labels:

```
normalized_label = (raw_label - target_mean) / target_std
```

`target_mean` and `target_std` are the UHN training-set statistics for that task. Labels in your UCMC test CSV must be **raw float values in native units** — the pipeline does the normalization at runtime. At evaluation time, predictions are un-normalized back to raw units before metrics are reported.

The pipeline (in `evals/video_classification_frozen/eval.py:479`) looks for these params in this order:

1. **YAML config** (`experiment.data.target_mean`, `experiment.data.target_std`)
2. **Checkpoint metadata** (`target_mean`, `target_std` keys inside `best.pt`) ← expected location, but missing on the GDrive copies
3. **`zscore_params.json`** sidecar next to the training CSV (`dirname(dataset_train)/zscore_params.json`)
4. Compute from `train.csv` if training (not applicable for `val_only: true`)
5. Error: `RuntimeError: Regression inference requires zscore params but no source found.`

Since #2 is broken on the GDrive copies, we need to supply #1 or #3.

---

## 2. The parameters

UHN training-set statistics for each regression task. These are the ground truth — same values embedded in the local (non-GDrive) copies of the probes, identical across all four encoder models (G, L-K, EchoPrime, PanEcho) for a given task.

| Task | `target_mean` | `target_std` | Unit |
|------|--------------:|-------------:|------|
| **LV function** | | | |
| `lvef` | 57.9137039185 | 10.9787607193 | % |
| `edv` | 110.5755615234 | 45.6866912842 | mL |
| `esv` | 48.9779205322 | 33.8625030518 | mL |
| `cardiac_output` | 4.8316020966 | 1.6277904510 | L/min |
| **RV function** | | | |
| `tapse` | 1.9553818703 | 0.4970227480 | cm |
| `rvsp` | 35.0985984802 | 13.9064950943 | mmHg |
| `rv_fac` | 37.0955810547 | 10.2812814713 | % |
| `rv_sp` | 0.1156020537 | 0.0314756520 | m/s |
| **Hemodynamic** | | | |
| `aov_mean_grad` | 11.3073425293 | 12.5528430939 | mmHg |
| `aov_vmax` | 1.5986993313 | 0.7661124468 | m/s |
| **Diastolic (regression)** | | | |
| `mv_ee_medial` | 11.4293422699 | 5.2504682541 | cm/s (tissue-Doppler velocity, **not** the E/e' ratio) |

Classification tasks (valve severity, disease detection, diastolic function grading, trajectory onset) do **not** use Z-score params — ignore this file for those.

---

## 3. Usage — Option A: add to YAML (recommended)

Add `target_mean` and `target_std` under `experiment.data` in each regression inference config. Example for `echojepa_g_lvef.yaml`:

```yaml
app: vjepa
eval_name: video_classification_frozen
val_only: true
resume_checkpoint: true
tag: ucmc-echojepa-g-lvef
probe_checkpoint: checkpoints/probes/lvef/echojepa-g/best.pt

experiment:
  classifier:
    num_heads: 16
    num_probe_blocks: 1

  data:
    dataset_type: VideoDataset
    task_type: regression
    dataset_train: data/csv/ucmc_lvef_test.csv
    dataset_val: data/csv/ucmc_lvef_test.csv
    num_classes: 1
    resolution: 224
    frames_per_clip: 16
    frame_step: 2
    num_segments: 2
    num_views_per_segment: 1
    study_sampling: true
    target_mean: 57.9137039185      # <-- add
    target_std: 10.9787607193       # <-- add

  optimization:
    batch_size: 4
    num_epochs: 1
    use_bfloat16: true
    multihead_kwargs:
    - {lr: 0.0, start_lr: 0.0, final_lr: 0.0, warmup: 0.0, weight_decay: 0.0, final_weight_decay: 0.0}

model_kwargs:
  # ... rest unchanged ...
```

The same two values are used for all 4 models on a given task (e.g. `lvef` uses the same `mean=57.9137, std=10.9788` for `echojepa-g`, `echojepa-l-k`, `echoprime`, and `panecho`).

### Quick sed snippet (optional)

If you already have all the YAML configs generated and just want to bulk-patch them:

```bash
# From your EchoJEPA repo root, for each task, add the two lines under experiment.data
patch_task() {
  local task=$1 mean=$2 std=$3
  for f in configs/inference/chicago/*_${task}.yaml; do
    # Insert the two lines after "study_sampling: true" (last line in data block in the template)
    grep -q "target_mean:" "$f" || \
      sed -i "/study_sampling: true/a\\    target_mean: ${mean}\\n    target_std: ${std}" "$f"
  done
}

patch_task lvef           57.9137039185  10.9787607193
patch_task edv            110.5755615234 45.6866912842
patch_task esv            48.9779205322  33.8625030518
patch_task cardiac_output 4.8316020966   1.6277904510
patch_task tapse          1.9553818703   0.4970227480
patch_task rvsp           35.0985984802  13.9064950943
patch_task rv_fac         37.0955810547  10.2812814713
patch_task rv_sp          0.1156020537   0.0314756520
patch_task aov_mean_grad  11.3073425293  12.5528430939
patch_task aov_vmax       1.5986993313   0.7661124468
patch_task mv_ee_medial   11.4293422699  5.2504682541
```

### Verify before running

When you launch inference, check the log for:

```
INFO  - Z-score params from YAML: mean=57.9137, std=10.9788
```

If you instead see `RuntimeError: Regression inference requires zscore params...`, the YAML edit didn't take effect.

---

## 4. Usage — Option B: drop `zscore_params.json` next to each CSV

Alternative if you don't want to touch the YAMLs. The pipeline looks for `zscore_params.json` in the same directory as your `dataset_train` CSV. For each regression task, create a JSON file with that name.

Example: if `dataset_train: data/csv/ucmc_lvef_test.csv`, create `data/csv/zscore_params.json`.

**Important caveat:** if multiple regression tasks share the same CSV directory, they will fight over this file (different mean/std per task). Use **Option A (YAML)** if you keep all CSVs in one folder, or put each task's CSV in its own subdirectory.

Per-task JSON file contents (one per task):

```json
{"target_mean": 57.9137039185, "target_std": 10.9787607193}
```

Quick generation script:

```bash
# Create one subdirectory per task, containing its CSV + zscore_params.json
mkdir -p data/csv/{lvef,edv,esv,cardiac_output,tapse,rvsp,rv_fac,rv_sp,aov_mean_grad,aov_vmax,mv_ee_medial}

cat > data/csv/lvef/zscore_params.json <<EOF
{"target_mean": 57.9137039185, "target_std": 10.9787607193}
EOF
cat > data/csv/edv/zscore_params.json <<EOF
{"target_mean": 110.5755615234, "target_std": 45.6866912842}
EOF
cat > data/csv/esv/zscore_params.json <<EOF
{"target_mean": 48.9779205322, "target_std": 33.8625030518}
EOF
cat > data/csv/cardiac_output/zscore_params.json <<EOF
{"target_mean": 4.8316020966, "target_std": 1.6277904510}
EOF
cat > data/csv/tapse/zscore_params.json <<EOF
{"target_mean": 1.9553818703, "target_std": 0.4970227480}
EOF
cat > data/csv/rvsp/zscore_params.json <<EOF
{"target_mean": 35.0985984802, "target_std": 13.9064950943}
EOF
cat > data/csv/rv_fac/zscore_params.json <<EOF
{"target_mean": 37.0955810547, "target_std": 10.2812814713}
EOF
cat > data/csv/rv_sp/zscore_params.json <<EOF
{"target_mean": 0.1156020537, "target_std": 0.0314756520}
EOF
cat > data/csv/aov_mean_grad/zscore_params.json <<EOF
{"target_mean": 11.3073425293, "target_std": 12.5528430939}
EOF
cat > data/csv/aov_vmax/zscore_params.json <<EOF
{"target_mean": 1.5986993313, "target_std": 0.7661124468}
EOF
cat > data/csv/mv_ee_medial/zscore_params.json <<EOF
{"target_mean": 11.4293422699, "target_std": 5.2504682541}
EOF
```

Then move each CSV into its task subdirectory and update `dataset_train`/`dataset_val` paths in the configs accordingly (e.g. `data/csv/lvef/ucmc_lvef_test.csv`).

Verify in the log:

```
INFO  - Z-score params from data/csv/lvef/zscore_params.json: mean=57.9137, std=10.9788
```

---

## 5. Machine-readable bundle

The same values in a single JSON blob, in case you want to consume them programmatically:

```json
{
  "lvef":           {"target_mean": 57.9137039185,  "target_std": 10.9787607193, "unit": "%"},
  "edv":            {"target_mean": 110.5755615234, "target_std": 45.6866912842, "unit": "mL"},
  "esv":            {"target_mean": 48.9779205322,  "target_std": 33.8625030518, "unit": "mL"},
  "cardiac_output": {"target_mean": 4.8316020966,   "target_std": 1.6277904510,  "unit": "L/min"},
  "tapse":          {"target_mean": 1.9553818703,   "target_std": 0.4970227480,  "unit": "cm"},
  "rvsp":           {"target_mean": 35.0985984802,  "target_std": 13.9064950943, "unit": "mmHg"},
  "rv_fac":         {"target_mean": 37.0955810547,  "target_std": 10.2812814713, "unit": "%"},
  "rv_sp":          {"target_mean": 0.1156020537,   "target_std": 0.0314756520,  "unit": "m/s"},
  "aov_mean_grad":  {"target_mean": 11.3073425293,  "target_std": 12.5528430939, "unit": "mmHg"},
  "aov_vmax":       {"target_mean": 1.5986993313,   "target_std": 0.7661124468,  "unit": "m/s"},
  "mv_ee_medial":   {"target_mean": 11.4293422699,  "target_std": 5.2504682541,  "unit": "cm/s"}
}
```

---

## 6. FAQ

**Q: Do I need to Z-score my UCMC labels myself before putting them in the CSV?**
No. Labels in the CSV are raw float values in native units. The pipeline applies `(raw - target_mean) / target_std` internally during dataloading.

**Q: Will predictions come out Z-scored?**
No. Predictions are un-normalized (multiplied by `target_std`, then offset by `target_mean`) before being written to `study_predictions.csv`, so you'll see raw-unit predictions. Same for the MAE / R² / Pearson metrics.

**Q: What if my UCMC population has a different mean/std?**
You should still use the UHN parameters listed here — those are what the probe was trained on, so the probe's internal prediction head outputs values calibrated against UHN distribution statistics. If you re-normalize using UCMC statistics, the predictions will be systematically biased. Distribution shift will show up in the metrics (MAE may be higher than UHN held-out MAE), which is the expected and correct signal for external validation.

**Q: Does this affect classification probes too?**
No. Classification probes (valve severity, disease detection, diastolic function 4-class, trajectory onset binary) don't use z-score normalization. Only the 11 regression tasks in the table above.

**Q: One of my runs is failing with `RuntimeError: Regression inference requires zscore params...` — what do I check?**
1. You're running a regression task from Section 2 of this file.
2. Your YAML has `target_mean` and `target_std` under `experiment.data` (Option A) OR `zscore_params.json` sits next to your `dataset_train` CSV (Option B).
3. Log shows either `Z-score params from YAML: ...` or `Z-score params from <path>/zscore_params.json: ...` during startup.
