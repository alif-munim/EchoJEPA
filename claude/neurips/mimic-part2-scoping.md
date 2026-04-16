# MIMIC Part 2 Validation: Probe Training Scoping Doc

**Date:** 2026-04-08
**Purpose:** Train d=1 attentive probes for the four controlled-comparison models on a curated MIMIC task battery that tests the Part 2 predictions from `paper-outline.md` §5. **Not yet launched.** This doc specifies exactly what needs to be trained, which configs to clone, which CSVs to use, and the recommended launch strategy.

**Prerequisites for launching:**
1. `paper-outline.md` §5 predictions are agreed and finalized (i.e., don't train probes until the predictions are locked — otherwise retrospective cherry-picking risk).
2. Nature Medicine deconfliction carve-out is agreed with co-authors (see `paper-outline.md` §5.2 Task Selection Rationale, Note).
3. HyperPod compute window is available — estimated 1 day on an 8-GPU A100/H100 node.

---

## Scope: what needs to be trained

**4 encoders × 6 tasks = 24 probes**, plus 2 existing tasks (CAMUS segmentation + UHN LVEF + Pediatric transfer) that reuse existing decoders for 3 additional evaluations.

### Encoders (the four controlled-comparison models)

| Model | Local encoder path | `model_name` | `checkpoint_key` | `use_rope` | Notes |
|---|---|---|---|---|---|
| **JEPA IN21K e100** | `checkpoints/pretrain/mimic/vjepa2_vitl_224px_16f_in1k/e100.pt` (or equivalent from job 376 S3) | `vit_large` | `target_encoder` | `true` | The primary JEPA row. Confirmed init-matched with BYOL/MAE. |
| **BYOL e100** | `checkpoints/byol_vitl_imagenet_v2_e100.pt` | `vit_large` | `encoder` | `true` | BYOL backbone; projector/predictor heads discarded for probing. |
| **MAE e99** | `checkpoints/videomae_l_mimic_ep99.pth` | (VideoMAE-specific loader) | `model` | N/A (VideoMAE uses its own PE) | **Gotcha:** MAE uses the external VideoMAE codebase. Cannot use the standard `vit_encoder_multiclip` module; needs the `videomae_encoder` adapter at `evals/video_classification_frozen/modelcustom/videomae_encoder.py`. See existing ICML probe configs (e.g. `configs/eval/vitl/icml/echomae_l_e99_end_lvef_d4.yaml`) for the correct `module_name`. |
| **SALT v1 e79** | `checkpoints/salt_s2_vitl_e79.pt` | `vit_large` | `encoder` | `true` | Primary SALT row (locked 2026-04-08). 8 `norms_block` keys (v1 hierarchical) are ignored by the standard ViT loader — no special handling needed. |

### Tasks (Part 2 validation battery — 6 MIMIC + 3 reuse)

All MIMIC CSVs live at `experiments/nature_medicine/mimic/probe_csvs/{task}/{train,val,test}.csv` with z-score params (for regression) at `zscore_params.json`.

| Task | Type | Tests predictions | train/val/test CSVs exist? | Probe epochs | Notes |
|---|---|---|---|---|---|
| `nt_probnp` | Regression | **P1, P6** | ✅ `.../nt_probnp/{train,val,test}.csv` + `zscore_params.json` | 35 | Study-level sampling. Cardiac stress biomarker. Expected: JEPA > BYOL > MAE ≈ SALT, gap ≥0.05. |
| `ef_note_extracted` | Regression | **P1, P6** | ✅ `.../ef_note_extracted/` | 35 | LVEF from clinical notes. Fundamentally dynamic. Expected: JEPA > BYOL > MAE. |
| `disease_afib` | Classification (binary) | **P2, P6** | ✅ `.../disease_afib/` | 15–20 | AFib rhythm. Expected: JEPA >> MAE (largest gap of any task). |
| `disease_hcm` | Classification (binary) | **P3, P6, P7** | ✅ via `disease_hcm/` (multi-version available: v1, v2, v4.1) | 15–20 | Hypertrophic cardiomyopathy. Expected: MAE ≈ JEPA (static anatomy). **Use latest version** (`disease_hcm_v4.1` if available; otherwise base `disease_hcm`). |
| `disease_dcm` | Classification (binary) | **P3, P6, P7** | ✅ via `disease_dcm/` (multi-version) | 15–20 | Dilated cardiomyopathy. Same framing as HCM — structural task. |
| `troponin_t` | Regression | **P4, P6** | ✅ `.../troponin_t/` (if present; confirm) | 35 | Acute cardiac injury biomarker. Expected: JEPA > MAE ≈ BYOL. |
| `mortality_1yr` | Classification (binary) | **P4, P6** | ✅ `.../mortality_1yr/` | 15–20 | 1-year all-cause mortality. Expected: JEPA > MAE ≈ BYOL, BYOL not leading. |
| **CAMUS segmentation** | Segmentation (Dice) | **P3, P7** | (existing decoders) | Reuse ICML decoder | **Already done for JEPA/BYOL/MAE** (MAE 0.827 > BYOL 0.825 > JEPA 0.815). **Need SALT v1 e79 CAMUS eval** — reuse existing decoder training protocol. |
| **Pediatric zero-shot LVEF** | Regression (transfer) | **P1, P5** | (zero-shot from UHN probes) | No probe training | **Already done for JEPA/BYOL/MAE** (JEPA 0.670 > MAE 0.617 > BYOL 0.500). **Need SALT v1 e79 pediatric** — just inference, no new training. |
| **UHN LVEF** | Regression | **P1** | (existing UHN probes) | 20 | Need init-matched 4-way probes on UHN 53K (not just the ICML Meta-init JEPA pt50). |

**Task selection is minimal — 6 MIMIC + 4 reuse = 10 data points total.** Not kitchen-sink. Each task tests at least one specific prediction with a pre-committed expected ranking. Tasks like `creatinine`, `readmission_30d`, `mortality_30d`, `mortality_90d`, `lactate`, `discharge_destination` are **intentionally excluded** — they would inflate the task count without testing distinct predictions. If a reviewer asks "why these and not those," the answer is "each of these tests a specific §5 prediction, and the excluded tasks do not add new predictions."

### Summary: new probes needed

| Category | Count | Wall-clock (8-GPU) |
|---|---|---|
| MIMIC probe training (6 tasks × 4 models) | 24 | ~6 hours (sequential, 4 probes in parallel) |
| CAMUS segmentation decoder for SALT | 1 | ~2 hours |
| UHN LVEF probes for 4 init-matched models | 4 | ~4 hours |
| **Total new work** | **29 probes/decoders** | **~12 hours (1 HyperPod day)** |

Pediatric zero-shot SALT evaluation requires no new training — it is probe inference on the existing SALT-on-UHN-LVEF probe (which also needs to be trained as part of the UHN LVEF row above). Estimated inference time: ~30 minutes.

---

## Config templates and swap instructions

### MIMIC probe template

Use `configs/eval/vitg-384/nature_medicine/echojepa_g_mortality_1yr.yaml` as the structural template. Clone per task × model and swap:

```yaml
# Header
app: vjepa
tag: {model}-{task}  # e.g. jepa-in21k-e100-nt-probnp
folder: /path/to/output/evals/vitl/neurips/{model}_{task}

eval_name: video_classification_frozen
resume_checkpoint: false

experiment:
  classifier:
    num_heads: 16
    num_probe_blocks: 1  # d=1 attentive probe (matches ICML/NatMed)

  data:
    dataset_type: VideoDataset
    dataset_train: /mnt/custom-file-systems/efs/.../vjepa2/experiments/nature_medicine/mimic/probe_csvs/{task}/train.csv
    dataset_val:   /mnt/custom-file-systems/efs/.../vjepa2/experiments/nature_medicine/mimic/probe_csvs/{task}/val.csv
    num_classes: {2 for binary, 1 for regression}
    resolution: 224
    frames_per_clip: 16
    frame_step: 2
    num_segments: 2
    num_views_per_segment: 1
    study_sampling: true  # DistributedStudySampler — 1 clip per study per epoch

  optimization:
    batch_size: 1
    num_epochs: {35 for regression, 15-20 for classification}
    use_bfloat16: true
    use_pos_embed: false
    multihead_kwargs:
      # 20-element HP grid from echojepa_g_mortality_1yr.yaml
      # (5 LRs × 4 WDs = 20 heads)
      - {lr: 0.001,  ..., weight_decay: 0.001, ...}
      # ... (copy verbatim from the template)

model_kwargs:
  checkpoint: {ENCODER PATH FROM TABLE ABOVE}
  module_name: evals.video_classification_frozen.modelcustom.vit_encoder_multiclip
  pretrain_kwargs:
    encoder:
      checkpoint_key: {target_encoder | encoder | model}  # per-model, from table above
      img_temporal_dim_size: null
      model_name: vit_large
      patch_size: 16
      tubelet_size: 2
      uniform_power: true
      use_rope: true
  wrapper_kwargs:
    max_frames: 128
    use_pos_embed: false
```

### Model-specific overrides

**JEPA IN21K e100:**
```yaml
model_kwargs:
  checkpoint: /opt/dlami/nvme/checkpoints/pretrain/mimic/vjepa2_vitl_224px_16f_in1k/e100.pt
  pretrain_kwargs:
    encoder:
      checkpoint_key: target_encoder
      model_name: vit_large
      use_rope: true
```

**BYOL e100:**
```yaml
model_kwargs:
  checkpoint: /opt/dlami/nvme/checkpoints/byol_vitl_imagenet_v2_e100.pt
  pretrain_kwargs:
    encoder:
      checkpoint_key: encoder
      model_name: vit_large
      use_rope: true
```

**MAE e99 (VideoMAE external codebase — DIFFERENT module_name):**
```yaml
model_kwargs:
  checkpoint: /opt/dlami/nvme/checkpoints/videomae_l_mimic_ep99.pth
  module_name: evals.video_classification_frozen.modelcustom.videomae_encoder  # <-- DIFFERENT
  pretrain_kwargs:
    encoder:
      checkpoint_key: model
      model_name: vit_large_patch16_224  # VideoMAE naming
      # no use_rope flag — VideoMAE uses its own positional embedding
```

Reference: see any existing ICML MAE probe config (e.g. `configs/eval/vitl/icml/echomae_l_e99_end_lvef_d4.yaml`) for the exact VideoMAE loader invocation.

**SALT v1 e79:**
```yaml
model_kwargs:
  checkpoint: /opt/dlami/nvme/checkpoints/salt_s2_vitl_e79.pt
  pretrain_kwargs:
    encoder:
      checkpoint_key: encoder
      model_name: vit_large
      use_rope: true
  # Note: 8 norms_block.{0..3}.{weight,bias} keys will be flagged as
  # unexpected by the standard ViT loader with strict=False. This is
  # expected (they belong to SALT's hierarchical predictor head, not
  # the encoder backbone). No special handling needed.
```

### Task-specific overrides

**Classification tasks** (`disease_afib`, `disease_hcm`, `disease_dcm`, `mortality_1yr`):
```yaml
experiment:
  data:
    num_classes: 2  # binary; adjust if multi-class
  optimization:
    num_epochs: 20  # classification converges faster
task_type: classification  # ensure CSV loader uses discrete labels
```

**Regression tasks** (`nt_probnp`, `ef_note_extracted`, `troponin_t`):
```yaml
experiment:
  data:
    num_classes: 1  # regression
  optimization:
    num_epochs: 35  # match NatMed MIMIC regression convention
# Load z-score params from zscore_params.json at runtime:
target_mean: {from zscore_params.json}
target_std:  {from zscore_params.json}
task_type: regression
```

---

## Launch strategy

### Phase 0: Sanity check (1 hour)

Before launching 24 probes, verify the four encoders load correctly into the standard d=1 attentive probe pipeline:

```bash
# Launch a single-head, single-epoch smoke test per encoder
for model in jepa_in21k_e100 byol_e100 mae_e99 salt_v1_e79; do
  python -m evals.main \
    --fname configs/eval/vitl/neurips/${model}_nt_probnp_smoke.yaml \
    --devices cuda:0
done
```

Check: encoder loads, forward pass works, loss decreases over a few steps. Kill after ~100 steps. Fix any loading issues before launching full sweep. Common issues:
- `checkpoint_key` mismatch → wrong tensor shapes
- BYOL's projector/predictor prefix leaking into encoder load → use `encoder.backbone.load_state_dict(..., strict=False)` pattern
- MAE wrapper issues → use the VideoMAE-specific loader module
- SALT `norms_block` keys unexpected → expected; check they appear in the `unexpected_keys` list, not `missing_keys`

### Phase 1: MIMIC probe training (1 HyperPod day)

Launch 4 probes in parallel on 8-GPU node (2 GPUs per probe × 4 probes = 8 GPUs utilized). Chain through the 6 tasks sequentially.

```bash
# Approximate layout (pseudocode — needs a proper sbatch wrapper)
for task in nt_probnp ef_note_extracted disease_afib disease_hcm disease_dcm mortality_1yr troponin_t; do
  parallel -j 4 python -m evals.main --fname configs/eval/vitl/neurips/{}_${task}.yaml --devices cuda:0 cuda:1 ::: \
    jepa_in21k_e100 byol_e100 mae_e99 salt_v1_e79
done
```

Wall-clock estimate per task:
- Regression (35 epochs, ~5K studies): ~1 hour
- Classification (20 epochs): ~40 minutes

Total MIMIC: ~6 hours wall-clock if 4 probes truly parallelize, ~12 hours if they have to run partially sequentially due to memory constraints.

### Phase 2: Supplementary evaluations (~4 hours)

- **CAMUS SALT segmentation decoder** (1 probe, ~2 hours) — clone `scripts/neurips/noised_segmentation.py` or the existing CAMUS training script, swap encoder to SALT v1 e79
- **Pediatric zero-shot SALT inference** (~30 min) — reuse SALT UHN LVEF probe; just run pred-avg inference on `echonet_pediatric_test_*.csv`
- **UHN LVEF 4-way probes** (4 probes, ~4 hours) — clone `configs/eval/vitl/icml/echojepa_l_pt50_lvef_d4.yaml` and swap encoders for init-matched comparison

### Phase 3: Results aggregation (1 hour)

Collect per-task, per-model test-set numbers into a single CSV:
```
task,model,metric,value,seed,n_test
nt_probnp,jepa_in21k_e100,R2,0.XXX,42,5000
nt_probnp,byol_e100,R2,0.XXX,42,5000
...
```

Then compute per-prediction pass/fail, the P6 Spearman correlation, and populate `paper-outline.md` §5.3 "Results by prediction."

---

## Compute budget summary

| Phase | Time | GPUs | Notes |
|---|---|---|---|
| Phase 0 sanity checks | 1 hour | 1 | Pre-launch verification |
| Phase 1 MIMIC probes (24) | 6–12 hours | 8 | Main work |
| Phase 2 supplementary | 4 hours | 4–8 | CAMUS + UHN + Pediatric |
| Phase 3 aggregation | 1 hour | 0 | CPU/analysis |
| **Total** | **~12–18 hours wall-clock** | **1 HyperPod day** | |

This is compatible with the remaining NeurIPS timeline (deadline May 4 2026, ~4 weeks out). Budget the work for the week 2-3 window after the paper-outline.md restructure is reviewed.

---

## Known gotchas

1. **VideoMAE MAE module.** The external VideoMAE codebase uses a different encoder wrapper (`videomae_encoder.py`) with different checkpoint key conventions. **Do not** use `vit_encoder_multiclip` for MAE — use the VideoMAE-specific adapter. Reference: existing ICML MAE probe configs at `configs/eval/vitl/icml/echomae_l_*`.

2. **BYOL encoder key.** BYOL checkpoints save encoder weights under `encoder` (not `target_encoder` — that's JEPA convention). Verify with `torch.load(path, weights_only=False).keys()` before launching.

3. **SALT `norms_block` unexpected keys.** SALT v1 has 8 `norms_block.{0..3}.{weight,bias}` keys that belong to the hierarchical predictor head, not the encoder. These will appear as `unexpected_keys` in the `load_state_dict` message with `strict=False`. This is benign — the encoder blocks load correctly. If they appear in `missing_keys`, something is wrong.

4. **Pred-avg `study_sampling` for EchoNet-Dynamic-style tasks.** EchoNet-Dynamic pred-avg must use `study_sampling: false` (each video is a study). MIMIC tasks can use `study_sampling: true` (DistributedStudySampler, 1 clip per study per epoch). See `salt-comparison.md` § Artifacts Inventory gotcha note.

5. **CSV paths assume EFS mount.** CSV paths in `experiments/nature_medicine/mimic/probe_csvs/{task}/*.csv` assume the EFS mount at `/mnt/custom-file-systems/...`. On HyperPod compute nodes, paths may differ (typically `/opt/dlami/nvme/...` after `deploy.sh` sync). Verify CSV paths resolve on the compute node before launching.

6. **Nature Medicine labels have multiple versions.** `disease_hcm`, `disease_dcm`, `disease_afib` etc. have multiple versions in `probe_csvs/` (e.g. `disease_hcm`, `disease_hcm_v1`, `disease_hcm_v2`, `disease_hcm_v4.1`). **Use the latest version** (typically `_v4.1`) unless a specific reason exists to pin to an older one. Confirm with the Nature Medicine label provenance docs at `experiments/nature_medicine/uhn/DATASET_PROVENANCE.md` (UHN labels) or the MIMIC equivalent.

7. **`target_mean`/`target_std` for regression.** Regression tasks z-score labels at runtime; values must match the training CSV's distribution. See `zscore_params.json` in each task directory. Hardcode these into the probe config (the loader doesn't read the JSON automatically).

---

## What this does NOT cover

- **Bootstrap CIs for statistical significance.** §5 results should include bootstrap 95% CIs for the pairwise comparisons. Run separately after initial results are collected.
- **Multi-seed probes.** The plan above runs one probe per (model, task). For tighter error bars, run 3 seeds per probe — adds ~3× compute. Recommended for the Nature Medicine draft, not strictly required for NeurIPS unless a reviewer asks.
- **Failure-mode analysis.** If predictions P4 (BYOL mortality) or P5 (SALT always last) fail, what does that tell us? This is analysis work to do after the numbers come in, not scoping work.
- **Pre-registration.** If you want to pre-register the predictions publicly (e.g. on OSF or as a dated commit in this repo) before seeing results, do that as a separate step before launching Phase 1.

---

## Readiness checklist before launching

- [ ] `paper-outline.md` §5 predictions agreed and committed (prevent retroactive adjustment)
- [ ] Nature Medicine deconfliction carve-out agreed with co-authors
- [ ] Phase 0 sanity checks passed for all 4 encoders
- [ ] Config templates cloned for 6 MIMIC tasks × 4 models (24 configs)
- [ ] CSV paths resolve on the target compute node
- [ ] HyperPod 8-GPU node reserved for ~1 day
- [ ] Output directory structure decided (e.g. `evals/vitl/neurips/mimic_part2/`)
- [ ] `sbatch` wrapper script written for parallel launch
- [ ] Results aggregation script template ready (Phase 3)

Check all items before launching to avoid wasted GPU time on misconfigured runs.
