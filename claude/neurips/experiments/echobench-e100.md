# EchoBench Noise Robustness — Init-Matched e100 Models (4-Way with Bootstrap CIs)

**Date:** 2026-04-07 (3-model), 2026-04-09 (SALT added + bootstrap CIs)
**Models:** JEPA IN21K e100, BYOL e100, MAE e99, SALT S2v1 e79 (all ImageNet-initialized, trained on MIMIC)

---

## Checkpoints

| Model | Checkpoint path | S3 path |
|-------|----------------|---------|
| JEPA IN21K e100 | `checkpoints/pretrain/mimic/jepa_in21k_e100.pt` | `HYP/runs/jepa_in21k_pretrain_376/checkpoints/e100.pt` |
| BYOL e100 | `checkpoints/pretrain/mimic/echobyol_l_e100.pt` | `HYP/runs/byol_pretrain_resume_342/checkpoints/e100.pt` |
| MAE e99 | `checkpoints/pretrain/mimic/echomae_l_e99.pt` | `HYP/runs/videomae_resume_e54_354/checkpoints/checkpoint-99.pth` |
| SALT S2v1 e79 | `checkpoints/pretrain/mimic/salt_s2v1_e79.pt` | `HYP/runs/salt_s2_pretrain_388/checkpoints/e79.pt` |

**LVEF Probes** (AttentiveRegressor, d=4, 16 heads, 6-head HP grid):

| Model | S3 path |
|-------|---------|
| JEPA | `s3://echodata25/neurips/probes/end_lvef_e100/jepa_in21k_e100/best.pt` |
| BYOL | `s3://echodata25/neurips/probes/end_lvef_e100/byol_e100/best.pt` |
| MAE | `s3://echodata25/neurips/probes/end_lvef_e100/mae_e99/best.pt` |
| SALT | `s3://echodata25/neurips/probes/end_lvef_e100/salt_v1_e79/best.pt` |

**CAMUS Segmentation Decoders:** Trained via `evals/segmentation_frozen/`, 1x1 conv decoders on frozen features. SALT decoder trained using the `salt_s2v1_e79.pt` checkpoint loaded via `vit_encoder_multiclip.py` into standard ViT (not the v2.1 architecture).

---

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/rebuttal/noised_inference_persample.py` | LVEF per-sample noised inference (10 conditions × 1277 videos) |
| `scripts/rebuttal/noised_segmentation_persample.py` | CAMUS per-sample noised segmentation (10 conditions × 100 samples) |
| `scripts/rebuttal/lvef_noised_bootstrap.py` | Bootstrap CIs for LVEF R² from per-sample data |
| `scripts/rebuttal/camus_noised_bootstrap.py` | Bootstrap CIs for CAMUS Dice from per-sample data |
| `scripts/rebuttal/echo_perturbations.py` | Physics-based perturbation implementations |

**Sbatch scripts:**

| Job | Script | Node | Models |
|-----|--------|------|--------|
| 778 | `scripts/camus_noised_persample_node83.sbatch` | ip-10-0-50-83 | JEPA, BYOL, MAE (CAMUS) |
| 779 | `scripts/camus_noised_persample_node184.sbatch` | ip-10-0-50-184 | SALT (CAMUS) |
| 797 | `scripts/lvef_noised_persample_node83.sbatch` | ip-10-0-50-83 | JEPA, BYOL, MAE (LVEF) |
| 828 | `scripts/lvef_noised_persample_node184.sbatch` | ip-10-0-50-184 | SALT (LVEF) |

---

## LVEF Regression — R² with 95% Bootstrap CIs

**Data:** EchoNet-Dynamic test set, 1,277 videos. Single clip per video (16 frames, frame_step=2).
**Bootstrap:** 10K resamples, seed=42, percentile CIs. Paired bootstrap for degradation.

### Absolute R²

| Condition | JEPA | BYOL | MAE | SALT |
|-----------|------|------|-----|------|
| clean | **0.591** [0.538, 0.638] | 0.465 [0.401, 0.523] | 0.445 [0.377, 0.505] | 0.293 [0.215, 0.362] |
| depth_atten/mild | **0.540** [0.482, 0.593] | 0.436 [0.368, 0.496] | 0.354 [0.295, 0.408] | 0.258 [0.193, 0.319] |
| depth_atten/moderate | **0.460** [0.393, 0.521] | 0.395 [0.336, 0.450] | 0.211 [0.162, 0.258] | 0.178 [0.127, 0.227] |
| depth_atten/severe | **0.396** [0.321, 0.463] | 0.342 [0.288, 0.392] | 0.090 [0.051, 0.129] | 0.137 [0.094, 0.179] |
| shadow/mild | **0.581** [0.526, 0.629] | 0.449 [0.384, 0.507] | 0.444 [0.379, 0.503] | 0.282 [0.203, 0.352] |
| shadow/moderate | **0.544** [0.485, 0.597] | 0.414 [0.345, 0.475] | 0.435 [0.372, 0.492] | 0.258 [0.181, 0.327] |
| shadow/severe | **0.457** [0.385, 0.518] | 0.320 [0.240, 0.390] | 0.400 [0.340, 0.455] | 0.208 [0.128, 0.278] |
| haze/mild | **0.588** [0.536, 0.635] | 0.455 [0.391, 0.513] | 0.425 [0.361, 0.482] | 0.281 [0.210, 0.347] |
| haze/moderate | **0.576** [0.523, 0.624] | 0.443 [0.379, 0.500] | 0.343 [0.282, 0.401] | 0.248 [0.179, 0.313] |
| haze/severe | **0.553** [0.498, 0.603] | 0.431 [0.368, 0.488] | 0.159 [0.099, 0.217] | 0.217 [0.145, 0.283] |

### Paired R² Degradation (clean → severe)

| Perturbation | JEPA | BYOL | MAE | SALT |
|---|---|---|---|---|
| Depth attenuation | 0.195 [0.154, 0.239] | 0.123 [0.073, 0.172] | **0.355** [0.292, 0.413] | 0.156 [0.091, 0.214] |
| Acoustic shadow | **0.134** [0.104, 0.168] | 0.145 [0.098, 0.196] | 0.045 [0.014, 0.076] | 0.085 [0.038, 0.133] |
| Haze artifact | 0.038 [0.016, 0.060] | 0.035 [0.013, 0.057] | **0.286** [0.233, 0.334] | 0.076 [0.035, 0.114] |

### Additional Metrics — Severe

**MAE (EF points):**

| Condition | JEPA | BYOL | MAE | SALT |
|---|---|---|---|---|
| clean | **5.77** [5.49, 6.07] | 6.41 [6.07, 6.75] | 6.58 [6.23, 6.92] | 7.35 [6.96, 7.74] |
| depth_atten/severe | **6.86** [6.50, 7.22] | 7.10 [6.72, 7.48] | 7.90 [7.43, 8.37] | 7.90 [7.45, 8.35] |
| shadow/severe | **6.59** [6.25, 6.93] | 7.17 [6.79, 7.56] | 6.85 [6.50, 7.21] | 7.97 [7.57, 8.38] |
| haze/severe | **6.04** [5.74, 6.34] | 6.66 [6.31, 7.02] | 7.80 [7.36, 8.25] | 7.63 [7.22, 8.05] |

**Pearson correlation:**

| Condition | JEPA | BYOL | MAE | SALT |
|---|---|---|---|---|
| clean | **0.771** [0.739, 0.801] | 0.690 [0.647, 0.729] | 0.674 [0.626, 0.716] | 0.564 [0.510, 0.614] |
| depth_atten/severe | **0.650** [0.603, 0.693] | 0.596 [0.547, 0.641] | 0.594 [0.545, 0.639] | 0.433 [0.376, 0.488] |
| shadow/severe | **0.693** [0.650, 0.731] | 0.578 [0.523, 0.629] | 0.639 [0.589, 0.683] | 0.491 [0.433, 0.543] |
| haze/severe | **0.744** [0.709, 0.777] | 0.670 [0.625, 0.711] | 0.632 [0.582, 0.675] | 0.505 [0.446, 0.560] |

---

## CAMUS Segmentation — Dice with 95% Bootstrap CIs

**Data:** CAMUS test set, 50 patients × 2 views = 100 samples.
**Bootstrap:** 10K resamples, seed=42, percentile CIs.

### Absolute Dice

| Condition | JEPA | BYOL | MAE | SALT |
|-----------|------|------|-----|------|
| clean | 0.815 [0.801, 0.829] | 0.823 [0.811, 0.835] | **0.827** [0.814, 0.838] | 0.777 [0.759, 0.794] |
| depth_atten/mild | 0.792 [0.775, 0.808] | 0.721 [0.701, 0.740] | **0.805** [0.790, 0.819] | 0.739 [0.718, 0.758] |
| depth_atten/moderate | 0.746 [0.728, 0.764] | 0.530 [0.508, 0.553] | **0.753** [0.731, 0.773] | 0.656 [0.631, 0.679] |
| depth_atten/severe | **0.683** [0.663, 0.703] | 0.368 [0.345, 0.391] | 0.654 [0.625, 0.681] | 0.508 [0.486, 0.529] |
| shadow/mild | 0.804 [0.789, 0.819] | 0.805 [0.791, 0.818] | **0.815** [0.801, 0.827] | 0.763 [0.743, 0.781] |
| shadow/moderate | 0.778 [0.760, 0.794] | 0.751 [0.732, 0.769] | **0.789** [0.773, 0.804] | 0.734 [0.711, 0.754] |
| shadow/severe | 0.717 [0.697, 0.736] | 0.587 [0.556, 0.616] | **0.737** [0.717, 0.755] | 0.645 [0.621, 0.668] |
| haze/mild | 0.814 [0.799, 0.827] | **0.824** [0.812, 0.836] | 0.825 [0.812, 0.836] | 0.777 [0.758, 0.794] |
| haze/moderate | 0.810 [0.795, 0.824] | **0.823** [0.811, 0.834] | 0.817 [0.805, 0.829] | 0.774 [0.756, 0.791] |
| haze/severe | 0.794 [0.778, 0.808] | **0.815** [0.804, 0.826] | 0.778 [0.763, 0.792] | 0.767 [0.749, 0.785] |

### Paired Dice Degradation (clean → severe, % drop)

| Perturbation | JEPA | BYOL | MAE | SALT |
|---|---|---|---|---|
| Depth attenuation | 16.2% [14.3, 18.3] | **55.3%** [52.9, 57.7] | 20.9% [18.5, 23.5] | 34.7% [31.9, 37.4] |
| Acoustic shadow | 12.1% [10.6, 13.6] | **28.8%** [25.6, 31.9] | **10.9%** [9.5, 12.2] | 17.0% [14.9, 19.1] |
| Haze artifact | 2.7% [2.0, 3.4] | **1.0%** [0.5, 1.4] | 5.9% [5.2, 6.6] | 1.3% [0.8, 1.8] |
| **Avg severe drop** | **10.3%** [9.4, 11.3] | 28.4% [26.8, 29.9] | 12.6% [11.4, 13.8] | 17.6% [16.4, 18.9] |

### Per-Structure Breakdown (severe perturbation, absolute Dice)

**depth_attenuation/severe:**

| Structure | JEPA | BYOL | MAE | SALT |
|---|---|---|---|---|
| LV | 0.763 [0.748, 0.777] | 0.353 [0.325, 0.381] | **0.825** [0.807, 0.842] | 0.720 [0.692, 0.747] |
| MYO | **0.650** [0.623, 0.676] | 0.217 [0.186, 0.249] | 0.627 [0.595, 0.658] | 0.636 [0.615, 0.656] |
| LA | **0.637** [0.605, 0.667] | 0.534 [0.498, 0.568] | 0.509 [0.463, 0.554] | 0.167 [0.133, 0.202] |

**gaussian_shadow/severe:**

| Structure | JEPA | BYOL | MAE | SALT |
|---|---|---|---|---|
| LV | 0.795 [0.776, 0.814] | 0.624 [0.580, 0.665] | **0.817** [0.801, 0.832] | 0.682 [0.650, 0.712] |
| MYO | 0.621 [0.595, 0.646] | 0.542 [0.512, 0.571] | **0.634** [0.606, 0.661] | 0.622 [0.599, 0.644] |
| LA | 0.735 [0.705, 0.763] | 0.594 [0.560, 0.629] | **0.759** [0.730, 0.785] | 0.632 [0.593, 0.667] |

---

## Combined Analysis

### Robustness rankings

| Task | Clean ranking | Robustness ranking (avg severe drop) |
|------|--------------|--------------------------------------|
| **LVEF (functional)** | JEPA > BYOL > MAE > SALT | JEPA > BYOL > SALT > MAE |
| **CAMUS (spatial)** | MAE > BYOL > JEPA > SALT | JEPA (10.3%) > MAE (12.6%) > SALT (17.6%) >> BYOL (28.4%) |

### Key findings (4-model)

1. **JEPA is most robust on both tasks.** On LVEF, JEPA retains more absolute R² under every perturbation. On CAMUS, JEPA has the lowest avg severe drop (10.3%).

2. **Clean performance fails to predict robustness.** MAE leads clean CAMUS (0.827) but drops 12.6% under severe noise. SALT has lowest clean CAMUS (0.777) but drops only 17.6% — less than BYOL (28.4%) despite lower clean.

3. **BYOL collapses under depth attenuation.** On CAMUS, BYOL drops from 0.823 to 0.368 Dice (−55.3%) — the single worst degradation across all model-task pairs.

4. **MAE collapses on functional tasks under noise.** Depth attenuation: R² drops from 0.445 to 0.090 (−0.355). Haze: 0.445 → 0.159 (−0.286). But MAE is relatively robust on spatial tasks (CAMUS shadow: only −10.9%).

5. **SALT sits between JEPA and BYOL** on robustness. Moderate degradation on both tasks. SALT's lower clean performance limits its ceiling, but it doesn't catastrophically collapse like BYOL or MAE.

### Comparison with pt50 results (ICML rebuttal)

| | pt50 LVEF avg drop | e100 LVEF avg drop | pt50 CAMUS avg drop | e100 CAMUS avg drop |
|---|---|---|---|---|
| JEPA | −19% | −20% | −10% | −10% |
| BYOL | −40% | −22% | −25% | −29% |
| MAE | −37% | −51% | −8% | −13% |

**Note:** pt50 JEPA used wrong init (fully-trained 235ep). The e100 init-matched results are the authoritative comparison. SALT not in pt50 experiments.

---

## Errors and Issues

### SALT LVEF checkpoint mismatch (jobs 807, 812 → 828)

**Problem:** SALT LVEF inference initially produced R²=−0.63 (expected 0.293). Predictions clustered in [63, 67] regardless of label.

**Root cause:** The sbatch used `salt_s2_vitl_224px_16f/e79.pt` (on NVMe scratch), but the LVEF probe was trained with `salt_s2v1_e79.pt` (in the repo `checkpoints/` dir). Despite identical epoch numbers and architecture, these are **different training runs**:
- File sizes: 3,912,175,009 vs 3,976,768,109 bytes
- Loss: 0.448 vs 0.429, LR: 0.000255 vs 0.000175
- 299 of 300 encoder tensors differ, max abs diff 0.72
- `salt_s2v1_e79.pt` is from S3 job 388 (the canonical SALT run). `salt_s2_vitl_224px_16f/e79.pt` is a separate training run that happened to reach the same epoch count

**Debugging path:**
1. First suspected missing `norm.weight`/`norm.bias` (SALT v2.1 uses `norms_block` instead of single final LayerNorm). Added `norms_block.3` → `norm` key mapping. R² unchanged at −0.62.
2. Investigated v2.1 ViT architecture differences (mode parameter, modality embeddings). But eval pipeline uses standard ViT with `strict=False` — same as our script.
3. Single-video test showed reasonable prediction (64 vs label 56) but 20-video batch revealed predictions clustered at [63, 67] with Pearson=0.33 — collapsed dynamic range.
4. Compared file sizes of the two checkpoint files → different sizes → compared tensor values → 299/300 tensors differ → **confirmed checkpoint mismatch**.

**Fix:** Changed sbatch to use `${REPO_DIR}/checkpoints/pretrain/mimic/salt_s2v1_e79.pt`. Clean R² immediately matched expected 0.293. The `norms_block.3` mapping was also removed (not needed — default LayerNorm init is what the probe was trained with).

**Lesson:** SALT has multiple training runs with similar naming conventions (`salt_s2_vitl_224px_16f/` vs `salt_s2v1_`). Always verify the checkpoint matches the probe by checking R² on clean data matches the expected value from the eval pipeline.

---

## Output CSVs

### Per-sample CSVs (for bootstrap CIs)

| Model | LVEF CSV | CAMUS CSV |
|-------|---------|-----------|
| JEPA | `jepa_in21k_e100_noised_lvef_persample.csv` | `jepa_in21k_e100_noised_seg_persample.csv` |
| BYOL | `byol_e100_noised_lvef_persample.csv` | `byol_e100_noised_seg_persample.csv` |
| MAE | `mae_e99_noised_lvef_persample.csv` | `mae_e99_noised_seg_persample.csv` |
| SALT | `salt_v1_e79_noised_lvef_persample.csv` | `salt_s2v1_e79_noised_seg_persample.csv` |

**Local path:** `scripts/rebuttal/samples/`

**S3 paths:**
- LVEF: `s3://echodata25/neurips/results/lvef_noised/{filename}`
- CAMUS: `s3://echodata25/neurips/results/camus_frame_shuffle/{filename}`

### CSV formats

**LVEF** (12,771 rows per model = 1 header + 1,277 samples × 10 conditions):
```
sample_idx,condition,prediction,label
0,clean,59.41,55.95
```

**CAMUS** (1,001 rows per model = 1 header + 100 samples × 10 conditions):
```
sample_idx,condition,mean_dice,ed_LV_dice,ed_MYO_dice,ed_LA_dice,es_LV_dice,es_MYO_dice,es_LA_dice,mean_LV_dice,mean_MYO_dice,mean_LA_dice
```

### Aggregated CSVs (original, 3-model only)

| Model | LVEF CSV | CAMUS CSV |
|-------|---------|-----------|
| JEPA | `jepa_in21k_e100_noised_inference.csv` | `jepa_in21k_e100_noised_segmentation.csv` |
| BYOL | `byol_e100_noised_inference.csv` | `byol_e100_noised_segmentation.csv` |
| MAE | `mae_e99_noised_inference.csv` | `mae_e99_noised_segmentation.csv` |
| SALT | — | `salt_s2v1_e79_noised_segmentation.csv` |

---

## Bootstrap methodology

- **R²:** Set-level metric. Bootstrap resamples (prediction, label) pairs (n=1277 for LVEF, n=100 for CAMUS) with replacement, recomputes R²/Dice for each resample. 10K resamples, seed=42.
- **Paired degradation:** Same resampled indices for clean and perturbed conditions, computes metric_clean − metric_perturbed for each resample. This accounts for sample-level correlation between conditions.
- **CIs:** 95% percentile intervals [2.5th, 97.5th percentile].

All CIs computed on the controller (no GPU needed) using `lvef_noised_bootstrap.py` and `camus_noised_bootstrap.py`.
