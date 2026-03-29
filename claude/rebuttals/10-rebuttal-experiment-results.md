# ICML Rebuttal — Experiment Results Tracker (2026-03-29)

Consolidated results from all rebuttal experiments. **Single source of truth** for run status and numbers.
See `08-rebuttal-v2.md` for reviewer concerns, narrative framing, and contingency plans.
See `09-three-way-comparison-results.md` for BYOL architecture audit and interpretation notes.

---

## 1. Three-Way Controlled Comparison (JEPA vs BYOL vs MAE, 50ep)

**Purpose:** Reviewer Concern 3b (contrastive/self-distillation baseline) + Concern 4 (only one controlled comparison).
All models: ViT-L (304M), MIMIC 525K, 50 pretraining epochs. Probes: d=4 attentive, 16 HP heads.

### Checkpoints

| Model | Objective | Checkpoint | Status |
|-------|-----------|-----------|--------|
| EchoJEPA-L (50ep) | Latent prediction | `checkpoints/echojepa-l-pt50.pt` | Done |
| EchoBYOL-L (50ep) | Self-distillation | `checkpoints/byol_vitl_imagenet_v2_e50.pt` | Done |
| EchoMAE-L (50ep) | Pixel reconstruction | — | Planned (retrain with corrected LR 1.5e-4) |

### 1a. LVEF Regression (Single-View, 10K train / 1K val)

| Model | Objective | Best Val MAE | Val R² | Val Pearson | Status |
|-------|-----------|-------------|--------|-------------|--------|
| EchoJEPA-L (50ep) | Latent prediction | 6.329 (ep17) | **0.436** (ep18) | **0.667** (ep17) | DONE |
| EchoBYOL-L (50ep) | Self-distillation | **6.297** (ep18) | 0.421 (post-hoc) | 0.652 (post-hoc) | DONE |
| EchoMAE-L (ep99) | Pixel reconstruction | 8.05 | ~0 | ~0 | DONE (no signal) |
| EchoMAE-L (50ep) | Pixel reconstruction | — | — | — | NOT STARTED |

**Predict-mean baseline MAE:** ~9.0. Z-score: mean=57.07, std=11.28.

**Finding:** JEPA and BYOL near-identical on LVEF (R² 0.436 vs 0.421, Pearson 0.667 vs 0.652). MAE shows zero signal at ep99 (2x more training). Matches "BYOL ~80%+" contingency framing: EMA-based methods both succeed; the shared ingredient is the momentum teacher filtering noise.

**BYOL R²/Pearson:** Originally NaN due to scipy libstdc++ mismatch at runtime. Computed post-hoc via `val_only` inference on best checkpoint (ep18). R² 0.421, Pearson 0.652, best head 1.

<details>
<summary>EchoJEPA-L pt50 LVEF epoch table</summary>

| Epoch | Train MAE | Val MAE | Val R² | Val Pearson |
|-------|-----------|---------|--------|-------------|
| 1 | 8.196 | 9.046 | -0.014 | 0.216 |
| 5 | 7.486 | 7.005 | 0.285 | 0.544 |
| 10 | 7.051 | 6.870 | 0.352 | 0.617 |
| 15 | 6.787 | 6.371 | 0.418 | 0.656 |
| 17 | 6.760 | **6.329** | 0.434 | **0.667** |
| 18 | 6.783 | 6.352 | **0.436** | 0.667 |
| 20 | 6.739 | 6.361 | 0.430 | 0.663 |

</details>

<details>
<summary>EchoBYOL-L pt50 LVEF epoch table</summary>

| Epoch | Train MAE | Val MAE |
|-------|-----------|---------|
| 1 | 8.141 | 8.264 |
| 5 | 7.622 | 7.415 |
| 10 | 7.180 | 6.860 |
| 13 | 7.013 | 6.387 |
| 15 | 6.910 | 6.372 |
| 17 | 6.886 | 6.334 |
| 18 | 6.885 | **6.297** |
| 20 | 6.841 | 6.378 |

</details>

### 1b. RVSP Regression (Multi-View, d=4 factorized, 2 views × 2 clips)

Z-score: mean=34.465, std=14.013.

| Model | Objective | Dataset | Best Val MAE | Val R² | Val Pearson | Status |
|-------|-----------|---------|-------------|--------|-------------|--------|
| EchoJEPA-L (50ep) | Latent prediction | 5K/1K subset | 9.771 (ep20) | 0.092 | 0.376 | DONE (insufficient) |
| EchoJEPA-L (50ep) | Latent prediction | Full 41K/5K | 9.124 (ep12) | 0.235 | 0.485 | **RUNNING** (ep 12/20) |
| EchoBYOL-L (50ep) | Self-distillation | Full 41K/5K | — | — | — | QUEUED |
| EchoMAE-L (ep163) | Pixel reconstruction | Full 41K/5K | 10.529 (ep1) | -0.031 | 0.124 | PAUSED (ep2) |
| EchoMAE-L (50ep) | Pixel reconstruction | — | — | — | — | NOT STARTED |

**Finding (5K subset):** Insufficient data for multi-view RVSP. Pearson plateaued at 0.376, R² peaked at 0.092. All three models should use full 41K.

**Finding (41K, in progress):** Dramatically better. R² 0.235, Pearson 0.485 at epoch 12 — on par with pt210-an25 (R² 0.235, Pearson 0.504 at epoch 9) despite 4× less pretraining. Expect ~0.50+ Pearson by epoch 20.

<details>
<summary>EchoJEPA-L pt50 RVSP — full 41K epoch table (in progress)</summary>

| Epoch | Train MAE | Val MAE | Val R² | Val Pearson |
|-------|-----------|---------|--------|-------------|
| 1 | 9.823 | 10.544 | -0.007 | 0.206 |
| 2 | 9.568 | 9.882 | 0.079 | 0.336 |
| 3 | 9.320 | 9.701 | 0.138 | 0.382 |
| 4 | 9.183 | 9.839 | 0.167 | 0.419 |
| 6 | 9.006 | 9.339 | 0.133 | 0.445 |
| 8 | 8.864 | 9.344 | 0.161 | 0.452 |
| 10 | 8.788 | 9.234 | 0.221 | 0.475 |
| 12 | 8.717 | 9.124 | 0.235 | 0.485 |

</details>

<details>
<summary>EchoJEPA-L pt50 RVSP — 5K subset FINAL</summary>

| Epoch | Train MAE | Val MAE | Val R² | Val Pearson |
|-------|-----------|---------|--------|-------------|
| 1 | 10.139 | 10.595 | -0.040 | 0.044 |
| 4 | 9.774 | 10.604 | -0.003 | 0.065 |
| 8 | 9.610 | 10.550 | -0.106 | 0.210 |
| 12 | 9.351 | 10.197 | 0.084 | 0.313 |
| 16 | 9.117 | 9.858 | 0.067 | 0.368 |
| 20 | 9.044 | 9.771 | 0.089 | 0.376 |

</details>

**Reference — pt210-an25 RVSP on full 41K (prior session, killed at ep9):**

| Epoch | Train MAE | Val MAE | Val R² | Val Pearson |
|-------|-----------|---------|--------|-------------|
| 1 | 9.825 | 10.525 | -0.034 | 0.167 |
| 5 | 8.927 | 9.284 | 0.199 | 0.459 |
| 9 | 8.703 | 9.040 | 0.235 | 0.504 |

### 1c. CAMUS Segmentation (Frozen Linear Decoder)

**Decoder:** 1×1 conv + bilinear upsample (~4.1K params), 50 epochs, 7 HP configs.

**Fully-trained model results (from prior runs):**

| Model | Pretrain Epochs | LV Dice | MYO Dice | LA Dice | Mean Dice |
|-------|----------------|---------|----------|---------|-----------|
| EchoJEPA-L (210+25) | 235 | 0.884 | 0.762 | 0.807 | **0.818** |
| EchoMAE-L (163) | 163 | 0.852 | 0.735 | 0.783 | 0.790 |
| EchoJEPA-L-K (220+55) | 275 | 0.811 | 0.687 | 0.739 | 0.746 |
| PanEcho | — | 0.814 | 0.652 | 0.736 | 0.734 |
| EchoJEPA-G 384px | 361 | 0.853 | 0.606 | 0.726 | 0.729 |
| EchoPrime | — | 0.774 | 0.579 | 0.654 | 0.669 |

**50-epoch controlled comparison:**

| Model | Objective | Best Val Dice | Test Dice | Status |
|-------|-----------|--------------|-----------|--------|
| EchoJEPA-L (50ep) | Latent prediction | 0.818 (ep48) | **0.815** | DONE |
| EchoBYOL-L (50ep) | Self-distillation | — | — | NOT STARTED |
| EchoMAE-L (50ep) | Pixel reconstruction | — | — | NOT STARTED |

**Per-structure test Dice (EchoJEPA-L pt50, best config lr=5e-2, wd=1e-4):**

| Structure | ED Dice | ES Dice | Mean Dice | ED HD95 | ES HD95 |
|-----------|---------|---------|-----------|---------|---------|
| LV | 0.898 | 0.859 | **0.878** | 6.54 | 6.46 |
| MYO | 0.755 | 0.765 | **0.760** | 10.59 | 8.28 |
| LA | 0.778 | 0.836 | **0.807** | 11.77 | 10.37 |

**Finding:** pt50 nearly matches the fully-trained pt210-an25 (test Dice 0.815 vs 0.818, Δ=0.3pp). Per-structure gaps are negligible: LV 0.878 vs 0.884 (-0.6pp), MYO 0.760 vs 0.762 (-0.2pp), LA 0.807 vs 0.807 (0pp). 50 pretraining epochs already capture nearly all spatial feature quality needed for dense prediction — consistent with the RVSP finding (§5d) that pt50 shows diminishing returns vs longer training.

<details>
<summary>Full HP grid results (7 configs)</summary>

| Config | LR | WD | Val Dice | Test Dice |
|--------|-----|------|----------|-----------|
| **lr5e-02_wd1e-04** | 5e-2 | 1e-4 | **0.818** | **0.815** |
| lr5e-02_wd1e-02 | 5e-2 | 1e-2 | 0.816 | 0.815 |
| lr2e-02_wd1e-04 | 2e-2 | 1e-4 | 0.814 | 0.812 |
| lr1e-02_wd1e-04 | 1e-2 | 1e-4 | 0.810 | 0.810 |
| lr5e-03_wd1e-04 | 5e-3 | 1e-4 | 0.806 | 0.806 |
| lr1e-03_wd1e-02 | 1e-3 | 1e-2 | 0.784 | 0.790 |
| lr1e-03_wd1e-04 | 1e-3 | 1e-4 | 0.784 | 0.789 |

</details>

---

## 2. Model Scaling Analysis (B -> L -> G)

**Purpose:** Reviewer Concern 1 (novelty) — latent prediction benefits from scale.

### LVEF (10K train / 1K val, d=4 attentive probe)

| Model | Params | Architecture | Pretrain Data | Best Val MAE | Val R² | Val Pearson | Status |
|-------|--------|-------------|---------------|-------------|--------|-------------|--------|
| EchoJEPA-B (V-JEPA 2.1) | 86M | ViT-B | MIMIC 525K | **5.244** | **0.650** | **0.806** | DONE |
| EchoJEPA-L (50ep, V-JEPA 2.0) | 304M | ViT-L | MIMIC 525K | 6.329 | 0.436 | 0.667 | DONE |
| EchoJEPA-G (V-JEPA 2.0) | 1,012M | ViT-g | UHN 18M | — | 0.778 (NM) | — | NM result |

**Caveat:** B uses V-JEPA 2.1 (dense loss, multi-layer heads) and 229 total epochs; L pt50 uses V-JEPA 2.0 and 50 epochs. B→L is NOT a clean scaling comparison. L→G (both 2.0) is cleaner but confounds data scale (MIMIC vs UHN).

**Key finding:** Even small (86M) JEPA models show strong clinical signal (R²=0.650). B outperforms L pt50 due to more pretraining + improved architecture version.

---

## 3. EchoMAE-L Baseline Probes

**Purpose:** Establish MAE performance for comparison.

| Task | Checkpoint | Best Val Metric | Notes |
|------|-----------|----------------|-------|
| LVEF (5K) | ep99 | R² ~0, MAE 8.05 | **No signal.** MAE objective fails for hemodynamics. |
| View (5K) | ep99 | Acc 44.1%, AUROC **0.847** | Good. MAE encodes spatial appearance but not dynamics. |
| RVSP (41K) | ep163 | MAE 10.53 (ep1) | PAUSED at ep2. Early Pearson 0.124 suggests some signal unlike LVEF. |

---

## 4. Execution Status Summary

### Currently Running

| Experiment | GPU | Progress | ETA |
|-----------|-----|----------|-----|
| EchoJEPA-L pt50 RVSP (full 41K) | 8×A100 | Epoch 12/20 | ~6.5h (~48 min/epoch) |

### Queued

| Experiment | Waiting For | Config |
|-----------|-------------|--------|
| EchoBYOL-L pt50 RVSP (full 41K) | JEPA RVSP finish + GPU availability | To be created (use full 41K, not 5K) |

### Completed

| Experiment | Key Result | Date |
|-----------|-----------|------|
| EchoJEPA-L pt50 LVEF (10K, 20ep) | R²=0.436, Pearson=0.667, MAE=6.329 | 2026-03-29 |
| EchoBYOL-L pt50 LVEF (10K, 20ep) | R²=0.421, Pearson=0.652, MAE=6.297 | 2026-03-29 |
| EchoJEPA-L pt50 RVSP (5K, 20ep) | R²=0.092, Pearson=0.376 (insufficient data) | 2026-03-29 |
| EchoJEPA-B LVEF (10K, 19ep) | R²=0.650, Pearson=0.806, MAE=5.244 | 2026-03-28 |
| EchoMAE-L ep99 LVEF (5K, 20ep) | R²~0, MAE=8.05 (no signal) | 2026-03-28 |
| EchoMAE-L ep99 View (5K, 20ep) | Acc=44.1%, AUROC=0.847 | 2026-03-28 |
| CAMUS seg (6 fully-trained models) | JEPA-L=0.818, MAE=0.790 (+2.8pp) | 2026-03-27 |
| EchoJEPA-L pt50 CAMUS (50ep, 7 HP) | Test Dice=0.815, Val Dice=0.818 (ep48) | 2026-03-29 |

### Paused

| Experiment | Reason | How to Resume |
|-----------|--------|---------------|
| EchoMAE-L ep163 RVSP (full 41K) | GPU priority | Set `resume_checkpoint: true`, relaunch |

### Not Started

| Experiment | Priority | Notes |
|-----------|----------|-------|
| EchoMAE-L 50ep retrain | High | Corrected LR (1.5e-4). Needed for clean 3-way comparison. |
| EchoBYOL-L pt50 CAMUS seg | Medium | After RVSP comparison is done |
| ~~EchoJEPA-L pt50 CAMUS seg~~ | ~~Medium~~ | DONE — Test Dice 0.815 |
| CKA speckle invariance (all models) | High (Tier 1) | Hours. Reviewer ncQn. |
| Frame shuffling temporal ablation (all models) | High (Tier 1) | Hours. All reviewers. |
| Noise-level linear probe (all models) | High (Tier 1) | Hours. Reviewer ncQn. |

---

## 5. Key Findings So Far

### 5a. JEPA vs BYOL: Near-Identical on LVEF, RVSP Will Differentiate

LVEF (10K subset): JEPA R²=0.436 vs BYOL R²=0.421 (Pearson 0.667 vs 0.652). The 1.5pp gap is not meaningful — both EMA-based methods succeed equally on a global cardiac function metric. The shared ingredient is the momentum teacher filtering speckle noise.

RVSP (multi-view, spatial reasoning) is where JEPA's local prediction should pull ahead over BYOL's global pooling. Full 41K run in progress; BYOL RVSP queued.

### 5b. MAE Fails for Hemodynamics, Succeeds for Appearance

EchoMAE-L ep99 shows R²~0 on LVEF but AUROC=0.847 on view classification. Pixel reconstruction encodes spatial appearance (which view was recorded) but not hemodynamic function (how well the heart pumps). This directly supports the paper's thesis.

### 5c. 5K Subset Insufficient for Multi-View RVSP

RVSP on 5K: Pearson 0.376. RVSP on 41K: Pearson 0.485 at epoch 12 (still climbing). The 8× more data dramatically improves both R² (0.092→0.235) and Pearson. All controlled comparison models should use the full 41K dataset.

### 5d. pt50 Matches pt210-an25 on RVSP (Full 41K)

Despite 4× less pretraining, pt50 on full 41K matches pt210-an25's performance at comparable epochs (R² 0.235, Pearson 0.485 vs 0.504). This suggests the 50-epoch checkpoint already captures most of the RVSP-relevant information — more pretraining helps but has diminishing returns.

### 5f. pt50 Matches Fully-Trained on CAMUS Segmentation

EchoJEPA-L pt50 test Dice 0.815 vs fully-trained pt210-an25 test Dice 0.818 (Δ=0.3pp). The gap is negligible across all structures (LV -0.6pp, MYO -0.2pp, LA 0pp). Combined with RVSP (§5d), this confirms that 50 pretraining epochs capture most representation quality — the 50-epoch controlled comparison is not handicapped by insufficient pretraining.

### 5e. Rebuttal Narrative: "EMA Targets Filter Noise"

The emerging story is: EMA-based methods (JEPA, BYOL) >> pixel reconstruction (MAE). Within EMA methods, JEPA's local prediction provides spatial precision advantages on dense tasks (CAMUS: 0.818 vs TBD for BYOL), while BYOL's global pooling is sufficient for global metrics (LVEF). The novel finding is "EMA targets filter noise in stochastic domains" — a general SSL principle, not just "JEPA beats everything."

---

## 6. Config and Checkpoint Reference

| Experiment | Config | Encoder Checkpoint | Data |
|-----------|--------|-------------------|------|
| EchoJEPA-L pt50 LVEF | `configs/eval/vitb/icml/echojepa_l_pt50_lvef_d4.yaml` | `echojepa-l-pt50.pt` | 10K/1K rebuttal |
| EchoBYOL-L pt50 LVEF | `configs/eval/vitb/icml/echobyol_l_pt50_lvef_d4.yaml` | `byol_vitl_imagenet_v2_e50.pt` | 10K/1K rebuttal |
| EchoJEPA-L pt50 RVSP (5K) | `configs/eval/vitl/icml/echojepa_l_pt50_rvsp_d4.yaml` | `echojepa-l-pt50.pt` | 5K/1K rebuttal |
| EchoJEPA-L pt50 RVSP (41K) | `configs/eval/vitl/icml/echojepa_l_pt50_rvsp_d4_full.yaml` | `echojepa-l-pt50.pt` | 41K/5K full |
| EchoJEPA-B LVEF | `configs/eval/vitb/icml/echojepa_b_lvef_d4.yaml` | `vjepa2_1_vitb_mimic_p169_c60.pt` | 10K/1K rebuttal |
| EchoMAE-L ep99 LVEF | `configs/eval/vitb/icml/echomae_l_lvef_d4.yaml` | `echomae_l_mimic_ep99.pth` | 5K subset |
| EchoMAE-L ep99 View | `configs/eval/vitb/icml/echomae_l_view_d4.yaml` | `echomae_l_mimic_ep99.pth` | 5K subset |
| EchoMAE-L ep163 RVSP | `configs/eval/vitb/icml/echomae_l_rvsp_d4_ep163.yaml` | `videomae-ep163.pth` | 41K/5K full |

### Known Bugs Encountered

| Bug | Impact | Fix |
|-----|--------|-----|
| Bug 017: Multi-view missing z-score | All ICML RVSP numbers invalid | Runtime z-scoring added to multi-view eval.py |
| Bug 017b: Stale zscore_params.json | Wrong task params silently poison results | Explicit `target_mean`/`target_std` in all YAML configs |
| Bug 018: Port collision → single-GPU fallback | 5K RVSP took same time as 41K (world_size=1) | Set `MASTER_PORT` env var |
| Bug 019: Orphan GPU processes | 19 processes accumulated, blocked ports | Kill ppid=1 orphans before relaunch |
| scipy libstdc++ mismatch | BYOL R²/Pearson NaN at runtime | `LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH` |
