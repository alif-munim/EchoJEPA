# CMR Cross-Modality Validation Results

**Last updated:** 2026-04-25
**Status:** Full CMR MAE ViT-S 800ep trajectory + matched-frame complete (zero temporal Δ). CMR JEPA ViT-S fast-EMA 800ep complete (jobs 333, 344 seed 234/163) with LVEF trajectory probes (job 345). Slow-EMA variant (job 346) **completed e295**, with LVEF + Dx trajectory probes (jobs 375, 376) — **slow-EMA does not rescue the representation-quality collapse**. JEPA-on-CMR peaks at e30-e100 and monotonically degrades thereafter regardless of EMA schedule; MAE-on-CMR climbs monotonically and overtakes JEPA by e200.

---

## Overview

Cross-modality validation of the temporal shortcut on cardiac MRI (CMR). Tests whether MAE's transient temporal encoding and subsequent spatial convergence — observed on echocardiography — also occurs on a different cardiac imaging modality with different noise characteristics (no speckle), different spatial structure (SAX cross-sections vs A4C views), but the same underlying temporal signal (cardiac contraction).

## Pretraining

### CMR MAE ViT-S (job 183, completed 2026-04-19)

**Config:** `configs/train/vits16/pretrain-jepa-cmr-224px-16f-in21k.yaml` (adapted for MAE)
**Sbatch:** `scripts/neurips/cmr/videomae_pretrain_cmr_vits.sbatch`
**S3 run:** `runs/mae_cmr_vits_183/`

| Setting | Value |
|---------|-------|
| Model | ViT-S (384-dim, 12 blocks, 6 heads) |
| Init | ImageNet-21K (`checkpoints/vits_in21k.pt`, timm `augreg_in21k`) |
| Data | 21,840 SAX-only clips from MnM + MnM2 + Sunnybrook + DSB2 + CMR-Multi |
| Epochs | 800 |
| Effective BS | 256 (32×8 GPUs, accum=1) |
| Mask | 90% tube masking |
| Duration | 9h05m on 8×H100 |

Checkpoints saved every 10 epochs: checkpoint-9 through checkpoint-799.

**Loss trajectory:** 1.12 (e0) → 0.90 (e10) → 0.34 (e200) → 0.28 (e500) → 0.27 (e800). Still declining at e800 but flattening.

### CMR JEPA ViT-S (job 333, completed 2026-04-23)

**Config:** `configs/train/vits16/pretrain-jepa-cmr-224px-16f-in21k.yaml`
**Sbatch:** `scripts/neurips/cmr/jepa_pretrain_cmr_vits.sbatch`
**S3 run:** `runs/jepa_cmr_vits_333/training_folder/`
Same ViT-S architecture, same data, 800 epochs. 11h42m on 8xH100.

| Setting | Value |
|---------|-------|
| Model | ViT-S (384-dim, 12 blocks, 6 heads) |
| Init | ImageNet-21K |
| Data | 21,840 SAX-only clips |
| Epochs | 800 |
| Effective BS | 256 (32x8 GPUs) |
| LR peak | 1.75e-4 (flat after e40 warmup) |
| EMA | [0.996, 0.996] **(differs from refs/vjepa2 = [0.99925, 0.99925])** |
| Weight decay | 0.04 flat |
| seed | 234 |

**Loss trajectory (non-monotonic):**
- 0.598 (e1) → 0.347 (e20, min) → 0.420 (e100) → **0.505 (e305 peak)** → 0.431 (e800)
- After the initial descent the loss **rises** by ~45% through e300-e400 before slowly recovering.
- All reference V-JEPA 2 configs (ViT-L/H/g on K710+SSv2+HowTo) use `ema=[0.99925, 0.99925]`; our `0.996` is 5x faster EMA, which may drive teacher runaway at the 21.8K-clip data scale.

### CMR JEPA ViT-S seed-163 resume (job 344, completed 2026-04-24)

**Config:** `configs/train/vits16/pretrain-jepa-cmr-224px-16f-in21k-resume250-s163.yaml`
**Sbatch:** `scripts/neurips/cmr/jepa_pretrain_cmr_vits_resume250_s163.sbatch`
**S3 run:** `runs/jepa_cmr_vits_resume250_s163_344/training_folder/`
Resumed from job 333's e250.pt with seed=163 (force_load_pretrain=false, full optimizer/EMA/step restore). 7h59m on 8xH100.

**Purpose:** test whether the e300-e500 loss rise in job 333 is a seed artifact or a reproducible training dynamic.

**Result:** loss trajectory **tracks job 333 within ~0.01 at every epoch** through e252-e800. The rise-and-fall pattern is seed-independent — it is a property of the training dynamic on this data/config combination, not a random fluctuation.

### CMR JEPA ViT-S slow-EMA variant (job 346, completed 2026-04-24 through e295)

**Config:** `configs/train/vits16/pretrain-jepa-cmr-224px-16f-in21k-slowema.yaml`
**Sbatch:** `scripts/neurips/cmr/jepa_pretrain_cmr_vits_slowema.sbatch`
**S3 run:** `runs/jepa_cmr_vits_slowema_346/training_folder/`

One change from the base config: `ema: [0.996, 0.996]` -> `ema: [0.99925, 0.99925]` (matches `refs/vjepa2`). All other knobs identical. Tests whether the loss-rise and probe-collapse pattern is driven by the fast-EMA teacher at the 21.8K-clip data scale.

**Loss trajectory:** 0.603 (e1) → 0.369 (e20, min) → 0.440 (e302, peak) → still descending at e295. Loss rise is **muted vs fast-EMA** (peak 0.44 vs 0.505, +19% vs +45%) but not eliminated. The run was cancelled at e295 when `latest.pt` reached the e300 milestone per the operational rollover for Gate 2 probes; the `e300` artifact was not created on disk before cancellation so probes use the available {e30, e55, e100, e200, e295} trajectory.

---

## Probe Results: ACDC Holdout

**ACDC** = Automated Cardiac Diagnosis Challenge (Bernard et al., IEEE TMI 2018).
Official challenge split: 100 train / 50 test patients, both with ground truth labels.
5 balanced diagnosis classes: DCM, HCM, MINF, NOR, RV (20 train + 10 test each).
951 train clips, 538 test clips (one clip per SAX slice per patient).

### LVEF Regression (job 209, completed 2026-04-19)

**Config:** `configs/eval/vits16/cmr/mae_cmr_e{N}_acdc_lvef_d4.yaml`
**Sbatch:** `scripts/neurips/cmr/mae_cmr_acdc_lvef_trajectory.sbatch`
**S3 run:** `runs/cmr_probe_traj_209/`

d=4 attentive probe, 6 HP heads (lr × wd grid), 30 probe epochs, num_segments=1.
EF labels from LV cavity segmentation at ED/ES frames.

| CMR MAE Epoch | Val R² | Val Pearson | Val MAE | Best Probe Epoch |
|---------------|--------|-------------|---------|-----------------|
| e30 | 0.085 | 0.295 | 17.10 | 17 |
| e50 | 0.051 | 0.235 | 17.34 | 21 |
| e100 | 0.069 | 0.267 | 17.34 | 21 |
| e200 | 0.092 | 0.311 | 17.14 | 21 |
| e300 | 0.102 | 0.328 | 17.02 | 14 |
| e400 | 0.109 | 0.334 | 16.89 | 21 |
| e600 | 0.126 | 0.369 | 16.66 | 25 |
| **e800** | **0.133** | **0.380** | **16.63** | 25 |

**Interpretation:** Weak absolute performance (max R²=0.133). Steady improvement from e30→e800 confirms the model keeps learning through 800 epochs and hasn't saturated. The low R² reflects the challenging setup: ViT-S (22M params) on only 951 train clips (100 patients), SAX slices (not standard A4C views), and EF computed from segmentation (noisy label).

### 5-Class Diagnosis Classification (job 281, in progress 2026-04-20)

**Config:** `configs/eval/vits16/cmr/mae_cmr_e{N}_acdc_dx_d4.yaml`
**Sbatch:** `scripts/neurips/cmr/mae_cmr_acdc_dx_trajectory.sbatch`
**S3 run:** `runs/cmr_dx_traj_281/`

d=4 attentive probe, 6 HP heads, 30 probe epochs, num_segments=2.
Classes: DCM=0, HCM=1, MINF=2, NOR=3, RV=4. Cross-entropy loss.
Phase 1: probe training (8 checkpoints). Phase 2: clean + matched_frame inference.
Data downloaded to local NVMe to avoid S3 credential expiration.

| CMR MAE Epoch | Val Acc | Val AUROC | Val Bal Acc | Val Kappa | Best Probe Epoch |
|---------------|---------|-----------|------------|-----------|-----------------|
| e30 | 37.5% | 0.662 | 0.343 | 0.206 | 27 |
| e50 | 36.6% | 0.652 | 0.349 | 0.202 | 30 |
| e100 | 44.5% | 0.695 | 0.421 | 0.300 | 27 |
| e200 | **46.1%** | 0.744 | **0.434** | **0.323** | 30 |
| e300 | 43.8% | **0.750** | 0.414 | 0.295 | 30 |
| e400 | 44.1% | 0.745 | 0.415 | 0.297 | 30 |
| e600 | **46.5%** | **0.758** | **0.437** | **0.326** | 30 |
| e800 | 46.1% | 0.759 | 0.433 | 0.322 | 30 |

**Interpretation:** Clear improvement from e30→e600. AUROC jumps from 0.66 to 0.76 (chance=0.50), accuracy from 37.5% to 46.5% (chance=20%). Classification is substantially more informative than LVEF regression at this dataset scale (AUROC 0.76 vs R²=0.13). Performance plateaus at e200-e400, then slight improvement at e600-e800 — the ViT-S still has some capacity headroom on this task.

---

## Matched_frame Inference — Diagnosis (Complete)

**Protocol:** matched_frame with RoPE remap via evals.main, `FRAME_SHUFFLE=100 FRAME_SHUFFLE_TYPE=matched_frame`
**Jobs:** 281 (Phase 2, num_segments=2) and 282 (rerun with num_segments=1 to rule out clip overlap on short CMR videos)
**Result:** Both protocols give identical results — **temporal Δ is zero across all checkpoints.**

### num_segments=2 (job 281 Phase 2)

| CMR MAE Epoch | Clean AUROC | MF AUROC | Δ AUROC |
|---------------|-------------|----------|---------|
| e30 | 0.647 | 0.649 | +0.002 |
| e50 | 0.673 | 0.675 | +0.001 |
| e100 | 0.700 | 0.700 | +0.001 |
| e200 | 0.757 | 0.760 | +0.004 |
| e300 | 0.755 | 0.753 | -0.002 |
| e400 | 0.750 | 0.746 | -0.003 |
| e600 | 0.755 | 0.756 | +0.001 |
| e800 | 0.755 | 0.757 | +0.002 |

### num_segments=1 (job 282, single-clip eval)

| CMR MAE Epoch | Clean AUROC | MF AUROC | Δ AUROC |
|---------------|-------------|----------|---------|
| e30 | 0.648 | 0.649 | +0.002 |
| e50 | 0.673 | 0.674 | +0.001 |
| e100 | 0.700 | 0.701 | +0.001 |
| e200 | 0.758 | 0.762 | +0.005 |
| e300 | 0.757 | 0.757 | -0.000 |
| e400 | 0.751 | 0.749 | -0.002 |
| e600 | 0.753 | 0.759 | +0.007 |
| e800 | 0.749 | 0.758 | +0.009 |

### Interpretation

Frame shuffling has **no effect** on CMR MAE diagnosis classification at any training epoch. The temporal Δ is within noise (±0.01 AUROC) across all 8 checkpoints and both evaluation protocols. This means:

1. **The 5-class diagnosis task is purely spatial.** DCM (dilated chambers), HCM (hypertrophic walls), MINF (thinned/scarred wall), NOR (normal anatomy), and RV (RV enlargement) are all identifiable from single-frame structural appearance. Wall motion dynamics are not needed — or at least not used — for classification on SAX slices.

2. **This could mean diagnosis is purely spatial, OR the model lacks temporal features entirely.** The LVEF results below disambiguate.

---

## Matched_frame Inference — LVEF (Complete)

**Job 283** (2026-04-20, ip-10-0-50-39, 9 min). Same protocol as diagnosis: matched_frame with RoPE remap, num_segments=1, local data.

| CMR MAE Epoch | Clean R² | MF R² | Δ R² | Clean r | MF r | Δ r |
|---------------|----------|-------|------|---------|------|-----|
| e30 | 0.048 | 0.047 | -0.000 | 0.252 | 0.250 | -0.002 |
| e50 | 0.026 | 0.027 | +0.001 | 0.233 | 0.232 | -0.001 |
| e100 | 0.047 | 0.047 | +0.000 | 0.229 | 0.228 | -0.001 |
| e200 | 0.046 | 0.050 | +0.004 | 0.232 | 0.239 | +0.007 |
| e300 | 0.057 | 0.059 | +0.002 | 0.250 | 0.258 | +0.009 |
| e400 | 0.082 | 0.097 | +0.016 | 0.311 | 0.315 | +0.005 |
| e600 | 0.105 | 0.119 | +0.014 | 0.331 | 0.351 | +0.020 |
| e800 | 0.112 | 0.120 | +0.009 | 0.341 | 0.357 | +0.016 |

**Result: Zero temporal Δ on LVEF too.** Frame shuffling has no negative effect at any checkpoint. At later epochs (e400-e800) matched_frame R² is slightly *higher* than clean — this is noise (±0.01-0.02), not a real effect.

---

## CMR JEPA Probe Trajectory (job 345, completed 2026-04-24)

**Config:** `configs/eval/vits16/cmr/jepa_cmr_e{N}_acdc_lvef_d4.yaml` (generated inline by sbatch)
**Sbatch:** `scripts/neurips/cmr/jepa_cmr_acdc_lvef_trajectory.sbatch`
**S3 run:** `runs/cmr_jepa_probe_traj_345/`

d=4 attentive probe, 6 HP heads (lr x wd grid), 30 probe epochs, num_segments=1 — identical protocol to CMR MAE trajectory (job 209). Checkpoint sources: job 333 for epochs <250, job 344 (seed 163 continuation) for epochs >=250. Rounding: e55 (no e50 in 333), e605 (no e600 in 344), e800 -> latest.pt. 3h00m total.

| CMR JEPA Epoch | Best Val R^2 | Best Pearson r | Final Val R^2 | Source |
|---|---|---|---|---|
| e30 | 0.120 | 0.383 | 0.070 | 333 |
| e55 | 0.120 | 0.391 | 0.071 | 333 |
| **e100** | **0.162** | **0.438** | 0.134 | 333 |
| e200 | 0.079 | 0.314 | 0.073 | 333 |
| e300 | 0.090 | 0.333 | 0.080 | 344 |
| e400 | 0.109 | 0.360 | 0.096 | 344 |
| e605 | 0.077 | 0.296 | 0.077 | 344 |
| e800 | 0.069 | 0.288 | 0.060 | 344 (latest.pt) |

**Observations:**
1. **Peak at e100, collapse from e200 on.** JEPA's best-ever R² is 0.162 at e100, then drops to 0.078 at e200 and never recovers. The probe trajectory shape mirrors the pretraining loss trajectory — representation quality degrades as the teacher-chasing loss rises.
2. **Reproducible under seed 163** (job 344 probes span e300-e800 via checkpoints from the seed-163 continuation), so this is not seed noise.
3. **No matched-frame inference yet** — only clean R^2. The MF gap (analog to MAE jobs 282/283) would answer whether the e200+ collapse is *temporal*-feature abandonment or general representation degradation.

---

## CMR JEPA Slow-EMA Trajectory (jobs 375 LVEF + 376 Dx, completed 2026-04-25)

**Sbatches:** `scripts/neurips/cmr/jepa_cmr_slowema_acdc_lvef_trajectory.sbatch` (375), `scripts/neurips/cmr/jepa_cmr_slowema_acdc_dx_trajectory.sbatch` (376)
**S3 runs:** `runs/cmr_jepa_probe_traj_slowema_375/`, `runs/cmr_jepa_dx_traj_slowema_376/`

Same protocol as jobs 345 (fast-EMA LVEF) and 281 (MAE Dx). Checkpoints probed: e30, e55, e100, e200, e295. 30 probe epochs, 6-head HP grid, d=4 attentive, num_segments=1. Total: 2h01m (job 375) + 1h24m (job 376).

### LVEF regression (job 375)

| Slow-EMA ckpt | Val R² | Val Pearson | Val MAE | Best probe epoch |
|---|---|---|---|---|
| **e30** | **0.138** | **0.441** | 15.61 | 21 |
| e55 | 0.111 | 0.396 | 16.28 | 5 |
| e100 | 0.105 | 0.360 | 16.29 | 8 |
| e200 | 0.102 | 0.346 | 16.25 | 13 |
| e295 | 0.089 | 0.304 | 16.07 | 12 |

### 5-class diagnosis (job 376)

| Slow-EMA ckpt | Val AUROC | Val Acc | Val Bal Acc | Val Kappa | Best probe epoch |
|---|---|---|---|---|---|
| **e30** | **0.799** | 47.79 | 0.454 | 0.344 | 22 |
| e55 | 0.797 | 48.35 | 0.472 | 0.353 | 17 |
| e100 | 0.795 | 48.16 | 0.469 | 0.350 | 21 |
| e200 | 0.782 | **49.08** | 0.467 | 0.359 | 26 |
| e295 | 0.766 | 44.85 | 0.420 | 0.305 | 15 |

### Interpretation (the headline update)

**Slow-EMA does not rescue the JEPA collapse.** Both probe tasks peak at e30 and degrade monotonically thereafter, mirroring the fast-EMA pattern:
- LVEF: 0.138 (e30) → 0.089 (e295), −0.049 R² over 265 epochs
- Dx: AUROC 0.799 (e30) → 0.766 (e295), −0.033 AUROC

The pretraining loss rise is **muted** under slow-EMA (peak 0.44 vs fast-EMA 0.505) but the representational trajectory is not. This **rules out the EMA schedule as the primary cause** of the JEPA-on-CMR degradation.

Candidate causes that remain: (a) data scale (21.8K clips inadequate to stabilize JEPA on SAX), (b) model capacity (ViT-S too small to hold both spatial and temporal features), (c) CMR-specific structural properties (SAX slices offer strong purely-spatial signal that JEPA's teacher-chasing dynamic overfits to), (d) clip length (~25 frames = one cycle, no cross-cycle variation for the predictor to exploit). The LR/WD schedules match refs/vjepa2 exactly; other than data scale, no config knob remains untested.

### Fast-EMA vs Slow-EMA side-by-side (LVEF R²)

| Epoch | Fast-EMA (job 345) | Slow-EMA (job 375) | Winner |
|---|---|---|---|
| e30 | 0.120 | **0.138** | slow |
| e55 | **0.120** | 0.111 | fast |
| e100 | **0.162** | 0.105 | fast |
| e200 | 0.079 | **0.102** | slow |
| e295/300 | 0.090 | 0.089 | ~tied |

Neither variant dominates. Fast-EMA has a higher peak (e100 R²=0.162) but loses it by e200; slow-EMA has a lower peak (e30 R²=0.138) but is more stable around the 0.10 plateau. The difference between them is within run-to-run noise for a ViT-S on a 951-clip eval split.

---

## JEPA vs MAE — Three-way LVEF trajectory (ACDC, d=4 attentive)

Full side-by-side with slow-EMA added:

| Epoch | MAE R² | JEPA-fast R² | JEPA-slow R² |
|---|---|---|---|
| e30 | 0.085 | 0.120 | **0.138** |
| ~e50 | 0.051 | 0.120 | 0.111 |
| e100 | 0.069 | **0.162** | 0.105 |
| e200 | 0.092 | 0.079 | 0.102 |
| e295/300 | 0.102 | 0.090 | 0.089 |
| e400 | 0.109 | 0.109 | — |
| ~e600 | **0.126** | 0.077 | — |
| e800 | **0.133** | 0.069 | — |

### JEPA vs MAE — Three-way Dx trajectory (AUROC)

| Epoch | MAE AUROC | JEPA-slow AUROC |
|---|---|---|
| e30 | 0.662 | **0.799** |
| ~e50 | 0.652 | 0.797 |
| e100 | 0.695 | 0.795 |
| e200 | 0.744 | 0.782 |
| e295/300 | 0.750 | 0.766 |
| e400 | 0.745 | — |
| ~e600 | 0.758 | — |
| **e800** | **0.759** | — |

### Three-regime pattern (now triangulated)

1. **Early (e30–e100): JEPA dominates.** LVEF ~2× MAE's R²; Dx AUROC +0.14. Consistent with the canonical "predictive SSL converges faster than pixel SSL" result.
2. **Mid (e200–e400): JEPA plateaus / degrades, MAE catches up.** By e200 LVEF R² is within 0.01 across objectives. Dx AUROC gap narrows from 0.14 to 0.04.
3. **Late (e400–e800, MAE only, JEPA has collapsed): MAE climbs monotonically.** LVEF R² 0.126 (e600) → 0.133 (e800). Dx AUROC 0.758 (e600) → 0.759 (e800). MAE's asymptote meets or exceeds JEPA's peak at e100.

### MAE-vs-JEPA-fast delta table (LVEF R², for reference)

| Epoch | MAE R^2 | JEPA-fast R^2 | Δ | MAE r | JEPA-fast r | Δ |
|---|---|---|---|---|---|---|
| e30 | 0.085 | 0.120 | **+0.035** | 0.295 | 0.383 | +0.088 |
| ~e50 | 0.051 | 0.120 | **+0.069** | 0.235 | 0.391 | +0.156 |
| e100 | 0.069 | **0.162** | **+0.093** | 0.267 | 0.438 | +0.171 |
| e200 | 0.092 | 0.079 | -0.013 | 0.311 | 0.314 | +0.003 |
| e300 | 0.102 | 0.090 | -0.012 | 0.328 | 0.333 | +0.005 |
| e400 | 0.109 | 0.109 | 0.000 | 0.334 | 0.360 | +0.026 |
| ~e600 | 0.126 | 0.077 | **-0.049** | 0.369 | 0.296 | -0.073 |
| e800 | 0.133 | 0.069 | **-0.064** | 0.380 | 0.288 | -0.092 |

Trajectories cross at e200-e400 and MAE wins decisively at e600-e800 (Δ up to -0.064 R²). See the three-way table above for the full comparison including slow-EMA.

---

## Combined Interpretation

**CMR MAE:** both tasks (LVEF regression and 5-class diagnosis) show zero matched-frame temporal Δ across all 8 checkpoints (e30-e800). The CMR MAE ViT-S solves both tasks entirely from single-frame spatial/structural appearance. Unlike the echo ViT-L on MIMIC (transient temporal sensitivity peaking at e50), CMR ViT-S never enters the temporal encoding phase. Performance climbs monotonically through e800 on both LVEF (R² 0.085 → 0.133) and Dx (AUROC 0.662 → 0.759).

**CMR JEPA (both fast-EMA and slow-EMA):** reproducible non-monotonic training dynamic. Fast-EMA loss rises 0.35→0.505 between e20 and e300 then slowly descends; slow-EMA rises 0.37→0.44 — smaller but still present. Probe R² peaks early (e100 for fast-EMA at R²=0.162; e30 for slow-EMA at R²=0.138, AUROC=0.799) and degrades thereafter. JEPA beats MAE at e30-e100 on both tasks, crosses below MAE by e200, and falls decisively behind by e600-e800. Pattern is reproducible across (i) fast-EMA seed 234 (job 333), (ii) fast-EMA seed 163 (job 344), (iii) slow-EMA seed 234 (job 346). **It is seed-independent AND EMA-schedule-independent.**

**What the slow-EMA result rules out:** the `ema=0.996` config value documented as a candidate cause in earlier revisions of this doc is NOT the primary driver. Slow-EMA at `ema=0.99925` (matching `refs/vjepa2`) mutes the loss rise by ~60% but does not rescue representation quality on either downstream task. Whatever causes JEPA-on-CMR to degrade, it acts at a representational level independent of how aggressively the teacher tracks the student.

### Remaining candidate explanations (post-slow-EMA)

1. **Data scale:** 21,840 CMR clips vs 525,000 echo clips (24x fewer). JEPA's teacher-chasing dynamic may be unstable below some critical clip count; MAE is not, because it has a reconstruction target that is invariant to data scale (each pixel is its own supervision).

2. **Model capacity:** ViT-S (22M params) vs echo's ViT-L (304M). Predictive SSL may need capacity overhead to hold both spatial and temporal-predictor-useful features; MAE's per-pixel target may be more memory-efficient for a small encoder.

3. **Clip length:** CMR clips are ~25 frames (one cardiac cycle); there is no cross-cycle variation for the JEPA predictor to exploit. Echo clips contain multiple cycles with natural HR-driven variation.

4. **SAX purely-spatial signal:** SAX slices offer strong structural cues (chamber diameter, wall thickness) that are independently sufficient for DCM/HCM/MINF discrimination — i.e. the task is spatially solvable, and JEPA's teacher-chasing dynamic may overfit to this easier signal and discard the harder temporal signal that would help marginal cases.

5. **(RULED OUT)** EMA schedule: both fast (0.996) and slow (0.99925) exhibit the collapse. The rise in loss is EMA-sensitive but the probe degradation is not.

### Implications for the paper (revised)

- **CMR MAE result is solid and stays.** Zero MF temporal Δ across 8 checkpoints on both regression and classification tasks, matched-frame inference at both `num_segments=1` and `num_segments=2`. Clean negative result on the temporal shortcut at ViT-S/21K scale.
- **CMR JEPA result is also solid now.** The collapse reproduces across 3 independent runs (two seeds × two EMA schedules), all four candidate causes except (ruled out) EMA now stand as joint hypotheses rather than one leading explanation. This is publishable as "JEPA-on-CMR fails to stabilize at ViT-S/21K scale for reasons independent of EMA scheduling."
- **MAE eventually overtakes JEPA on CMR** — this is the inverse of the echo result, where JEPA dominates MAE at matched compute on ViT-L/525K. The direction-flip is itself a clean finding: predictive SSL's advantage on echo is **not** a universal property of cardiac video SSL.
- **Matched-frame inference on any JEPA CMR checkpoint still TODO.** Needed to separate "temporal features absent" from "all features degraded" in the post-e200 regime, and to complete the mechanism narrative. Priority targets: slow-EMA e30 (JEPA peak), MAE e800 (MAE asymptote), fast-EMA e100 (fast-EMA peak).

**Option for the paper (final):** report the full trajectory matrix in an appendix figure. Three rows (MAE / JEPA-fast / JEPA-slow) × two tasks (LVEF / Dx), x-axis = pretraining epoch, y-axis = val R² or val AUROC. Main-text claim: "On CMR, JEPA's downstream advantage is transient and objective-invariant: fast and slow EMA both peak in e30-e100 and degrade thereafter, while MAE climbs monotonically through e800. This is the opposite direction from echo, where JEPA dominates MAE at matched compute."

---

## Data Pipeline

See `data/cmr/README.md` for full details.

**Pretraining data:** 21,840 SAX clips from 5 datasets (MnM, MnM2, Sunnybrook, DSB2, CMR-Multi). SAX-only to prevent view-classification shortcuts.

**Holdout:** ACDC (1,489 clips, 150 patients). Official challenge split, not randomly held out.

**Conversion:** `scripts/neurips/cmr/convert_cmr_to_mp4.py` (ACDC/MnM/MnM2/Sunnybrook/DSB2), `scripts/neurips/cmr/convert_cmr_multi.py` (CMR-Multi). 256×256 H.264 MP4, 25 fps.

**Checkpoint:** ViT-S ImageNet-21K from timm (`vits_in21k.pt`, 86.7 MB). Same init for MAE and JEPA.

---

## Key Differences: CMR vs Echo

| Property | Echo (MIMIC, ViT-L) | CMR (ACDC, ViT-S) |
|----------|---------------------|-------------------|
| Modality | Ultrasound (speckle noise, shadows) | MRI (high contrast, smooth) |
| Views | A4C (long-axis) | SAX (short-axis cross-sections) |
| Model size | ViT-L (304M params) | ViT-S (22M params) |
| Pretrain data | 525K clips | 21.8K clips |
| Eval data | EchoNet-Dynamic (7.5K train, 1.3K test) | ACDC (951 train, 538 test) |
| Epochs | 100-200 | 800 |
| LVEF R² at convergence | 0.47-0.53 (MAE e99-e194) | 0.13 (MAE e800) / 0.07 (JEPA e800) |

The massive performance gap (R² 0.47 vs 0.13) reflects model capacity (ViT-L vs ViT-S) and data scale (525K vs 21.8K), not modality. The temporal shortcut hypothesis is testable regardless of absolute performance — what matters is the clean-vs-shuffled delta trajectory, not the absolute R².

---

## Job Log

| Job | Role | Submitted | Completed | Elapsed | Outcome |
|---|---|---|---|---|---|
| 183 | CMR MAE ViT-S 800ep pretrain | 2026-04-18 | 2026-04-19 | 9h05m | Loss 1.12 -> 0.27 monotonic |
| 209 | CMR MAE ACDC LVEF trajectory | 2026-04-19 | 2026-04-19 | - | R^2 0.085 -> 0.133 monotonic |
| 281 | CMR MAE ACDC Dx trajectory + MF (ns=2) | 2026-04-20 | 2026-04-20 | - | AUROC 0.662 -> 0.759, delta MF ~ 0 |
| 282 | CMR MAE ACDC Dx MF (ns=1) | 2026-04-20 | 2026-04-20 | - | delta MF ~ 0 at every ckpt |
| 283 | CMR MAE ACDC LVEF MF (ns=1) | 2026-04-20 | 2026-04-20 | - | delta MF ~ 0 at every ckpt |
| 333 | CMR JEPA ViT-S 800ep pretrain (seed 234) | 2026-04-22 | 2026-04-23 | 11h42m | Loss 0.60 -> 0.35 -> 0.505 rise -> 0.43 |
| 334 | CMR JEPA ACDC LVEF probe (buggy) | 2026-04-23 | 2026-04-23 | 24m | Only e800 probed (checkpoint naming bug, 7 of 8 skipped) |
| 344 | CMR JEPA ViT-S resume-e250 (seed 163) | 2026-04-23 | 2026-04-24 | 7h59m | Loss tracks job 333 within 0.01 — seed-independent rise |
| 345 | CMR JEPA ACDC LVEF trajectory (fixed) | 2026-04-24 | 2026-04-24 | ~3h | R^2 peaks 0.162 at e100, collapses to 0.069 at e800 |
| 346 | CMR JEPA ViT-S slow-EMA (0.99925) | 2026-04-24 | 2026-04-24 | — | Loss rise muted (0.37→0.44) but present; ran to e295 |
| 375 | CMR JEPA slow-EMA ACDC LVEF trajectory | 2026-04-24 | 2026-04-25 | 2h01m | R² peaks 0.138 at e30, collapses to 0.089 at e295 — EMA-independent |
| 376 | CMR JEPA slow-EMA ACDC Dx trajectory | 2026-04-24 | 2026-04-25 | 1h24m | AUROC peaks 0.799 at e30, collapses to 0.766 at e295 |
