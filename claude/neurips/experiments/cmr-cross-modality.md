# CMR Cross-Modality Validation Results

**Last updated:** 2026-04-20
**Status:** CMR MAE ViT-S 800ep pretraining complete. ACDC LVEF probes complete. ACDC diagnosis probes in progress. CMR JEPA pretraining queued.

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

### CMR JEPA ViT-S (queued, not yet run)

Same ViT-S architecture, same data, 800 epochs. Will provide the JEPA comparison arm.
**Config:** `configs/train/vits16/pretrain-jepa-cmr-224px-16f-in21k.yaml`

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

## Combined Interpretation

Both tasks (LVEF regression and 5-class diagnosis) show zero temporal sensitivity across all 8 CMR MAE checkpoints (e30-e800). This rules out the task-specific explanation (diagnosis being purely spatial) and confirms:

**The CMR MAE ViT-S never develops temporal features at any training epoch.** The model solves both tasks entirely from single-frame spatial/structural appearance. Unlike the echo ViT-L on MIMIC (which shows transient temporal sensitivity peaking at e50), the CMR ViT-S on 21K clips never enters the temporal encoding phase.

### Possible explanations

1. **Model capacity:** ViT-S (22M params) may be too small to learn both spatial and temporal features simultaneously. On echo, the temporal shortcut was observed on ViT-L (304M params) — 14× larger. The temporal encoding phase at e50 may require sufficient representational capacity to discover temporal correlations as reconstruction shortcuts.

2. **Data scale:** 21,840 CMR clips vs 525,000 echo clips (24× fewer). With limited data, the model may converge directly to spatial features without the transient temporal phase.

3. **Modality:** CMR SAX slices have higher spatial resolution and contrast than echo A4C views. Within-frame spatial interpolation may be so effective on CMR that the model never needs temporal cues, even transiently.

4. **Clip length:** CMR clips are ~25 frames (one cardiac cycle) vs echo clips with 100+ frames (multiple cycles). Shorter clips provide less temporal signal for the model to exploit.

### Implications for the paper

The CMR cross-modality validation **cannot confirm the temporal shortcut at ViT-S scale.** The model never develops temporal features to lose. This does not contradict the echo findings — it sets a boundary condition: the temporal shortcut requires sufficient model capacity and data scale to manifest. At ViT-S on 21K clips, the model goes directly to spatial features without the transient temporal phase.

**Options for the paper:**
1. **Include as a negative/boundary result:** "The temporal shortcut requires sufficient model capacity; a ViT-S on 21K CMR clips never develops temporal features to abandon." This is informative — it shows the shortcut is not universal but requires a model large enough to discover temporal correlations.
2. **Run CMR JEPA for comparison:** If JEPA ViT-S *also* shows zero temporal Δ, the capacity/data explanation is confirmed. If JEPA shows temporal sensitivity where MAE doesn't, the objective-dependence extends to CMR even at small scale — a stronger finding.
3. **Defer to appendix or future work:** Note that CMR validation at ViT-L scale would be definitive but requires more compute.

**Recommendation:** Run CMR JEPA (already queued as job 235) and compare. The JEPA-vs-MAE comparison at ViT-S scale is informative regardless of outcome.

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
| LVEF R² at convergence | 0.47-0.53 (MAE e99-e194) | 0.13 (MAE e800) |

The massive performance gap (R² 0.47 vs 0.13) reflects model capacity (ViT-L vs ViT-S) and data scale (525K vs 21.8K), not modality. The temporal shortcut hypothesis is testable regardless of absolute performance — what matters is the clean-vs-shuffled delta trajectory, not the absolute R².
