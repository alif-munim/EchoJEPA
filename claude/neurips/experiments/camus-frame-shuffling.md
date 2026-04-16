# CAMUS Segmentation Frame Shuffling

**Date:** 2026-04-08
**Status:** Complete — all 4 models × severity gradient + 6-condition + per-sample bootstrap CIs + tracked extraction (Version A vs B)

---

## Overview

Frame shuffling applied to a **segmentation** (per-frame spatial) task on CAMUS echocardiography, complementing the LVEF (global temporal) frame shuffling in `frame-shuffling.md` and `6-condition-shuffling.md`. Tests whether temporal disruption affects a task that should depend only on spatial features at a single temporal position.

**Key finding:** Segmentation degrades 5–9% under full shuffle — more than the predicted <2% "temporal invariance" hypothesis. Reverse is catastrophic (12–15%), revealing that all encoders embed cardiac phase into temporal position. MYO (myocardium) is the most sensitive structure.

## Experimental Setup

**Dataset:** CAMUS test set — 50 patients × 2 views (4CH, 2CH) = 100 samples
**Task:** Cardiac segmentation (LV, MYO, LA) at ED and ES frames
**Pipeline:** Frozen encoder → frozen LinearSegDecoder (1×1 conv, ~4K params, trained via 7-config HP grid search, 50 epochs)
**Metric:** Dice score per structure, mean across ED+ES
**Data source:** `s3://echodata25/neurips/data/camus.zip` (3.6 GB, 500 patients, NIfTI)

### Encoder Checkpoints

| Model | Checkpoint | S3 Path |
|-------|-----------|---------|
| JEPA IN21K e100 | `latest.pt` (key: `target_encoder`) | `s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts/runs/jepa_in21k_pretrain_376/checkpoints/latest.pt` |
| BYOL e100 | `e100.pt` (key: `target_encoder`) | `s3://.../checkpoints/byol-vitl-imagenet-v2-resume/e100.pt` |
| MAE e99 | `checkpoint-99.pth` (key: `model`) | `s3://.../runs/videomae_resume_e54_354/training_folder/checkpoint-99.pth` |
| SALT S2 e79 | `e79.pt` (key: `target_encoder`) | `s3://.../runs/salt_s2_pretrain_388/checkpoints/e79.pt` |

### Decoder Checkpoints (trained CAMUS segmentation probes)

| Model | S3 Path | Training Job |
|-------|---------|-------------|
| JEPA IN21K e100 | `s3://echodata25/neurips/probes/camus_segmentation/jepa_in21k_e100/best_decoder.pt` | Local |
| BYOL e100 | `s3://echodata25/neurips/probes/camus_segmentation/byol_e100/best_decoder.pt` | Local |
| MAE e99 | `s3://echodata25/neurips/probes/camus_segmentation/mae_e99/best_decoder.pt` | Local |
| SALT S2 e79 | `s3://echodata25/neurips/probes/camus_segmentation/salt_s2v1_e79/best_decoder.pt` | Job 702, node 184 |

SALT probe training: best config `lr5e-02_wd1e-04`, test mean Dice = 0.737, grid summary at `s3://echodata25/neurips/probes/camus_segmentation/salt_s2v1_e79/grid_summary.json`.

### Scripts

| Script | Purpose |
|--------|---------|
| `scripts/neurips/frame_shuffle_segmentation.py` | Basic full-shuffle (clean vs shuffled × 3 seeds) |
| `scripts/neurips/frame_shuffle_segmentation_extended.py` | Severity gradient + 6-condition (aggregate output) |
| `scripts/neurips/frame_shuffle_segmentation_persample.py` | Per-sample output for bootstrap CIs |
| `scripts/neurips/frame_shuffle_segmentation_tracked.py` | Version B: tracked extraction (inverse perm) |
| `scripts/neurips/camus_bootstrap_persample.py` | Paired bootstrap CI computation (10K resamples) |

### Sbatch Scripts

| Script | Node | Job IDs | Purpose |
|--------|------|---------|---------|
| `scripts/salt_camus_probe_node184.sbatch` | 184 | 702 | SALT probe training + basic shuffle |
| `scripts/camus_shuffle_node83.sbatch` | 83 | 713 | JEPA/BYOL/MAE basic shuffle |
| `scripts/camus_shuffle_mae_node83.sbatch` | 83 | 723 | MAE retry (cuBLAS bf16 fix) |
| `scripts/camus_extended_shuffle_node83.sbatch` | 83 | 729 | JEPA/BYOL/MAE severity + 6-cond |
| `scripts/camus_extended_shuffle_node184.sbatch` | 184 | 730 | SALT severity + 6-cond |
| `scripts/camus_persample_node83.sbatch` | 83 | 747 | JEPA/BYOL/MAE per-sample output |
| `scripts/camus_persample_node184.sbatch` | 184 | 748 | SALT per-sample output |
| `scripts/camus_tracked_node83.sbatch` | 83 | 761 | JEPA/BYOL/MAE tracked extraction (Version B) |
| `scripts/camus_tracked_node184.sbatch` | 184 | 762 | SALT tracked extraction (Version B) |

### Output CSVs (all on controller at `scripts/neurips/samples/`)

**Basic shuffle:**
- `{jepa_in21k_e100,byol_e100,mae_e99,salt_s2v1_e79}_frame_shuffle_segmentation.csv`

**Aggregate severity + 6-condition:**
- `{model}_camus_severity.csv` — 5 fractions × 3 seeds
- `{model}_camus_6cond.csv` — 6 conditions × 1–3 seeds

**Per-sample (for bootstrap):**
- `{model}_camus_severity_persample.csv` — 100 samples × 5 fractions × 3 seeds = 1500 rows
- `{model}_camus_6cond_persample.csv` — 100 samples × 14 condition-seeds = 1400 rows

**Tracked extraction (Version B):**
- `{model}_camus_tracked_persample.csv` — 100 samples × (1 clean + 2 conditions × 3 seeds) = 700 rows

All uploaded to `s3://echodata25/neurips/results/camus_frame_shuffle/`.

---

## Results

### 1. Clean Baselines (Mean Dice)

| Model | Mean Dice | LV | MYO | LA |
|-------|----------|-----|-----|-----|
| MAE e99 | **0.827** [0.814, 0.838] | 0.891 | 0.765 | 0.825 |
| BYOL e100 | **0.823** [0.811, 0.835] | 0.883 | 0.781 | 0.806 |
| JEPA IN21K e100 | **0.815** [0.801, 0.829] | 0.884 | 0.758 | 0.805 |
| SALT S2 e79 | **0.777** [0.759, 0.794] | 0.847 | 0.720 | 0.764 |

All 95% CIs from paired bootstrap (n=100 samples, 10K resamples).

### 2. Severity Gradient (% degradation from clean)

Partial shuffle: randomly permute N% of 16 frames, keep rest in original order.

| Shuffle % | JEPA | BYOL | MAE | SALT |
|-----------|------|------|-----|------|
| 25% | 1.6% [1.1, 2.0] | 1.3% [0.9, 1.7] | 2.1% [1.7, 2.5] | 0.8% [0.5, 1.1] |
| 50% | 3.6% [2.9, 4.3] | 2.7% [2.2, 3.3] | 4.9% [4.1, 5.6] | 2.1% [1.5, 2.6] |
| 75% | 5.0% [4.1, 5.8] | 4.1% [3.5, 4.8] | 6.8% [6.0, 7.7] | 3.2% [2.4, 4.0] |
| 100% | 7.0% [6.0, 8.0] | 6.0% [5.2, 6.7] | **8.6% [7.6, 9.6]** | 4.9% [4.0, 5.8] |

**Pattern:** Monotonic, roughly linear degradation. No threshold/cliff behavior. MAE degrades fastest, SALT slowest. All CIs exclude zero — degradation is statistically significant at every level.

### 3. Six-Condition Ablation (% degradation from clean)

| Condition | JEPA | BYOL | MAE | SALT |
|-----------|------|------|-----|------|
| reverse | **14.5% [13.4, 15.7]** | **12.2% [11.0, 13.4]** | **12.7% [11.6, 13.8]** | **11.9% [10.6, 13.3]** |
| tubelet | 5.8% [5.0, 6.7] | 4.6% [3.9, 5.3] | 5.5% [4.6, 6.3] | 5.0% [4.1, 5.9] |
| matched | 6.3% [5.7, 6.9] | 5.3% [4.7, 5.9] | 6.1% [5.5, 6.7] | 5.5% [4.8, 6.1] |
| shuffle | 7.1% [6.0, 8.1] | 6.0% [5.2, 6.7] | 8.5% [7.5, 9.5] | 4.8% [4.0, 5.7] |
| matched_frame | 7.6% [6.6, 8.5] | 7.1% [6.2, 7.9] | 8.5% [7.6, 9.3] | 5.7% [4.8, 6.6] |

**Condition definitions:**
- **reverse** — play video backwards (cardiac cycle reversed)
- **tubelet** — permute at 2-frame tubelet granularity (per-video random)
- **matched** — fixed tubelet-level permutation (same perm for all videos per seed)
- **shuffle** — full random frame permutation (per-video random)
- **matched_frame** — fixed frame-level permutation (same perm for all videos per seed)

### 4. Per-Structure Breakdown (matched_frame)

| Structure | JEPA | BYOL | MAE | SALT |
|-----------|------|------|-----|------|
| LV | 5.8% [5.1, 6.5] | 5.6% [4.9, 6.3] | 5.3% [4.6, 5.9] | 4.6% [3.8, 5.3] |
| **MYO** | **8.8% [7.8, 9.8]** | **9.4% [8.3, 10.5]** | **12.7% [11.5, 14.1]** | **5.0% [4.2, 5.8]** |
| LA | 8.4% [6.6, 10.2] | 6.4% [5.1, 7.8] | 7.9% [6.7, 9.3] | 7.5% [5.8, 9.3] |

MYO is the most sensitive structure for all models. MAE MYO degradation (12.7%) is nearly 2.5× its LV degradation (5.3%).

### 5. ED vs ES Phase (matched_frame)

| Phase | JEPA | BYOL | MAE | SALT |
|-------|------|------|-----|------|
| ED | 7.9% [6.5, 9.3] | **9.5% [8.1, 10.9]** | 6.6% [5.6, 7.6] | **8.3% [6.9, 9.6]** |
| ES | 7.2% [6.3, 8.1] | 4.6% [3.8, 5.5] | **10.3% [9.1, 11.4]** | 3.1% [2.3, 3.9] |

ED degrades more than ES for BYOL (9.5 vs 4.6%) and SALT (8.3 vs 3.1%). MAE shows the opposite: ES degrades more (10.3 vs 6.6%).

### 6. Tracked Extraction — Version A vs Version B

Isolates **content misalignment** from **temporal encoding disruption**. Two versions of the same shuffle experiment:

- **Version A (original positions):** Extract features at the original ED/ES tubelet positions after shuffling. The content at those positions is wrong (random frame landed there). This is the standard frame shuffling result.
- **Version B (tracked positions):** After shuffling, compute the inverse permutation to find where the ED/ES content actually landed. Extract features at those new positions. The content is correct, but the positional encoding is wrong and temporal attention context is disrupted.

**Script:** `scripts/neurips/frame_shuffle_segmentation_tracked.py`
**Implementation:** `inv_perm = np.argsort(perm); new_ed_t = inv_perm[ed_sampled_idx] // 2`

| Model | Condition | Clean | V-A (orig pos) | V-B (tracked) | A drop | B drop |
|-------|-----------|-------|-----------------|----------------|--------|--------|
| JEPA | shuffle | 0.816 | 0.758 | 0.761 | **5.8%** | **5.5%** |
| JEPA | matched_frame | 0.816 | 0.754 | 0.774 | **6.2%** | **4.2%** |
| BYOL | shuffle | 0.823 | 0.774 | 0.680 | **4.9%** | **14.3%** |
| BYOL | matched_frame | 0.823 | 0.765 | 0.632 | **5.8%** | **19.2%** |
| MAE | shuffle | 0.827 | 0.756 | 0.724 | **7.0%** | **10.3%** |
| MAE | matched_frame | 0.827 | 0.757 | 0.737 | **7.0%** | **9.0%** |
| SALT | shuffle | 0.777 | 0.740 | 0.749 | **3.7%** | **2.8%** |
| SALT | matched_frame | 0.777 | 0.733 | 0.760 | **4.4%** | **1.8%** |

**Two distinct behavioral groups emerge:**

**SALT/JEPA — tracked extraction recovers performance (B drop < A drop):**
SALT: 1.8–2.8% B drop vs 3.7–4.4% A drop. JEPA: 4.2–5.5% vs 5.8–6.2%. Most degradation was content misalignment — extracting the correct cardiac phase content at the wrong position still works. These encoders produce relatively position-invariant spatial features.

**BYOL/MAE — tracked extraction is *worse* (B drop > A drop):**
BYOL: 14.3–19.2% B drop vs 4.9–5.8% A drop. MAE: 9.0–10.3% vs 7.0%. Extracting the correct content at an unfamiliar position is more damaging than extracting wrong content at the familiar position.

**Explanation — decoder position lock-in:** During training, the decoder exclusively sees ED features from tubelet ~0 and ES features from tubelet ~7. With RoPE positional encoding, features at each temporal position occupy a different rotated coordinate subspace. The 1×1 conv decoder implicitly learns position-specific linear mappings. When Version B extracts ED features from tubelet 5 (where ED content landed), the features have position 5's RoPE rotation, which the decoder has never seen for ED. For BYOL/MAE, this position mismatch is more destructive than the content mismatch in Version A.

**Implication for the A→B decomposition:** The clean decomposition "A drop = content + temporal, B drop = temporal only" holds for SALT/JEPA where features are position-invariant. For BYOL/MAE, Version B conflates temporal context disruption with decoder position bias (a third confound), making the decomposition unclean. The result nonetheless reveals a meaningful architectural difference: JEPA/SALT learn more position-invariant representations than BYOL/MAE.

---

## Interpretation

### Reverse is catastrophic — temporal direction matters for segmentation

The reverse condition (12–15%) produces nearly **2× the degradation** of full random shuffle (5–9%). This is the opposite of what a "temporal-invariant spatial task" would show. The encoder maps specific cardiac phases (diastole, systole) to specific temporal positions. Reversing the sequence puts late-cycle frames into early temporal slots, scrambling which anatomy lands at the ED/ES extraction positions.

### MYO is the canary

Myocardium segmentation is the hardest structure and the most dependent on temporal context. The thin, variable-thickness MYO boundary requires the encoder to correctly resolve which cardiac phase is at each temporal position. When frame order is disrupted, the MYO boundary becomes ambiguous. This is most extreme for MAE (12.7% degradation), which processes frames independently but still encodes temporal position via positional embeddings.

### Model ranking differs from LVEF

For LVEF (global temporal task), JEPA degrades least and MAE most. For segmentation (spatial task), **SALT degrades least** (4.9–5.7%) and MAE still degrades most (8.5–8.6%). SALT's lower sensitivity may reflect its auxiliary supervised loss providing more robust spatial features that are less entangled with temporal structure.

### Segmentation is NOT temporally invariant

The initial hypothesis was <2% degradation. The actual 5–9% (and 12–15% for reverse) shows that modern video encoders entangle spatial and temporal information even for nominally per-frame spatial tasks. The segmentation decoder extracts features at a single temporal position, but the encoder's contextual processing across all 16 frames means the features at that position are influenced by what appears at other positions. Frame shuffling changes the semantic content at each temporal slot, degrading the spatial features used for segmentation.

### Tracked extraction reveals position-invariance spectrum

The Version A vs B experiment (Section 6) reveals that encoders fall on a spectrum of positional invariance. JEPA/SALT produce features where the spatial content is relatively independent of the temporal position it occupies — the decoder can interpret ED features even when they arrive from an unusual position. BYOL/MAE produce strongly position-dependent features — the RoPE rotation at each temporal slot is entangled with the learned spatial representation, making the decoder unable to transfer across positions. This suggests JEPA/SALT learn more generalizable spatial features, while BYOL/MAE rely more heavily on positional scaffolding.

---

## Comparison with LVEF Frame Shuffling

| | LVEF (R² degradation) | Segmentation (Dice degradation) |
|--|----------------------|-------------------------------|
| JEPA matched_frame | −33.6% | −7.6% |
| BYOL matched_frame | −18.2% | −7.1% |
| MAE matched_frame | +1.4% (immune) | −8.5% |
| SALT matched_frame | −191% (cliff) | −5.7% |

LVEF degradation is measured as R² change (can go negative). Segmentation Dice degradation is bounded [0, 100%]. The key contrast: MAE is **immune** to frame shuffling for LVEF but **sensitive** for segmentation, suggesting MAE's positional embeddings encode spatial-temporal coupling that matters for per-frame extraction even when global temporal pooling washes it out.

---

## Issues Encountered

1. **nibabel not installed** on compute nodes — required for CAMUS NIfTI loading. Fixed by adding `pip install nibabel` to sbatch scripts.
2. **Root filesystem full** on node 83 (97G, 100% used) — encoder checkpoint downloads (5–10 GiB each) failed with `ENOSPC`. Fixed by downloading to NVMe (`/opt/dlami/nvme/encoder_ckpts/`) which has 26TB free.
3. **cuBLAS bf16 error** for VideoMAE on H100 — system cuBLAS 12.9 incompatible with PyTorch's expected 12.8. Fixed by prepending PyTorch-bundled NVIDIA libs to `LD_LIBRARY_PATH` (see `h100_cublas_fix.md`).
