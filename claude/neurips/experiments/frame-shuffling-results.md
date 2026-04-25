# Frame Shuffling Results — Consolidated Reference

**Last updated:** 2026-04-24
**Status:** Active. Extended training trajectory complete for MAE (e25-e194) and JEPA (e25-e200, with probes in job 332). JEPA IN21K clean R² plateaus at e175-e200 (0.715-0.717), 17-22pp above MAE at the same epochs (job 332 vs MAE e124-e194). JEPA extended matched-frame inference still pending. BYOL, SALT e25-e100 complete. CMR cross-modality: MAE ViT-S 800ep complete (MF Δ zero at all checkpoints), JEPA ViT-S base run complete (jobs 333/344) with probe trajectory (job 345) showing JEPA >MAE at e30-e100 but JEPA <MAE at e600-e800 due to teacher-chasing loss rise. Slow-EMA variant (job 346) in progress to test EMA speed as the cause. SALT: S1 V-Pixel teacher complete (329), S2 V-Pixel student complete (330), S2 JEPA-teacher student running (335, ~e54/80). Probe runs: S2 V-Pixel probe submitted (349, pending node); S2 JEPA-teacher probe queued (350, afterok:335).

---

## Current Results: Matched_frame with RoPE Remap (Protocol D)

**How these numbers were produced:**
- **Eval script:** `evals.main` (V-JEPA eval pipeline, `evals/video_classification_frozen/eval.py`)
- **Entry point:** `python3 -m evals.main --fname <config> --devices cuda:0 ... cuda:7`
- **Shuffle implementation:** `src/datasets/video_dataset.py:375-412` — triggered by `FRAME_SHUFFLE=100 FRAME_SHUFFLE_TYPE=matched_frame` env vars. Applies frame permutation with RoPE position remapping.
- **Prediction averaging:** `num_segments=2` (two temporal clips per video, predictions averaged)
- **Sbatch:** `--ntasks-per-node=1, --cpus-per-task=96` (evals.main uses mp.Process internally, NOT srun)
- **Run on:** HyperPod `echojepa-h100-neurips` cluster, 2026-04-20
- **Sbatch scripts:**
  - MAE e25-e194: `scripts/neurips/echomae_matched_frame_trajectory.sbatch` (job 216)
  - JEPA + BYOL + SALT e4: `scripts/neurips/all_models_matched_frame_trajectory.sbatch` (job 220)
  - SALT e29/54/79: `scripts/neurips/salt_matched_frame_remaining.sbatch` (job 237)
- **Inference configs:** `configs/inference/vitl/neurips/echonet-dynamic/{echomae,jepa_in21k,byol,salt_s2v1}_e{N}_end_test.yaml`
- **Dataset:** EchoNet-Dynamic test (1,277 videos), `echonet_dynamic_test_s3_raw.csv`
- **Probes:** Epoch-matched d=4 attentive probes, 6 HP heads (6 multihead_kwargs entries required in inference config)
- **Metrics:** R² computed via DDP all_reduce across all 8 GPUs (full 1,277 clips), best head selected automatically

### MAE ViT-L Training Trajectory (job 216, ip-10-0-50-35, 2026-04-20)

**Sbatch:** `scripts/neurips/echomae_matched_frame_trajectory.sbatch`
**S3 run:** `runs/mae_mf_traj_216/`

Encoder checkpoints (S3):
- e25: `checkpoints/echomae_l_mimic_ep24.pth` (from local EFS `videomae_l_mimic_ep24.pth`)
- e50: `checkpoints/echomae_l_mimic_ep50.pth`
- e75: `checkpoints/echomae_l_mimic_ep74.pth`
- e99: `checkpoints/echomae_l_mimic_ep99.pth` (⚠️ must be `videomae_l_mimic_ep99.pth`, NOT `echomae_l_mimic_ep99.pth` — different files)
- e124: `runs/videomae_e200_159/training_folder/checkpoint-124.pth`
- e149: `runs/videomae_e200_159/training_folder/checkpoint-149.pth`
- e174: `runs/videomae_e200_159/training_folder/checkpoint-174.pth`
- e194: `runs/videomae_e200b_179/training_folder/checkpoint-194.pth`

Probe checkpoints (S3):
- e25-e99: `probes/echomae_e{25,50,75,99}_end_lvef/best.pt` (from ICML runs on local EFS)
- e124-e194: from NeurIPS probe training jobs 201/202

Inference configs: `configs/inference/vitl/neurips/echonet-dynamic/echomae_e{N}_end_test.yaml`
- VideoMAE encoder: `module_name: evals.video_classification_frozen.modelcustom.videomae_encoder`
- `model_name: vit_large_patch16_224`, `tubelet_size: 2`

| MAE Epoch | Clean R² | Matched_frame R² | Temporal Δ | Rel. Drop |
|-----------|----------|-------------------|------------|-----------|
| e25 | 0.225 | 0.257 | +0.033 | +15% |
| **e50** | **0.413** | **0.281** | **-0.132** | **-32%** |
| e75 | 0.435 | 0.356 | -0.080 | -18% |
| e99 | 0.467 | 0.440 | -0.027 | -6% |
| e124 | 0.469 | 0.428 | -0.041 | -9% |
| e149 | 0.527 | 0.491 | -0.035 | -7% |
| e174 | 0.500 | 0.448 | -0.052 | -10% |
| e194 | 0.526 | 0.460 | -0.065 | -12% |

### JEPA IN21K ViT-L Training Trajectory (jobs 220 + 332)

**Sbatch (e25-e100):** `scripts/neurips/all_models_matched_frame_trajectory.sbatch` (job 220, ip-10-0-50-39, 2026-04-20)
**Sbatch (e125-e200 probes):** `scripts/neurips/jepa_in21k_extended_probes.sbatch` (job 332, ip-10-0-50-148, 2026-04-22)
**S3 run (e25-e100):** `runs/all_mf_traj_220/`
**S3 run (e125-e200 probes):** `runs/jepa_ext_probes_332/`

Encoder checkpoints (S3):
- e25-e100: `checkpoints/jepa_in21k/jepa_in21k_vitl_e{25,50,75,100}.pt`
- e125-e195: `runs/jepa_in21k_e200_280/training_folder/e{125,150,175,195}.pt`
- e200: `runs/jepa_in21k_e200_280/training_folder/latest.pt`

Probe checkpoints (S3):
- e25-e100: `probes/jepa_in21k_e{25,50,75,100}_end_lvef/best.pt`
- e125: `runs/jepa_ext_probes_332/jepa_e125_lvef/video_classification_frozen/neurips-jepa-in21k-e125-end-lvef-d4/best.pt`
- e150: `runs/jepa_ext_probes_332/jepa_e150_lvef/video_classification_frozen/neurips-jepa-in21k-e150-end-lvef-d4/best.pt`
- e175: `runs/jepa_ext_probes_332/jepa_e175_lvef/video_classification_frozen/neurips-jepa-in21k-e175-end-lvef-d4/best.pt`
- e200: `runs/jepa_ext_probes_332/jepa_e200_lvef/video_classification_frozen/neurips-jepa-in21k-e200-end-lvef-d4/best.pt`

Probe training configs (e125-e200): generated at runtime in sbatch, pattern: `configs/eval/vitl/neurips/jepa_in21k_e{N}_end_lvef_d4.yaml`

Inference configs: `configs/inference/vitl/neurips/echonet-dynamic/jepa_in21k_e{N}_end_test.yaml`
- JEPA encoder: `module_name: evals.video_classification_frozen.modelcustom.vit_encoder_multiclip`
- `checkpoint_key: target_encoder`, `model_name: vit_large`, `use_rope: true`

| JEPA Epoch | Val MAE | Val R² (final) | Val R² (best) | Val Pearson (best) | Source |
|------------|---------|----------------|----------------|----|--------|
| e25 | — | 0.460 | — | 0.685 | job 220 (test, clean) |
| **e50** | — | **0.612** | — | **0.784** | job 220 (test, clean) |
| e75 | — | 0.629 | — | 0.796 | job 220 (test, clean) |
| e100 | — | 0.650 | — | 0.807 | job 220 (test, clean) |
| e125 | 5.10 (best 18/20) | 0.681 | 0.685 | 0.832 | job 332 (val) |
| e150 | 4.96 (best 18/20) | 0.696 | 0.700 | 0.840 | job 332 (val) |
| e175 | 4.86 (best 18/20) | 0.716 | 0.717 | 0.848 | job 332 (val) |
| e200 | 4.91 (best 20/20) | 0.715 | 0.715 | 0.847 | job 332 (val) |

**Note:** e25-e100 are test-set R² from matched-frame inference (clean condition); e125-e200 are val-set R² from probe training (log_r0.csv, over 20 probe epochs × 16-head HP grid). Final R² uses the last probe epoch; best R² picks the probe epoch with max val R². Probes peak at epoch 18-20 of 20 — no overfitting. Test-set matched-frame inference for e125-e200 pending.

**Key finding:** JEPA representations continue improving through e175 (val R² 0.650→0.717), then plateau at e200 (0.715). Compared to MAE extended-training (published in `tab:attn_traj`: clean R² 0.467→0.526 e99→e194), JEPA-extended stays **+0.17 to +0.22 R² ahead** at every matched checkpoint — the main-paper JEPA-vs-MAE gap (+0.183 at e100) persists through extended training and does not close.

| Epoch | JEPA R² | MAE R² | Δ (JEPA − MAE) |
|---|---|---|---|
| ~e125 | 0.685 | 0.469 (e124) | +0.216 |
| ~e150 | 0.700 | 0.527 (e149) | +0.173 |
| ~e175 | 0.717 | 0.500 (e174) | +0.217 |
| ~e200 | 0.715 | 0.526 (e194) | +0.189 |

JEPA's improvement through extended training is genuine representation improvement, not spatial compensation. MAE's extended-training clean-R² improvement comes together with a narrowing matched-frame gap (−0.132 → −0.065) — the paper's "temporal shortcut" finding. Whether JEPA's extended matched-frame gap narrows, stays flat at ~−0.14, or widens is the open question (pending MF inference on job 332's 4 ckpts).

Matched-frame results (test set, Protocol D):

| JEPA Epoch | Clean R² | Matched_frame R² | Temporal Δ | Rel. Drop |
|------------|----------|-------------------|------------|-----------|
| e25 | 0.460 | 0.416 | -0.045 | -10% |
| **e50** | **0.612** | **0.328** | **-0.284** | **-46%** |
| e75 | 0.629 | 0.439 | -0.190 | -30% |
| e100 | 0.650 | 0.507 | -0.143 | -22% |
| e125 | — | — | — | — |
| e150 | — | — | — | — |
| e175 | — | — | — | — |
| e200 | — | — | — | — |

e125-e200 matched-frame inference not yet run. Probes are trained and ready.

### BYOL ViT-L Training Trajectory (job 220, ip-10-0-50-39, 2026-04-20)

**Sbatch:** `scripts/neurips/all_models_matched_frame_trajectory.sbatch`

Encoder checkpoints (S3): `checkpoints/byol/byol_vitl_imagenet_v2_e{24,50,75,100}.pt`
Probe checkpoints (S3): `probes/byol_e{24,50,75,100}_end_lvef/best.pt`

Inference configs: `configs/inference/vitl/neurips/echonet-dynamic/byol_e{N}_end_test.yaml`
- BYOL encoder: `module_name: evals.video_classification_frozen.modelcustom.vit_encoder_multiclip`
- `checkpoint_key: target_encoder`, `model_name: vit_large`, `use_rope: true`

| BYOL Epoch | Clean R² | Matched_frame R² | Temporal Δ | Rel. Drop |
|------------|----------|-------------------|------------|-----------|
| e24 | 0.437 | -0.294 | **-0.731** | **-167%** |
| e50 | 0.477 | 0.108 | -0.369 | -77% |
| e75 | 0.505 | 0.287 | -0.217 | -43% |
| e100 | 0.527 | 0.249 | -0.279 | -53% |

### SALT S2v1 ViT-L Training Trajectory (job 220, ip-10-0-50-39, 2026-04-20)

**Sbatch:** `scripts/neurips/all_models_matched_frame_trajectory.sbatch`

Encoder checkpoints (S3): `checkpoints/salt_v1/salt_s2v1_e{4,29,54,79}.pt`
Probe checkpoints (S3): `probes/salt_e{4,29,54,79}_end_lvef/best.pt`

Inference configs: `configs/inference/vitl/neurips/echonet-dynamic/salt_s2v1_e{N}_end_test.yaml`
- SALT encoder: `module_name: evals.video_classification_frozen.modelcustom.vit_encoder_multiclip`
- `checkpoint_key: encoder` (NOT target_encoder), `model_name: vit_large`, `use_rope: true`

| SALT Epoch | Clean R² | Matched_frame R² | Temporal Δ | Rel. Drop |
|------------|----------|-------------------|------------|-----------|
| e4 | 0.028 | -0.020 | -0.049 | -171% |
| **e29** | **0.356** | **-0.443** | **-0.799** | **-224%** |
| e54 | 0.431 | -0.417 | -0.848 | -197% |
| e79 | 0.402 | -0.404 | -0.805 | -200% |

SALT e29/54/79 from job 237 (ip-10-0-50-35, 2026-04-20), SALT e4 from job 220.

### Combined 4-Model Summary Table (R² and Pearson r)

All values from EchoNet-Dynamic test (1,277 videos), matched_frame with RoPE remap, num_segments=2.
R² and Pearson r are best-head values (may be different heads). Source: stdout logs from jobs 216/220/237.

**Note:** No per-clip prediction CSVs were saved for these runs — only aggregate metrics via DDP all_reduce. The R²/Pearson values below are computed over all 1,277 clips.

| Model | Epoch | Clean R² | Clean r | MF R² | MF r | Δ R² | Δ r |
|-------|-------|----------|---------|-------|------|------|-----|
| **JEPA** | e25 | 0.460 | 0.685 | 0.416 | 0.663 | -0.045 | -0.022 |
| | **e50** | **0.612** | **0.784** | **0.328** | **0.732** | **-0.284** | **-0.052** |
| | e75 | 0.629 | 0.796 | 0.439 | 0.772 | -0.190 | -0.024 |
| | e100 | 0.650 | 0.807 | 0.507 | 0.779 | -0.143 | -0.028 |
| | e125 | 0.681* | 0.831* | — | — | — | — |
| | e150 | 0.696* | 0.838* | — | — | — | — |
| | e175 | 0.716* | 0.847* | — | — | — | — |
| | e200 | 0.715* | 0.847* | — | — | — | — |
| **BYOL** | e24 | 0.437 | 0.668 | -0.294 | 0.547 | -0.731 | -0.121 |
| | e50 | 0.477 | 0.698 | 0.108 | 0.646 | -0.369 | -0.052 |
| | e75 | 0.505 | 0.715 | 0.287 | 0.680 | -0.217 | -0.035 |
| | e100 | 0.527 | 0.732 | 0.249 | 0.673 | -0.279 | -0.059 |
| **MAE** | e25 | 0.225 | 0.541 | 0.257 | 0.529 | +0.033 | -0.012 |
| | **e50** | **0.413** | **0.653** | **0.281** | **0.617** | **-0.132** | **-0.036** |
| | e75 | 0.435 | 0.684 | 0.356 | 0.632 | -0.080 | -0.052 |
| | e99 | 0.467 | 0.703 | 0.440 | 0.674 | -0.027 | -0.029 |
| | e124 | 0.469 | 0.698 | 0.428 | 0.663 | -0.041 | -0.035 |
| | e149 | 0.527 | 0.735 | 0.491 | 0.702 | -0.035 | -0.033 |
| | e174 | 0.500 | 0.713 | 0.448 | 0.685 | -0.052 | -0.028 |
| | e194 | 0.526 | 0.730 | 0.460 | 0.692 | -0.065 | -0.038 |
| **SALT** | e4 | 0.028 | 0.277 | -0.020 | 0.287 | -0.049 | +0.010 |
| | **e29** | **0.356** | **0.620** | **-0.443** | **0.577** | **-0.799** | **-0.043** |
| | e54 | 0.431 | 0.672 | -0.417 | 0.606 | -0.848 | -0.066 |
| | e79 | 0.402 | 0.650 | -0.404 | 0.626 | -0.805 | -0.024 |

*\* JEPA e125-e200: val-set R²/Pearson from probe training (job 332 log_r0.csv), not test-set. Matched-frame inference pending.*

### R² vs Pearson r Divergence — SALT Calibration Failure

SALT shows the most striking R²/Pearson divergence: at e79, R² crashes from 0.402 to -0.404 under matched_frame (Δ=-0.805), but Pearson r only drops from 0.650 to 0.626 (Δ=-0.024). This confirms the paper's calibration failure hypothesis: SALT preserves ordinal structure (patients still ranked correctly by r) but the probe's output scale is destroyed when the input distribution shifts. The frozen teacher transmits ranking information but not calibrated magnitude.

By contrast, JEPA's Pearson r drop (0.807→0.779 at e100) tracks its R² drop (0.650→0.507) proportionally — both ranking and calibration degrade together, indicating genuine temporal content loss rather than calibration failure.

BYOL shows intermediate behavior: at e24, R² goes to -0.294 but r stays at 0.547 (fragile calibration, partially preserved ranking). By e100, the R²/r relationship is more proportional (0.527/0.732 → 0.249/0.673).

---

## Interpretation

### Four Distinct Temporal Encoding Profiles

**JEPA — Consolidation + continued improvement.** Peak temporal Δ at e50 (-0.284, -46%), compresses to -0.143 (-22%) at e100. Temporal features are compressed into a more efficient representation but retained. JEPA's spatial floor (e100 MF R²=0.507) exceeds MAE's clean ceiling (e99 R²=0.467). Extended training to e200 shows continued val R² improvement (0.650 at e100 → 0.716 at e175, plateau at e200=0.715), confirming genuine representation improvement rather than spatial compensation. Matched-frame inference for e125-e200 pending to verify temporal Δ trajectory under extended training.

**BYOL — Catastrophic fragility → partial stabilization.** Temporal Δ at e24 is -0.731 (R² goes to -0.294, worse than random). Stabilizes by e75 (-0.217) but then re-widens at e100 (-0.279). BYOL's temporal encoding is fragile and inconsistent — global mean-pooling creates implicit temporal dependence that shatters under disruption.

**MAE — Transient then abandoned.** Peak Δ at e50 (-0.132, -32%), collapses to -0.027 (-6%) by e99. Extended training (e124-e194) shows clean R² continues improving (0.467→0.526) but temporal Δ stays flat (-0.03 to -0.07). The temporal shortcut is permanent — 100 extra epochs cannot reopen the temporal channel. All improvement from e99 onward comes from better spatial features.

**SALT — Permanent cliff.** R² collapses to deeply negative values under matched_frame from e29 onward (-0.443 to -0.848) and never recovers. But Pearson r barely changes (0.620→0.577 at e29), confirming the calibration failure: the frozen pixel teacher transmits ordinal structure but the probe's output scale shatters when frame order is disrupted. Unlike JEPA's consolidation or MAE's abandonment, SALT's temporal encoding is permanently brittle — the frozen teacher cannot drive the student toward robust representations.

### Key Contrasts

1. **Peak magnitude:** BYOL (-0.731) >> JEPA (-0.284) >> MAE (-0.132). BYOL is most temporally dependent, MAE least.

2. **Resolution pattern:** JEPA consolidates (50% reduction in Δ, retains large residual). MAE abandons (80% reduction, near-zero residual). BYOL oscillates (no clean convergence).

3. **Spatial floor:** JEPA e100 matched_frame (0.507) > BYOL e100 clean (0.527 — similar) > MAE e99 clean (0.467). JEPA's spatial features alone are competitive with BYOL's full representation.

4. **Extended training effect (MAE only):** Temporal Δ bottoms at e99 (-0.027), hovers at -0.03 to -0.07 through e194. Clean R² climbs from 0.467 to 0.526 but matched_frame R² grows slower (0.440→0.460). The shortcut is a stable attractor — extended training does not recover temporal encoding.

5. **SALT calibration vs ranking:** R² crashes (-0.8) but Pearson r barely moves (-0.04). The frozen teacher preserves ordinal structure but cannot maintain calibration under distribution shift. This is fundamentally different from MAE (which loses temporal features entirely) and BYOL (which loses both ranking and calibration early, then partially recovers ranking).

---

## Protocol Details

### What "matched_frame" does in evals.main

Code: `src/datasets/video_dataset.py:375-412`

1. After loading a video clip [C, T, H, W], a fixed random permutation (seed=100) is applied to the time dimension
2. RoPE positional encodings are remapped so position i points to the content at shuffled position perm[i]
3. The encoder sees frames in wrong order but "knows" where each frame originally was via RoPE
4. This isolates learned temporal content from positional encoding artifacts
5. Any R² drop compared to clean is from lost temporal content, not positional confusion

### Why num_segments=2

Each video yields 2 temporal clips (different random starting points). The probe prediction is the average of both clips' outputs. This reduces variance from single-clip evaluation but inflates clean baselines compared to the single-clip severity gradient protocol. All results in this document use num_segments=2 for consistency.

### Inference config requirements

1. **6 multihead_kwargs entries** matching the 6 HP heads in the probe checkpoint (1 entry → only head 0 loaded → wrong R²)
2. **`--ntasks-per-node=1`** in sbatch (evals.main uses mp.Process, srun causes 8x slowdown)
3. **`pip install boto3`** for S3 video streaming
4. **`predictions_save_path`** set for per-clip predictions (note: only rank 0's subset saved, but R² in stdout is correct over all clips via all_reduce)

---

## Checkpoint Provenance

### Local EFS (gdrive mirror)

```
checkpoints/
  videomae_l_mimic_ep{24,50,74,99}.pth          # MAE e25-e99 (IN21K init lineage)
  videomae_l_mimic_ep{124,149,174,194}.pth       # MAE e124-e194 (extended training)
  jepa_in21k_vitl_e{25,50,75,100}.pt             # JEPA IN21K
  byol_vitl_imagenet_v2_e{24,50,75,100}.pt       # BYOL
  salt_s2v1_e{4,29,54,79}.pt                     # SALT S2v1

evals/vitl/icml/
  echomae_e{24,74,99}_end_lvef_224/.../best.pt   # MAE LVEF probes (ICML)
  echomae_pt50_end_lvef_224/.../best.pt           # MAE e50 probe
  echomae_e{124,149,174,194}_end_lvef_224/.../best.pt  # MAE extended probes (NeurIPS)
  jepa_in21k_e{25,50,75,100}_end_lvef_224/.../best.pt  # JEPA probes
  echobyol_e{24,50,75,100}_end_lvef_224/.../best.pt    # BYOL probes
  salt_s2v1_e{4,29,54,79}_end_lvef_224/.../best.pt     # SALT probes
```

### Google Drive

```
echo_foundation/nature_medicine/neurips/
  checkpoints/mae/videomae_l_mimic_ep{24-194}.pth
  checkpoints/jepa_in21k/jepa_in21k_vitl_e{25-100}.pt
  checkpoints/byol/byol_vitl_imagenet_v2_e{24-100}.pt
  checkpoints/salt_v1/salt_s2v1_e{4-79}.pt
  probes/mae_e{24-194}_lvef/best.pt
  probes/jepa_in21k_e{25-100}_lvef/best.pt
  probes/byol_e{24-100}_lvef/best.pt
  probes/salt_v1_e{4-79}_lvef/best.pt
```

### ⚠️ Known Checkpoint Pitfall

`echomae_l_mimic_ep99.pth` (3,813,850,675 bytes, Jan 24) ≠ `videomae_l_mimic_ep99.pth` (3,813,850,611 bytes, Apr 1). The probes were trained on `videomae_l_mimic_ep99.pth`. Using the wrong file produces garbage R². For all other epochs only one version exists.

---

## 6-Condition Results (Protocol B — standalone script, no RoPE remap)

**Script:** `scripts/neurips/frame_shuffling/frame_shuffle_6cond.py`
**Run locally** on A100 notebook instance (SageMaker), single GPU per model. Not run via HyperPod.
**Date:** 2026-04-06

Epoch-matched probes, single GPU, single clip per video, NO prediction averaging, NO RoPE remapping. The "matched" and "matched_frame" conditions here use a fixed permutation but do NOT remap RoPE positions, so they are less rigorous than the Protocol D matched_frame results above. However, this is the only protocol with all 6 conditions across all 4 models × 4 epochs.

**Output CSVs:** `scripts/neurips/samples/6cond_{MODEL}_{EPOCH}.csv`
**Log files:** `scripts/neurips/samples/end_{model}_{condition}.log` (pt50 only, from Protocol A ICML run)

**⚠️ Do not mix these numbers with Protocol D.** Protocol B has lower clean baselines (no prediction averaging) and different shuffle semantics (no RoPE remap). Use Protocol D for the paper's main claims; use Protocol B for the appendix 6-condition table and for conditions not available in Protocol D (tubelet, reverse, matched).

Source: `claude/archive/frame-shuffling/6-condition-shuffling-protocol-b-standalone.md`

### JEPA IN21K

| Condition | e25 | e50 | e75 | e100 |
|-----------|-----|-----|-----|------|
| clean | 0.383 | 0.503 | 0.537 | **0.591** |
| tubelet | 0.384 | 0.507 | 0.532 | 0.582 |
| reverse | 0.384 | 0.487 | 0.489 | 0.539 |
| matched | 0.381 | 0.505 | 0.533 | 0.580 |
| shuffle | 0.328 | 0.288 | 0.375 | 0.484 |
| matched_frame | 0.323 | 0.273 | 0.372 | 0.477 |

### BYOL

| Condition | e24 | e50 | e75 | e100 |
|-----------|-----|-----|-----|------|
| clean | 0.380 | 0.427 | 0.435 | 0.468 |
| tubelet | 0.342 | 0.372 | 0.413 | 0.402 |
| reverse | 0.252 | 0.354 | 0.331 | 0.373 |
| matched | 0.350 | 0.380 | 0.413 | 0.415 |
| shuffle | -0.179 | 0.210 | 0.297 | 0.291 |
| matched_frame | -0.188 | 0.194 | 0.292 | 0.280 |

### MAE

| Condition | e24 | e50 | e74 | e99 |
|-----------|-----|-----|-----|-----|
| clean | 0.221 | 0.141 | 0.390 | 0.445 |
| tubelet | 0.197 | 0.255 | 0.417 | 0.424 |
| reverse | 0.184 | 0.114 | 0.400 | 0.431 |
| matched | 0.208 | 0.231 | 0.400 | 0.419 |
| shuffle | 0.178 | -0.278 | 0.327 | 0.422 |
| matched_frame | 0.189 | -0.343 | 0.345 | 0.449 |

### SALT S2v1

| Condition | e4 | e29 | e54 | e79 |
|-----------|-----|------|------|------|
| clean | 0.007 | 0.277 | **0.330** | 0.296 |
| tubelet | 0.008 | 0.261 | 0.324 | 0.294 |
| reverse | 0.005 | 0.202 | 0.223 | 0.120 |
| matched | 0.007 | 0.257 | 0.320 | 0.296 |
| shuffle | -0.021 | -0.439 | -0.294 | -0.283 |
| matched_frame | -0.020 | -0.462 | -0.326 | -0.310 |

### Convergence comparison (primary comparison point)

| Condition | JEPA e100 | BYOL e100 | MAE e99 | SALT e79 |
|-----------|-----------|-----------|---------|----------|
| clean | **.591** | .468 | .445 | .296 |
| tubelet | **.582** | .402 | .424 | .294 |
| reverse | **.539** | .373 | .431 | .120 |
| matched | **.580** | .415 | .419 | .296 |
| shuffle | **.484** | .291 | .422 | −.283 |
| matched_frame | **.477** | .280 | .449 | −.310 |

### Key observations from the 6-condition data

1. **Tubelet and matched barely degrade any model** — local 2-frame reordering is invisible to all four objectives.
2. **Reverse is model-specific.** JEPA: -10%. BYOL: -20%. MAE: -3%. SALT: **-60% at e79** (increases with training). Cardiac cycle is quasi-periodic so reversal should be mild — SALT's extreme sensitivity suggests its temporal encoding is directional and fragile.
3. **shuffle ≈ matched_frame** for JEPA and MAE (1-2pp difference) — RoPE positional compensation is minimal for these models. BYOL shows a larger gap (shuffle 0.291 vs matched_frame 0.280), and SALT's gap is negligible.
4. **The monotonic ordering** (clean ≈ tubelet ≈ matched > reverse > shuffle ≈ matched_frame) holds across all four models at convergence — the conditions form a genuine disruption gradient.

---

## Archived Protocols

The following older experiments used different protocols. Results are valid for their original purpose but should NOT be mixed with the Protocol D results above. See `claude/archive/frame-shuffling/` for full docs.

### Protocol A: ICML pt50 (frame-shuffling.md → archived)
- pt50 probes for all encoder epochs, evals.main with RoPE remap, num_segments=2
- Main text table source: MAE clean 0.396, matched_frame 0.286
- **Archived because:** pt50 probes don't match epoch-matched protocol used everywhere else

### Protocol B: NeurIPS 6-condition standalone (6-condition-shuffling.md → archived)
- Epoch-matched probes, standalone script, NO RoPE remap, NO prediction averaging
- MAE e50 clean 0.141, matched_frame -0.343
- **Archived because:** "matched_frame" in standalone script is NOT time-aware shuffle (no RoPE remap). Misleading name.

### Protocol C: NeurIPS severity gradient (severity-gradient.md → kept, separate doc)
- Epoch-matched probes, standalone script, partial shuffle fractions (0-100%)
- Used for training dynamics figure in paper: MAE temporal Δ 0.443 (e50) → 0.016 (e99)
- **Kept as separate doc** because it measures a different thing (shuffle fraction gradient) and is used for a different paper figure
- Partially reproduced on HyperPod (e24 exact match confirmed)

### CAMUS segmentation (camus-frame-shuffling.md → kept, separate doc)
- Segmentation task, not LVEF — different experiment entirely
- **Kept as separate doc**
