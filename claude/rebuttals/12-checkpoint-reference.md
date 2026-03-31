# ICML Rebuttal — Checkpoint Reference

All encoder and probe checkpoint locations for rebuttal experiments.
Paths are relative to the repo root unless marked with full path.

---

## 1. Pretrained Encoder Checkpoints

### Rebuttal 3-Way Comparison (ViT-L, MIMIC, 50ep)

| Model | Checkpoint | Notes |
|-------|-----------|-------|
| EchoJEPA-L (50ep) | `checkpoints/echojepa-l-pt50.pt` | V-JEPA 2.0, key: `target_encoder` |
| EchoBYOL-L (50ep) | `checkpoints/byol_vitl_imagenet_v2_e50.pt` | BYOL-Video v2, key: `target_encoder` |
| EchoMAE-L (50ep) | `checkpoints/videomae_l_mimic_ep50.pth` | VideoMAE, key: `model` (pretrain format, auto-converted) |

**S3 copies for HyperPod** (uploaded 2026-03-29):

| Model | S3 Path |
|-------|---------|
| EchoJEPA-L (50ep) | `s3://.../vjepa2-artifacts/checkpoints/echojepa_l_mimic_ep50.pt` |
| EchoBYOL-L (50ep) | `s3://.../vjepa2-artifacts/checkpoints/echobyol_l_mimic_ep50.pt` |
| EchoMAE-L (50ep) | `s3://.../vjepa2-artifacts/runs/videomae_matched_2n_245/training_folder/checkpoint-49.pth` |

S3 bucket: `sagemaker-hyperpod-lifecycle-495467399120-usw2`

### Other EchoMAE-L

| Model | Checkpoint | Notes |
|-------|-----------|-------|
| EchoMAE-L (ep99) | `checkpoints/echomae_l_mimic_ep99.pth` | Used for initial LVEF/View probes |
| EchoMAE-L (ep163) | `checkpoints/videomae-ep163.pth` | Fully-trained, used for RVSP + CAMUS |

### EchoJEPA-L Scaling Checkpoints

| Model | Checkpoint | Notes |
|-------|-----------|-------|
| EchoJEPA-L (pt210+an25) | `checkpoints/vitl.pt` | Fully-trained (235 total epochs) |
| EchoJEPA-L-K | `checkpoints/vitl_in21k.pt` | Kinetics pretrain → MIMIC (275 total epochs) |
| EchoJEPA-L (pt50-pt230) | `checkpoints/echojepa-l-pt{50,70,90,110,150,180,200,220}.pt` | Intermediate checkpoints |
| EchoJEPA-L (anneal) | `checkpoints/echojepa-l-pt230-an{10,20,30}.pt` | Annealing phase |

### EchoJEPA-B (V-JEPA 2.1)

| Model | Checkpoint | Notes |
|-------|-----------|-------|
| EchoJEPA-B (p169+c60) | `checkpoints/vjepa2_1_vitb_mimic_p169_c60.pt` | V-JEPA 2.1, 229 total epochs |
| EchoJEPA-B (p169+c52) | `checkpoints/vjepa2_1_vitb_mimic_p169_c52.pt` | Earlier cooldown checkpoint |

### EchoJEPA-G / Other

| Model | Checkpoint | Notes |
|-------|-----------|-------|
| EchoJEPA-G (384px) | `checkpoints/vitg-384.pt` | ViT-g, 384px, UHN 18M |
| SSv2-ViTG-384 | `checkpoints/ssv2-vitg-384-64x2x3.pt` | Meta reference checkpoint |

### BYOL-Video Pretraining Checkpoints

| Checkpoint | Notes |
|-----------|-------|
| `checkpoints/byol_vitl_e0.pt` | Random init / epoch 0 |
| `checkpoints/byol_vitl_e10.pt` | Epoch 10 |
| `checkpoints/byol_vitl_e44.pt` | Epoch 44 |
| `checkpoints/byol_vitl_e50.pt` | Epoch 50 (v1, no ImageNet init) |
| `checkpoints/byol_vitl_imagenet_v2_e50.pt` | **v2, 50ep, ImageNet init — used in rebuttal** |
| `checkpoints/byol_vitl_latest.pt` | Latest (in-progress pretrain) |

---

## 2. ICML Rebuttal Probe Checkpoints

### LVEF Probes — UHN (Single-View, d=4 attentive)

| Model | Probe Location | Best Result |
|-------|---------------|-------------|
| EchoJEPA-L pt50 | `evals/vitb/icml/echojepa_l_pt50_lvef/.../icml-echojepa-l-pt50-lvef-d4/best.pt` (EFS) | R²=0.436, test R²=0.409 |
| EchoBYOL-L pt50 | `evals/vitb/icml/byol_pt50_lvef/.../icml-echobyol-l-pt50-lvef-d4/best.pt` (EFS) | R²=0.421, test R²=0.384 |
| **EchoMAE-L pt50** | **S3**: `s3://.../runs/echomae_pt50_lvef_274/.../icml-echomae-l-pt50-lvef-d4/best.pt` | **R²=0.325, Pearson=0.584, MAE=6.866** (HyperPod job 274) |
| EchoMAE-L ep99 | `evals/vitb/icml/echomae_lvef/.../icml-echomae-l-lvef-d4/best.pt` (EFS) | R²≈0 (no signal) |
| EchoJEPA-B | `evals/vitb/icml/lvef/.../icml-echojepa-b-lvef-d4/best.pt` (EFS) | R²=0.650 |

### LVEF Probes — UHN Biplane (Multi-View A4C+A2C, d=4 attentive)

| Model | Probe Location (S3) | Best Result |
|-------|---------------------|-------------|
| **EchoJEPA-L pt50** | `s3://.../runs/echojepa_pt50_biplane_lvef_310/.../best.pt` | **IN PROGRESS** (HyperPod job 310) |
| **EchoBYOL-L pt50** | `s3://.../runs/echobyol_pt50_biplane_lvef_311/.../best.pt` | **IN PROGRESS** (HyperPod job 311) |

Data: 9,990 train / 1,000 val (biplane A4C+A2C, matched to rebuttal subset). Configs: `configs/eval/vitl/icml/echo{jepa,byol}_l_pt50_biplane_lvef_d4.yaml`. Sbatch: `scripts/echo{jepa,byol}_pt50_biplane_lvef_probe.sbatch`.

Single-view comparison: JEPA R²=0.436, BYOL R²=0.421 (same data, A4C only).

### LVEF Probes — EchoNet-Dynamic (pt50, d=4 attentive)

| Model | Probe Location | Best Result |
|-------|---------------|-------------|
| **EchoJEPA-L pt50** | **S3**: `s3://.../runs/echojepa_pt50_end_lvef_294/.../best.pt` | **DONE (224px) — R²=0.621, Pearson=0.793, MAE=5.506** |
| **EchoBYOL-L pt50** | `evals/vitl/icml/echobyol_pt50_end_lvef_224/.../best.pt` (EFS) | **DONE (224px) — R²=0.528, Pearson=0.729, MAE=6.174** |
| **EchoMAE-L pt50** | **S3**: `s3://.../runs/echomae_pt50_end_lvef_296/.../best.pt` | **DONE (224px) — R²=0.495, Pearson=0.706, MAE=6.410** |

### RVSP Probes (Multi-View, d=4 attentive)

| Model | Probe Location | Best Result |
|-------|---------------|-------------|
| EchoJEPA-L pt50 (5K) | `evals/vitl/icml/rvsp/.../icml-echojepa-l-pt50-rvsp-d4/best.pt` (EFS) | Pearson=0.376 (insufficient) |
| **EchoJEPA-L pt50 (41K)** | `evals/vitl/icml/rvsp/.../icml-echojepa-l-pt50-rvsp-d4-full/best.pt` (EFS) | **DONE — Pearson=0.504, R²=0.241, MAE=9.044** |
| EchoBYOL-L pt50 (5K) | `evals/vitl/icml/byol_pt50_rvsp/.../icml-echobyol-l-pt50-rvsp-d4/best.pt` (EFS) | — |
| EchoBYOL-L pt50 (41K) | `evals/vitl/icml/byol_pt50_rvsp_full/.../icml-echobyol-l-pt50-rvsp-d4-full/best.pt` (EFS) | Killed ep1, needs restart |
| **EchoMAE-L pt50 (41K)** | **S3**: `s3://.../runs/echomae_pt50_rvsp_260/.../icml-echomae-l-pt50-rvsp-d4/best.pt` | **DONE — Pearson=0.453, R²=0.198, MAE=9.287** |
| EchoMAE-L ep99 (5K) | `evals/vitb/icml/echomae_rvsp/.../icml-echomae-l-rvsp-d4/best.pt` (EFS) | — |
| EchoMAE-L ep163 (41K) | `evals/vitb/icml/echomae_rvsp_ep163/.../icml-echomae-l-rvsp-d4-ep163-full/best.pt` (EFS) | Paused ep2 |

### RVSP Probes — Single-View Ablation (d=4 attentive, 41K train / 5K val)

| Model | View | Probe Location (S3) | Best Val Pearson | Best Val R² | Best Val MAE |
|-------|------|---------------------|-----------------|-------------|-------------|
| EchoJEPA-L pt50 | A4C | `s3://.../runs/echojepa_pt50_rvsp_a4c_301/.../icml-echojepa-l-pt50-rvsp-a4c-d4/best.pt` | **0.492** (ep18) | 0.224 (ep17) | 9.173 (ep15) |
| EchoJEPA-L pt50 | PSAX-AV | `s3://.../runs/echojepa_pt50_rvsp_psax_305/.../icml-echojepa-l-pt50-rvsp-psax-d4/best.pt` | **0.478** (ep16) | 0.212 (ep15) | 9.198 (ep17) |

**Multi-view comparison (same encoder):** Pearson=0.504, R²=0.241, MAE=9.044. Multi-view advantage: +1.2pp Pearson, +1.7pp R² over A4C single-view.

Configs: `configs/eval/vitl/icml/echojepa_l_pt50_rvsp_{a4c,psax}_d4.yaml`. Sbatch: `scripts/echojepa_pt50_rvsp_{a4c,psax}_probe.sbatch`.

**Test inference pending** — need to build `rvsp_test_a4c.csv` / `rvsp_test_psax.csv` from `rvsp_multiview_test.csv`.

### RVSP Probes (Prior / Fully-Trained Encoders)

| Model | Probe Location | Notes |
|-------|---------------|-------|
| EchoJEPA-L (pt210-an25, 5K) | `evals/vitl/icml/rvsp/.../icml-echojepa-l-mimic-rvsp-d4/best.pt` | — |
| EchoJEPA-L (pt210-an25, 41K) | `evals/vitl/icml/rvsp/.../icml-echojepa-l-mimic-full-rvsp-d4/best.pt` | Killed ep9, Pearson=0.504 |
| EchoJEPA-L (pt210-an25, UHN 41K) | `evals/vitl/icml/rvsp/.../icml-echojepa-l-full-rvsp-d4-uhn41k/best.pt` | — |
| EchoJEPA-L v2 | `evals/vitl/icml/rvsp/.../icml-echojepa-l-mimic-v2-rvsp-d4/best.pt` | — |
| EchoJEPA-L-K | `evals/vitl/icml/lvef/.../icml-echojepa-l-k-lvef-d4/best.pt` | — |

### EchoNet-Pediatric LVEF Probes (pt50 3-Way, Rebuttal)

All d=4 attentive, 6 HP heads, 20 epochs, raw-label S3 CSVs (mean=61.03, std=10.44).

| Model | Config | Probe Location | Best Val MAE |
|-------|--------|---------------|-------------|
| EchoJEPA-L pt50 | `configs/eval/vitl/icml/echojepa_l_pt50_enp_lvef_d4.yaml` | Retraining on A100 (224px) | 6.130 (ep11, 224px in progress) |
| EchoBYOL-L pt50 | `configs/eval/vitl/icml/echobyol_l_pt50_enp_lvef_d4.yaml` | Retraining on A100 (224px) | 6.184 (ep14, 224px in progress) |
| EchoMAE-L pt50 | `configs/eval/vitl/icml/echomae_l_pt50_enp_lvef_d4.yaml` | Retraining on A100 (224px) | **6.081** (ep11, 224px in progress) |

### View Classification Probes

| Model | Probe Location |
|-------|---------------|
| EchoMAE-L ep99 | `evals/vitb/icml/echomae_view/.../icml-echomae-l-view-d4/best.pt` |
| EchoJEPA-L (pt210-an25) | `evals/vitl/icml/view/.../icml-echojepa-l-mimic-view-d4/best.pt` |

---

## 3. CAMUS Segmentation Probe Checkpoints

All in `results/segmentation/<model>/<lr_wd>/best_decoder.pt` with `grid_summary.json` per model.

### pt50 3-Way (Rebuttal)

| Model | Best Config | Test Dice | Location |
|-------|------------|-----------|----------|
| EchoMAE-L pt50 | lr1e-02_wd1e-04 | **0.822** | `results/segmentation/echomae_l_pt50/lr1e-02_wd1e-04/best_decoder.pt` |
| EchoBYOL-L pt50 | lr5e-02_wd1e-04 | 0.821 | `results/segmentation/echobyol_l_pt50/lr5e-02_wd1e-04/best_decoder.pt` |
| EchoJEPA-L pt50 | lr5e-02_wd1e-04 | 0.815 | `results/segmentation/echojepa_l_pt50/lr5e-02_wd1e-04/best_decoder.pt` |

### Fully-Trained Models

| Model | Best Config | Test Dice | Location |
|-------|------------|-----------|----------|
| EchoJEPA-L (235ep) | lr5e-02_wd1e-04 | 0.818 | `results/segmentation/echojepa_l/lr5e-02_wd1e-04/best_decoder.pt` |
| EchoMAE-L (163ep) | — | 0.790 | `results/segmentation/echomae_l/*/best_decoder.pt` |
| EchoJEPA-L-K | — | 0.746 | `results/segmentation/echojepa_l_k/*/best_decoder.pt` |
| PanEcho | — | 0.734 | `results/segmentation/panecho/*/best_decoder.pt` |
| EchoJEPA-G (384px) | — | 0.729 | `results/segmentation/echojepa_g_384_fixed/*/best_decoder.pt` |
| EchoPrime | — | 0.669 | `results/segmentation/echoprime/*/best_decoder.pt` |

**⚠️ CAMUS Orientation Issue:** CAMUS A4C images have a ~45° clockwise-rotated sector scan (apex top-left, LV center-left, LA right). This differs from standard North American (UHN) A4C convention (sector pointing downward, apex top-center). The issue is a rotation, not just a horizontal flip. The frozen encoder (pretrained on UHN) has RoPE positional embeddings that encode absolute spatial position — rotated inputs produce different representations. Impact:
- **G (384px)** most affected: higher spatial precision = more sensitive to orientation. Also has resolution mismatch (384px pretrain → 224px eval).
- **L (224px)** partially compensated: random horizontal flip during probe training helps the decoder learn some spatial invariance, and lower spatial precision reduces sensitivity.
- **Fix attempted**: `--fix_orientation` flag (rot270 + flipH). Results: **G worsened** (0.729→0.606), L pt50 marginal improvement (0.815→0.826). The fix is WRONG for G — the G model learned orientation-invariant features from 18M diverse echos. Do not apply for G.
- **Output dirs**: G → `results/segmentation/echojepa_g_384_fixed_orient/` (test Dice 0.606, DO NOT USE). L pt50 → `results/segmentation/echojepa_l_pt50_fixed_orient/` (test Dice 0.826, marginal improvement).
- **Conclusion**: G < L gap is primarily resolution mismatch (384px→224px), not orientation.
- **Visual reference**: `scripts/rebuttal/samples/camus_orientation_fix_comparison.png`.

**⚠️ Resolution Note (EchoNet-Pediatric):** EchoNet-Pediatric is natively 112px. We tested both 112px and 224px probes — results are nearly identical (JEPA R²=0.157 at 112px vs 0.123 at 224px). Resolution is NOT the issue for the pediatric pt50 results. The fully-trained JEPA (pt210-an25) gets R²=0.568 at 112px on the same test set, confirming the pt50 encoder simply needs more pretraining for pediatric transfer. UHN probes train at 224px; EchoNet-Dynamic probes at 224px (upscaled from native 112px, works fine with 7.5K training data).

---

## 4. Fully-Trained Probes (ICML Preprint / EchoBench)

Consolidated in `checkpoints/eval_probes/`, organized by task.

### LVEF — UHN

| Model | Probe |
|-------|-------|
| EchoJEPA-G (336px) | `checkpoints/eval_probes/lvef/echojepa_336px.pt` |
| EchoJEPA-G (224px) | `checkpoints/eval_probes/lvef/echojepa_g_224px.pt` |
| EchoJEPA-L | `checkpoints/eval_probes/lvef/echojepa-l-uhn.pt` |
| EchoPrime | `checkpoints/eval_probes/lvef/echoprime_224px.pt` |
| PanEcho | `checkpoints/eval_probes/lvef/panecho_224px.pt` |
| VideoMAE | `checkpoints/eval_probes/lvef/videomae_224px.pt` |

### LVEF — EchoNet-Dynamic (for EchoBench)

| Model | Probe |
|-------|-------|
| EchoJEPA-G | `checkpoints/eval_probes/lvef/echonet-dynamic/echojepa-g.pt` |
| EchoJEPA-L | `checkpoints/eval_probes/lvef/echonet-dynamic/echojepa-l.pt` |
| EchoPrime | `checkpoints/eval_probes/lvef/echonet-dynamic/echoprime.pt` |
| PanEcho | `checkpoints/eval_probes/lvef/echonet-dynamic/panecho.pt` |
| VideoMAE | `checkpoints/eval_probes/lvef/echonet-dynamic/videomae.pt` |

### LVEF — EchoNet-Pediatric (for EchoBench)

| Model | Probe |
|-------|-------|
| EchoJEPA-G | `checkpoints/eval_probes/lvef/echonet-pediatric/echojepa-g.pt` |
| EchoJEPA-L | `checkpoints/eval_probes/lvef/echonet-pediatric/echojepa-l.pt` |
| EchoPrime | `checkpoints/eval_probes/lvef/echonet-pediatric/echoprime.pt` |
| PanEcho | `checkpoints/eval_probes/lvef/echonet-pediatric/panecho.pt` |
| VideoMAE | `checkpoints/eval_probes/lvef/echonet-pediatric/videomae.pt` |

### RVSP — UHN (Fully-Trained)

| Model | Probe |
|-------|-------|
| EchoJEPA-G (224px) | `checkpoints/eval_probes/rvsp/echojepa_224px.pt` |
| EchoJEPA-G (336px) | `checkpoints/eval_probes/rvsp/echojepa_336px.pt` |
| EchoJEPA-L | `checkpoints/eval_probes/rvsp/echojepa-l-rvsp.pt` |
| EchoPrime | `checkpoints/eval_probes/rvsp/echoprime_224px.pt` |
| PanEcho | `checkpoints/eval_probes/rvsp/panecho_224px.pt` |
| VideoMAE (ep16) | `checkpoints/eval_probes/rvsp/videomae_ep16.pt` |

### View Classification — UHN

| Model | Probe |
|-------|-------|
| EchoJEPA-G (224px) | `checkpoints/eval_probes/classification/echojepa_224px.pt` |
| EchoJEPA-G (224px, multi) | `checkpoints/eval_probes/classification/echojepa_224px_multi.pt` |
| EchoPrime | `checkpoints/eval_probes/classification/echoprime_224px.pt` |
| PanEcho | `checkpoints/eval_probes/classification/panecho_224px.pt` |
| VideoMAE | `checkpoints/eval_probes/classification/videomae_224px.pt` |

### RVSP Ablation Probes

| Ablation | Probe |
|----------|-------|
| No slot embeddings | `checkpoints/eval_probes/rvsp/vjepa-g-rvsp-nse.pt` |
| No slot + late fusion | `checkpoints/eval_probes/rvsp/vjepa-g-rvsp-nse-lf.pt` |
| No slot + late fusion + no MSA | `checkpoints/eval_probes/rvsp/vjepa-g-rvsp-nse-lf-nmsa.pt` |

---

## 5. Test Set Inference Tracker

Tracks which probes have been run on held-out test sets, with prediction CSV locations.

### UHN LVEF Test (53,637 clips)

| Model | Probe | Inference Config | Predictions CSV | Test R² | Test Pearson | Status |
|-------|-------|-----------------|----------------|---------|-------------|--------|
| EchoJEPA-L pt50 | EFS: `.../icml-echojepa-l-pt50-lvef-d4/best.pt` | `configs/inference/vitl/icml/echojepa_l_pt50_lvef_test.yaml` | `predictions/icml/echojepa_l_pt50_lvef_test.csv` | 0.409 | 0.650 | DONE |
| EchoBYOL-L pt50 | EFS: `.../icml-echobyol-l-pt50-lvef-d4/best.pt` | `configs/inference/vitl/icml/echobyol_l_pt50_lvef_test.yaml` | `predictions/icml/echobyol_l_pt50_lvef_test.csv` | 0.384 | 0.625 | DONE |
| EchoMAE-L pt50 | EFS: `.../echomae_pt50_lvef/.../icml-echomae-l-pt50-lvef-d4/best.pt` | `/tmp/echomae_lvef_test.yaml` | `predictions/icml-echomae-l-pt50-lvef-test.csv` | 0.283 | 0.572 | **DONE** |

### UHN RVSP Test (5,103 studies)

| Model | Probe | Inference Config | Predictions CSV | Test R² | Test Pearson | Status |
|-------|-------|-----------------|----------------|---------|-------------|--------|
| EchoJEPA-L pt50 | EFS: `.../icml-echojepa-l-pt50-rvsp-d4-full/best.pt` | `configs/inference/vitl/icml/echojepa_l_pt50_rvsp_test.yaml` | `predictions/icml-echojepa-l-pt50-rvsp-test.csv` | 0.220 | 0.484 | DONE |
| EchoMAE-L pt50 | S3: `.../echomae_pt50_rvsp_260/.../best.pt` | — | — | — | — | **NOT RUN** (probe done, needs inference config) |
| EchoBYOL-L pt50 | — | — | — | — | — | **BLOCKED** (no probe — BYOL RVSP killed ep1) |

### UHN RVSP Single-View Test (pending — test CSVs need building)

| Model | View | Probe (S3) | Test CSV | Status |
|-------|------|-----------|----------|--------|
| EchoJEPA-L pt50 | A4C | `.../echojepa_pt50_rvsp_a4c_301/.../best.pt` | `rvsp_test_a4c.csv` (NOT BUILT) | **BLOCKED** (need test CSV) |
| EchoJEPA-L pt50 | PSAX-AV | `.../echojepa_pt50_rvsp_psax_305/.../best.pt` | `rvsp_test_psax.csv` (NOT BUILT) | **BLOCKED** (need test CSV) |

### EchoNet-Dynamic LVEF Test (1,277 videos)

| Model | Probe | Predictions CSV | Test R² | Test Pearson | Status |
|-------|-------|----------------|---------|-------------|--------|
| EchoJEPA-L pt50 (224px) | S3: `.../echojepa_pt50_end_lvef_294/.../best.pt` | — | — | — | **NOT RUN** (probe done) |
| EchoMAE-L pt50 (224px) | S3: `.../echomae_pt50_end_lvef_296/.../best.pt` | — | — | — | **NOT RUN** (probe done) |
| EchoBYOL-L pt50 (224px) | Running on A100 | — | — | — | **PROBE IN PROGRESS** (ep11/20) |

### EchoNet-Pediatric LVEF Test (368 videos)

| Model | Probe | Predictions CSV | Test MAE | Test R² | Test Pearson | Status |
|-------|-------|----------------|----------|---------|-------------|--------|
| EchoJEPA-L (pt210-an25, fine-tuned) | `checkpoints/eval_probes/lvef/echonet-pediatric/echojepa-l.pt` | `predictions/echojepa-l-echonet-pediatric-lvef-test.csv` | 5.122 | 0.568 | 0.763 | DONE (reproduces preprint 5.12) |
| EchoJEPA-L (pt210-an25, zero-shot) | `checkpoints/eval_probes/lvef/echonet-dynamic/echojepa-l.pt` | `predictions/echojepa-l-echonet-pediatric-lvef-zeroshot.csv` | 7.713 | — | 0.402 | DONE (preprint claims 6.31 — different eval pipeline) |
| **EchoBYOL-L pt50 (112px)** | `evals/vitl/icml/enp_lvef/.../icml-echobyol-l-pt50-enp-lvef-d4/best.pt` | `predictions/icml-echobyol-l-pt50-enp-lvef-test.csv` | 5.618 | **0.415** | **0.668** | **DONE** (112px is correct — see §5k) |
| EchoJEPA-L pt50 d=4 (112px) | `.../icml-echojepa-l-pt50-enp-lvef-d4/best.pt` | `predictions/icml-echojepa-l-pt50-enp-lvef-test.csv` | 6.598 | 0.157 | 0.489 | DONE (d=4 overfitting — use d=1 instead) |
| **EchoJEPA-L pt50 d=1 (224px, 50ep)** | `.../icml-echojepa-l-pt50-enp-lvef-d1-224px/best.pt` | `predictions/icml-echojepa-l-pt50-enp-lvef-d1-224px-test.csv` | **5.167** | **0.568** | **0.771** | **DONE — matches fully-trained probe** |
| EchoMAE-L pt50 d=4 (112px) | `.../icml-echomae-l-pt50-enp-lvef-d4/best.pt` | `predictions/icml-echomae-l-pt50-enp-lvef-test.csv` | 6.776 | -0.065 | 0.195 | DONE (collapsed) |

**d=4 probe overfitting on pediatric:** d=4 attentive probes overfit on 2,580 training samples — JEPA R² drops from 0.568 (d=1) to 0.123 (d=4). d=1 (cross-attention only, no SA blocks, 50 epochs) is the correct probe for small-data cross-population transfer. See §5k in doc 10 for full analysis.

**224px retrain (inconclusive — resolution was not the issue):**
- `predictions/icml-echojepa-l-pt50-enp-lvef-test-224px.csv` (224px, R²=0.123 — same attenuation as 112px)
- `predictions/icml-echobyol-l-pt50-enp-lvef-test-224px.csv` (224px, R²=0.316)
- `predictions/icml-echomae-l-pt50-enp-lvef-test-224px.csv` (224px, R²=0.247)

**Zero-shot runs (112px, UHN probes → pediatric — cross-resolution, less reliable):**
- `predictions/icml-echojepa-l-pt50-enp-lvef-zeroshot.csv`
- `predictions/icml-echobyol-l-pt50-enp-lvef-zeroshot.csv`

---

## 6. Eval Config Reference

All rebuttal configs in `configs/eval/vit{b,l}/icml/`. See `10-rebuttal-experiment-results.md` §6 for the full config-to-checkpoint mapping table.

### Key Configs (Rebuttal 3-Way)

| Task | Config |
|------|--------|
| JEPA pt50 LVEF | `configs/eval/vitb/icml/echojepa_l_pt50_lvef_d4.yaml` |
| BYOL pt50 LVEF | `configs/eval/vitb/icml/echobyol_l_pt50_lvef_d4.yaml` |
| MAE pt50 LVEF | `configs/eval/vitb/icml/echomae_l_pt50_lvef_d4.yaml` |
| JEPA pt50 RVSP (41K) | `configs/eval/vitl/icml/echojepa_l_pt50_rvsp_d4_full.yaml` |
| BYOL pt50 RVSP (41K) | `configs/eval/vitl/icml/echobyol_l_pt50_rvsp_d4_full.yaml` |
| MAE pt50 RVSP (41K) | `configs/eval/vitl/icml/echomae_l_pt50_rvsp_d4_full.yaml` |
| **JEPA pt50 RVSP test inference** | `configs/inference/vitl/icml/echojepa_l_pt50_rvsp_test.yaml` |
| JEPA pt50 RVSP A4C (single-view) | `configs/eval/vitl/icml/echojepa_l_pt50_rvsp_a4c_d4.yaml` |
| JEPA pt50 RVSP PSAX (single-view) | `configs/eval/vitl/icml/echojepa_l_pt50_rvsp_psax_d4.yaml` |
| JEPA pt50 Biplane LVEF (A4C+A2C) | `configs/eval/vitl/icml/echojepa_l_pt50_biplane_lvef_d4.yaml` |
| BYOL pt50 Biplane LVEF (A4C+A2C) | `configs/eval/vitl/icml/echobyol_l_pt50_biplane_lvef_d4.yaml` |

### Key Configs (EchoBench — EchoNet-Dynamic/Pediatric)

| Task | Config |
|------|--------|
| JEPA pt50 EchoNet-Dynamic LVEF | `configs/eval/vitl/icml/echojepa_l_pt50_end_lvef_d4.yaml` |
| BYOL pt50 EchoNet-Dynamic LVEF | `configs/eval/vitl/icml/echobyol_l_pt50_end_lvef_d4.yaml` |
| MAE pt50 EchoNet-Dynamic LVEF | `configs/eval/vitl/icml/echomae_l_pt50_end_lvef_d4.yaml` |
| JEPA pt50 EchoNet-Pediatric LVEF | `configs/eval/vitl/icml/echojepa_l_pt50_enp_lvef_d4.yaml` |
| BYOL pt50 EchoNet-Pediatric LVEF | `configs/eval/vitl/icml/echobyol_l_pt50_enp_lvef_d4.yaml` |
| MAE pt50 EchoNet-Pediatric LVEF | `configs/eval/vitl/icml/echomae_l_pt50_enp_lvef_d4.yaml` |

### HyperPod Data CSVs (S3)

All at `s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts/data/csv/`.

#### UHN LVEF (single-view A4C)

| CSV | Format | Rows | Notes |
|-----|--------|------|-------|
| `rebuttal/lvef/lvef_train_10k.csv` | `a4c_s3_path label` | 10,000 | Rebuttal subset, raw LVEF (mean=57.07, std=11.28) |
| `rebuttal/lvef/lvef_val_1k.csv` | `a4c_s3_path label` | 1,000 | Rebuttal subset |
| `a4c_b_lvef_train_224px.csv` | `a4c_s3_path label` | 176,791 | Full UHN A4C B-mode, 224px pre-resized |
| `a4c_b_lvef_val_224px.csv` | `a4c_s3_path label` | 26,166 | Full UHN |
| `a4c_b_lvef_test_224px.csv` | `a4c_s3_path label` | 53,637 | Full UHN test |

#### UHN LVEF (biplane A4C + A2C)

| CSV | Format | Rows | Notes |
|-----|--------|------|-------|
| `rebuttal/lvef/biplane_lvef_train_10k.csv` | `a4c_path a2c_path label` | 9,990 | Rebuttal subset matched, 99.9% coverage |
| `rebuttal/lvef/biplane_lvef_val_1k.csv` | `a4c_path a2c_path label` | 1,000 | 100% coverage |
| `rebuttal/lvef/biplane_lvef_test.csv` | `a4c_path a2c_path label` | 53,611 | Full test, 100% coverage |
| `biplane_lvef_train.csv` | `a4c_path a2c_path label` | 34,792 | Full UHN study-level |
| `biplane_lvef_val.csv` | `a4c_path a2c_path label` | 5,013 | Full UHN study-level |
| `biplane_lvef_test.csv` | `a4c_path a2c_path label` | 10,039 | Full UHN study-level |

A4C = highest-confidence A4C per study. A2C = highest-confidence A2C per study. Both from 18M view classifier predictions. Built for `VideoGroupDataset` (multi-view probing).

#### UHN RVSP (multi-view A4C + PSAX-AV)

| CSV | Format | Rows | Notes |
|-----|--------|------|-------|
| `rvsp_train.csv` | `a4c_path psax_path label` | 40,969 | Full 41K, raw RVSP (mean=34.47, std=14.01) |
| `rvsp_val.csv` | `a4c_path psax_path label` | 5,102 | Full 5K |
| `rvsp_test.csv` | `a4c_path psax_path label` | 5,103 | Full test |

#### UHN RVSP (single-view ablation)

| CSV | Format | Rows | Notes |
|-----|--------|------|-------|
| `rvsp_train_a4c.csv` | `a4c_path label` | 40,969 | Column 1 of multi-view train |
| `rvsp_val_a4c.csv` | `a4c_path label` | 5,102 | Column 1 of multi-view val |
| `rvsp_test_a4c.csv` | `a4c_path label` | 5,103 | Column 1 of multi-view test |
| `rvsp_train_psax.csv` | `psax_path label` | 40,969 | Column 2 of multi-view train |
| `rvsp_val_psax.csv` | `psax_path label` | 5,102 | Column 2 of multi-view val |
| `rvsp_test_psax.csv` | `psax_path label` | 5,103 | Column 2 of multi-view test |

#### EchoNet-Dynamic LVEF

| CSV | Format | Rows | Notes |
|-----|--------|------|-------|
| `echonet_dynamic_train_s3_raw.csv` | `s3_path label` | 7,465 | Raw LVEF (mean=55.78, std=12.41), from FileList.csv |
| `echonet_dynamic_val_s3_raw.csv` | `s3_path label` | 1,288 | Raw LVEF |
| `echonet_dynamic_test_s3_raw.csv` | `s3_path label` | 1,277 | Raw LVEF |
| `echonet_dynamic_*_s3.csv` | `s3_path z_label` | — | Pre-z-scored, **legacy — do not use** |

#### EchoNet-Pediatric LVEF

| CSV | Format | Rows | Notes |
|-----|--------|------|-------|
| `echonet_pediatric_train_s3_raw.csv` | `s3_path label` | 2,580 | Raw LVEF (mean=61.03, std=10.44), folds 0-7, from FileList.csv |
| `echonet_pediatric_val_s3_raw.csv` | `s3_path label` | 336 | Fold 8 |
| `echonet_pediatric_test_s3_raw.csv` | `s3_path label` | 368 | Fold 9 |
| `echonet_pediatric_*_s3.csv` | `s3_path z_label` | — | Pre-z-scored, **legacy — do not use** |

#### Data Sources

| Dataset | Videos (S3) | FileList.csv (S3) | Archive (GDrive) |
|---------|-------------|-------------------|------------------|
| EchoNet-Dynamic | `.../data/EchoNet-Dynamic/Videos/` (10,031) | `.../data/EchoNet-Dynamic/FileList.csv` | `echo_foundation/nature_medicine/datasets/echonet_data.zip` (6.6GB) |
| EchoNet-Pediatric (A4C) | `.../data/echonetpediatric/.../A4C/Videos/` (3,284) | `.../data/echonetpediatric/.../A4C/FileList.csv` | `echo_foundation/nature_medicine/datasets/echonet_pediatric.tar.gz` (2.1GB) |
| UHN 18M | `s3://echodata25/results/echo-study/` | 18M view classifier: `classifier/output/view_inference_18m/master_predictions.csv` (4GB) | — |

**Z-score convention:** All `*_raw.csv` files have raw label values. The code computes z-score at runtime from training data and embeds params in the checkpoint. Legacy `*_s3.csv` files have pre-z-scored labels — do not use (double z-scoring risk).

**Pediatric split mapping:** FileList.csv folds 0-7 = train (2,580), fold 8 = val (336), fold 9 = test (368).

**Biplane LVEF construction:** For each study, select highest-confidence A4C + A2C clips from 18M view classifier predictions (`classifier/output/view_inference_18m/master_predictions.csv`). Study→clip mapping from `experiments/nature_medicine/uhn/study_to_clips_index.pkl` (319K studies, 18M clips).

### HyperPod Sbatch Scripts

| Script | Task | Node |
|--------|------|------|
| `scripts/echojepa_pt50_end_lvef_probe.sbatch` | JEPA pt50 EchoNet-Dynamic LVEF | node 83 |
| `scripts/echobyol_pt50_end_lvef_probe.sbatch` | BYOL pt50 EchoNet-Dynamic LVEF | node 83 |
| `scripts/echomae_pt50_end_lvef_probe.sbatch` | MAE pt50 EchoNet-Dynamic LVEF | node 83 |
| `scripts/echomae_pt50_lvef_probe.sbatch` | MAE pt50 UHN LVEF | node 83 |
| `scripts/echomae_pt50_rvsp_probe.sbatch` | MAE pt50 UHN RVSP | node 184 |
| `scripts/echomae_pt50_lvef_test.sbatch` | MAE pt50 LVEF test inference | node 83 |

### EchoNet Dataset Sources

| Dataset | Videos (S3) | FileList.csv (S3) | Archive (GDrive) |
|---------|-------------|-------------------|------------------|
| EchoNet-Dynamic | `.../data/EchoNet-Dynamic/Videos/` (10,031) | `.../data/EchoNet-Dynamic/FileList.csv` | `echo_foundation/nature_medicine/datasets/echonet_data.zip` (6.6GB) |
| EchoNet-Pediatric (A4C) | `.../data/echonetpediatric/.../A4C/Videos/` (3,284) | `.../data/echonetpediatric/.../A4C/FileList.csv` | `echo_foundation/nature_medicine/datasets/echonet_pediatric.tar.gz` (2.1GB) |

**Raw-label CSVs** (built from FileList.csv, uploaded to S3 `data/csv/`): `echonet_{dynamic,pediatric}_{train,val,test}_s3_raw.csv`. Z-scored legacy versions (`*_s3.csv`) also exist but should not be used — the code computes z-score at runtime from raw labels.

**Z-score params:** Dynamic mean=55.78, std=12.41. Pediatric mean=61.03, std=10.44. UHN LVEF mean=57.06, std=11.28.

**Pediatric split mapping:** FileList.csv folds 0-7 = train (2,580), fold 8 = val (336), fold 9 = test (368).

### pt50 Encoder Checkpoints on S3

All at `s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts/checkpoints/`:

| S3 Key | Local Path | Model |
|--------|-----------|-------|
| `echojepa_l_mimic_ep50.pt` | `checkpoints/echojepa-l-pt50.pt` | EchoJEPA-L (V-JEPA 2.0, 50ep) |
| `echobyol_l_mimic_ep50.pt` | `checkpoints/byol_vitl_imagenet_v2_e50.pt` | EchoBYOL-L (BYOL-Video v2, 50ep, ImageNet init) |
| `echomae_l_mimic_ep50.pth` | `checkpoints/videomae_l_mimic_ep50.pth` | EchoMAE-L (VideoMAE, 50ep) |
