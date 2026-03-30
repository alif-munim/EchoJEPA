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

### LVEF Probes — EchoNet-Dynamic (pt50, d=4 attentive)

| Model | Probe Location | Best Result |
|-------|---------------|-------------|
| **EchoJEPA-L pt50** | **S3**: `s3://.../runs/echojepa_pt50_end_lvef_294/.../best.pt` | **DONE (224px) — R²=0.621, Pearson=0.793, MAE=5.506** |
| **EchoBYOL-L pt50** | Running on A100 (224px) | **IN PROGRESS** ep11/20, R²=0.491, Pearson=0.706 |
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
- **Fix**: `--fix_orientation` flag in `evals/segmentation_frozen/eval.py` applies `rot270 + flipH`. Already implemented in `camus_dataset.py` line 124.
- **Retraining**: G (384px) → `results/segmentation/echojepa_g_384_fixed_orient/`. L pt50 (224px) → `results/segmentation/echojepa_l_pt50_fixed_orient/`. Both in progress.
- **Visual reference**: `scripts/rebuttal/samples/camus_orientation_fix_comparison.png` (left=original, right=fixed).

**⚠️ Resolution Confound (general):** Comparing results across datasets at different resolutions is invalid. UHN probes train at 224px (1568 tokens); EchoNet-Pediatric native is 112px (392 tokens). All cross-dataset comparisons must use matched resolution. See §5k in doc 10 for the pediatric resolution confound and retraining at 224px.

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
| EchoMAE-L pt50 | S3: `.../echomae_pt50_lvef_274/.../best.pt` | — | — | — | — | **NOT RUN** (probe done, needs inference config) |

### UHN RVSP Test (5,103 studies)

| Model | Probe | Inference Config | Predictions CSV | Test R² | Test Pearson | Status |
|-------|-------|-----------------|----------------|---------|-------------|--------|
| EchoJEPA-L pt50 | EFS: `.../icml-echojepa-l-pt50-rvsp-d4-full/best.pt` | `configs/inference/vitl/icml/echojepa_l_pt50_rvsp_test.yaml` | `predictions/icml-echojepa-l-pt50-rvsp-test.csv` | 0.220 | 0.484 | DONE |
| EchoMAE-L pt50 | S3: `.../echomae_pt50_rvsp_260/.../best.pt` | — | — | — | — | **NOT RUN** (probe done, needs inference config) |
| EchoBYOL-L pt50 | — | — | — | — | — | **BLOCKED** (no probe — BYOL RVSP killed ep1) |

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
| EchoJEPA-L pt50 (224px) | Retraining on A100 | — | — | — | — | **PROBE IN PROGRESS** (ep17/20) |
| EchoBYOL-L pt50 (224px) | Retraining on A100 | — | — | — | — | **PROBE IN PROGRESS** (ep17/20) |
| EchoMAE-L pt50 (224px) | Retraining on A100 | — | — | — | — | **PROBE IN PROGRESS** (ep14/20) |

**⚠️ Invalid 112px runs** (resolution artifact — do not use):
- `predictions/icml-echojepa-l-pt50-enp-lvef-test.csv` (112px, MAE=6.598)
- `predictions/icml-echobyol-l-pt50-enp-lvef-test.csv` (112px, MAE=5.618)
- `predictions/icml-echomae-l-pt50-enp-lvef-test.csv` (112px, MAE=6.776)
- `predictions/icml-echojepa-l-pt50-enp-lvef-zeroshot.csv` (112px, UHN probe at wrong resolution)
- `predictions/icml-echobyol-l-pt50-enp-lvef-zeroshot.csv` (112px, UHN probe at wrong resolution)

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

| CSV | Labels | Notes |
|-----|--------|-------|
| `echonet_dynamic_train_s3_raw.csv` | Raw LVEF (mean=55.78, std=12.41) | 7,465 videos, S3 paths |
| `echonet_dynamic_val_s3_raw.csv` | Raw LVEF | 1,288 videos |
| `echonet_dynamic_test_s3_raw.csv` | Raw LVEF | 1,277 videos |
| `echonet_pediatric_train_s3_raw.csv` | Raw LVEF (mean=61.03, std=10.44) | 2,580 videos, folds 0-7 |
| `echonet_pediatric_val_s3_raw.csv` | Raw LVEF | 336 videos, fold 8 |
| `echonet_pediatric_test_s3_raw.csv` | Raw LVEF | 368 videos, fold 9 |
| `echonet_dynamic_train_s3.csv` | Pre-z-scored | Legacy, do not use |
| `rebuttal/lvef/lvef_train_10k.csv` | Raw LVEF (mean=57.06, std=11.28) | UHN 10K rebuttal subset |
| `rebuttal/lvef/lvef_val_1k.csv` | Raw LVEF | UHN 1K rebuttal subset |
| `rvsp_train.csv` / `rvsp_val.csv` | Raw RVSP (mean=34.47, std=14.01) | UHN full 41K/5K |

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
