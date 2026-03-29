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

### LVEF Probes (Single-View, d=4 attentive)

| Model | Probe Location | Best Result |
|-------|---------------|-------------|
| EchoJEPA-L pt50 | `evals/vitb/icml/echojepa_l_pt50_lvef/.../icml-echojepa-l-pt50-lvef-d4/best.pt` | R²=0.436, test R²=0.409 |
| EchoBYOL-L pt50 | `evals/vitb/icml/byol_pt50_lvef/.../icml-echobyol-l-pt50-lvef-d4/best.pt` | R²=0.421, test R²=0.384 |
| EchoMAE-L ep99 | `evals/vitb/icml/echomae_lvef/.../icml-echomae-l-lvef-d4/best.pt` | R²≈0 (no signal) |
| EchoJEPA-B | `evals/vitb/icml/lvef/.../icml-echojepa-b-lvef-d4/best.pt` | R²=0.650 |

### RVSP Probes (Multi-View, d=4 attentive)

| Model | Probe Location | Best Result |
|-------|---------------|-------------|
| EchoJEPA-L pt50 (5K) | `evals/vitl/icml/rvsp/.../icml-echojepa-l-pt50-rvsp-d4/best.pt` | Pearson=0.376 (insufficient) |
| EchoJEPA-L pt50 (41K) | `evals/vitl/icml/rvsp/.../icml-echojepa-l-pt50-rvsp-d4-full/best.pt` | Pearson=0.503 (ep16), **running ep19** |
| EchoBYOL-L pt50 (5K) | `evals/vitl/icml/byol_pt50_rvsp/.../icml-echobyol-l-pt50-rvsp-d4/best.pt` | — |
| EchoBYOL-L pt50 (41K) | `evals/vitl/icml/byol_pt50_rvsp_full/.../icml-echobyol-l-pt50-rvsp-d4-full/best.pt` | Killed ep1, needs restart |
| EchoMAE-L ep99 (5K) | `evals/vitb/icml/echomae_rvsp/.../icml-echomae-l-rvsp-d4/best.pt` | — |
| EchoMAE-L ep163 (41K) | `evals/vitb/icml/echomae_rvsp_ep163/.../icml-echomae-l-rvsp-d4-ep163-full/best.pt` | Paused ep2 |

### RVSP Probes (Prior / Fully-Trained Encoders)

| Model | Probe Location | Notes |
|-------|---------------|-------|
| EchoJEPA-L (pt210-an25, 5K) | `evals/vitl/icml/rvsp/.../icml-echojepa-l-mimic-rvsp-d4/best.pt` | — |
| EchoJEPA-L (pt210-an25, 41K) | `evals/vitl/icml/rvsp/.../icml-echojepa-l-mimic-full-rvsp-d4/best.pt` | Killed ep9, Pearson=0.504 |
| EchoJEPA-L (pt210-an25, UHN 41K) | `evals/vitl/icml/rvsp/.../icml-echojepa-l-full-rvsp-d4-uhn41k/best.pt` | — |
| EchoJEPA-L v2 | `evals/vitl/icml/rvsp/.../icml-echojepa-l-mimic-v2-rvsp-d4/best.pt` | — |
| EchoJEPA-L-K | `evals/vitl/icml/lvef/.../icml-echojepa-l-k-lvef-d4/best.pt` | — |

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

## 5. Eval Config Reference

All rebuttal configs in `configs/eval/vit{b,l}/icml/`. See `10-rebuttal-experiment-results.md` §6 for the full config-to-checkpoint mapping table.

### Key Configs (Rebuttal 3-Way)

| Task | Config |
|------|--------|
| JEPA pt50 LVEF | `configs/eval/vitb/icml/echojepa_l_pt50_lvef_d4.yaml` |
| BYOL pt50 LVEF | `configs/eval/vitb/icml/echobyol_l_pt50_lvef_d4.yaml` |
| JEPA pt50 RVSP (41K) | `configs/eval/vitl/icml/echojepa_l_pt50_rvsp_d4_full.yaml` |
| BYOL pt50 RVSP (41K) | `configs/eval/vitl/icml/echobyol_l_pt50_rvsp_d4_full.yaml` |
| MAE pt50 RVSP (41K) | `configs/eval/vitl/icml/echomae_l_pt50_rvsp_d4_full.yaml` |
