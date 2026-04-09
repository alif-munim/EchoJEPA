# Checkpoint Registry

All model and probe checkpoints used across the project. Descriptive symlinks in `checkpoints/encoders/` point to the original files.

## Encoder Checkpoints

### EchoJEPA (JEPA-trained)

| Descriptive name | Original path | Architecture | Pretraining data | Epochs | Notes |
|------------------|---------------|-------------|------------------|--------|-------|
| `echojepa_g_uhn_pt280_an81.pt` | `anneal/keep/pt-280-an81.pt` | ViT-G (1,012M) | UHN 18M echos | 280 pretrain + 81 anneal | Primary encoder for Nature Medicine. Only model pretrained on UHN. |
| `echojepa_l_mimic_pt210_an25.pt` | `anneal/keep/vitl-pt-210-an25.pt` | ViT-L (304M) | MIMIC-IV-Echo | 210 pretrain + 25 anneal | ICML preprint + Nature Medicine L model. MIMIC-only (not UHN). |
| `echojepa_l_k_mimic_pt220_an55.pt` | `anneal/keep/vitl-kinetics-pt220-an55.pt` | ViT-L (304M) | Kinetics-400 -> MIMIC anneal | 220 pretrain (K400) + 55 anneal (MIMIC) | Kinetics init, MIMIC fine-tune. |
| `echojepa_l_mimic_run125_e100.pt` | `pretrain_21/mimic/vjepa2_1_vitl_224px_16f/e100_run125.pt` | ViT-L (304M), V-JEPA 2.1 | MIMIC-IV-Echo | 100 (stored epoch=101, next-to-resume) | V-JEPA 2.1 architecture. For ICML rebuttal probes. |
| `echojepa_b_mimic_p169_c60.pt` | `vjepa2_1_vitb_mimic_p169_c60.pt` | ViT-B (86M), V-JEPA 2.1 | MIMIC-IV-Echo | 169 pretrain + 60 cooldown | Scaling analysis (B -> L -> G). Rebuttal only. |

### EchoMAE (VideoMAE-trained)

| Descriptive name | Original path | Architecture | Pretraining data | Epochs | Notes |
|------------------|---------------|-------------|------------------|--------|-------|
| `echomae_l_mimic_run115_ep99.pth` | `echomae_l_mimic_ep99.pth` | ViT-L (304M) | MIMIC-IV-Echo | 99 (stored epoch=99, current) | VideoMAE stores `epoch: N` (current epoch convention). |
| `echomae_l_mimic_run115_ep163.pth` | `videomae-ep163.pth` | ViT-L (304M) | MIMIC-IV-Echo | 163 | Later checkpoint, same run. Used for Nature Medicine. |

### EchoBYOL (BYOL-Video-trained)

| Descriptive name | Original path | Architecture | Pretraining data | Epochs | Notes |
|------------------|---------------|-------------|------------------|--------|-------|
| `echobyol_l_mimic_latest.pt` | `byol_vitl_latest.pt` | ViT-L (304M) | MIMIC-IV-Echo | 45/240 (v2) | ICML rebuttal contrastive comparison. H100 2-node (Job 241). ImageNet-21k init, constant EMA 0.99925. S3: `s3://sagemaker-echojepa-h100-march-0d224785-bucket/checkpoints/byol-vitl-imagenet-v2/` |

### SALT (frozen-teacher pixel-reconstruction → latent student)

| Descriptive name | Original path | Architecture | Pretraining data | Epochs | Notes |
|------------------|---------------|-------------|------------------|--------|-------|
| `salt_s2_vitl_e79.pt` **(primary)** | `checkpoints/salt_s2_vitl_e79.pt` | ViT-L (304M), v1 hierarchical predictor | MIMIC-IV-Echo | S1:20 + S2:79 | **Primary SALT checkpoint for NeurIPS** (locked 2026-04-08). Best of three SALT variants. END LVEF test R²=0.414, MAE=6.66. Constant LR 1.75e-4, weak aug. See `claude/neurips/experiments/salt-comparison.md` § FINAL DECISION. |
| `salt_s2_vitl_e199.pt` | `checkpoints/salt_s2_vitl_e199.pt` | ViT-L (304M), v1 hierarchical | MIMIC-IV-Echo | S1:20 + S2:199 | Appendix robustness line on END LVEF only (R²=0.360). Extended v1 run; marginally worse than e79 due to overfitting from constant LR. Do not use for new tasks. |
| SALT v3 e79 (S3-only) | `HYP/runs/salt_s2v2_pretrain_446/checkpoints/e79.pt` | ViT-L (304M), v3 single-level predictor | MIMIC-IV-Echo | S1:20 + S2:79 | Appendix robustness line on END LVEF only (R²=0.348). Paper-spec single-level predictor + cosine LR + paper augmentation. Do not use for new tasks. |

### External Models (not self-supervised by us)

EchoPrime and PanEcho checkpoints are loaded at runtime by their respective model adapters. They are not stored in `checkpoints/encoders/` — each adapter downloads or expects a fixed path.

| Model | Architecture | Training | Adapter |
|-------|-------------|----------|---------|
| EchoPrime | MViT-v2-S | CLIP (echo + text) | `modelcustom/echoprime_encoder_multiclip.py` |
| PanEcho | ConvNeXt-T | Multi-task supervised | `modelcustom/panecho_encoder_multiclip.py` |
| EchoFM | ViT-B | MAE + triplet on echo | `modelcustom/echofm_encoder_multiclip.py` |

### Init Weights (pretrained starting points)

| Descriptive name | Original path | Source | Used by |
|------------------|---------------|--------|---------|
| `vjepa2_vitl_meta.pt` | `vitl.pt` | Meta V-JEPA 2 ViT-L | EchoJEPA-L (pt210_an25) |
| `vjepa2_vitg384_meta.pt` | `vitg-384.pt` | Meta V-JEPA 2 ViT-G 384px | EchoJEPA-G (pt280_an81) |
| `vjepa2_1_vitl_meta.pt` | `vjepa2_1_vitl.pt` | Meta V-JEPA 2.1 ViT-L | EchoJEPA-L run125 |
| `vjepa2_1_vitb_meta.pt` | `vjepa2_1_vitb.pt` | Meta V-JEPA 2.1 ViT-B | EchoJEPA-B |
| `vitl_imagenet21k.pt` | `vitl_in21k.pt` | ImageNet-21K ViT-L (timm) | EchoBYOL-L, EchoJEPA-L in21k, EchoMAE-L init (+ S3: `vitl_raw.pth`) |
| `vitb_imagenet1k.pt` | `vitb_in1k.pt` | ImageNet-1K ViT-B (torchvision `vit_b_16-c867db91.pth`, DeIT recipe, 81.07% top-1) | ViT-B JEPA/BYOL/VideoMAE MIMIC pretraining (in1k-hp configs). Upload to S3 as `vitb_raw.pth` for HyperPod sbatches. See `pretraining-and-cooldown.md` § ImageNet Initialization for the remap script. |

## Probe Checkpoints

### ICML Preprint Probes (`checkpoints/eval_probes/`)

d=4 attentive probes, 6-head HP grid, 20 epochs. UHN data, single-view eval module.

```
eval_probes/
  classification/          # View classification (13 classes)
    echojepa_224px.pt        # G, single-view
    echojepa_224px_multi.pt  # G, multi-view
    echoprime_224px.pt       # EchoPrime
    panecho_224px.pt         # PanEcho
    videomae_224px.pt        # EchoMAE
  lvef/                    # LVEF regression
    echojepa_224px_multi.pt  # G, multi-view
    echojepa_336px.pt        # G, 336px
    echojepa_g_224px.pt      # G, 224px
    echojepa-l-uhn.pt        # L, UHN
    echojepa-l-lvef-ep15.pt  # L, partial (15 epochs)
    echoprime_224px.pt
    panecho_224px.pt
    videomae_224px.pt
    videomae_224px_old.pt    # Earlier run
  rvsp/                    # RVSP multi-view regression
    echojepa_224px.pt        # G
    echojepa_224px_old.pt    # G, earlier run
    echojepa_336px.pt        # G, 336px
    echojepa-l-rvsp.pt       # L
    echojepa-l-rvsp-old.pt   # L, earlier run
    echoprime_224px.pt
    panecho_224px.pt
    videomae_ep16.pt         # EchoMAE (early stop)
    vjepa-g-rvsp-nse.pt      # G, NSE loss
    vjepa-g-rvsp-nse-lf.pt   # G, NSE + LF
    vjepa-g-rvsp-nse-lf-nmsa.pt  # G, NSE + LF + no MSA
  lvef/echonet-dynamic/    # EchoNet-Dynamic transfer
    echojepa-g.pt
    echojepa-l.pt
    echoprime.pt
    panecho.pt
    videomae.pt
  lvef/echonet-pediatric/  # EchoNet-Pediatric transfer
    echojepa-g.pt
    echojepa-l.pt
    echoprime.pt
    panecho.pt
    videomae.pt
```

### Nature Medicine UHN Probes (`checkpoints/probes/`)

d=1 attentive probes, 12-head HP grid (4 LR x 3 WD), 15 epochs. Strategy E: view-filtered training, prediction averaging.

Each task has `{model}/best.pt` and `{model}/latest.pt`. Models: echojepa-g, echojepa-l, echojepa-l-k, echoprime, panecho (+ echojepa-b for some).

**Regression tasks (8):**
| Task dir | Metric | View-filtered |
|----------|--------|---------------|
| `lvef/` | R² | Yes (A4C, PLAX) |
| `tapse/` | R² | Yes (A4C) |
| `rvsp/` | R² | Yes (A4C, RV) |
| `rv_fac/` | R² | Yes (RV focused) |
| `rv_sp/` | R² | Yes (A4C) |
| `mv_ee_medial/` | R² | Yes (A4C) |
| `aov_mean_grad/` | R² | Yes (A5C, PLAX) |
| `aov_vmax/` | R² | Yes (A5C, PLAX) |

**Classification tasks (5):**
| Task dir | Metric | View-filtered |
|----------|--------|---------------|
| `tr_severity/` | AUROC | Yes (A4C, RV) |
| `mr_severity/` | AUROC | Yes (A4C, PLAX) |
| `as_severity/` | AUROC | Yes (A5C, PLAX) |
| `ar_severity/` | AUROC | Yes (PLAX, A5C) |
| `trajectory_lvef_onset/` | AUROC | No (study-level) |

**Disease detection (8):**
| Task dir | Metric |
|----------|--------|
| `disease_amyloidosis/` | AUROC |
| `disease_bicuspid_av/` | AUROC |
| `disease_dcm/` | AUROC |
| `disease_hcm/` | AUROC |
| `disease_myxomatous_mv/` | AUROC |
| `disease_rheumatic_mv/` | AUROC |
| `disease_stemi/` | AUROC |
| `disease_takotsubo/` | AUROC |

**Other:**
| Task dir | Notes |
|----------|-------|
| `trajectory_lvef/` | LVEF trajectory regression (current) |
| `trajectory_lvef_v1/` | Earlier version |
| `trajectory_mr_severity_onset/` | MR severity onset classification |
| `rv_function/` | RV function grade |
| `diastolic_function/` | Diastolic function grade |
| `cardiac_output/` | Cardiac output regression |
| `edv/` | End-diastolic volume |
| `esv/` | End-systolic volume |

### Nature Medicine Cross-Institution MR Probes (HyperPod, S3 only)

d=1 attentive probes, 16 multihead HP sweep, EchoJEPA-G encoder (`pt-280-an81.pt`). MR severity 4-class.

| Probe | Training Data | Epochs | Best Val Acc | Job | S3 Path |
|-------|---------------|--------|-------------|-----|---------|
| MIMIC MR | MIMIC-IV-Echo MR labels | 35 | 57.87% | 436 | `runs/echojepa_g_mitral_regurg_436/.../echojepa-g-mitral-regurg/best.pt` |
| UHN MR severity | UHN MR severity labels | 28 (stopped 29/35) | 69.99% (e22) | 443 | `runs/echojepa_g_mr_severity_uhn_443/.../echojepa-g-mr-severity-uhn/best.pt` |

Cross-dataset comparison (job 549): MIMIC AUROC 0.806, UHN→MIMIC AUROC 0.799. See `claude/neurips/experiments/mr-cross-dataset-transfer.md`.

### Nature Medicine MIMIC Probes (`checkpoints/probes/mimic/`)

d=1 attentive probes, study-level sampling + prediction averaging. 35 epochs for most tasks.

Tasks (13): `mortality_1yr`, `mortality_30d`, `mortality_90d`, `in_hospital_mortality`, `discharge_destination`, `readmission_30d`, `los_remaining`, `troponin_t`, `nt_probnp`, `creatinine`, `lactate`, `ef_note_extracted`, `disease_afib`.

Models per task: echojepa-g, echojepa-b, echojepa-l-k, echoprime, panecho. (echojepa-l pending for some.)

## Epoch Convention

| Framework | Stored `epoch` value | Interpretation |
|-----------|---------------------|----------------|
| V-JEPA / V-JEPA 2.1 | `N+1` (next epoch to resume) | File `e100` stores `epoch: 101`, meaning 100 epochs completed |
| VideoMAE | `N` (current epoch) | File `ep99` stores `epoch: 99`, meaning 99 epochs completed |

## S3 Sources

| Checkpoint | S3 path |
|-----------|---------|
| Meta V-JEPA 2 weights | `torch.hub` (`facebookresearch/vjepa2`) |
| EchoJEPA-G (pt280_an81) | Local only (UHN pretraining on A100 node) |
| EchoJEPA-L (pt210_an25) | Local only (MIMIC pretraining on A100 node) |
| EchoJEPA-L-K (pt220_an55) | Local only |
| VideoMAE ep99/ep163 | Downloaded from CY's checkpoint share |
| ImageNet-21K ViT-L | `timm` model registry (local: `vitl_in21k.pt`; S3: `s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts/checkpoints/vitl_raw.pth`) |
| ImageNet-1K ViT-B | `torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1` (local: `vitb_in1k.pt`; S3: upload as `checkpoints/vitb_raw.pth` before launching ViT-B sbatches) |

## Intermediate Pretraining Checkpoints

Saved during L pretraining at regular intervals (in `checkpoints/`):
`echojepa-l-pt50.pt`, `echojepa-l-pt70.pt`, `echojepa-l-pt90.pt`, `echojepa-l-pt110.pt`, `echojepa-l-pt150.pt`, `echojepa-l-pt180.pt`, `echojepa-l-pt200.pt`, `echojepa-l-pt220.pt`, `echojepa-l-pt230-an10.pt`, `echojepa-l-pt230-an20.pt`, `echojepa-l-pt230-an30.pt`.

V-JEPA 2.1 ViT-B intermediate: `vjepa2_1_vitb_mimic_p169_c52.pt` (pretrain 169 + cooldown 52).

V-JEPA 2.1 ViT-L run 125 intermediate: `e109.pt`, `e114.pt`, `e119.pt`, `e125_latest_backup.pt` (in `pretrain_21/mimic/vjepa2_1_vitl_224px_16f/`).
