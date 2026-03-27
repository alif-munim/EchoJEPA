# Evaluation Protocols: ICML Preprint vs Nature Medicine vs ICML Rebuttal

Three distinct probe evaluation protocols are used across the project. This document is the single source of truth for what differs between them.

## Protocol Comparison Table

| Parameter | ICML Preprint | Nature Medicine (Strategy E) | ICML Rebuttal |
|-----------|--------------|------------------------------|---------------|
| **Probe depth** | d=4 (3 SA + 1 CA) | d=1 (CA only, no SA) | d=4 (match preprint) |
| **Attention heads** | 16 | 16 | 16 |
| **HP grid** | 6 heads: 2 LR x 3 WD | **12 heads: 4 LR x 3 WD** | 6 heads: 2 LR x 3 WD |
| **LR values** | {1e-4, 5e-5} | **{5e-4, 1e-4, 5e-5, 1e-5}** | {1e-4, 5e-5} |
| **WD values** | {0.01, 0.1, 0.4} | **{0.001, 0.01, 0.1}** | {0.01, 0.1, 0.4} |
| **Warmup** | None (start_lr = lr) | **2 epochs (start_lr=0)** | None (start_lr = lr) |
| **Epochs** | 20 | **15** (35 for small/MIMIC datasets) | 20 |
| **Resolution** | 224px | 224px | 224px |
| **Frames** | 16, frame_step=2 | 16, frame_step=2 | 16, frame_step=2 |
| **Temporal clips** | num_segments=2 | num_segments=2 | num_segments=2 |
| **Batch size** | 1 per GPU | **2 per GPU** | 1 per GPU |
| **Precision** | bfloat16 | bfloat16 | bfloat16 |
| **LR schedule** | Cosine decay to 0 | Cosine decay to 0 | Cosine decay to 0 |
| **View classes** | 13 | 13 | 13 |
| **View filtering** | No | Yes (pre-filtered CSVs) | No |
| **Study sampling** | No | Yes (DistributedStudySampler) | No |
| **Prediction avg** | No | Yes (all clips per study) | No |
| **Models (UHN)** | G, L, MAE, EP, Pan | G, L, L-K, MAE, EP, Pan | G, L, L-K, B, MAE, EP, Pan, BYOL |
| **ICML tasks** | View, LVEF, RVSP | (different task set) | View, LVEF, RVSP |

**Note on NatMed grid evolution:** Early NatMed YAML configs (mortality, TR, etc.) used a 20-head grid (5 LR × 4 WD) with warmup 3.0, BS1, 35 epochs. The production `run_uhn_probe.sh` later standardized to 12 heads (4 LR × 3 WD), warmup 2.0, BS2, 15 epochs. The 12-head run script is the authoritative NatMed protocol; the 20-head YAML configs are earlier iterations.

## What Changed: ICML Preprint -> Nature Medicine

The following changes were made between the ICML preprint protocol and the Nature Medicine protocol (Strategy E, adopted 2026-03-11):

### 1. Probe depth: d=4 -> d=1

**Rationale:** d=1 attentive probes are fairer across architectures with different token counts. At d=1, only cross-attention operates (no self-attention blocks), which is equally expressive whether the encoder outputs 1568 tokens (ViT-G/L) or 1 token (EchoPrime CLS). d=4 probes with 3 SA blocks give token-rich encoders a larger advantage via inter-token self-attention, which is vacuous for 1-token models.

**Verification:** d=1 was verified non-harmful for all 4 tested models: G +1.2pp, L +17.3pp, EchoPrime +9.3pp, PanEcho +7.1pp over linear probes. d=1 attentive mathematically contains linear probing as a strict special case.

**Config:** `num_probe_blocks: 1` (Nature Medicine) vs `num_probe_blocks: 4` (ICML)

### 2. View filtering for view-specific tasks

**Rationale:** Without filtering, ~81% of training clips for view-specific tasks (TAPSE, LVEF) are non-informative views. The probe wastes gradient steps learning to predict the population mean from irrelevant views. Pre-filtered CSVs (`train_vf.csv`) contain only task-relevant views.

**ICML:** No view filtering. All clips used for training regardless of view relevance.
**Nature Medicine:** View-filtered training CSVs for view-specific tasks; unfiltered for global tasks (mortality, biomarkers).

### 3. Study-level evaluation with prediction averaging

**Rationale:** Nature Medicine includes study-level tasks (MIMIC clinical outcomes) where each study has ~72 clips from different echo views. `DistributedStudySampler` selects 1 random clip/study/epoch for training. At eval, all clips are scored independently and predictions are averaged per study.

**ICML:** Single-clip evaluation (no study-level aggregation).
**Nature Medicine:** Prediction averaging across all clips per study.

### 4. Additional models

**ICML:** 5 models (G, L, MAE, EchoPrime, PanEcho)
**Nature Medicine:** 7 models (adds L-K Kinetics-pretrained, EchoFM)

### 5. Task expansion

**ICML:** 3 primary tasks (view classification, LVEF regression, RVSP multi-view regression) + standard video benchmarks (SSv2, K400, etc.)
**Nature Medicine:** 13 UHN tasks (RV mechanics, hemodynamics, valvular disease severity) + 7 disease detection + MIMIC outcomes/biomarkers + trajectory prediction

### 6. HP grid expansion: 6-head -> 12-head

**ICML:** 6 heads — LR ∈ {1e-4, 5e-5} × WD ∈ {0.01, 0.1, 0.4}, no warmup, start_lr = lr
**Nature Medicine (run_uhn_probe.sh):** 12 heads — LR ∈ {5e-4, 1e-4, 5e-5, 1e-5} × WD ∈ {0.001, 0.01, 0.1}, warmup 2 epochs, start_lr = 0

Changes: wider LR range (added 5e-4 and 1e-5), shifted WD range (dropped 0.4, added 0.001), added warmup, increased batch size to 2.

### 7. Epoch reduction: 20 -> 15 (UHN), 35 (MIMIC/small datasets)

The `run_uhn_probe.sh` defaults to 15 epochs for UHN tasks. MIMIC configs use 35 epochs to compensate for smaller study counts.

### What did NOT change

- Resolution (224px)
- Temporal sampling (16 frames, step=2, 2 segments)
- Attention heads (16)
- Precision (bfloat16)
- Frozen encoder (no fine-tuning)

## ICML Rebuttal Protocol

The rebuttal adds new models (L-K, ViT-B, BYOL-L) and mechanistic experiments to the ICML evaluation framework. The probe protocol matches the ICML preprint exactly (d=4, 6-head grid, 20 epochs) to ensure results are directly comparable to preprint tables.

### New models for rebuttal
- **EchoJEPA-L-K** (ViT-L, Kinetics pretrained) — fills scaling table gap
- **EchoJEPA-B** (ViT-B, V-JEPA 2.1 on MIMIC) — scaling analysis B -> L -> G
- **EchoBYOL-L** (ViT-L, BYOL-Video on MIMIC) — contrastive/self-distillation comparison

### Rebuttal-only experiments (no probe training needed)
- Frame shuffling (temporal ablation)
- CKA speckle invariance
- Noise-level linear probe

## Config Archaeology

The HP grid evolved over time. For reference:

| Config generation | Grid size | LR range | WD range | Depth | Epochs | Warmup | BS | Notes |
|-------------------|-----------|----------|----------|-------|--------|--------|----|-------|
| `old/classification_1221_old.yaml` | 20 (5x4) | 5e-3 to 1e-4 | 0.01 to 0.8 | d=4 | 30 | 0 | 1 | First experiment, 336px, 2 classes |
| `old/ssv2.yaml` (Meta reference) | 20 (5x4) | 5e-3 to 1e-4 | 0.01 to 0.8 | d=4 | 20 | 0 | 2 | V-JEPA 2 official protocol |
| `old/classification_pruned.yaml` | 6 (2x3) | 1e-4, 5e-5 | 0.01, 0.1, 0.4 | d=4 | 30 | 0 | 1 | ICML pruned grid for echo |
| Production ICML configs | 6 (2x3) | 1e-4, 5e-5 | 0.01, 0.1, 0.4 | d=4 | 20 | 0 | 1 | LVEF, RVSP, view, baselines |
| Early NatMed YAML (MIMIC) | 20 (5x4) | 1e-3 to 1e-5 | 0.001 to 0.4 | d=1 | 35 | 3 | 1 | mortality, TR, MR YAML configs |
| Early NatMed YAML (UHN) | 6 (2x3) | 1e-4, 5e-5 | 0.01, 0.1, 0.4 | d=1 | 20 | 0 | 1 | view.yaml, lvef.yaml etc. |
| **NatMed `run_uhn_probe.sh`** | **12 (4x3)** | **5e-4 to 1e-5** | **0.001, 0.01, 0.1** | **d=1** | **15** | **2** | **2** | **Authoritative NatMed UHN protocol** |
| Verification d=4 configs | 20 (5x4) | 1e-3 to 1e-5 | 0.001 to 0.4 | d=4 | 20 | 2 | 1 | Post-ICML, expanded grid |

**Pruning history:** Meta's 20-head grid (5 LR × 4 WD) was first pruned to 6 heads (2 LR × 3 WD) for ICML, observing that LR > 1e-3 and WD = 0.8 never won. For Nature Medicine, the grid was re-expanded to 12 heads (4 LR × 3 WD) with a shifted range: wider LR coverage (5e-4 through 1e-5), lower WD range (0.001 replaces 0.4), and 2-epoch warmup was added.

## Reference Configs by Protocol

**ICML Preprint templates** (d=4, 6-head, echo tasks):
- `configs/eval/vitg-384/lvef/vjepa_lvef_224px.yaml` (but has d=1 — was later modified for NatMed)
- `configs/eval/vitg-384/old/classification_pruned.yaml` (d=4, 6-head, 30 epochs — earlier iteration)

**Nature Medicine templates:**
- **UHN production (d=1, 12-head):** `scripts/run_uhn_probe.sh` generates configs on the fly
- **Early MIMIC YAML (d=1, 20-head):** `configs/eval/vitg-384/nature_medicine/echojepa_g_mortality_1yr.yaml`
- **Early UHN YAML (d=1, 6-head):** `configs/eval/vitg-384/view/echojepa_view_classification_224px.yaml`, `configs/eval/vitl/view.yaml`

**Verification templates** (d=4, 20-head — NOT for rebuttal):
- `configs/eval/vitg-384/view/verification/echomae_d4.yaml`
- `configs/eval/vitg-384/view/verification/echojepa_g_d4.yaml`

**ICML Rebuttal configs** (d=4, 6-head, 20 epochs):
- `configs/eval/vitl/icml/echojepa_l_k_view_d4.yaml`
- `configs/eval/vitl/icml/echojepa_l_k_lvef_d4.yaml`
- `configs/eval/vitl/icml/echojepa_l_k_rvsp_d4.yaml`
- `configs/eval/vitb/icml/echojepa_b_view_d4.yaml` (Phase 2)
- `configs/eval/vitb/icml/echojepa_b_lvef_d4.yaml` (Phase 2)
- `configs/eval/vitb/icml/echojepa_b_rvsp_d4.yaml` (Phase 2)
- `configs/eval/vitl/icml/echobyol_l_view_d4.yaml` (Phase 3)
- `configs/eval/vitl/icml/echobyol_l_lvef_d4.yaml` (Phase 3)
- `configs/eval/vitl/icml/echobyol_l_rvsp_d4.yaml` (Phase 3)
