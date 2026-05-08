# Evaluation Protocols: ICML Preprint vs Nature Medicine vs ICML Rebuttal

Three distinct probe evaluation protocols are used across the project. This document is the single source of truth for what differs between them.

**How to use this doc when evaluating the pretraining variants in `flops-matched-probe-results.md`:** the d=4 / 6-head / no-PA protocol in column 3 is the one the FLOPs-matched experiments use. Section "Interpreting FLOPs-matched results through the protocol lens" below summarises what the d=4 probe, the HP grid, and the lack of prediction averaging mean for reading deltas between V-JEPA†, MV-PhaseRel, MCC-Anchored, FullJoint-Study, and friends.

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
**Nature Medicine:** 7 models (adds L-K Kinetics→MIMIC, EchoFM)

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
- **EchoJEPA-L-K** (ViT-L, Kinetics→MIMIC) — fills scaling table gap
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

---

## Interpreting FLOPs-matched results through the protocol lens

The pretraining variants in `flops-matched-probe-results.md` (V-JEPA†, V-JEPA‡, MV-PhaseMatched, MV-PairedIntra, MV-PhaseRel, TokenRel-Motion, MCC-Anchored, FullJoint-Study) are all evaluated under the **ICML Rebuttal protocol** (d=4 attentive, 6-head HP grid, 20 epochs, no prediction averaging, no view filtering). Several non-obvious properties of this protocol should shape how inter-variant deltas are read.

### 1. The d=4 attentive probe is doing substantial work

The probe is not a passive readout. V-JEPA reports a **16–17 pp gap between mean-pool linear and attentive probes on standard video benchmarks** (Bardes et al., 2024, Table 3; see `context_files/literature_review/probing.md` Q1). On 1568-token ViT-L encoders the probe's cross-attention layer is learning spatial/temporal aggregation that mean-pooling destroys. **Source:** `context_files/literature_review/probing.md` lines 11–48.

Consequence for variant evaluation: the probe can partially compensate for a weaker encoder by learning a better aggregation strategy, so variant-vs-variant deltas can reflect both representation quality *and* aggregation-friendliness of the token geometry. Variants that produce more structured spatial/temporal tokens (e.g. FullJoint-Study with explicit study-level structure, phase-relational variants with phase-discriminative tokens) may get disproportionate lifts from a d=4 probe compared to a hypothetical d=1 or linear probe.

### 2. d=1 attentive mathematically contains linear probing as a strict special case

At d=1, the probe reduces to cross-attention with one query over all tokens. For any linear probe $w$, there exists a cross-attention configuration that reproduces it exactly (uniform attention weights + identity value projection), so d=1 attentive ≥ linear by construction. Depth=1 verification on view classification showed this empirically: **EchoJEPA-G +1.2pp, EchoJEPA-L +17.3pp, EchoPrime +9.3pp, PanEcho +7.1pp** over linear probes (all four tested models). No model was harmed. **Source:** `context_files/decisions/probe_implementation_analysis.md` §4.

Consequence for FLOPs-matched variants: if a linear-probe-only rerun is ever performed on the same checkpoints (cheap to add), any variant whose d=4 advantage disappears under linear probing owes its delta to the probe rather than to the encoder. A variant whose ordering is preserved under d=1 or linear is the more robust claim.

### 3. d=4 probe parameter counts per sample

At d=4, 16-head, on a 1408-dim encoder (EchoJEPA-G/ViT-L), the attentive probe has approximately **93.3M trainable parameters**, or roughly **18,654 params per training sample** on a 5K-sample split. For ViT-L at 1024-dim the probe is 49.3M params. These are the largest probes in the entire protocol family. **Source:** `context_files/decisions/probe_implementation_analysis.md` §2.

Consequence for FLOPs-matched variants: on the small MIMIC A4C splits used for TAPSE (2K), HCM (2,165), MR (4,482), and RV-function (2,122), the d=4 probe is genuinely under-determined. HP grid winners can flip across seeds. Ranking robustness across V-JEPA† / V-JEPA‡ (same recipe, different seed) at ~0.02–0.04 R² is the right yardstick for whether a variant's claimed delta is above the seed-noise floor.

### 4. No prediction averaging — single-clip test inference

Strategy E adds prediction averaging across ~72 clips per study; the ICML protocol does not. Empirically on MIMIC clinical tasks, prediction averaging typically shifts R² up by **0.02–0.05** and AUC up by **2–5 points**. **Source:** `context_files/literature_review/aggregation.md` lines 625, 712, and `flops-matched-probe-results.md` line 357 ("Strategy E (PA across clips per study) typically shifts R² up 0.02–0.05").

Consequence for FLOPs-matched variants: the reported numbers are single-clip test metrics. If a variant's entire claimed lift over V-JEPA† is ≤ 0.05 R², it is comparable in magnitude to the PA lift that *every* variant would receive if probed under Strategy E. Claims bigger than 0.05 R² (e.g. MV-PhaseRel +0.054 on EchoNet LVEF, per `phase-relational-design.md` §3.4) are more robust to PA normalisation.

### 5. Why the rebuttal intentionally uses d=4 despite the Nature Medicine switch to d=1

Two reasons, both explicit in the decision docs:

- **Apples-to-apples with the preprint.** Any claim of the form "under the published ICML protocol, variant X outperforms V-JEPA†" requires the published protocol. Swapping to d=1 changes the comparison and invites reviewer pushback that the lift is a probe artefact.
- **All rebuttal/FLOPs-matched encoders are 1568-token ViT-L.** The d=4 self-attention degeneration only harms models with very few tokens (EchoPrime 1, PanEcho 32 under legacy configs). For 1568-token encoders, d=4 SA blocks perform real attention over the token grid and help; V-JEPA 2 Table 18 reports +1.0–1.6pp from the extra SA depth. **Source:** `context_files/decisions/probe_implementation_analysis.md` §3 and §4.

So "d=4 for FLOPs-matched, d=1 for Nature Medicine cross-model comparison" is not inconsistent — both are the defensible choice for their respective token-geometry situation.

### 6. HP grid: 6-head pruned grid may miss optima for new objectives

The 6-head grid (LR ∈ {1e-4, 5e-5} × WD ∈ {0.01, 0.1, 0.4}) was pruned from V-JEPA 2's 20-head Meta reference grid by observing that the dropped configs never won on ICML echo tasks with baseline V-JEPA objectives. For objectives with different latent geometry (FullJoint-Study, MCC-Anchored, phase-relational variants), the pruned grid may systematically under-select. **Source:** `evaluation-protocols.md` §"Config Archaeology" line 115; `context_files/decisions/evaluation_protocol_decision.md` §4.

Consequence for FLOPs-matched variants: a variant that requires a wider LR range to tune well will look worse under the 6-head grid than under the 12-head NatMed grid. Check whether the variant's best HP sits at a grid edge — if so, the 6-head grid is likely underselling it.

### 7. What to check when a variant claims a lift

A short sanity-check checklist distilled from the four supporting docs:

| Check | What it rules out | Where it's authoritative |
|---|---|---|
| Does the variant's HP optimum sit at a grid edge? | HP-grid-induced ranking flips | `evaluation_protocol_decision.md` §4, §7 |
| Is the lift > 0.05 R² or 5 pp AUC? | Comparable to unreported PA lift | `aggregation.md` line 712 |
| Is V-JEPA† and V-JEPA‡ seed-noise < claimed lift? | Within-recipe variance floor | `flops-matched-probe-results.md` §"Known gotchas" |
| Does the ordering survive at d=1 (if re-run)? | Probe-induced ranking | `probe_implementation_analysis.md` §4; `probing.md` §Q1 |
| Is the split ≥ 5K train samples? | Probe over-parameterisation | `probe_implementation_analysis.md` §2 |
| Is this a 1568-token encoder? | d=4 SA degeneration | `probe_implementation_analysis.md` §3 |

All six checks pass by construction for the current FLOPs-matched V-JEPA† baseline (it is the recipe the protocol was tuned on). They need to be re-verified for each new pretraining variant.

### Authoritative supporting docs

| Doc | What it authoritatively covers | Key sections for FLOPs-matched eval |
|---|---|---|
| `context_files/decisions/evaluation_protocol_decision.md` | Strategy A–E scoring matrix; why Nature Medicine switched from d=4 to d=1 | §4 (score matrix), §5 (peer-review simulation), §8 (final recommendation) |
| `context_files/literature_review/aggregation.md` | Six aggregation approaches (A–F); prediction-averaging precedent; per-model aggregation strategies | Line 625 (PA lift 0.02–0.05 R²), line 712 (PA convention), §"Approach Comparison" |
| `context_files/decisions/probe_implementation_analysis.md` | Probe parameter counts, V-JEPA 1 vs V-JEPA 2 SA/CA ordering, d=4 degeneration mechanism | §1 (architecture trace), §2 (parameter counts), §3 (V-JEPA comparison), §4 (depth recommendation) |
| `context_files/literature_review/probing.md` | V-JEPA linear/attentive 16–17pp gap; contrastive vs predictive geometry; containment property | Q1 (V-JEPA gap), Q2 (mean-pool information loss) |
| `vjepa2/claude/archive/rebuttals/05-probe-fairness.md` | Root-cause analysis of the ICML 5-model attentive-probe "inversion" (artefact, not structural) | §"Root Cause: Implementation Artifacts (NOT Token Starvation)" |
| `vjepa2/claude/preprint/experiment-issues.md` | ICML pipeline pitfalls (BS=48 failure, normalization bugs); Nature Medicine resolution | §1 (batch size scaling), §"Strategy E" entry |
