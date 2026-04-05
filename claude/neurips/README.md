# NeurIPS 2025 Resubmission Plan

## Submission Target

**NeurIPS 2025**, Representation Learning or Datasets & Benchmarks track. New paper from scratch (not a revision of the ICML submission).

## Title Candidates

1. "What Gets Predicted Gets Preserved: How SSL Objectives Filter Noise and Determine Clinical Utility in Echocardiography"
2. "What Do Self-Supervised Video Models Learn from Echocardiography? Prediction Targets Determine Clinical Utility"
3. "The Prediction Target Determines Robustness: A Controlled Study of Self-Supervised Video Objectives Under Stochastic Noise"

## Core Thesis

The choice of prediction target in self-supervised video learning determines what clinical information is encoded and preserved under noise. Through a controlled four-way comparison (JEPA, BYOL, MAE, SALT — same ViT-L, same data, same compute), we demonstrate that (1) prediction target rankings invert across task types, with latent prediction leading functional tasks and pixel reconstruction leading spatial tasks; (2) this dissociation is explained mechanistically by differential noise encoding, confirmed via speckle probing and temporal ablation; (3) clean-condition benchmarks fail to predict noise robustness, motivating EchoBench, the first physics-informed evaluation framework for ultrasound foundation models.

## Key Differences from ICML

| Aspect | ICML submission | NeurIPS submission |
|--------|----------------|-------------------|
| **Title** | "EchoJEPA: A Foundation Model for Echocardiography" | Scientific question about SSL objectives |
| **Framing** | Model paper (we built X) | Understanding paper (we discovered Y) |
| **Track** | Applications / Health | Representation Learning / SSL |
| **Comparison** | 2-way (JEPA vs MAE) | 4-way (JEPA vs BYOL vs MAE vs SALT) |
| **SALT** | Not included | Completes {pixel,latent} × {EMA,frozen} 2×2 design |
| **Frame shuffling** | Absent | Main experiment (6 conditions, Figure 2) |
| **Speckle probing** | Absent | Main experiment (representation-level mechanism) |
| **Segmentation** | Absent | Included (anatomy-function dissociation evidence) |
| **EchoBench** | Absent | Own section with formal definition |
| **Multi-view probe** | Headline contribution | Methods section tool, ablation in appendix |
| **EchoJEPA-G scaling** | Main text hero result | Brief scaling subsection |
| **Novelty preemption** | Absent | Explicit paragraph with precedent citations |

---

## Experiment Status Matrix

### Completed (from ICML rebuttal) — Ready to Use

| # | Experiment | Models | Key Result | NeurIPS Section | Source |
|---|-----------|--------|-----------|----------------|--------|
| 1 | **LVEF regression (UHN, 53K test)** | JEPA/BYOL/MAE | JEPA R²=0.409, BYOL 0.384, MAE 0.283 | §3 Core finding | `rebuttals/10-rebuttal-experiment-results.md` §1a |
| 2 | **RVSP regression (UHN, 5K test, multi-view)** | JEPA/BYOL/MAE | JEPA Pearson=0.484, BYOL 0.446, MAE 0.438 | §3 Core finding | `rebuttals/10-*` §1b |
| 3 | **CAMUS segmentation (50 test patients)** | JEPA/BYOL/MAE | MAE **0.822**, BYOL 0.821, JEPA 0.815 | §3 Ranking inversion | `rebuttals/10-*` §1c |
| 4 | **EchoNet-Dynamic LVEF (1,277 test)** | JEPA/BYOL/MAE | JEPA R²=0.552 >> BYOL 0.440 >> MAE 0.351 | §3 Transfer | `rebuttals/10-*` §3a |
| 5 | **Pediatric zero-shot (UHN probes, 368 test)** | JEPA/BYOL/MAE | JEPA Pearson=0.705, BYOL 0.602, MAE 0.626 | §3 Transfer | `rebuttals/10-*` §4b |
| 6 | **Pediatric zero-shot (END probes)** | JEPA/BYOL/MAE | JEPA Pearson=0.615, BYOL 0.498, MAE 0.531 | §3 Transfer | `rebuttals/10-*` §4c |
| 7 | **Noise robustness — LVEF (3 perturbations × 3 severities)** | JEPA/BYOL/MAE | JEPA -19% avg, BYOL -40%, MAE -37% | §5 EchoBench | `rebuttals/10-*` §5m |
| 8 | **Noise robustness — CAMUS (3 × 3)** | JEPA/BYOL/MAE | MAE -8% avg, JEPA -10%, BYOL -25% | §5 EchoBench | `rebuttals/10-*` §5o |
| 9 | **Noise robustness — Pediatric ZS (3 × 2 source datasets)** | JEPA/BYOL/MAE | JEPA highest Pearson at all severity levels | §5 EchoBench | `rebuttals/10-*` §5n |
| 10 | **Frame shuffling (6 conditions, 1,277 test)** | JEPA/BYOL/MAE | JEPA 0.365 post-shuffle > MAE clean 0.396; BYOL collapses to 0.099 | §4 Mechanism | `rebuttals/experiments/frame-shuffling.md` |
| 11 | **Speckle probing (partial R², 5-fold CV)** | JEPA/BYOL/MAE | JEPA 0.674, BYOL 0.775, MAE 0.875 (23% less speckle in JEPA) | §4 Mechanism | `rebuttals/10-*` §6e |
| 12 | **Pathology-stratified LVEF (EchoNet-Dynamic)** | JEPA/BYOL/MAE | JEPA advantage 8× larger on reduced EF (<40%): 12.4 vs 19.3 MAE | §3 Clinical | `rebuttals/10-*` §6d |
| 13 | **Bootstrap CIs (UHN 53K, 10K resamples)** | JEPA/BYOL/MAE | All 6 pairwise CIs exclude zero | §3 Stats | `rebuttals/10-*` §6a |
| 14 | **Bootstrap CIs (EchoNet-Dynamic 1,277)** | JEPA/BYOL/MAE | All 3 pairwise Pearson CIs exclude zero | §3 Stats | `rebuttals/10-*` §3a |
| 15 | **RVSP multi-view noise robustness** | JEPA (MV vs SV) | MV severe (0.449) ≈ SV clean (0.448); MV halves degradation | §5 EchoBench | `rebuttals/10-*` §6c |
| 16 | **RVSP single-view ablation** | JEPA (A4C vs PSAX vs MV) | MV +3.9pp R² over best SV | Appendix | `rebuttals/10-*` §6b |
| 17 | **LVEF scaling (B → L → G)** | JEPA only | B R²=0.650, L 0.436, G 0.778 (confounded) | §6 Scaling | `rebuttals/10-*` §2 |

### Completed (2026-04-05) — Training Dynamics & SALT

| # | Experiment | Models | Key Result | NeurIPS Section | Source |
|---|-----------|--------|-----------|----------------|--------|
| 18 | **Training dynamics LVEF probes (END, d=4)** | BYOL/MAE at e24/e50/e75/e100 | BYOL: 6.37→6.17→5.99→5.94; MAE: 7.54→6.41→6.11→6.05 | §4 Mechanism / §3 | `rebuttals/12-checkpoint-reference.md` §EchoNet-Dynamic LVEF Probes |
| 19 | **SALT S2 END LVEF probe (d=4)** | SALT S2 (e79) | **IN PROGRESS** — training on 8×A100 | §3 Core finding | `configs/eval/vitl/icml/salt_s2_e79_end_lvef_d4.yaml` |

### In Progress (2026-04-05)

| Experiment | Status | ETA | Notes |
|-----------|--------|-----|-------|
| **SALT S2 END LVEF probe** | Training epoch 1/20 on 8×A100 | ~1.5 hrs (BYOL-like arch) | Checkpoint: `checkpoints/salt_s2_vitl_e79.pt`, key: `encoder` |
| **JEPA IN21K (100ep)** | Epoch 71/100 on HyperPod node 184 | ~9 hrs | Job 376 |
| **Frame shuffling severity gradient (P1.5a)** | Scripts ready, queued after SALT probe | ~2 hrs | 9 models (JEPA e50 + BYOL/MAE e24/e50/e75/e100), `scripts/rebuttal/frame_shuffle_severity.py --all` |

### New Experiments Needed for NeurIPS

| Priority | Experiment | Models | Compute | Depends On | Status |
|----------|-----------|--------|---------|-----------|--------|
| ~~**P0**~~ | ~~SALT Stage 1 (V-Pixel, 20ep)~~ | ~~SALT teacher~~ | — | — | **DONE** (HyperPod job, S1 checkpoint used by S2) |
| ~~**P0**~~ | ~~SALT Stage 2 (student, 80ep)~~ | ~~SALT student~~ | — | — | **DONE** (HyperPod job 388, 16 checkpoints e4–e79) |
| **P0** | SALT 5-task evaluation | SALT | ~1 day probes | SALT S2 checkpoint | **IN PROGRESS** — END LVEF probe training now; RVSP, CAMUS, Pediatric ZS queued |
| **P0** | SALT EchoBench (9 noise conditions) | SALT | ~4 hours | SALT probes | Queued after probes |
| **P0** | SALT frame shuffling (6 conditions) | SALT | ~2 hours | SALT END probe | Queued after probe |
| **P0** | SALT speckle probing | SALT | ~1 hour | SALT encoder | Queued |
| **P1** | V-JEPA 2.1 probe evaluation | V-JEPA 2.1 | ~1 day probes | Check if ckpt exists | Not started |
| **P1.5a** | Frame shuffling severity gradient | JEPA/BYOL/MAE (4 epochs each) | ~2 hours | Training dynamics probes | **READY** — scripts + all 9 probes done, queued after SALT |
| **P1.5b** | CAMUS segmentation under shuffling | JEPA/BYOL/MAE | ~1 hour | Existing decoders | **READY** — script done (`frame_shuffle_segmentation.py`) |
| **P2** | View classification (all 4 paradigms) | JEPA/BYOL/MAE/SALT | ~2 hours each | Existing + SALT ckpt | Not started |
| **P2** | EchoBench reference baselines | DINOv2, random ViT | ~4 hours | Public checkpoints | Not started |
| **P3** | Training dynamics (speckle across epochs) | JEPA/MAE | ~1 day | Epoch checkpoints | Partially addressed by #18 (LVEF probes at 4 epochs) |

### SALT Experiment Design (2×2 Matrix)

The key addition for NeurIPS. SALT decouples two confounded variables:

| | Pixel target | Latent target |
|---|---|---|
| **No teacher (reconstruction)** | MAE | — |
| **EMA teacher** | — | JEPA |
| **Frozen pixel teacher** | V-Pixel (SALT S1) | SALT S2 |
| **Global self-distillation** | — | BYOL |

**Predictions to test:**
- If SALT S2 ≈ JEPA → frozen teacher suffices, EMA is unnecessary
- If SALT S2 > MAE but < JEPA → latent target helps, but EMA adds filtering
- If SALT S1 ≈ MAE on downstream → pixel teacher has same anatomy-function profile as MAE
- Any outcome is publishable — the 2×2 isolates the mechanism

**Code:** `app/salt/` (built 2026-04-04). Configs: `configs/train/vitl16/pretrain-salt-s{1,2}-mimic-224px-16f.yaml`
**Docs:** `claude/architecture/salt-pretraining.md`

---

## Timeline Sketch

| Phase | Duration | Activity |
|-------|----------|---------|
| **Compute: SALT pretraining** | ~1 week | Stage 1 (50ep) + Stage 2 (50ep) on 8×A100/H100 |
| **Compute: SALT evaluation** | ~3 days | 5-task battery + EchoBench + frame shuffling + speckle probing |
| **Writing: New paper draft** | ~2 weeks (parallel with compute) | Sections 1-6, figures, tables |
| **P1/P2 experiments** | ~1 week (if time) | V-JEPA 2.1, view classification, DINOv2 baselines |
| **Revision + polish** | ~1 week | Internal review, figure refinement, appendix |

**Total: ~5-6 weeks from start to submission-ready.**

---

## Nature Medicine Deconfliction

**In NeurIPS scope** (SSL understanding, public data):
- 3-way + SALT controlled comparison on MIMIC-IV-Echo
- EchoBench noise robustness methodology
- Speckle probing, frame shuffling, information probing
- EchoNet-Dynamic/Pediatric transfer
- CAMUS segmentation
- Multi-view probing framework (methods section)

**Excluded from NeurIPS** (reserved for Nature Medicine):
- Cross-modal hemodynamic prediction (MR/AS severity from B-mode)
- Mortality, blood biomarkers, ICD code prediction
- Longitudinal cardiomyopathy onset (93K pairs)
- SAE interpretability / Goodfire concept discovery
- Fairness analysis across demographics
- UHN-specific clinical outcomes
- EchoJEPA-G as flagship result (NeurIPS uses it only in a brief scaling section)

---

## Source Reference Index

All rebuttal experiment data lives in `claude/rebuttals/`. The NeurIPS docs reference but don't duplicate:

| File | What to extract for NeurIPS |
|------|----------------------------|
| `rebuttals/10-rebuttal-experiment-results.md` | All numbers, CIs, per-structure segmentation results |
| `rebuttals/experiments/frame-shuffling.md` | 6-condition R² table, log file inventory, correct framing |
| `rebuttals/09-three-way-comparison-results.md` | BYOL architecture audit table |
| `rebuttals/12-checkpoint-reference.md` | Encoder + probe paths for all models |
| `rebuttals/13-post-rebuttal-outcome.md` | ICML lessons (novelty framing, reviewer feedback) |
| `uhn_echo/nature_medicine/icml_rebuttal_final.tex` | Submitted rebuttal text (~5.5 pages, zero TBDs) |
| `claude/preprint/icml_preprint.tex` | Original method section, figures, baselines (1519 lines) |
| `claude/papers/vjepa-salt/arxiv.tex` | SALT paper details for concurrent work section |
| `claude/papers/us-jepa/example_paper.tex` | US-JEPA for independent validation citation |
| `claude/architecture/salt-pretraining.md` | SALT implementation guide |

---

## Other Files in This Directory

| File | Purpose |
|------|---------|
| `completed-experiments.md` | Summary inventory of all done experiments organized by NeurIPS section |
| `new-experiments.md` | What needs to be run, with compute estimates and dependency chains |
| `paper-outline.md` | Section-by-section NeurIPS paper structure |
| `competitive-landscape.md` | US-JEPA, SALT paper, concurrent work positioning |

## Experiment Writeups (`experiments/`)

Standalone writeups for each major experiment with full results, methodology, scripts, and data references.

| File | Experiment | NeurIPS Section |
|------|-----------|----------------|
| `frame-shuffling.md` | 6-condition temporal ablation (clean→matched_frame) | §4 Mechanism |
| `three-way-comparison.md` | LVEF + RVSP + CAMUS core results with bootstrap CIs | §3 Core Finding |
| `noise-robustness.md` | EchoBench: LVEF + CAMUS + Pediatric, 3 perturbations × 3 severities | §5 EchoBench |
| `speckle-probing.md` | Information probing: partial R² for speckle/intensity/texture | §4 Mechanism |
| `cross-dataset-transfer.md` | EchoNet-Dynamic + Pediatric zero-shot transfer | §3 Transfer |
| `clinical-stratification.md` | Pathology-stratified LVEF by EF severity bin | §3 Clinical |
| `multi-view-ablation.md` | RVSP single-view vs multi-view + noise robustness | Appendix |
