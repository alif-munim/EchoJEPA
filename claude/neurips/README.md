# NeurIPS 2025 Resubmission Plan

## Submission Target

**NeurIPS 2025**, Representation Learning or Datasets & Benchmarks track. Abstract deadline: **May 4, 2026**. New paper from scratch (not a revision of the ICML submission). ~6 weeks from 2026-04-05.

## Title Candidates

1. "What Gets Predicted Gets Preserved: How SSL Objectives Filter Noise and Determine Clinical Utility in Echocardiography"
2. "What Do Self-Supervised Video Models Learn from Echocardiography? Prediction Targets Determine Clinical Utility"
3. "The Prediction Target Determines Robustness: A Controlled Study of Self-Supervised Video Objectives Under Stochastic Noise"

## Core Thesis

The choice of prediction target in self-supervised video learning determines what clinical information is encoded and preserved under noise. Through a controlled three-way comparison (JEPA, BYOL, MAE — same ViT-L, same data, same compute) + SALT as a mechanistic probe, we demonstrate that (1) prediction target rankings invert across task types, with latent prediction leading functional tasks and pixel reconstruction leading spatial tasks; (2) this dissociation is explained mechanistically by differential noise encoding, confirmed via speckle probing and temporal ablation; (3) clean-condition benchmarks fail to predict noise robustness, motivating EchoBench.

## Contribution Framing (Analysis → Tool)

To avoid "ablation paper" perception, the paper must deliver actionable tools, not just observations:

1. **Discovery** — The prediction target determines clinical utility (ranking inversion across task types)
2. **Mechanism** — Frame shuffling + speckle probing explain *why* (temporal encoding + noise filtering)
3. **Framework** — EchoBench (physics-based evaluation) + noise autocorrelation sweep (causal proof)
4. **Generalization** — Cross-domain transfer to fetal ultrasound (appendix) and/or calcium imaging (if viable)

## Key Differences from ICML

| Aspect | ICML submission | NeurIPS submission |
|--------|----------------|-------------------|
| **Title** | "EchoJEPA: A Foundation Model for Echocardiography" | Scientific question about SSL objectives |
| **Framing** | Model paper (we built X) | Understanding paper (we discovered Y) |
| **Track** | Applications / Health | Representation Learning / SSL |
| **Comparison** | 2-way (JEPA vs MAE) | 3-way (JEPA vs BYOL vs MAE), SALT as mechanistic probe |
| **Frame shuffling** | Absent | Main experiment (6 conditions + severity gradient) |
| **Speckle probing** | Absent | Main experiment (representation-level mechanism) |
| **Noise autocorrelation sweep** | Absent | **NEW: causal proof of mechanism** |
| **Segmentation** | Absent | Included (anatomy-function dissociation evidence) |
| **EchoBench** | Absent | Own section with formal definition |
| **SALT** | Not included | Mechanistic probe (conditional inclusion — see decision gate below) |
| **Second modality** | Absent | Calcium imaging (if viable) or fetal US (appendix) |

---

## Known Framing Issues

### EchoBench perturbations are spatially static (2026-04-05)

All three EchoBench perturbations (depth attenuation, gaussian shadow, haze) apply the same spatial corruption map to every frame in a clip. Code: `scripts/neurips/echo_perturbations.py` — all maps broadcast via `unsqueeze(0).unsqueeze(0)` over time. This means EchoBench tests robustness to **clinical image quality degradation**, not to frame-varying speckle noise.

**Impact on claims:** Results are valid (JEPA really does degrade less). But the mechanistic explanation (EMA filters frame-varying noise) is tested by speckle probing and frame shuffling (§4), not by EchoBench (§5). Keep these sections separate. Do NOT claim EchoBench tests "speckle robustness."

**Correct framing:** §4 explains the mechanism (speckle probing + frame shuffling on native data). §5 shows practical consequence (does the model work with degraded image quality?). One sentence in §5 noting the perturbations are static.

### Init confound in the controlled comparison (2026-04-05)

| Model | Init | Source | Config |
|-------|------|--------|--------|
| JEPA pt50 | **Fully-trained JEPA (235ep on MIMIC)** | `anneal_ckpt: vitl.pt` | `pretrain-mimic-224px-16f.yaml` |
| JEPA IN21K | ImageNet-21K | Job 376 on HyperPod | Matches BYOL/MAE init |
| BYOL | ImageNet-21K | `anneal_ckpt: vitl_in21k.pt` | `pretrain-byol-mimic-224px-16f-h100.yaml` |
| MAE | ImageNet | `args.finetune = pretrained_videomae_init.pth` | VideoMAE pretrain script |
| SALT S1 | ImageNet-21K | `anneal_ckpt: vitl_in21k.pt` | `pretrain-salt-s1-mimic-224px-16f-hp.yaml` |
| SALT S2 student | **Random** | `force_load_pretrain: false` | Correct per SALT paper recipe |

**JEPA pt50 has the strongest init by far (full MIMIC pretraining). Do not use in primary comparison table.** Use JEPA IN21K (job 376) instead for init-matched comparison with BYOL/MAE.

SALT S2's random student init is the paper's standard recipe. The SALT paper matches total steps (S1+S2), not init. Note in a footnote.

### 🔒 Final SALT checkpoint: v1 e79 (locked 2026-04-08)

**Use `checkpoints/salt_s2_vitl_e79.pt` for ALL final NeurIPS experiments.** Test R²=0.414, MAE=6.66 on END LVEF — the best of three SALT variants evaluated (v1 e79, v1 e199, v3 e79). Do **not** re-run v3 or e199 on new downstream tasks. For new tasks (UHN LVEF, RVSP, CAMUS, Pediatric ZS, EchoBench, frame shuffling, speckle probing), clone the JEPA-IN21K-e100 config and swap the encoder checkpoint. v3 and e199 stay as appendix robustness rows on END LVEF only. Full rationale: `experiments/salt-comparison.md` § FINAL DECISION.

### SALT decision: single-row probe, no retrain needed (revised 2026-04-07)

**Decision gate resolved, verified across two implementation variants.** SALT S2 has been trained in two valid variants and both underperform the EMA baselines by 0.03–0.24 R² on EchoNet-Dynamic LVEF:

| Variant | Predictor | HP regime | Test R² | Test MAE |
|---|---|---|---|---|
| **SALT v1 e79** (best) | hierarchical 4-layer | LR 1.75e-4 constant, weak aug | **0.414** | **6.66** |
| SALT v1 e199 | hierarchical 4-layer | same as v1, extended | 0.360 | 7.02 |
| SALT v3 e79 | single-level (paper-spec) | LR 2.55e-4 cosine, paper aug | 0.348 | 7.03 |

**Earlier "SALT invalidated / must retrain" claim retracted (2026-04-07).** Config inspection confirms both v1 and v3 used `loss_exp: 1.0` (L1 loss, matching SALT paper Eq 2.1). The "L1 vs L2" claim in earlier ops-notes was never true. **No retraining required.** Use v1 e79 as the primary SALT row.

**The finding is robust to implementation choice.** Three variants span predictor architecture (hierarchical vs single-level), HP regime (constant vs cosine LR, weak vs paper aug), and training duration (80 vs 200 S2 epochs). All land within ±0.03 R² and ±0.4 MAE of each other. The SALT gap to EMA-based methods is intrinsic to the frozen-teacher mechanism, not an artifact of implementation.

**Extended-teacher experiments (2026-04-24/25):** new results at matched teacher compute (e100) add three findings. Extended V-Pixel teacher (job 349, S1 e100 → S2 e79) reaches clean R²=0.445, +0.03 over the original SALT v1 e79, with ΔR²=−0.93 under matched-frame (vs −0.33 for the short-teacher variant). The S1 V-Pixel teacher trajectory (jobs 358/359/360/361 at e24/e49/e74/e99) shows matched-frame ΔR² deepening monotonically from −0.02 → −1.14 as the teacher trains longer. JEPA-teacher SALT (job 350, JEPA e100 teacher → S2 e80 student) probes at R²=0.252 vs raw JEPA e100 R²=0.650 — a **0.398 R² loss attributable to the distillation step alone** at matched teacher compute. **Extended MF trajectory for JEPA itself (e125-e200)** completed as job 379: clean R² plateaus at 0.68 (+0.16 persistent lead over MAE), matched-frame ΔR² contracts from −0.143 (e100) to a stable −0.045 to −0.050 band at e150-e200 — equalizing with MAE's late-training ΔR² range. The "JEPA retains frame-order dependence, MAE abandons it" diagnostic is specifically an e100 finding. Full tables in `experiments/salt-comparison.md` §Extended-Teacher Experiments and `experiments/frame-shuffling-results.md`.

**Effective dimensionality (2026-04-07):** JEPA 245, BYOL 221, MAE 206, SALT 203. **No dimensionality collapse** — SALT has enough capacity, it just can't organize features usefully without the evolving teacher signal.

**Conservative framing (web Claude recommendation, 2026-04-07):** One row in the §3 comparison table + two sentences in §4.5. Don't oversell, don't layer mechanisms. See `experiments/salt-comparison.md` for the full writeup.

> **Paper text (two sentences):** "Replacing JEPA's co-evolving EMA teacher with a frozen pixel-reconstruction teacher (SALT) reduces LVEF R² from 0.591 to 0.414 (−30%), placing it below all three EMA-based objectives. This suggests co-evolution of the target encoder contributes to representation quality independent of the prediction target choice."

**Does not contradict the SALT paper.** SALT paper uses 3.6M diverse clips; US-JEPA uses URFM (strong external BiomedCLIP-distilled teacher). Both succeed because the teacher has broad coverage. Our V-Pixel teacher on 525K narrow-domain echo clips hits a ceiling. The frozen teacher mechanism needs data diversity OR a strong external teacher.

**Inclusion:** Add SALT as a 4th row in the primary comparison table (alongside JEPA/BYOL/MAE). Keep the ranking inversion figure as 3-way (JEPA/BYOL/MAE) for visual clarity.

### Second modality assessment (2026-04-05)

**Explored and evaluated:**
- **Fetal intrapartum US** (774 videos, Zenodo): Downloaded to `data/fetal_ultrasound/DatasetV3/`. Has segmentation + AOP/HSD regression labels. BUT AOP/HSD are single-frame measurements (spatial tasks), NOT temporal/functional. Both fetal tasks are spatial → no ranking inversion possible. Useful as appendix cross-anatomy transfer only.
- **Allen Brain Observatory calcium imaging** (1,518 sessions, ~60GB each): Has genuine temporal task (transient detection) + spatial task (cell segmentation). BUT requires pretraining from scratch (domain gap too large for frozen transfer from echo). 5-10 sessions = 200-400K clips, viable volume. High engineering cost.
- **Noise autocorrelation sweep** (synthetic): Sweep temporal correlation of noise from static to iid per-frame on echo data. Directly proves causal mechanism with one controllable parameter. Fast to implement on existing pipeline.

**Decision:** Noise autocorrelation sweep is P0 (week 1). Fetal is P1 (appendix, 1 day). Calcium imaging is P2 (attempt weeks 2-4, hard kill at week 4 if not working).

---

## Experiment Status Matrix

### Completed (from ICML rebuttal) — Ready to Use

| # | Experiment | Models | Key Result | NeurIPS Section | Source |
|---|-----------|--------|-----------|----------------|--------|
| 1 | **LVEF regression (UHN, 53K test)** | JEPA/BYOL/MAE | JEPA R²=0.409, BYOL 0.384, MAE 0.283 | §3 Core finding | `rebuttals/10-*` §1a |
| 2 | **RVSP regression (UHN, 5K test, multi-view)** | JEPA/BYOL/MAE | JEPA Pearson=0.484, BYOL 0.446, MAE 0.438 | §3 Core finding | `rebuttals/10-*` §1b |
| 3 | **CAMUS segmentation (50 test patients)** | JEPA/BYOL/MAE | MAE **0.822**, BYOL 0.821, JEPA 0.815 | §3 Ranking inversion | `rebuttals/10-*` §1c |
| 4 | **EchoNet-Dynamic LVEF (1,277 test)** | JEPA/BYOL/MAE | JEPA R²=0.552 >> BYOL 0.440 >> MAE 0.351 | §3 Transfer | `rebuttals/10-*` §3a |
| 5 | **Pediatric zero-shot (UHN probes, 368 test)** | JEPA/BYOL/MAE | JEPA Pearson=0.705, BYOL 0.602, MAE 0.626 | §3 Transfer | `rebuttals/10-*` §4b |
| 6 | **Pediatric zero-shot (END probes)** | JEPA/BYOL/MAE | JEPA Pearson=0.615, BYOL 0.498, MAE 0.531 | §3 Transfer | `rebuttals/10-*` §4c |
| 7 | **Noise robustness — LVEF (3×3)** | JEPA/BYOL/MAE | JEPA -19% avg, BYOL -40%, MAE -37% | §5 EchoBench | `rebuttals/10-*` §5m |
| 8 | **Noise robustness — CAMUS (3×3)** | JEPA/BYOL/MAE | MAE -8% avg, JEPA -10%, BYOL -25% | §5 EchoBench | `rebuttals/10-*` §5o |
| 9 | **Noise robustness — Pediatric ZS (3×2)** | JEPA/BYOL/MAE | JEPA highest at all severity levels | §5 EchoBench | `rebuttals/10-*` §5n |
| 10 | **Frame shuffling (6 conditions)** | JEPA/BYOL/MAE | JEPA 0.365 post-shuffle; BYOL collapses to 0.099 | §4 Mechanism | `rebuttals/experiments/frame-shuffling.md` |
| 10b | **Tube masking does not prevent the shortcut** (2026-04-08 reframe) | MAE ViT-L | VideoMAE ViT-L pretrained with `tube mask @ 0.9` (canonical Tong 2022). MAE e99 still invariant (−4%, flat across 6 conditions). Shortcut is within-frame spatial interpolation, not cross-frame copying. No masking fix possible → drops frame-gap MAE ViT-B pilot. | §4.6 Mechanism (new subsection) | `experiments/tube-masking-failure.md`, `paper-outline.md` §4.6 |
| ~~11~~ | ~~Speckle probing (partial R², pt50)~~ | ~~JEPA/BYOL/MAE~~ | ~~JEPA 0.674, BYOL 0.775, MAE 0.875~~ — **RETRACTED** (init confound). e100 init-matched: BYOL 0.716 < JEPA 0.848 < MAE 0.885 (gap is 4%, not 23%, and ranking changed). Demoted to non-load-bearing. See `experiments/speckle-probing.md` and `experiments/representation-analysis.md`. | §4 (secondary) | `experiments/representation-analysis.md` |
| 11b | **Effective dimensionality (e100, init-matched)** | JEPA/BYOL/MAE/SALT | ⚠️ **REVISED:** Consistent 4-model RankMe (`scripts/neurips/rankme.py`): JEPA 245, BYOL 221, MAE 206, SALT 203. All 200-245 range — **no 3× collapse**. Prior MAE=63 not reproducible. Demoted from primary mechanism. | §4 (appendix) | `experiments/representation-analysis.md` |
| 12 | **Pathology-stratified LVEF** | JEPA/BYOL/MAE | JEPA advantage 8× larger on reduced EF | §3 Clinical | `rebuttals/10-*` §6d |
| 13 | **Bootstrap CIs (UHN 53K)** | JEPA/BYOL/MAE | All 6 pairwise CIs exclude zero | §3 Stats | `rebuttals/10-*` §6a |
| 14 | **Bootstrap CIs (END 1,277)** | JEPA/BYOL/MAE | All 3 pairwise CIs exclude zero | §3 Stats | `rebuttals/10-*` §3a |
| 15 | **RVSP multi-view noise robustness** | JEPA | MV severe ≈ SV clean | §5 EchoBench | `rebuttals/10-*` §6c |
| 16 | **RVSP single-view ablation** | JEPA | MV +3.9pp R² over best SV | Appendix | `rebuttals/10-*` §6b |
| ~~17~~ | ~~LVEF scaling (B→L→G)~~ | ~~JEPA~~ | ~~B 0.650, L 0.436, G 0.778 (confounded)~~ | ~~§6 Scaling~~ — **CUT**: G uses private 18M UHN data, confounds target with scale. One sentence in discussion instead. | — |

### Completed (2026-04-05) — Training Dynamics, SALT, Severity Gradient

| # | Experiment | Models | Key Result | Source |
|---|-----------|--------|-----------|--------|
| 18 | **Training dynamics LVEF probes** | BYOL/MAE at e24/e50/e75/e100 | BYOL: 6.37→6.17→5.99→5.94; MAE: 7.54→6.41→6.11→6.05 | `rebuttals/12-checkpoint-reference.md` |
| 19 | **SALT S2 e79 END LVEF probe** | SALT S2 | Val MAE=6.47 (11.6%) | `evals/vitl/icml/salt_s2_e79_end_lvef_224/.../best.pt` |
| 20 | **Frame shuffling severity gradient** | BYOL e50/e100, MAE e50/e99, SALT S2 e79 | See results table below | `scripts/neurips/samples/severity_*.csv` |

#### Severity Gradient Results (R² vs Shuffle Fraction, EchoNet-Dynamic test)

| Fraction | BYOL e50 | BYOL e100 | MAE e50 | MAE e99 | SALT S2 e79 |
|----------|----------|-----------|---------|---------|-------------|
| 0.00 | 0.427 | 0.468 | 0.141 | **0.445** | 0.293 |
| 0.25 | 0.360 | 0.410 | 0.091 | **0.421** | -0.037 |
| 0.50 | 0.278 | 0.336 | -0.103 | **0.436** | -0.277 |
| 0.75 | 0.220 | 0.300 | -0.271 | **0.414** | -0.382 |
| 1.00 | 0.219 | 0.291 | -0.301 | **0.428** | -0.397 |

**Key findings:**
- **MAE e99 is invariant to shuffling** — R² stays ~0.43 from clean to fully shuffled. MAE converges to purely spatial representations over training, losing whatever temporal encoding it had at e50.
- **MAE e50 collapses** under shuffling (0.14→-0.30) — early MAE had temporal reliance that disappeared by e99.
- **BYOL degrades linearly** — ~40% relative drop, consistent at both e50 and e100. Stable temporal encoding.
- **SALT S2 collapses dramatically** — 0.29→-0.40, steepest relative drop. Frozen teacher doesn't maintain temporal features.
- **Training dynamics insight:** MAE's temporal encoding is transient — present early, lost by convergence. This is a novel finding.

### Completed (2026-04-08) — Cross-Dataset MR Transfer

| # | Experiment | Models | Key Result | Source |
|---|-----------|--------|-----------|--------|
| 21 | **MR severity cross-dataset (UHN→MIMIC)** | EchoJEPA-G (two probes) | AUROC 0.806 (MIMIC) vs 0.799 (UHN→MIMIC), −0.9% | `experiments/mr-cross-dataset-transfer.md` |
| 22 | **ViT-B temporal shortcut pilot** | Standard MAE vs Frame-Gap MAE | IN PROGRESS (jobs 570/571, ~2026-04-10) | `experiments/vitb-temporal-shortcut-pilot.md` |

### In Progress / Queued (2026-04-08)

| Experiment | Status | Location | ETA |
|-----------|--------|----------|-----|
| **ViT-B Standard MAE pilot** | RUNNING (job 570, node 83) | `scripts/videomae_pilot_standard_vitb.sbatch` | ~43 hrs (~2026-04-10) |
| **ViT-B Frame-Gap MAE pilot** | RUNNING (job 571, node 184) | `scripts/videomae_pilot_framegap_vitb.sbatch` | ~43 hrs (~2026-04-10) |
| **SALT S2 e29 probe** | Training on GPUs 5-7 | `configs/eval/vitl/icml/salt_s2_e29_end_lvef_d4.yaml` | ~4 hrs |
| **JEPA IN21K (100ep)** | Epoch ~80/100 on HyperPod | Job 376, node 184 | ~7 hrs |
| **SALT S2 e80→e100** | Running on HyperPod | Job 391, node 83 | ~2 hrs |
| **SALT S2 e100→e200** | Queued after job 391 | Job 392, node 83 | ~13 hrs after 391 |
| **BYOL/MAE resume to e200** | Not started | Resume from e108/e116 on S3 | Queue after JEPA/SALT finish |

### Active Queue (2026-04-24)

| Job | Experiment | State | Depends on | Notes |
|-----|------------|-------|------------|-------|
| 374 | phi-JEPA Run D (phase-conditioned V-JEPA, 200ep) | RUNNING on 148 (e60/200) | — | Headline phi-JEPA run; see `experiments/phase-jepa.md` |
| 377 | phi-JEPA ACDC LVEF trajectory probes | PENDING | afterany:374 | Probes {e25, e50, e75, e100} |
| 378 | JEPA IN21K extended MF trajectory (e125-e200) | FAILED (TMPDIR bug, 46s) | — | Superseded by 379 |
| 379 | JEPA IN21K extended MF trajectory (retry) | **COMPLETED 2026-04-25 (~40min)** | — | Results below. ΔR² contracts from −0.143 (e100) to −0.045 to −0.050 band at e150–e200; equalizes with MAE's late-training ΔR² range. Clean-R² lead over MAE persists at +0.16 across extended training. See `experiments/frame-shuffling-results.md`. |

Recent completions: 329 (SALT S1 V-Pixel teacher e20→e100); 330 (SALT S2 V-Pixel student 80ep); 333/344 (CMR JEPA base run seed 234/163, loss 0.60→0.35→0.505→0.43); 345 (CMR JEPA LVEF trajectory probes — fast-EMA, JEPA peaks at e100 R²=0.162, collapses to e800 R²=0.069); **346 (CMR JEPA slow-EMA through e295); 349/350 (SALT endpoint probes on V-Pixel and JEPA-teacher students); 358-361 (SALT S1 V-Pixel trajectory probes at e24/e49/e74/e99); 375 (CMR JEPA slow-EMA LVEF trajectory — peaks R²=0.138 at e30, collapses to 0.089 at e295); 376 (CMR JEPA slow-EMA Dx trajectory — peaks AUROC=0.799 at e30, collapses to 0.766 at e295)**.

**Key finding: slow-EMA rules itself out as the cause of JEPA-on-CMR collapse.** Both EMA schedules degrade from e100 onwards on both LVEF and Dx. The JEPA representational collapse on CMR is independent of the EMA schedule. See `experiments/cmr-cross-modality.md` for full details.

### New Experiments — Prioritized Plan (6 weeks to May 4)

| Week | Priority | Experiment | Compute | Notes |
|------|----------|-----------|---------|-------|
| 1 | **P0** | Noise autocorrelation sweep | ~2 days | Sweep noise temporal correlation τ from static to iid on echo data. Modify `echo_perturbations.py` for per-frame noise. **Best-paper-caliber if curve is clean.** |
| 1 | **P0** | Fetal US appendix experiment | 1 day | Freeze MIMIC encoders, probe on fetal AOP/HSD + segmentation. Both tasks spatial → MAE should lead both. Confirms cross-anatomy transfer. |
| 1 | **P0** | JEPA IN21K probe training | ~4 hrs | After job 376 finishes. Primary comparison table needs this. |
| 2 | **P0** | Primary comparison table at e100 | — | JEPA IN21K / BYOL / MAE / (SALT if clean). All at ~100 total epochs. |
| 2-3 | **P1** | Calcium imaging viability test | 1 day | Pretrain ViT-B MAE+JEPA for 10ep on 1 session. If features non-degenerate, proceed. |
| 3-4 | **P1** | Calcium imaging full experiment | ~1 week | Pretrain ViT-L on 5-10 sessions, evaluate segmentation + transient detection. Kill at week 4 if not clean. |
| ~~2~~ | ~~**P1**~~ | ~~SALT decision gate~~ | — | **RESOLVED 2026-04-07.** Both v1 and v3 variants are valid (configs confirmed L1 loss). v1 e79 selected as primary SALT row: R²=0.414, MAE=6.66. Single row in §3 table + two-sentence §4.5 framing. See `experiments/salt-comparison.md`. No retrain needed. |
| 5-6 | — | Writing + polish | 2 weeks | Full draft, figures, appendix |

### SALT Experiment Design (2×2 Matrix)

| | Pixel target | Latent target |
|---|---|---|
| **No teacher (reconstruction)** | MAE | — |
| **EMA teacher** | — | JEPA |
| **Frozen pixel teacher** | V-Pixel (SALT S1) | SALT S2 |
| **Global self-distillation** | — | BYOL |

**SALT S2 compute matching (per SALT paper):** Total S1+S2 steps must match other models.
- ~50ep comparison: S1(20) + S2(29) = 49 total → SALT S2 e29
- ~100ep comparison: S1(20) + S2(79) = 99 total → SALT S2 e79
- Extended: S2 to e200 (jobs 391+392) for scaling analysis

**Current SALT result:** SALT S2 e79 val MAE=6.47, worst of all models. Possible reasons: (1) random student init (BYOL/MAE have ImageNet), (2) frozen teacher only 20ep pixel recon, (3) SALT needs >200K student steps per paper.

---

## Checkpoint Inventory (Local EFS)

### Encoder Checkpoints

| Model | Epoch | Init | Local Path |
|-------|-------|------|-----------|
| JEPA | 50 | Fully-trained JEPA 235ep | `checkpoints/echojepa-l-pt50.pt` |
| JEPA | 90 | Fully-trained JEPA 235ep | `checkpoints/echojepa-l-pt90.pt` |
| JEPA IN21K | 100 | ImageNet-21K | **Training** (job 376) |
| BYOL | 24 | ImageNet-21K | `checkpoints/byol_vitl_imagenet_v2_e24.pt` |
| BYOL | 50 | ImageNet-21K | `checkpoints/byol_vitl_imagenet_v2_e50.pt` |
| BYOL | 75 | ImageNet-21K | `checkpoints/byol_vitl_imagenet_v2_e75.pt` |
| BYOL | 100 | ImageNet-21K | `checkpoints/byol_vitl_imagenet_v2_e100.pt` |
| MAE | 24 | ImageNet | `checkpoints/videomae_l_mimic_ep24.pth` |
| MAE | 50 | ImageNet | `checkpoints/videomae_l_mimic_ep50.pth` |
| MAE | 74 | ImageNet | `checkpoints/videomae_l_mimic_ep74.pth` |
| MAE | 99 | ImageNet | `checkpoints/videomae_l_mimic_ep99.pth` |
| SALT S2 | 29 | Random student | `checkpoints/salt_s2_vitl_e29.pt` |
| SALT S2 | 49 | Random student | `checkpoints/salt_s2_vitl_e49.pt` |
| SALT S2 | 79 | Random student | `checkpoints/salt_s2_vitl_e79.pt` |

### EchoNet-Dynamic LVEF Probes (d=4 attentive, 224px)

| Model | Epoch | Val MAE | Probe Path |
|-------|-------|---------|-----------|
| JEPA | 50 | 5.51 | `evals/vitl/icml/echojepa_pt50_end_lvef_224/.../icml-echojepa-l-pt50-end-lvef-d4/best.pt` |
| BYOL | 24 | 6.37 | `evals/vitl/icml/echobyol_e24_end_lvef_224/.../icml-echobyol-l-e24-end-lvef-d4/best.pt` |
| BYOL | 50 | 6.17 | `evals/vitl/icml/echobyol_pt50_end_lvef_224/.../icml-echobyol-l-pt50-end-lvef-d4-224/best.pt` |
| BYOL | 75 | 5.99 | `evals/vitl/icml/echobyol_e75_end_lvef_224/.../icml-echobyol-l-e75-end-lvef-d4/best.pt` |
| BYOL | 100 | 5.94 | `evals/vitl/icml/echobyol_e100_end_lvef_224/.../icml-echobyol-l-e100-end-lvef-d4/best.pt` |
| MAE | 24 | 7.54 | `evals/vitl/icml/echomae_e24_end_lvef_224/.../icml-echomae-l-e24-end-lvef-d4/best.pt` |
| MAE | 50 | 6.41 | `evals/vitl/icml/echomae_pt50_end_lvef_224/.../icml-echomae-l-pt50-end-lvef-d4/best.pt` |
| MAE | 74 | 6.11 | `evals/vitl/icml/echomae_e74_end_lvef_224/.../icml-echomae-l-e74-end-lvef-d4/best.pt` |
| MAE | 99 | 6.05 | `evals/vitl/icml/echomae_e99_end_lvef_224/.../icml-echomae-l-e99-end-lvef-d4/best.pt` |
| SALT S2 | 79 | 6.47 | `evals/vitl/icml/salt_s2_e79_end_lvef_224/.../icml-salt-s2-e79-end-lvef-d4/best.pt` |
| SALT S2 | 29 | Training | `configs/eval/vitl/icml/salt_s2_e29_end_lvef_d4.yaml` |

### Severity Gradient Output Files

| Model | CSV | JSON |
|-------|-----|------|
| BYOL e50 | `scripts/neurips/samples/severity_BYOL_e50.csv` | — |
| BYOL e100 | `scripts/neurips/samples/severity_BYOL_e100.csv` | — |
| MAE e50 | `scripts/neurips/samples/severity_MAE_e50.csv` | — |
| MAE e99 | `scripts/neurips/samples/severity_MAE_e99.csv` | — |
| SALT S2 e79 | `scripts/neurips/samples/severity_SALT_e79.csv` | — |

---

## Acceptance Assessment (2026-04-22)

**Estimate: 30–40% acceptance probability.** This range reflects reviewer-fit variance more than uncertainty about quality.

### Strengths
- Controlled four-way design (JEPA/MAE/BYOL/SALT, same ViT-L, same data, same compute)
- Striking clinical numbers (pediatric zero-shot, pathology-stratified LVEF)
- Novel domain framing (temporal shortcut in medical video SSL)
- Readable narrative arc; clinical consequences section lands regardless of methodological objections

### Three Predictable Vulnerabilities

1. **100-epoch hedge is load-bearing.** Section 7 itself flags the central finding is contingent on a question the paper doesn't answer. Mitigated by e194 extended training (pattern persists) but not fully resolved.
2. **Single-probe inferential structure.** Probe-class sensitivity is well-documented; SSL methods reviewers will ask why one probe is sufficient. Mitigated by multi-probe readout (attentive + linear-A-raw + linear-A-diff) but attentive probe carries the headline claim.
3. **Mechanism overclaim.** "The mechanism follows from the objective" is asserted, not demonstrated. The natural ablation (hold masking fixed, vary only target type) is not run.

### Reviewer-Fit Split
- **Sympathetic reviewers** (medical imaging focus): ~45% (weak accept territory)
- **Less sympathetic reviewers** (SSL methods focus, push on single-probe and mechanism): ~25% (borderline reject)
- **Average**: ~35%

### Dominant Risk
Not that the paper is wrong, but that the central claim ("temporal encoding collapses") is interpretively fragile in ways the paper doesn't acknowledge. A careful reviewer who reads Section 5 and then the appendices can see the tension.

### Prioritized Recommendations (ordered by impact-per-effort)

| # | Recommendation | Impact | Cost | Status |
|---|---------------|--------|------|--------|
| 1 | **CMR JEPA on matched setup** | +10–15pp | 1 JEPA ViT-S pretrain + probes | **COMPLETE, negative-cross-modality finding** (jobs 333/344/345/346/375/376). Fast- and slow-EMA JEPA both peak at e30-e100 on ACDC LVEF+Dx, then degrade. MAE climbs monotonically and overtakes JEPA by e200-e800. Replicable across seeds and EMA schedules. This **falsifies** the "JEPA uniformly beats MAE on cardiac video" claim and becomes the anchor for the diagnostic paper's direction-flip panel. See `experiments/cmr-cross-modality.md`. |
| 2 | **Integrate e194 + SALT convergence into main text** | +8–12pp | Writing only (results exist) | **DONE** (Changes A-D applied to neurips.tex, Apr 22) |
| 3 | **Multi-probe readout (linear-diff) in appendix with per-token controls** | +5–8pp | Writing only (results exist) | NOT STARTED |
| 4 | **Restore figures from April 16 draft** | +3–5pp | Zero cost | NOT STARTED |
| 5 | **Seed variance on attentive-probe trajectory tables** | +3–5pp | Probe re-runs at key checkpoints | NOT STARTED |
| 6 | **Masking-vs-target ablation** | +5–10pp | 2 pretraining runs (~60% of one full run) | NOT STARTED (expensive) |
| 7 | **SALT with extended teacher (V-Pixel e99)** | +3–5pp | 1 S1 extension + 2 S2 student runs | **PARTIAL** (329/330 complete, 335 running ~e54/80, probes 349/350 queued) |
| 8 | **Spectral/Fourier analysis of reconstruction targets** | +2–4pp | Analysis only, no training | NOT STARTED |
| 9 | **Pediatric transfer with multi-seed CIs** | +1–3pp | Probe re-runs only | NOT STARTED |
| 10 | **JEPA IN21K extended matched-frame (e125-e200)** | +2–4pp | 1 MF inference pass on 4 probes (already trained) | NOT STARTED — probes complete in job 332 (clean R² 0.685-0.717, +0.17-0.22 over MAE at matched epochs), MF pending |

### Acceptance Probability by Tier

| Tier | Recommendations | Estimated Probability |
|------|----------------|---------------------|
| Baseline (current paper) | — | ~35% |
| **Minimum path** | (1) + (2) + (3) | ~60–65% |
| **Strong path** | (1) + (2) + (3) + (4) + (5) | ~65–75% |
| **Full path** | All of (1)–(7) | ~75–85% |
| **Maximum** | All of (1)–(9) | ~80–90% |

### Minimum Honest Path (if only 3 things)

1. **CMR JEPA** — only item that adds new affirmative evidence for the broadest claim; earns "medical video SSL" framing rather than narrowing to "echo SSL"
2. **Integrate e194 + SALT convergence** — closes the central hedge in Section 7 with work already done
3. **Linear-diff in appendix with per-token controls** — pre-empts the most sophisticated reviewer objection (single-probe inference)

### The Thing to Most Strongly Resist Cutting

**CMR JEPA.** Not because it's individually the largest improvement, but because it's the only item that adds new affirmative evidence for the broadest claim. Everything else strengthens defenses; CMR JEPA earns the "medical video SSL" framing. If it shows differential replication, the title is earned. If it shows differential collapse at small scale, that's a defensible scope condition. If skipped, it's a single-modality study claiming cross-modality scope.

---

## Timeline (6 weeks, 2026-04-05 → 2026-05-04)

| Week | Focus | Deliverables |
|------|-------|-------------|
| 1 | Noise autocorrelation sweep + fetal appendix | Per-frame noise implementation, sweep figure, fetal probe results |
| 2 | Primary comparison table + SALT decision | JEPA IN21K probes, SALT e200 evaluation, go/no-go on SALT |
| 2-3 | Calcium imaging viability | ViT-B test pretrain on 1 session, kill if degenerate |
| 3-4 | Calcium imaging full (if viable) | ViT-L pretrain on 5-10 sessions, segmentation + transient detection |
| 5 | Writing | Full draft sections 1-7, all figures |
| 6 | Polish | Internal review, figure refinement, appendix, submission |

---

## Nature Medicine Deconfliction

**In NeurIPS scope:** 3-way comparison on MIMIC, EchoBench, speckle probing, frame shuffling, EchoNet-Dynamic/Pediatric transfer, CAMUS segmentation, multi-view probing (methods), noise autocorrelation sweep, cross-domain transfer (fetal, calcium imaging).

**Excluded from NeurIPS:** Cross-modal hemodynamic prediction, mortality/biomarkers/ICD, longitudinal cardiomyopathy, SAE interpretability, fairness analysis, UHN clinical outcomes, EchoJEPA-G as flagship.

---

## Source Reference Index

| File | What to extract for NeurIPS |
|------|----------------------------|
| `rebuttals/10-rebuttal-experiment-results.md` | All numbers, CIs, per-structure segmentation |
| `rebuttals/experiments/frame-shuffling.md` | 6-condition R² table, correct framing |
| `rebuttals/12-checkpoint-reference.md` | Encoder + probe paths, S3 locations, training dynamics |
| `rebuttals/13-post-rebuttal-outcome.md` | ICML lessons |
| `claude/papers/vjepa-salt/arxiv.tex` | SALT paper: fair comparison = matched total steps |
| `claude/architecture/salt-pretraining.md` | SALT implementation guide |
| `scripts/neurips/frame_shuffle_severity.py` | Severity gradient script (all model configs) |
| `scripts/neurips/frame_shuffle_segmentation.py` | P1.5b CAMUS segmentation script |
| `scripts/neurips/echo_perturbations.py` | EchoBench perturbations (static — see framing note) |

## Other Files in This Directory

| File | Purpose |
|------|---------|
| `completed-experiments.md` | Inventory of done experiments by NeurIPS section |
| `new-experiments.md` | What needs to be run with compute estimates |
| `paper-outline.md` | Section-by-section structure with framing notes |
| `competitive-landscape.md` | US-JEPA, SALT paper, concurrent work |

## Experiment Writeups (`experiments/`)

| File | Experiment | NeurIPS Section |
|------|-----------|----------------|
| `frame-shuffling.md` | 6-condition temporal ablation | §4 Mechanism |
| `severity-gradient.md` | 5-fraction shuffle × 4 epochs × 3+1 models | §4.2 Mechanism |
| `6-condition-shuffling.md` | 6-condition × 4 epochs × 3 models | §4.1 Mechanism |
| `tube-masking-failure.md` | Reframe: tube mask @ 0.9 cannot prevent MAE's shortcut | §4.6 Mechanism |
| `three-way-comparison.md` | LVEF + RVSP + CAMUS core results | §3 Core Finding |
| `noise-robustness.md` | EchoBench: 3 perturbations × 3 severities | §5 EchoBench |
| `speckle-probing.md` | Information probing: partial R² | §4 Mechanism |
| `cross-dataset-transfer.md` | EchoNet-Dynamic + Pediatric | §3 Transfer |
| `clinical-stratification.md` | Pathology-stratified LVEF | §3 Clinical |
| `multi-view-ablation.md` | RVSP single-view vs multi-view | Appendix |
| `mr-cross-dataset-transfer.md` | MR severity: UHN vs MIMIC probes on MIMIC test | §3 Transfer |
| `vitb-temporal-shortcut-pilot.md` | Standard MAE vs Frame-Gap MAE ViT-B: temporal shortcut hypothesis | §4 Mechanism |
| `vitb-in21k-paused.md` | ViT-B IN21K MAE capacity-scaling comparison; paused at e34 (job 307), resumable via `RESUME_FROM_JOB=307` | §4 Mechanism (optional) |

## Datasets

### Fetal Intrapartum Ultrasound (IUGC2024)

Downloaded to `data/fetal_ultrasound/DatasetV3/`. 774 videos, 512×512 AVI at 24fps.

| Split | Videos | Total Frames | Labels |
|-------|--------|-------------|--------|
| Train | 434 | 53,996 | 266 with segmentation, 2575 AOP/HSD landmarks, 434 classification |
| Val | 40 | 2,870 | 40 seg, 40 landmarks, 40 cls |
| Test | 300 | 8,665 | 300 seg, 300 landmarks, 300 cls |

**Assessment:** Both AOP/HSD and segmentation are single-frame spatial tasks. No temporal/functional task available → no ranking inversion possible. Useful for appendix cross-anatomy transfer only.

### Allen Brain Observatory (Visual Coding 2P)

S3: `s3://allen-brain-observatory/visual-coding-2p/ophys_movies/`. 1,518 sessions, 512×512 at 30Hz, ~90 min each.

**Assessment:** Has genuine 2×2: spatial (cell segmentation) + temporal (transient detection). Poisson shot noise is frame-varying. But requires pretraining from scratch (domain gap too large for frozen echo transfer). ~200-400K clips from 5-10 sessions. Allen SDK kernel set up at `allen` conda env. High engineering cost, high reward.
