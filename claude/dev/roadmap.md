# Roadmap

Consolidated view of outstanding work across UHN and MIMIC pipelines. Updated 2026-03-19.

Source of truth for paper scope and task priorities: `uhn_echo/nature_medicine/context_files/nature_medicine_task_overview.md`. Comprehensive manuscript task inventory (every experiment in sn-article.tex with status): `uhn_echo/nature_medicine/context_files/dev/manuscript-tasks.md`. Detailed specs in `planning/tasks/`. Per-person assignments in `roles/`. This file is the cross-cutting infrastructure and experiment summary.

## Paper Framing (from task overview)

The core novelty is organized around **Wendy's three pillars**:
1. **RV mechanics** — learned functional organization (TAPSE, S', FAC, RV basal dimension, RV function grade, RV size)
2. **Hemodynamics** — structure predicts flow (MR/AS/TR severity from B-mode only, AV gradients, E/e' from structural views). **Reframed 2026-03-15:** B-mode video contains hemodynamic information accessible to ALL models (EchoPrime, PanEcho also extract signal). The story is *scale advantage* (+8-9pp AUROC for EchoJEPA-G over baselines) and *no text supervision needed*, not unique capability.
3. **Forecasting** — trajectory prediction (future LVEF, TAPSE, LV mass, MR severity from 93K pairs)

Standard benchmarks are brief (one paragraph + Extended Data). Disease detection is supporting evidence, not headline. SAE interpretability and core lab evaluation are Strong Additions / deferred to revision.

**4 manuscript models** (updated 2026-03-19): EchoJEPA-G, EchoJEPA-L-K, EchoPrime, PanEcho. EchoJEPA-L is internal testing only (not in manuscript). EchoMAE dropped (undertrained checkpoint). ICML preprint carries JEPA-vs-MAE comparison and G-vs-L scale comparison.

---

## MIMIC Embeddings — COMPLETE

> **Historic (old NPZ pipeline).** MIMIC probes are being re-run with Strategy E (d=1 attentive probes from video). See Phase 2 below.

All 7 models extracted, study-level pooled, and split into train/val/test for 23 tasks.

| Model | Clips | Dim | Status |
|-------|-------|-----|--------|
| EchoJEPA-G | 525,320 | 1408 | Ready (shuffle-fixed post-hoc) |
| EchoJEPA-L | 525,320 | 1024 | Ready (shuffle-fixed post-hoc) |
| EchoJEPA-L-K | 525,320 | 1024 | Ready (shuffle-fixed post-hoc) |
| EchoMAE-L | 525,320 | 1024 | Ready (shuffle-fixed post-hoc) |
| PanEcho | 525,320 | 768 | Ready (re-extracted 2026-03-08) |
| EchoPrime | 525,320 | 512 | Ready (re-extracted 2026-03-08) |
| EchoFM | 525,320 | 1024 | Ready (re-extracted 2026-03-08) |

## MIMIC Probe Training — COMPLETE

> **Historic (old NPZ pipeline).** MIMIC probes are being re-run with Strategy E (d=1 attentive probes from video). See Phase 2 below.

161/161 probe jobs done (7 models x 23 tasks). Results in `results/probes/nature_medicine/mimic/`. UHN linear probes also complete for EchoJEPA-G (26 classification + 21 regression + 5 trajectory) and EchoJEPA-L.

## UHN Embeddings — In Progress

> **Historic (old NPZ pipeline).** UHN evaluation now uses d=1 attentive probes from video (no NPZ extraction needed). Extraction status is retained for reference.

### Extraction Status (HISTORIC — NPZ pipeline superseded by Strategy E)

| Model | Checkpoint | Clip Extraction | Study-Level | Notes |
|-------|-----------|-----------------|-------------|-------|
| EchoJEPA-G | `pt-280-an81.pt` | **DONE** (95GB, 18.1M clips) | **DONE** (319,815 x 1408) | |
| EchoJEPA-L | `vitl-pt-210-an25.pt` | **DONE** (70GB, 18.1M clips) | **DONE** (319,802 x 1024) | |
| EchoJEPA-L-K | `vitl-kinetics-pt220-an55.pt` | **~17% done** (crashed/interrupted) | Missing | Chunk-based, supports resume. ~8-9h remaining |
| EchoMAE-L | `videomae-ep163.pt` | **DROPPED** (undertrained ckpt) | — | Cite ICML for JEPA-vs-MAE comparison |
| Random Init | Untrained ViT-L | **DEPRIORITIZED** | — | Not needed for headline results |

PanEcho, EchoPrime, EchoFM UHN extractions are not planned (these models are primarily compared on MIMIC).

### UHN Per-Task Split Pipeline — DONE

Built `evals/regenerate_uhn_downstream.py`. Generated splits for G and L (53 tasks each). **Note:** This NPZ split pipeline is superseded by Strategy E (d=1 attentive probes from video using CSVs). No further NPZ splits needed.

---

## Blocking Work — Manuscript-Aligned Priorities (updated 2026-03-19)

Organized by what the manuscript (`sn-article.tex`) actually needs. `‡` markers = experiments not yet run. `\tbd` = values needing data. ~30 `‡` markers and ~70 `\tbd` placeholders remain.

### P1: Finish UHN Probe Pipeline (Alif, compute-only)
1. **RV S' training** — L-K at ep 10/15 on GPUs 0-3, EP/Pan queued. G done (0.491), L done (0.234).
2. **RV FAC training** — 5 models, CSV ready. Fills §2.3 RV mechanics.
3. **Pred avg** — MR sev (4), AS sev (5), AV Vmax (5), AR sev (5), E/e' medial (5), RV S'/FAC (after training) = ~29 runs. Use GPUs 4-7.
4. **Bland-Altman analysis** — post-processing for all regression tasks. Wendy: MAE alone insufficient for Nature Medicine.

### P2: MIMIC Strategy E Probes (Alif, large compute — §2.5 outcomes table)
5. **MIMIC outcome probes (Strategy E)** — 8 tasks × 4 models: mortality (1yr/90d/30d/in-hospital), readmission, discharge dest, remaining LOS, end-of-life. Currently only G has sklearn results (†-marked in manuscript).
6. **MIMIC biomarkers** — troponin, NT-proBNP, creatinine, lactate. G + L-K minimum.
7. **MIMIC cross-institution pred avg** — more tasks beyond LVEF (TAPSE, RVSP, etc.)

### P3: Novel/JEPA-Unique Experiments (Alif — §2.5 trajectory + §2.8 ablation)
8. **Forward prediction (Exp 6.1)** — JEPA predictor, first-N-frame inference. Key differentiator. Write `evals/forward_prediction.py`.
9. **Anomaly detection (Exp 6.4)** — JEPA prediction error as zero-shot anomaly score vs HCM/DCM/amyloidosis.
10. **Trajectory expansion** — EF change (UHN+MIMIC), multi-parameter trajectory, new HF diagnosis (MIMIC). §2.5 has ‡ markers on all.

### P4: Supporting Experiments (Alif/Reza — §2.6 diseases, §2.1 extended data)
11. **Disease panel** — 9 diseases × 4 models (CSVs ready): HCM, amyloidosis, takotsubo, STEMI, endocarditis, DCM, bicuspid AV, myxomatous MV, rheumatic MV.
12. **Additional hemodynamics** — diastolic function, PA pressure, cardiac output (CSVs need B-mode filter build).

### P5: Delegated Work (blocked on other people)
13. **CY**: Combined echo+EHR model (§2.5), acuity conditioning/H_confound, outcome baselines (EF baseline matrix).
14. **Ali/Wendy**: Core lab reader recruitment (3 minimum, §Extended Data).
15. **Adib/Goodfire**: SAE training + frame shuffling (§2.8, blocked on checkpoint delivery).
16. **Reza**: UMAP main figure (t-SNE done, need UMAP + UHN).

### P6: Fairness (CY — §2.7, blocked on Chicago demographics)
17. **Email Joe for Chicago demographics** — blocking Table 1 + entire §2.7.
18. **Fairness runs**: LVEF MAE, mortality AUC by sex/race/age/insurance. ECE, equalized odds.

### P7: Methods/Admin
19. **REB protocol number** — from Wendy.
20. **Methods placeholders** — masking strategy details, preprocessing params, frame count, FPS.
21. **UHN demographics for Table 1** — age, sex distribution.

### Pred Avg Summary (6/12 tasks fully done)

| Task | G | L | L-K | EP | Pan | Status |
|------|---|---|-----|----|----|--------|
| LVEF | 0.778 | 0.577 | 0.702 | 0.681 | 0.665 | **DONE** |
| TAPSE | 0.633 | 0.430 | 0.555 | 0.430 | 0.385 | **DONE** |
| RVSP | 0.504 | 0.168 | 0.317 | 0.169 | 0.274 | **DONE** |
| AV mean grad | 0.579 | 0.147 | 0.328 | 0.462 | 0.378 | **DONE** |
| TR severity | 0.854 | 0.787 | 0.817 | 0.780 | 0.778 | **DONE** |
| Onset | 0.793 | 0.514 | 0.677 | 0.776 | 0.759 | **DONE** |
| MR severity | 0.882 | -- | -- | -- | -- | 1/5 |
| AS severity | -- | -- | -- | -- | -- | 0/5 (G stale) |
| AV Vmax | -- | -- | -- | -- | -- | 0/5 |
| AR severity | -- | -- | -- | -- | -- | 0/5 |
| E/e' medial | -- | -- | -- | -- | -- | 0/5 |
| RV S' | -- | -- | -- | -- | -- | training |

**DONE since 2026-03-18:**
- Manuscript B-mode vs all-views clarification (6 sections of sn-article.tex edited)
- RVSP updated to pred-avg R²=0.504 in manuscript (from single-clip 0.463)
- TAPSE updated to pred-avg R²=0.633 in manuscript (from single-clip 0.537)
- LVEF pred avg complete: all 5 models (L-K 0.702, Pan 0.665 added)
- TR severity pred avg complete: all 5 models (EP 0.780, Pan 0.778 added)
- AV mean grad pred avg re-run VALID: all 5 models (G 0.579, Bug 008 fixed)
- MR severity pred avg: G 0.882
- AR severity, AS severity, MR severity, AV Vmax, E/e' medial: all 5/5 trained
- RV S': G 0.491, L 0.234 done; L-K training
- All Bug 007 checkpoint retraining complete (11 tasks × 5 models = 55 best.pt)
- Literature review: B-mode hemodynamic prior art documented

## View-Filtered Training Pipeline — DONE (2026-03-11)

All-clip CSVs built for 47 tasks. View-filtered CSVs built for 41 tasks (6 unfiltered). Trajectory CSVs built for 5 tasks.

**Script:** `experiments/nature_medicine/uhn/build_viewfiltered_csvs.py`
**Output:** `experiments/nature_medicine/uhn/probe_csvs/{task}/train_vf.csv`, `val_vf.csv`, `test_vf.csv`
**Lookup data:** `classifier/output/view_inference_18m/master_predictions.csv` (view) + `color_inference_18m/master_predictions.csv` (B-mode/Doppler)
**Run scripts:** `scripts/run_uhn_probe.sh` (generic single-task) + `scripts/run_phase1.sh` (Phase 1 orchestrator). Auto-detects task type, view filtering, study_sampling from CSV directory contents.

| Task | Filter | Train clips kept | Studies kept |
|------|--------|-----------------|--------------|
| tapse | A4C | 281K (18.4%) | 25,337 / 25,737 (98.4%) |
| rv_fac | A4C | 80K (19.3%) | 6,398 / 6,714 (95.3%) |
| rv_sp | A4C+Subcostal | 392K (26.2%) | 24,852 / 25,174 (98.7%) |
| rv_function | A4C+Subcostal+PLAX | 2.1M (39.4%) | 86,113 / 91,872 (93.7%) |
| rv_size | A4C+Subcostal+PLAX+PSAX | 2.2M (60.1%) | 57,230 / 61,422 (93.2%) |

**MR + AS severity COMPLETE** (2026-03-15). See Preliminary Hemodynamic Results below.

### B-Mode Hemodynamic Filters — DONE for all tasks (2026-03-16)

B-mode-only view filtering applied via `build_viewfiltered_csvs.py` with `bmode_only=True`. Excludes colour/spectral/tissue Doppler clips using color classifier predictions.

| Task | Views | B-mode filter | Status |
|------|-------|---------------|--------|
| mr_severity | A4C, A2C, A3C, PLAX | Yes | **DONE** (G 0.860) |
| as_severity | PLAX, PSAX-AV, A3C | Yes | **DONE** (G 0.908) |
| tr_severity | A4C, Subcostal, PLAX | Yes | **DONE** (G 0.838) |
| aov_vmax | PLAX, A3C, PSAX-AV | Yes | **DONE** (G R²=0.582) |
| aov_mean_grad | PLAX, A3C, PSAX-AV | Yes | **DONE** (5/5 ckpts; pred avg needs re-run, Bug 008) |
| ar_severity | A4C, A2C, A3C, PLAX | Yes | **PARTIAL** (G only ep 11, 4 models needed) |
| rvsp | A4C, Subcostal | Yes | **DONE** (all 5: G 0.463, L-K 0.268, Pan 0.248, EP 0.207, L 0.137) |
| mv_ee_medial | A4C | Yes | QUEUED |
| aov_area | PLAX, A3C, PSAX-AV | Yes | QUEUED |

### Strategy E Results (updated 2026-03-19)

All results: d=1 attentive probes, 15 epochs, 12-head HP grid. (PA) = prediction-averaged test-set metrics (final). Others are single-clip best-head val metrics.

**Pred Avg Complete (6 tasks, all 5 models)**

| Task | G | L-K | EchoPrime | L | PanEcho |
|------|---|-----|-----------|---|---------|
| LVEF R² | **0.778** | 0.702 | 0.681 | 0.577 | 0.665 |
| TAPSE R² | **0.633** | 0.555 | 0.430 | 0.430 | 0.385 |
| RVSP R² | **0.504** | 0.317 | 0.169 | 0.168 | 0.274 |
| AV mean grad R² | **0.579** | 0.328 | 0.462 | 0.147 | 0.378 |
| TR severity AUROC | **0.854** | 0.817 | 0.780 | 0.787 | 0.778 |
| Trajectory onset AUROC | **0.793** | 0.677 | 0.776 | 0.514 | 0.759 |
| MR severity AUROC | **0.882** (PA) | 0.803* | 0.770* | 0.765* | 0.724* |

**Training Complete, Pred Avg TODO (single-clip val shown)**

| Task | G | L-K | EchoPrime | L | PanEcho |
|------|---|-----|-----------|---|---------|
| AS severity AUROC | **0.908*** | 0.821* | 0.827* | 0.786* | 0.762* |
| AV Vmax R² | **0.582*** | 0.388* | 0.476* | 0.232* | 0.390* |
| E/e' medial R² | **0.558*** | 0.438* | 0.391* | 0.296* | 0.400* |
| AR severity AUROC | **0.739*** | 0.650* | 0.673* | 0.644* | 0.653* |
| RV S' R² | **0.491*** | training | — | 0.234* | — |

*Single-clip val metrics. Pred avg forthcoming.

### Checkpoint Inventory (2026-03-19)

All 11 trained tasks have best.pt for all 5 models (G, L, L-K, EP, Pan).

| Task | Pred Avg Status |
|------|----------------|
| LVEF | **ALL 5 DONE** |
| TAPSE | **ALL 5 DONE** |
| RVSP | **ALL 5 DONE** |
| AV mean grad | **ALL 5 DONE** |
| TR severity | **ALL 5 DONE** |
| Trajectory onset | **ALL 5 DONE** |
| MR severity | G DONE (0.882). L/L-K/EP/Pan TODO |
| AS severity | 0/5 (all TODO) |
| AV Vmax | 0/5 (all TODO) |
| E/e' medial | 0/5 (all TODO) |
| AR severity | 0/5 (all TODO) |
| RV S' | G+L done. L-K training (ep 10/15). EP/Pan queued |

**Key findings:**
- B-mode video contains hemodynamic severity information — this is a clinical discovery, not model-specific
- ALL models extract non-trivial signal (PanEcho 0.715-0.762 AUROC), so story is "scale advantage" not "unique capability"
- EchoJEPA-G leads by +8-10pp over next-best on hemodynamics
- G-vs-EchoPrime gap narrows to +1.7pp on trajectory (text-supervised models efficient for prognostic features)
- 15 epochs is sufficient (extending to 20 yielded <0.005 improvement)
- **Literature context:** AS from B-mode done by others (Holste 2023, Ahmadi 2024). MR, TR, AV Vmax from B-mode: genuinely novel. DELINEATE papers use color Doppler. PanEcho says regurgitation "requires Doppler."

---

## MVP — Required for submission

### Infrastructure

| Task | Owner | Status | Blocks |
|------|-------|--------|--------|
| Training/test CSVs (UHN + MIMIC) | Alif | **DONE** (47 UHN + 23 MIMIC + trajectory onset) | All probe training |
| View-filtered CSVs | Alif | **DONE** (41 UHN tasks) | View-specific probes |
| B-mode-only view filters (hemodynamic tasks) | Alif | **DONE** (all hemodynamic tasks) | Pillar 2 probes |
| Run scripts (training + inference, auto-config, archiving, Bug 008-010 fixes) | Alif | **DONE** | Batch execution |
| Study-level prediction aggregation | Alif | **DONE** | Study-level metrics |
| d=1 attentive probes (UHN) | Alif | **Training:** 5 tasks complete (5/5 ckpts), 5 partial. **Pred avg:** LVEF G done (R²=0.778), rest in progress. | UHN results tables |
| d=1 attentive probes (MIMIC) | Alif | TODO | MIMIC results tables |
| Bland-Altman post-processing | Alif | **TODO** | All regression reporting |
| MIMIC embeddings (5 models) | Alif | **DONE** (historic NPZ pipeline, superseded by Strategy E) | SAE, legacy |
| Ship checkpoints to Goodfire | Alif | **TODO** | Frame shuffling, SAE experiments |

### Core Experiments (Wendy's Pillars)

| Task | Owner | Status | Section |
|------|-------|--------|---------|
| Standard benchmarks: LVEF, RVSP, LV mass, IVSd, RWMA | Alif | **LVEF done**. RVSP, LV mass, IVSd in Phase 1 queue. | 2.1 (brief) |
| RV mechanics: TAPSE, S', FAC, RV function grade, RV basal dim | Alif | **TAPSE done**. Others in Phase 1 queue. | Pillar 1 (core novelty) |
| Hemodynamics: MR/AS/TR severity from B-mode only, E/e' | Alif | **MR+AS DONE** (G: MR 0.860, AS 0.908). 5 tasks remaining. 15ep sufficient. | Pillar 2 (core novelty) |
| Trajectory prediction: 93K pairs, 5 parameters | Alif | TODO (Phase 2) | Pillar 3 (core novelty) |
| MIMIC outcomes (sklearn, EchoJEPA-G) | CY | **DONE** — H3.1 passed. Ensemble AUC: 30d 0.912, 1yr 0.846. Echo ≈ EHR for mortality. ICU transfer 0.570 (deprioritize). | 2e |
| MIMIC EHR-only baseline (XGBoost + TabPFN) | CY | **DONE** — 54 features. Mortality AUC 0.856-0.959. TabPFN strongest. | 2e |
| Combined model (echo + EHR) | CY | **TODO** — key claim: does echo add to EHR? | 2e |
| Acuity conditioning (H_confound) | CY | **TODO** — likelihood ratio test | 2e |
| MIMIC fairness: discrimination parity, calibration equity | CY | TODO | Fairness |
| Frame shuffling (motion-dependence proof) | Goodfire | TODO (blocked on checkpoint delivery) | Interpretability |

**MVP progress (updated 2026-03-19):** Infrastructure complete (CSVs, scripts, pred avg pipeline, Bugs 007-014 fixed). **11 tasks × 5 models ALL TRAINED.** 6 tasks with ALL 5 pred avg DONE: LVEF, TAPSE, RVSP, AV mean grad, TR severity, trajectory onset. MR severity G pred avg done (0.882). Manuscript updated with all available results + vision-only/data-efficiency emphasis + ‡/\tbd markers for remaining work. **Remaining for tables**: pred avg for 5 more tasks (~29 runs), RV S' + FAC training, MIMIC Strategy E (8 outcomes × 4 models), Bland-Altman. **Remaining for sections**: disease panel (§2.6), fairness (§2.7, blocked), forward prediction (§2.8), methods placeholders. CY: combined model + confounding TODO. Ali: core lab TODO.

### Verification

| Task | Status | Notes |
|------|--------|-------|
| d=1 attentive probe verification (4 models) | **DONE** | G +1.2pp, L +17.3pp, EchoPrime +9.3pp, PanEcho +7.1pp. EchoMAE dropped. |

---

## Strong Additions — Elevates paper

| Task | Owner | Status | Manuscript Section |
|------|-------|--------|-------------------|
| **Forward prediction (JEPA predictor, Exp 6.1)** | Alif | **TODO** — Key JEPA differentiator | §2.8 Ablations |
| **Anomaly detection (Exp 6.4)** | Alif | **TODO** — Zero-shot, JEPA-only | §2.8 Ablations |
| **Disease panel (9 diseases × 4 models)** | Alif/Reza | **TODO** — CSVs ready | §2.6 Disease Detection |
| **MIMIC Strategy E outcomes (8 tasks × 4 models)** | Alif | **TODO** — currently sklearn G only (†) | §2.5 Outcomes Table |
| **Trajectory expansion (EF change, new HF diagnosis)** | Alif | **TODO** — ‡ in manuscript | §2.5 Trajectory |
| **Combined echo+EHR model** | CY | **TODO** — key additive value claim | §2.5 Outcomes |
| **Fairness analysis** | CY | **TODO** — blocked on Chicago demographics | §2.7 Fairness |
| **Core lab (3 readers minimum)** | Ali/Wendy | **TODO** | Extended Data |
| SAE training + analysis | Adib/Goodfire | **IN PROGRESS** (GPU issues) | §2.8 Interpretability |
| Frame shuffling | Goodfire | **TODO** (blocked on checkpoint delivery) | §2.8 |
| Attention map figures | Adib | **Infra DONE** | Extended Data |
| UMAP main figure | Reza/Alif | **t-SNE DONE**, need UMAP + UHN | Main Figure |
| Diastolic function, PA pressure, cardiac output | Alif | **TODO** — CSVs need B-mode filter | §2.2 Extended Data |
| EF baseline matrix (echo measurement ensemble) | CY | **TODO** | §2.5 |
| Troponin 48h biomarker labels | CY | **TODO** | §2.5 Biomarkers |

---

## Deferrable — Revision period

| Task | Owner | Notes |
|------|-------|-------|
| SAE full pipeline (feature visualization, cardiologist labelling, inter-rater) | Adib/Faraz/Wendy | Phase B/C blocked on mp4s + Wendy review |
| Core lab full evaluation (3+ readers, reading sessions, adjudication) | Ali/Wendy | Infrastructure built now, sessions in revision |
| Rare disease label verification (blinded 60-study review) | Wendy | Component B of core lab |
| Latent forward prediction (multi-cycle rollout) | Alif | Research experiment, untested |
| Cross-institution validation (UChicago) | Teodora | Paper does not depend on this |
| EchoNet-Dynamic / CAMUS external benchmarks | Teodora/CY | Standard, not novel |
| Intersectional fairness (age x sex x race) | CY | n >= 50 subgroups |
| Fabry disease / Marfan syndrome labels | — | Small cohorts, labels not built |
| Remaining ~20 UHN tasks for Extended Data | Alif | Labels built, run anytime |
| Layerwise probing (6 encoder depths) | Adib | Needs frozen ViT-G encoder |

---

## UHN Label Provenance Warning

Current rare disease cohort counts are inflated by union of HeartLab SENTENCE templates, Syngo observations, and Syngo Indication/PatientHistory fields. Indication-based labels record why the echo was ordered ("rule out amyloidosis"), not what was found. Cleaner cohorts using SENTENCE templates only (with rule-out exclusion) would be:
- HCM: ~5,000-8,000 (down from 12,291)
- Amyloidosis: ~800-900 (down from 1,174)
- Endocarditis: ~2,000 (down from 5,236)

See `nature_medicine_task_overview.md` for full analysis. Clinician validation (Wendy) of 50-100 samples per disease is the minimum needed for Nature Medicine — deferred to revision.

---

## What's Done

### UHN Pipeline
- [x] P0: UID -> StudyRef mapping (`aws_syngo.csv`, 320K studies)
- [x] P1: Clip index (`uhn_clip_index.npz`, 18.1M clips)
- [x] P2: Patient splits (`patient_split.json`, 138K patients, 70/10/20 temporal)
- [x] P3: Label NPZs (53 tasks: 20 regression + 17 classification + 9 rare disease + 6 trajectory + 1 RWMA)
- [x] P4a: EchoJEPA-G embeddings (319,815 studies, 1408-dim) [historic NPZ pipeline]
- [x] P4b: EchoJEPA-L embeddings (319,802 studies, 1024-dim) [historic NPZ pipeline]
- [x] P4c: EchoJEPA-L-K extraction started (17% complete, chunk-based resume available) [historic NPZ pipeline]
- [x] P5a: UHN all-clip probe CSVs (47 tasks)
- [x] P5b: UHN view-filtered probe CSVs (41 tasks)
- [x] P5c: UHN trajectory CSVs (5 tasks)
- [x] P5d: Run scripts built (Phase 1: 18 tasks x 5 models)
- [x] Label cleaning (physiological range filters, 1,582 rows dropped)
- [x] Rare disease confound audit (hard negative controls verified)
- [x] UHN linear probes: EchoJEPA-G (mean AUC 0.874, mean R² 0.625) and EchoJEPA-L (failed — embedding collapse on UHN) [historic NPZ pipeline]

### MIMIC Pipeline
- [x] 23 label CSVs (mortality, diseases, biomarkers, outcomes)
- [x] MIMIC probe CSVs rebuilt with Z-scored regression (23 tasks)
- [x] 7 model clip-level extractions (all correct) [historic NPZ pipeline]
- [x] 7 model study-level pooling + 23 task splits [historic NPZ pipeline]
- [x] Probe training: 161/161 jobs complete [historic NPZ pipeline]
- [x] Shared infrastructure (clip_index, patient_split, labels/)
- [x] Charlson/Elixhauser comorbidity scores (7,243 studies)
- [x] Demographics/fairness CSV (sex, race, age, insurance)
- [x] Acuity covariates (DRG severity, ICU flags, triage)
- [x] EHR features (54-feature baseline matrix)
- [x] S3 upload + presigned URLs for all 8 zips

### Bug Fixes
- [x] Shuffle ordering (bug 001) — code fix + post-hoc reordering
- [x] Encoder normalization (bug 002) — code fix + re-extraction complete
- [x] EchoFM temporal padding (bug 003) — code fix applied
- [x] Video load substitution tracking (bug 004) — logging added
- [x] drop_last forwarding (bug 005) — code fix applied
- [x] PanEcho hubconf.py local tasks.pkl cache
- [x] EchoFM simplejson dependency
- [x] MViT GPU memory leak (gc + empty_cache every 100 batches)
- [x] DataLoader resume logic (new DataLoader for resume)
- [x] TF32 matmul enabled (8x throughput for fp32 models)
- [x] Checkpoint loss prevention (bug 007) — per-epoch archiving to local + S3
- [x] Inference probe loading (bug 008) — `resume_checkpoint: true` required in YAML
- [x] /dev/shm exhaustion (bug 009) — orphan cleanup, reduced workers/batch size
- [x] Concurrent job safety (bug 010) — ppid=1 filtered orphan cleanup in all scripts
