# Roadmap

Consolidated view of outstanding work across UHN and MIMIC pipelines. Updated 2026-03-18.

Source of truth for paper scope and task priorities: `uhn_echo/nature_medicine/context_files/nature_medicine_task_overview.md`. Detailed specs in `planning/tasks/`. Per-person assignments in `roles/`. This file is the cross-cutting infrastructure and experiment summary.

## Paper Framing (from task overview)

The core novelty is organized around **Wendy's three pillars**:
1. **RV mechanics** — learned functional organization (TAPSE, S', FAC, RV basal dimension, RV function grade, RV size)
2. **Hemodynamics** — structure predicts flow (MR/AS/TR severity from B-mode only, AV gradients, E/e' from structural views). **Reframed 2026-03-15:** B-mode video contains hemodynamic information accessible to ALL models (EchoPrime, PanEcho also extract signal). The story is *scale advantage* (+8-9pp AUROC for EchoJEPA-G over baselines) and *no text supervision needed*, not unique capability.
3. **Forecasting** — trajectory prediction (future LVEF, TAPSE, LV mass, MR severity from 93K pairs)

Standard benchmarks are brief (one paragraph + Extended Data). Disease detection is supporting evidence, not headline. SAE interpretability and core lab evaluation are Strong Additions / deferred to revision.

**5 models** (updated 2026-03-13): EchoJEPA-G, EchoJEPA-L, EchoJEPA-L-K, EchoPrime, PanEcho. EchoMAE dropped (undertrained checkpoint). ICML preprint carries JEPA-vs-MAE comparison.

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

## Blocking Work — Priority Order (updated 2026-03-18)

1. **Fix failed pred avg runs** — LVEF L-K (socket error) + PanEcho (forward crash). Re-run on idle GPUs 0-3.
2. **Finish TR severity pred avg** — L-K running, EP/Pan queued (auto-chained). Then AS severity training.
3. **Re-run AV mean grad pred avg** — Bug 008 invalidated all 5 results. Checkpoints archived, just need inference.
4. **Train missing checkpoints** — AR severity (G at ep11, 4 models TODO), TAPSE (L-K ep8, EP/Pan TODO)
5. **JEPA-unique experiments** — 6.1 (forward prediction), 6.4 (anomaly detection via prediction error), 5.4 (trajectory onset expansion: MR severity + TAPSE)
6. **Complete Tier 1 hemodynamic tasks** — E/e' medial (novel), AV area
7. **Ship code + checkpoints to Goodfire** — repo with Claude docs, clean CSVs, L/L-K checkpoints, G NPZ embeddings
8. **Bland-Altman analysis** in eval post-processing (Wendy: MAE alone insufficient for Nature Medicine)
9. **Email Joe for Chicago demographics** (age, sex, race/ethnicity) — blocking for cross-site fairness

**DONE since last update (2026-03-18):**
- RVSP all 5 models complete: G R²=0.463, L-K 0.268, Pan 0.248, EP 0.207, L 0.137
- LVEF pred avg 3/5: G R²=0.778, EP 0.681, L 0.577 (L-K/Pan failed, re-run needed)
- TR severity pred avg: G AUROC=0.854 (+1.6pp), L 0.787 (+3.2pp). L-K running.
- TAPSE: G R²=0.552 (ep14), L R²=0.288 (ep15). L-K stopped at ep 8.
- Manuscript restructured around three-level world model (cross-modal, cross-system, cross-temporal)
- Bugs 008-010 found and fixed (inference checkpoint loading, shm exhaustion, concurrent job safety)
- Generic `scripts/run_pred_avg.sh` updated with all fixes

**DONE previously:**
- Study-level prediction aggregation: **DONE** (auto-enables when `val_only=True` + `study_sampling=True`)
- Hemodynamic training: MR (G 0.860), AS (G 0.908), TR (G 0.838), AV Vmax (G R²=0.582) all 5 models
- Trajectory onset: **DONE** (G 0.793)
- LVEF training: all 5 models retrained (15 ep)
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

### Strategy E Results (updated 2026-03-18)

All results: d=1 attentive probes, 15 epochs, 12-head HP grid. Results marked (PA) include prediction averaging; others are single-clip best-head val metrics.

**Hemodynamic Inference (B-mode only, single-clip)**

| Task | G | L-K | EchoPrime | L | PanEcho |
|------|---|-----|-----------|---|---------|
| AS severity (4-class AUROC) | **0.908** | 0.821 | 0.827 | 0.786 | 0.762 |
| MR severity (5-class AUROC) | **0.860** | 0.803 | 0.770 | 0.771 | 0.724 |
| TR severity (5-class AUROC) | **0.838** | 0.787 | 0.758 | 0.755 | 0.731 |
| AV Vmax (R²) | **0.582** | 0.388 | 0.476 | 0.232 | 0.390 |
| RVSP (R²) | **0.463** | 0.268 | 0.207 | 0.137 | 0.248 |
| AR severity (AUROC) | 0.740* | — | — | — | — |

*AR severity: G only at ep 11, training stopped. 4 models TODO.

**Trajectory Prediction (single-clip training, pred avg at test)**

| Task | G | EchoPrime | PanEcho | L-K | L |
|------|---|-----------|---------|-----|---|
| Onset cardiomyopathy (AUROC) | **0.793** | 0.776 | 0.759 | 0.677 | 0.514 |

**Standard Benchmarks**

| Task | G | L-K | EchoPrime | L | PanEcho |
|------|---|-----|-----------|---|---------|
| TAPSE (R²) | **0.552** | 0.427* | — | 0.288 | — |
| LVEF (R², single-clip) | **0.583** | 0.583 | 0.563 | 0.556 | 0.556 |
| LVEF (R², pred avg) | **0.778** | FAIL | 0.681 | 0.577 | FAIL |

*TAPSE L-K stopped at ep 8. EP/Pan not started. LVEF pred avg L-K/Pan failed (socket/forward errors).

Note: LVEF pred avg in progress (G done, L/L-K/EP/Pan running). TAPSE retraining (G epoch 13/15, ckpts lost per Bug 007).

### Checkpoint Inventory (2026-03-18)

| Task | G | L | L-K | EP | Pan | Pred Avg Status |
|------|---|---|-----|----|----|----------------|
| lvef | best.pt | best.pt | best.pt | best.pt | best.pt | G done (R²=0.778), L in progress |
| tr_severity | best.pt | best.pt | best.pt | best.pt | best.pt | Not started (ready) |
| aov_mean_grad | best.pt | best.pt | best.pt | best.pt | best.pt | **INVALID** (Bug 008, must re-run) |
| trajectory_lvef_onset | best.pt | best.pt | best.pt | best.pt | best.pt | Not started (ready) |
| trajectory_lvef_v1 | best.pt | best.pt | best.pt | best.pt | best.pt | Not started |
| tapse | best.pt | — | — | — | — | G retraining (epoch 13/15) |
| rvsp | best.pt | best.pt | best.pt | — | — | Needs 2 more models |
| aov_vmax | — | — | — | best.pt | best.pt | Needs 3 more models |
| ar_severity | best.pt | — | — | — | — | Needs 4 more models |
| trajectory_lvef | best.pt | best.pt | best.pt | — | — | Needs 2 more models |

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

**MVP progress (updated 2026-03-18):** All CSVs, run scripts, and prediction averaging pipeline built and debugged (Bugs 008-010 fixed). Training complete for 5 tasks (5/5 ckpts): LVEF, TR severity, AV mean grad, trajectory onset, trajectory v1. LVEF pred avg G: R²=0.778. TAPSE retraining in progress (G epoch 13/15, ckpts lost per Bug 007). 5 tasks partially trained. AV mean grad pred avg invalidated by Bug 008 (must re-run). CY's MIMIC outcome probes + EHR baseline done (H3.1 passed). Adib: attention map infra done, SAE training in progress. Remaining: finish TAPSE retrain, run pred avg for TR/trajectory/aov_mean_grad, train missing ckpts (AV Vmax G/L/L-K, AR sev 4 models, RVSP 2 models), Bland-Altman, combined model.

### Verification

| Task | Status | Notes |
|------|--------|-------|
| d=1 attentive probe verification (4 models) | **DONE** | G +1.2pp, L +17.3pp, EchoPrime +9.3pp, PanEcho +7.1pp. EchoMAE dropped. |

---

## Strong Additions — Elevates paper

| Task | Owner | Status | Section |
|------|-------|--------|---------|
| **Frame shuffling motion-dependence test** | **Goodfire** | **TODO** (blocked on checkpoint delivery) | Interpretability |
| SAE training + basic analysis (proxy classifiers, retention curves) | Adib/Goodfire | **IN PROGRESS** (training started, GPU broke, restarting) | Interpretability |
| Attention map visualization (supplementary) | Adib | **Infra DONE** (hooks for ViT-G + EchoPrime). Head specialization + temporal consistency findings. | Extended Data |
| Prosthetic valve detection (NPZs need building) | Alif | TODO | 2c |
| Core lab infrastructure + pilot | Ali | TODO | Human eval |
| Fine-tuned ViT-L baseline (DeLong vs frozen G) | Adib | TODO | Baselines |
| Disease detection: HCM, amyloidosis, takotsubo, endocarditis | Reza | TODO | 2d (supporting) |
| Representation visualization (UMAP + k-NN + spatial info gap) | Reza/Alif | **t-SNE DONE** (23 tasks x 7 models on MIMIC). Need UMAP regen + UHN main figure. | Figure X (main) |
| Troponin 48h biomarker labels | CY | TODO | 2f |
| Trajectory pair analysis | CY/Alif | TODO | 3a |
| EF baseline matrix (echo measurement ensemble) | CY | TODO | 2e |
| E/e', cardiac output, cardiac rhythm | Alif | TODO | 2.1 Extended Data |
| Forward prediction (JEPA predictor, first N frames) | Alif | TODO | 3b |

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
