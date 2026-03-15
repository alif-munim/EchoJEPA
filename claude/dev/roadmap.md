# Roadmap

Consolidated view of outstanding work across UHN and MIMIC pipelines. Updated 2026-03-15.

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

### Extraction Status

| Model | Checkpoint | Clip Extraction | Study-Level | Notes |
|-------|-----------|-----------------|-------------|-------|
| EchoJEPA-G | `pt-280-an81.pt` | **DONE** (95GB, 18.1M clips) | **DONE** (319,815 x 1408) | |
| EchoJEPA-L | `vitl-pt-210-an25.pt` | **DONE** (70GB, 18.1M clips) | **DONE** (319,802 x 1024) | |
| EchoJEPA-L-K | `vitl-kinetics-pt220-an55.pt` | **~17% done** (crashed/interrupted) | Missing | Chunk-based, supports resume. ~8-9h remaining |
| EchoMAE-L | `videomae-ep163.pt` | **IN PROGRESS** (~19h est.) | — | Controlled JEPA-vs-MAE comparison. MVP |
| Random Init | Untrained ViT-L | **TODO** | — | Baseline for all tables. MVP |

PanEcho, EchoPrime, EchoFM UHN extractions are not planned (these models are primarily compared on MIMIC).

### UHN Per-Task Split Pipeline — DONE

Built `evals/regenerate_uhn_downstream.py`. Generated splits for G and L (53 tasks each). Re-run for L-K, EchoMAE, Random Init when their study-level embeddings are available.

---

## Blocking Work — Priority Order

1. **Complete MR + AS severity 20-epoch runs** (in progress, see Preliminary Results below)
   - MR: echojepa-g done (20ep), echojepa-l at ep16, others at ep9-15 (need resume)
   - AS: echojepa-g done (24ep), echojepa-l at ep16, others at ep15 (need resume)
   - Known issue: shared memory exhaustion (shmmni=4096) crashes multi-worker DDP jobs. Fix: `num_workers: 4` + cleanup `/dev/shm/torch_*` between runs.

2. **Run remaining 5/7 hemodynamic tasks** (TR severity, AR severity, E/e', AV Vmax, RVSP from B-mode only)
   - B-mode-only view filters already built for MR + AS. Need to add: E/e', RVSP, E/A, MV DT, diastolic function, PA pressure.

3. **Run Phase 1 remaining UHN tasks** (14 remaining tasks x 5 models = 70 runs)
   - TAPSE + LVEF complete (5/5 models each). MR + AS severity in progress.

4. **Ship code + checkpoints to Goodfire** — repo with Claude docs, clean CSVs, L/L-K checkpoints, G NPZ embeddings

5. **Implement study-level prediction aggregation** in eval.py
   - Needed for study-level test metrics and fair multi-clip comparison
   - Current val uses `DistributedStudySampler` (1 random clip/study/epoch) — no prediction averaging
   - All reported results to date are single-clip, not study-averaged

6. **Implement Bland-Altman analysis** in eval post-processing (Wendy: MAE alone insufficient for Nature Medicine)

7. **Email Joe for Chicago demographics** (age, sex, race/ethnicity) — blocking for cross-site fairness

8. **Build Phase 2 run scripts** (trajectory + MIMIC)

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

**Run in progress:** MR severity + AS severity 20-epoch extension (B-mode only, class-balanced). Logs: `logs/jobA_mr_severity_ep20_v3.log`, `logs/jobB_as_severity_ep20_v3.log`.

### B-Mode Hemodynamic Filters — DONE for MR + AS (2026-03-14)

B-mode-only view filtering applied via `build_viewfiltered_csvs.py` with `bmode_only=True`. Excludes colour/spectral/tissue Doppler clips using color classifier predictions.

| Task | Views | B-mode filter | Train studies (class-balanced) |
|------|-------|---------------|-------------------------------|
| mr_severity | A4C, A2C, A3C, PLAX | Yes | ~29K (from 89K, ratio=3) |
| as_severity | PLAX, PSAX-AV, A3C | Yes | ~22K (from 131K, ratio=3) |
| tr_severity | A4C, Subcostal | TODO | — |
| ar_severity | PLAX, A3C | TODO | — |
| e_e_prime | A4C | TODO | — |
| av_vmax | PSAX-AV, A3C, A5C | TODO | — |
| rvsp | A4C, Subcostal | TODO (RVSP from B-mode is novel) | — |

### Preliminary Hemodynamic Results (2026-03-15, single-clip, no prediction averaging)

**AS Severity** (4-class ordinal, B-mode only, d=1 attentive, class_balance_ratio=3):

| Model | Epochs | Best AUROC | Status |
|-------|--------|-----------|--------|
| EchoJEPA-G | 24 | **0.908** | Done |
| EchoPrime | 15 | 0.827 | Needs resume to 20 |
| EchoJEPA-L-K | 15 | 0.819 | Needs resume to 20 |
| EchoJEPA-L | 16 | 0.784 | Running |
| PanEcho | 15 | 0.762 | Needs resume to 20 |

**MR Severity** (5-class ordinal, B-mode only, d=1 attentive, class_balance_ratio=3):

| Model | Epochs | Best AUROC | Status |
|-------|--------|-----------|--------|
| EchoJEPA-G | 20 | **0.860** | Done |
| EchoJEPA-L-K | 15 | 0.800 | Needs resume to 20 |
| EchoPrime | 15 | 0.770 | Needs resume to 20 |
| EchoJEPA-L | 16 | 0.767 | Running |
| PanEcho | 9 | 0.724 | Needs resume to 20 |

**Key findings (narrative reframing):**
- B-mode video contains hemodynamic severity information — this is a clinical discovery, not model-specific
- ALL models extract non-trivial signal (PanEcho 0.724-0.762 AUROC), so "no prior model" claims need softening
- EchoJEPA-G leads by +8-9pp over next-best (EchoPrime), suggesting scale + latent prediction captures more hemodynamic structure
- EchoPrime achieves this without any text about valve severity (CLIP training uses reports, but B-mode clips have no Doppler)
- These are **single-clip results** (1 random clip per study per val epoch). Prediction averaging will likely widen gaps for models with richer spatial representations (EchoJEPA: 1568 tokens benefit from diverse view sampling)
- Only 2/7 hemodynamic tasks complete. Breadth across all 7 tasks strengthens the story.

---

## MVP — Required for submission

### Infrastructure

| Task | Owner | Status | Blocks |
|------|-------|--------|--------|
| Training/test CSVs (UHN + MIMIC) | Alif | **DONE** (47 UHN + 23 MIMIC + 5 trajectory) | All probe training |
| View-filtered CSVs | Alif | **DONE** (41 UHN tasks) | View-specific probes |
| B-mode-only view filters (hemodynamic tasks) | Alif | **DONE** for MR+AS, TODO for 5 remaining | Pillar 2 probes |
| Run scripts for Phase 1-3 | Alif | **DONE** (Phase 1 built, Phase 2-3 TODO) | Batch execution |
| d=1 attentive probe training (UHN: 47×5) | Alif | **IN PROGRESS** (TAPSE+LVEF done, MR+AS severity running, 14 tasks remaining) | UHN results tables |
| d=1 attentive probe training (MIMIC: 23×5) | Alif | TODO | MIMIC results tables |
| Bland-Altman post-processing | Alif | **TODO** | All regression reporting |
| Study-level prediction aggregation | Alif | **TODO** | Study-level metrics |
| MIMIC embeddings (7 models) | Alif | **DONE** (historic NPZ pipeline) | SAE, legacy comparison |
| Ship checkpoints to Goodfire | Alif | **TODO** | Frame shuffling, SAE experiments |

### Core Experiments (Wendy's Pillars)

| Task | Owner | Status | Section |
|------|-------|--------|---------|
| Standard benchmarks: LVEF, RVSP, LV mass, IVSd, RWMA | Alif | **LVEF done**. RVSP, LV mass, IVSd in Phase 1 queue. | 2.1 (brief) |
| RV mechanics: TAPSE, S', FAC, RV function grade, RV basal dim | Alif | **TAPSE done**. Others in Phase 1 queue. | Pillar 1 (core novelty) |
| Hemodynamics: MR/AS/TR severity from B-mode only, E/e' | Alif | **MR+AS running** (G done: MR 0.860, AS 0.908). 5 tasks remaining. | Pillar 2 (core novelty) |
| Trajectory prediction: 93K pairs, 5 parameters | Alif | TODO (Phase 2) | Pillar 3 (core novelty) |
| MIMIC outcomes (sklearn, EchoJEPA-G) | CY | **DONE** — H3.1 passed. Ensemble AUC: 30d 0.912, 1yr 0.846. Echo ≈ EHR for mortality. ICU transfer 0.570 (deprioritize). | 2e |
| MIMIC EHR-only baseline (XGBoost + TabPFN) | CY | **DONE** — 54 features. Mortality AUC 0.856-0.959. TabPFN strongest. | 2e |
| Combined model (echo + EHR) | CY | **TODO** — key claim: does echo add to EHR? | 2e |
| Acuity conditioning (H_confound) | CY | **TODO** — likelihood ratio test | 2e |
| MIMIC fairness: discrimination parity, calibration equity | CY | TODO | Fairness |
| Frame shuffling (motion-dependence proof) | Goodfire | TODO (blocked on checkpoint delivery) | Interpretability |

**MVP progress:** All CSVs and run scripts built. TAPSE + LVEF complete (5 models each). MR + AS severity running (EchoJEPA-G done: MR 0.860, AS 0.908 AUROC from B-mode only). CY's MIMIC outcome probes + EHR baseline done (H3.1 passed, mortality AUC 0.846-0.912). Adib: attention map infra done, SAE training in progress. Reza: MIMIC t-SNE complete (23 tasks x 7 models), UMAP regeneration + UHN main figure needed. Remaining: 14 Phase 1 UHN tasks, 5 hemodynamic B-mode filters, prediction aggregation, Bland-Altman, combined model, acuity conditioning, Phase 2.

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
