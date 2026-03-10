# Roadmap

Consolidated view of outstanding work across UHN and MIMIC pipelines. Updated 2026-03-10.

Source of truth for paper scope and task priorities: `uhn_echo/nature_medicine/context_files/nature_medicine_task_overview.md`. Detailed specs in `planning/tasks/`. Per-person assignments in `roles/`. This file is the cross-cutting infrastructure and experiment summary.

## Paper Framing (from task overview)

The core novelty is organized around **Wendy's three pillars**:
1. **RV mechanics** — learned functional organization (TAPSE, S', FAC, RV function grade, RV size)
2. **Hemodynamics** — structure predicts flow (MR/AS/TR severity from structural views only, AV gradients). Nobody has done this with frozen self-supervised representations.
3. **Forecasting** — trajectory prediction (future LVEF, TAPSE, LV mass, MR severity from 93K pairs)

Standard benchmarks are brief (one paragraph + Extended Data). Disease detection is supporting evidence, not headline. SAE interpretability and core lab evaluation are Strong Additions / deferred to revision.

---

## MIMIC Embeddings — COMPLETE

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

161/161 probe jobs done (7 models x 23 tasks). Results in `results/probes/nature_medicine/mimic/`. UHN linear probes also complete for EchoJEPA-G (26 classification + 21 regression + 5 trajectory) and EchoJEPA-L.

## UHN Embeddings — In Progress

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

1. **Resume EchoJEPA-L-K UHN extraction** (~8-9h GPU)
   - Script supports auto-resume: just re-run same command
   - Then pool to study-level

2. ~~**Build UHN per-task split pipeline**~~ — **DONE**

3. **EchoMAE-L UHN extraction** (~19h GPU) — **IN PROGRESS**
   - Config: `configs/inference/vitl/extract_uhn_echomae.yaml`
   - 8×A100, bs=64, w=12. Log: `logs/echomae_uhn_extraction.log`

4. **Random Init UHN extraction** (~10-12h GPU)
   - Need config for untrained ViT-L
   - Baseline column (every table)

---

## MVP — Required for submission

### Infrastructure

| Task | Owner | Status | Blocks |
|------|-------|--------|--------|
| UHN L-K extraction (resume) | Alif | ~17% done | L-K in all UHN tables |
| UHN EchoMAE-L extraction | Alif | IN PROGRESS | JEPA-vs-MAE comparison |
| UHN Random Init extraction | Alif | TODO | Baseline column (every table) |
| UHN per-task split pipeline | Alif | **DONE** | All UHN probing |
| MIMIC embeddings (7 models) | Alif | **DONE** | All MIMIC experiments |
| MIMIC probe training (161 jobs) | Alif | **DONE** | Sections 2.3-2.4 |

### Core Experiments (Wendy's Pillars)

| Task | Owner | Status | Section |
|------|-------|--------|---------|
| Hemodynamics: MR/AS/TR severity from structural views | Alif | TODO | 2a (core novelty) |
| RV mechanics: TAPSE, S', FAC, RV function grade | Alif | TODO | 2b (core novelty) |
| Standard benchmarks: LVEF, RVSP, LV mass, IVSd, view, RWMA | Alif | TODO | 2.1 (brief) |
| Trajectory prediction: 93K pairs, 5 parameters | Alif | TODO | 3a (core novelty) |
| MIMIC outcomes: mortality, biomarkers, baselines | CY | Results available | 2e, 2f |
| MIMIC fairness: discrimination parity, calibration equity | CY | TODO | Fairness |
| JEPA vs MAE complexity gradient (H2.3) | CY | TODO | Ablations (narrative-breaking) |

### Verification (In Progress)

| Task | Status | Notes |
|------|--------|-------|
| Attentive probe verification (11 runs) | Run 8/11 | `scripts/overnight_run.sh`, log: `logs/overnight_node1_20260310_044409.log` |

---

## Strong Additions — Elevates paper

| Task | Owner | Status | Section |
|------|-------|--------|---------|
| Prosthetic valve detection (NPZs need building) | Alif | TODO | 2c |
| SAE training + basic analysis (proxy classifiers, retention curves) | Adib/Faraz | TODO | Interpretability |
| Core lab infrastructure + pilot | Ali | TODO | Human eval |
| Fine-tuned ViT-L baseline (DeLong vs frozen G) | Adib | TODO | Baselines |
| Disease detection: HCM, amyloidosis, takotsubo, endocarditis | Reza | TODO | 2d (supporting) |
| Clustering / UMAP visualizations | Reza | TODO | 2d (supporting) |
| Troponin 48h biomarker labels | CY | TODO | 2f |
| Trajectory pair analysis | CY/Alif | TODO | 3a |
| EF baseline matrix (echo measurement ensemble) | CY | TODO | 2e |
| E/e', cardiac output, cardiac rhythm | Alif | TODO | 2.1 Extended Data |
| Forward prediction (JEPA predictor, first N frames) | Alif | TODO | 3b |
| AV Vmax, mean gradient | Alif | TODO | 2a |
| RV size | Alif | TODO | 2b |

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
- [x] P4a: EchoJEPA-G embeddings (319,815 studies, 1408-dim)
- [x] P4b: EchoJEPA-L embeddings (319,802 studies, 1024-dim)
- [x] P4c: EchoJEPA-L-K extraction started (17% complete, chunk-based resume available)
- [x] Label cleaning (physiological range filters, 1,582 rows dropped)
- [x] Rare disease confound audit (hard negative controls verified)
- [x] UHN linear probes: EchoJEPA-G (mean AUC 0.874, mean R² 0.625) and EchoJEPA-L (failed — embedding collapse on UHN)

### MIMIC Pipeline
- [x] 23 label CSVs (mortality, diseases, biomarkers, outcomes)
- [x] 7 model clip-level extractions (all correct)
- [x] 7 model study-level pooling + 23 task splits
- [x] Probe training: 161/161 jobs complete
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
