# Roadmap

Consolidated view of outstanding work across UHN and MIMIC pipelines. Updated 2026-03-08.

Detailed specs live in `uhn_echo/nature_medicine/context_files/planning/tasks/`. Pipeline status at `uhn_echo/nature_medicine/context_files/dev/`. This file is the cross-cutting summary.

## MIMIC Embeddings — COMPLETE

All 7 models extracted, study-level pooled, and split into train/val/test for 23 tasks. Ready for probing.

| Model | Clips | Dim | Status |
|-------|-------|-----|--------|
| EchoJEPA-G | 525,320 | 1408 | Ready (shuffle-fixed post-hoc) |
| EchoJEPA-L | 525,320 | 1024 | Ready (shuffle-fixed post-hoc) |
| EchoJEPA-L-K | 525,320 | 1024 | Ready (shuffle-fixed post-hoc) |
| EchoMAE-L | 525,320 | 1024 | Ready (shuffle-fixed post-hoc) |
| PanEcho | 525,320 | 768 | Ready (re-extracted 2026-03-08) |
| EchoPrime | 525,320 | 512 | Ready (re-extracted 2026-03-08) |
| EchoFM | 525,320 | 1024 | Ready (re-extracted 2026-03-08) |

## UHN Embeddings — In Progress

### Extraction Status

| Model | Checkpoint | Clip Extraction | Study-Level | Notes |
|-------|-----------|-----------------|-------------|-------|
| EchoJEPA-G | `pt-280-an81.pt` | **DONE** (95GB, 18.1M clips) | **DONE** (319,815 × 1408) | |
| EchoJEPA-L | `vitl-pt-210-an25.pt` | **DONE** (70GB, 18.1M clips) | **DONE** (319,802 × 1024) | |
| EchoJEPA-L-K | `vitl-kinetics-pt220-an55.pt` | **~17% done** (crashed/interrupted) | Missing | Chunk-based, supports resume. ~8-9h remaining |
| EchoMAE-L | `videomae-ep163.pt` | **IN PROGRESS** (~19h est.) | — | Controlled JEPA-vs-MAE comparison. MVP |
| Random Init | Untrained ViT-L | **TODO** | — | Baseline for all tables. MVP |

PanEcho, EchoPrime, EchoFM UHN extractions are not planned (these models are primarily compared on MIMIC).

### UHN Per-Task Split Pipeline — DONE

Built `evals/regenerate_uhn_downstream.py`. Joins study-level embeddings with label NPZs on study_ids, splits by label-embedded split assignments. Handles both standard (single-study) and trajectory (paired-study) formats.

Generated splits for G and L (53 tasks each):
- `echojepa_g_splits/{task}/train.npz`, `val.npz`, `test.npz`
- `echojepa_l_splits/{task}/train.npz`, `val.npz`, `test.npz`

Re-run for L-K, EchoMAE, Random Init when their study-level embeddings are available.

## Blocking Work — Priority Order

1. **Resume EchoJEPA-L-K UHN extraction** (~8-9h GPU)
   - Script supports auto-resume: just re-run same command
   - Then pool to study-level

2. ~~**Build UHN per-task split pipeline**~~ — **DONE**
   - G and L splits generated (53 tasks each). Probing unblocked.
   - Re-run `evals/regenerate_uhn_downstream.py` for L-K/EchoMAE/Random Init when ready

3. **EchoMAE-L UHN extraction** (~19h GPU) — **IN PROGRESS**
   - Config: `configs/inference/vitl/extract_uhn_echomae.yaml`
   - 8×A100, bs=64, w=12. Log: `logs/echomae_uhn_extraction.log`
   - Then pool to study-level (automatic)

4. **Random Init UHN extraction** (~10-12h GPU)
   - Need config for untrained ViT-L
   - Baseline column (every table)

## MVP — Required for submission

| Task | Owner | Status | Blocks |
|------|-------|--------|--------|
| UHN L-K extraction (resume) | Alif | **~17% done** | L-K in all UHN tables |
| UHN per-task split pipeline | Alif | **DONE** | All UHN probing |
| UHN EchoMAE-L extraction | Alif | **IN PROGRESS** (~19h) | JEPA-vs-MAE comparison (every table) |
| UHN Random Init extraction | Alif | TODO | Baseline column (every table) |
| MIMIC PanEcho/EchoPrime/EchoFM re-extraction | Alif | **DONE** | 7-model comparison tables |
| MIMIC downstream pipeline re-run | Alif | **DONE** | All MIMIC probe results |
| UHN probe training (53 tasks × 5 models) | Alif | TODO | Section 2.1 benchmarks, Extended Data |
| MIMIC probe training (23 tasks × 7 models) | Team | TODO | Sections 2.3-2.4 |

## Strong Additions — Significantly elevates paper

| Task | Owner | Effort | Status | Section |
|------|-------|--------|--------|---------|
| Note phenotyping (rare disease label quality) | CY | ~2-3 days | TODO | 2.2 |
| Troponin 48h biomarker labels | CY | ~0.5 day | TODO | 2.3 |
| Trajectory pair analysis | CY/Adib | ~1-2 days | TODO | 2.3 |
| Intervention cohorts (forward prediction) | CY | ~1 day | TODO | 2.3 |
| EF baseline matrix (echo measurement ensemble) | CY | ~2 days | TODO | 2.3 |

## Deferrable — Can wait for revision

See `uhn_echo/nature_medicine/context_files/planning/tasks/deferrable.md` for the full list (10 items including cross-institution validation, SAE training data, extended fairness analysis).

## What's Done

### UHN Pipeline
- [x] P0: UID → StudyRef mapping (`aws_syngo.csv`, 320K studies)
- [x] P1: Clip index (`uhn_clip_index.npz`, 18.1M clips)
- [x] P2: Patient splits (`patient_split.json`, 138K patients, 70/10/20 temporal)
- [x] P3: Label NPZs (53 tasks: 20 regression + 17 classification + 9 rare disease + 6 trajectory + 1 RWMA)
- [x] P4a: EchoJEPA-G embeddings (319,815 studies, 1408-dim)
- [x] P4b: EchoJEPA-L embeddings (319,802 studies, 1024-dim)
- [x] P4c: EchoJEPA-L-K extraction started (17% complete, chunk-based resume available)
- [x] Label cleaning (physiological range filters, 1,582 rows dropped)
- [x] Rare disease confound audit (hard negative controls verified)

### MIMIC Pipeline
- [x] 23 label CSVs (mortality, diseases, biomarkers, outcomes)
- [x] 7 model clip-level extractions (all correct)
- [x] 7 model study-level pooling + 23 task splits
- [x] Shared infrastructure (clip_index, patient_split, labels/)
- [x] Charlson/Elixhauser comorbidity scores (7,243 studies)
- [x] Demographics/fairness CSV (sex, race, age, insurance)
- [x] Acuity covariates (DRG severity, ICU flags, triage)
- [x] EHR features (54-feature baseline matrix)

### Bug Fixes
- [x] Shuffle ordering (bug 001) — code fix + post-hoc reordering
- [x] Encoder normalization (bug 002) — code fix + re-extraction complete
- [x] EchoFM temporal padding (bug 003) — code fix applied
- [x] Video load substitution tracking (bug 004) — logging added (no threading.Lock)
- [x] drop_last forwarding (bug 005) — code fix applied
- [x] PanEcho hubconf.py local tasks.pkl cache — prevents GitHub rate-limiting
- [x] EchoFM simplejson dependency — installed
