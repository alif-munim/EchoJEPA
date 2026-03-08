# Roadmap

Consolidated view of outstanding work across UHN and MIMIC pipelines. Updated 2026-03-07.

Detailed specs live in `uhn_echo/nature_medicine/data_exploration/todo/` (MIMIC) and `data_exploration/ongoing/uhn_pipeline_status.md` (UHN). This file is the cross-cutting summary.

## Blocking — Must complete before any probe results

### UHN Embedding Extractions (GPU, ~10h each)

| Model | Checkpoint | Status | Notes |
|-------|-----------|--------|-------|
| EchoJEPA-G | `pt-280-an81.pt` | **DONE** | 319,815 studies, 1408-dim |
| EchoJEPA-L | `vitl-pt-210-an25.pt` | **DONE** | 319,802 studies, 1024-dim |
| EchoJEPA-L-K | `vitl-kinetics-pt220-an55.pt` | **DONE** | In progress or complete |
| EchoMAE-L | VideoMAE checkpoint | TODO | Controlled JEPA-vs-MAE comparison |
| Random Init | Untrained ViT-L | TODO | Baseline for all tables |

EchoJEPA-G/L embeddings are ready — UHN probes can run now for those two models.

PanEcho, EchoPrime, EchoFM UHN extractions are not planned yet (lower priority — these models are primarily compared on MIMIC).

### MIMIC Re-extractions (GPU, ~1h each)

3 models need re-extraction due to normalization bugs (bug 002):

| Model | Bug | Status |
|-------|-----|--------|
| PanEcho | Double ImageNet norm | **IN PROGRESS** — running via `scripts/reextract_mimic_3models.sh` |
| EchoPrime | Missing de-norm → [0,255] | **QUEUED** — runs after PanEcho |
| EchoFM | Missing de-norm → [0,1] + padding fix | **QUEUED** — runs after EchoPrime |

After re-extraction: re-run `pool_embeddings.py` and `remap_embeddings.py` to rebuild study-level NPZs and splits.

4 MIMIC models are correct: EchoJEPA-G, EchoJEPA-L, EchoJEPA-L-K, EchoMAE. Shuffle was fixed post-hoc (verified 100% label match). Downstream pipeline regenerated 2026-03-07.

### MIMIC Downstream Pipeline Re-run — DONE

Regenerated 2026-03-07 via `evals/regenerate_mimic_downstream.py`:
- All 7 models × 23 tasks: study-level NPZs + train/val/test splits
- Label NPZs in `labels/` were already correct (store CSV-order indices, which now match fixed master NPZs)
- 4 correct models (echojepa_g/l/l_kinetics, echomae) are fully ready for probing
- 3 norm-bugged models (panecho, echoprime, echofm) have correct ordering but wrong embedding values

## MVP — Required for submission

| Task | Owner | Status | Blocks |
|------|-------|--------|--------|
| UHN EchoMAE-L extraction | Alif | TODO | JEPA-vs-MAE comparison (every table) |
| UHN Random Init extraction | Alif | TODO | Baseline column (every table) |
| MIMIC PanEcho/EchoPrime/EchoFM re-extraction | Alif | **IN PROGRESS** | 9-model comparison tables |
| MIMIC downstream pipeline re-run | Alif | **DONE** | All MIMIC probe results |
| UHN probe training (53 tasks × 5 models) | Alif | TODO | Section 2.1 benchmarks, Extended Data |
| MIMIC probe training (23 tasks × 9 models) | Team | TODO | Sections 2.3-2.4 |

## Strong Additions — Significantly elevates paper

| Task | Owner | Effort | Status | Section |
|------|-------|--------|--------|---------|
| Note phenotyping (rare disease label quality) | CY | ~2-3 days | TODO | 2.2 |
| Troponin 48h biomarker labels | CY | ~0.5 day | TODO | 2.3 |
| Trajectory pair analysis | CY/Adib | ~1-2 days | TODO | 2.3 |
| Intervention cohorts (forward prediction) | CY | ~1 day | TODO | 2.3 |
| EF baseline matrix (echo measurement ensemble) | CY | ~2 days | TODO | 2.3 |

## Deferrable — Can wait for revision

See `data_exploration/todo/deferrable.md` for the full list (10 items including cross-institution validation, SAE training data, extended fairness analysis).

## What's Done

### UHN Pipeline
- [x] P0: UID → StudyRef mapping (`aws_syngo.csv`, 320K studies)
- [x] P1: Clip index (`uhn_clip_index.npz`, 18.1M clips)
- [x] P2: Patient splits (`patient_split.json`, 138K patients, 70/10/20 temporal)
- [x] P3: Label NPZs (53 tasks: 20 regression + 17 classification + 9 rare disease + 6 trajectory + 1 RWMA)
- [x] P4a: EchoJEPA-G embeddings (319,815 studies, 1408-dim)
- [x] P4b: EchoJEPA-L embeddings (319,802 studies, 1024-dim)
- [x] Label cleaning (physiological range filters, 1,582 rows dropped)
- [x] Rare disease confound audit (hard negative controls verified)

### MIMIC Pipeline
- [x] 23 label CSVs (mortality, diseases, biomarkers, outcomes)
- [x] 7 model clip-level extractions (shuffle-fixed)
- [x] Shared infrastructure (clip_index, patient_split, labels/)
- [x] Charlson/Elixhauser comorbidity scores (7,243 studies)
- [x] Demographics/fairness CSV (sex, race, age, insurance)
- [x] Acuity covariates (DRG severity, ICU flags, triage)
- [x] EHR features (54-feature baseline matrix)

### Bug Fixes
- [x] Shuffle ordering (bug 001) — code fix + post-hoc reordering
- [x] Encoder normalization (bug 002) — code fix applied, re-extraction pending
- [x] EchoFM temporal padding (bug 003) — code fix applied
- [x] Video load substitution tracking (bug 004) — logging added
- [x] drop_last forwarding (bug 005) — code fix applied
