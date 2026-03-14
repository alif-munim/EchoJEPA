# claude/ Directory

Reference documentation for the EchoJEPA project, organized by topic. These files provide context for Claude Code sessions and serve as persistent knowledge across conversations.

## architecture/

Technical documentation for the codebase internals.

| File | Contents |
|------|----------|
| `pretraining-and-cooldown.md` | Two-phase training (pretrain vs cooldown), LR schedules, masking strategies, kinetics vs echo config differences, resume/force-load behavior |
| `probe-system.md` | Frozen probe evaluation: attentive/linear/MLP heads, classification vs regression, view-filtered training pipeline, DistributedStudySampler, multi-view fusion, hyperparameter grid search, inference mode, prediction output |
| `classifier-pipeline.md` | ConvNeXt/Swin classifier pipeline: training, cooldown, 18M inference, data prep stages, label mappings, experiment history |

## data/

Datasets, database schemas, and manuscript scope.

| File | Contents |
|------|----------|
| `data-directory.md` | `data/` directory layout: CSV splits (153 files), scalers, labels, parquet exports, notebooks, scripts |
| `embedding-pipeline.md` | **(SUPERSEDED — historic reference)** Old NPZ-based multi-model embedding pipeline: extraction, label remapping, study-level pooling, patient-level splits. Replaced by Strategy E (d=1 attentive probes from video) |
| `nature-medicine-manuscript.md` | Manuscript scope, ICML vs Nature Medicine delineation, models, evaluation protocol (Strategy E), Wendy's three pillars |
| `uhn-database.md` | UHN echocardiography database (echo.db, Syngo/HeartLab schemas, rare disease cohorts) |
| `uhn-mapping.md` | DICOM UID → Syngo StudyRef mapping chain: deid key files, date extraction patterns, coverage analysis, patient ID systems, HeartLab bridge, action items |
| `mimic-database.md` | MIMIC-IV linked to echo (prediction targets, biomarker coverage, data engineering notes) |

## preprint/

Analysis of the ICML preprint's experimental methodology — probe fairness, encoder comparison confounds, and claim validity.

| File | Contents |
|------|----------|
| `encoder-fairness.md` | Encoder output comparison (5 models): token structure, embed dims, parameter counts. Four fairness confounds: dimensionality, token asymmetry, model scale, pretraining data. Controlled comparison identification |
| `probe-architecture-analysis.md` | Attentive vs linear probe inversion finding, root cause (token starvation), task-specific behavior (why LVEF works but view classification doesn't), UMAP interpretation, rebuttal strategy |
| `claim-validity.md` | Which preprint conclusions are bulletproof (controlled JEPA vs MAE), valid but overstated (system-level gaps), or confounded (clustering pattern). Implications for Nature Medicine |
| `hindsight-recommendations.md` | 9 recommendations for the camera-ready, ranked by impact/effort. Top 3: linear probes in main tables, PCA-512 baseline, comparison taxonomy table |
| `claude-chat-probes.md` | Raw conversation export (source material for the distilled docs above) |

## dev/

Development log: bug tracker, changelog, operational guides, and code review findings. Single source of truth for what's broken, fixed, and planned.

| File | Contents |
|------|----------|
| `README.md` | Bug tracker index (6 issues with severity/status), planned fixes with priority, file index |
| `roadmap.md` | Consolidated outstanding work: Phase 1/2/3 execution plan, MVP tasks, run scripts, strong additions, completion checklist |
| `changelog.md` | Chronological record of code changes, bug fixes, extraction runs, config changes |
| `code-review.md` | Full-repo review: 5 encoder adapters, extraction scripts, pooling, remapping, probe training, eval scaffold. Per-component verdict table |
| `ops.md` | UHN 18M extraction operational guide: launch commands, DataLoader tuning (prefetch_factor, batch_size, num_workers), S3 bottleneck, crash recovery, timing reference, failure modes |
| `bugs/001-shuffle-bug.md` | **CRITICAL**: DistributedSampler shuffle=True corrupts embedding-CSV alignment. Root cause, impact, post-hoc fix scripts, verification |
| `bugs/002-normalization-bugs.md` | **HIGH**: PanEcho double norm, EchoPrime/EchoFM missing de-norm. 3 MIMIC models need re-extraction |
| `bugs/003-echofm-padding.md` | Moderate: Last-frame repetition → linspace interleave. Fixed |
| `bugs/004-video-load-substitution.md` | **HIGH**: Failed video loads return random replacement, silent index misalignment. Fixed (tracking added) |
| `bugs/005-drop-last-not-forwarded.md` | **MEDIUM**: `drop_last` param ignored by DataLoader. Up to 248 clips silently dropped. Fixed |
| `bugs/006-labels-trainval-mode.md` | **MEDIUM**: `--labels` with `--train`/`--val` applies wrong indices. Latent, not triggered by documented workflows |

## rebuttals/

ICML rebuttal preparation — vulnerability inventory, response templates, worst-case scenarios. See `rebuttals/README.md` for the full index.

| File | Contents |
|------|----------|
| `01-paper-audit.md` | TIER 1-4 issue inventory with anticipated attacks, defense evidence, and response templates. Includes H5-H7 (probe mismatch, embedding dimensionality, pretraining data confounds) |
| `02-rebuttal-template.md` | Ready-to-submit reviewer response text. Leads with controlled comparison, covers compute-matched concerns, baseline fairness, task-specific patterns |
| `03-worst-case-scenarios.md` | Scenarios 1-7: broken VideoMAE, missing EchoCardMAE, model size, frozen probing, large gap, probe unfairness, inflated comparisons |
| `04-competitive-positioning.md` | EchoJEPA vs EchoCardMAE / Echo-Vision-FM / EFNet: objective, protocol, scale, multi-view, clinical breadth |
| `05-probe-fairness.md` | Attentive probe inversion analysis: root cause, three-pillar defense, updated clustering narrative, response template |
| `06-claim-validity.md` | Bulletproof vs confounded claims — which hills to die on, which to concede |
| `07-camera-ready-actions.md` | Final assessment + 13 prioritized action items for camera-ready |
| `claude-rebuttal-master.md` | Original unstructured source document (preserved) |
