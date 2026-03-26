# claude/ Directory

Reference documentation for the EchoJEPA project, organized by topic. These files provide context for Claude Code sessions and serve as persistent knowledge across conversations.

## architecture/

Technical documentation for the codebase internals.

| File | Contents |
|------|----------|
| `pretraining-and-cooldown.md` | Two-phase training (pretrain vs cooldown), LR schedules, masking strategies, kinetics vs echo config differences, resume/force-load behavior, MIMIC config analysis, V-JEPA 2.1 config reference |
| `vjepa2-paper-recipes.md` | Official V-JEPA 2 and 2.1 training recipes extracted from the papers: hyperparameters, progressive resolution, data, masking, scaling rules, ablation results |
| `vjepa21-code-diff.md` | V-JEPA 2.1 vs 2.0 code differences, EchoJEPA ports, checkpoint compatibility (distilled vs full, shape mismatches, ema_encoder). MIMIC configs for ViT-B and ViT-L. Operational notes from first training run: 7 issues fixed, verified perf (3.4s/iter on 8×A100), monitoring commands |
| `probe-system.md` | Frozen probe evaluation: attentive/linear/MLP heads, classification vs regression, view-filtered training pipeline, DistributedStudySampler, multi-view fusion, hyperparameter grid search, inference mode, prediction output |
| `classifier-pipeline.md` | ConvNeXt/Swin classifier pipeline: training, cooldown, 18M inference, data prep stages, label mappings, experiment history |
| `forward-prediction.md` | Zero-shot anomaly detection & forward prediction experiments exploiting the JEPA predictor network. Results across 4 approaches (prediction error, repr distance mean-pooled/token-level, forward prediction) on UHN (hard negatives) and MIMIC (population negatives). Key finding: takotsubo 0.711 AUROC zero-shot; predictor-based scoring uniformly at chance. Full results in `evals/forward_prediction/RESULTS.md` |

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
| `mimic-video-statistics.md` | MIMIC echo video statistics (n=50 sample): 30fps native, median 74 frames/2.5s, 42% stills, frame coverage at different FPC/fps settings, cooldown feasibility analysis |

## preprint/

Analysis of the ICML preprint's experimental methodology — probe fairness, encoder comparison confounds, and claim validity. Full preprint LaTeX source at `user-default-efs/vjepa2/claude/preprint/icml_preprint.tex`.

| File | Contents |
|------|----------|
| `icml_preprint.tex` | Full ICML preprint LaTeX source (in `user-default-efs/vjepa2/claude/preprint/`) |
| `encoder-fairness.md` | Encoder output comparison (5 models): token structure, embed dims, parameter counts. Four fairness confounds: dimensionality, token asymmetry, model scale, pretraining data. Controlled comparison identification |
| `probe-architecture-analysis.md` | Attentive vs linear probe inversion finding, root cause (token starvation), task-specific behavior (why LVEF works but view classification doesn't), UMAP interpretation, rebuttal strategy |
| `claim-validity.md` | Which preprint conclusions are bulletproof (controlled JEPA vs MAE), valid but overstated (system-level gaps), or confounded (clustering pattern). Implications for Nature Medicine |
| `experiment-issues.md` | Issues discovered during ICML experiments: batch size scaling failure, d=4 probe inversion root cause, normalization bugs, shuffle bug, pretraining loss divergence, video decode failures. What survived to Nature Medicine |
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
| `hyperpod-ops.md` | SageMaker HyperPod cluster operations: cluster creation checklist, SSM connectivity (special target format), conda-pack deployment, lifecycle scripts, Slurm job submission, S3 bucket layout, 8 troubleshooting issues with fixes. Covers echojepa-h100-march setup (2026-03-26) |
| `bugs/001-shuffle-bug.md` | **CRITICAL**: DistributedSampler shuffle=True corrupts embedding-CSV alignment. Root cause, impact, post-hoc fix scripts, verification |
| `bugs/002-normalization-bugs.md` | **HIGH**: PanEcho double norm, EchoPrime/EchoFM missing de-norm. 3 MIMIC models need re-extraction |
| `bugs/003-echofm-padding.md` | Moderate: Last-frame repetition → linspace interleave. Fixed |
| `bugs/004-video-load-substitution.md` | **HIGH**: Failed video loads return random replacement, silent index misalignment. Fixed (tracking added) |
| `bugs/005-drop-last-not-forwarded.md` | **MEDIUM**: `drop_last` param ignored by DataLoader. Up to 248 clips silently dropped. Fixed |
| `bugs/006-labels-trainval-mode.md` | **MEDIUM**: `--labels` with `--train`/`--val` applies wrong indices. Latent, not triggered by documented workflows |

## rebuttals/

ICML rebuttal — actual reviews received 2026-03-25 (scores 2/3/3/4). **`08-rebuttal-v2.md` is the active rebuttal plan.** Pre-review docs (`01`-`07`) anticipated different concerns than reviewers raised. See `rebuttals/README.md` for the full index and status of each file.

| File | Contents | Status |
|------|----------|--------|
| **`08-rebuttal-v2.md`** | **Active rebuttal plan based on actual reviews.** Reviewer-by-reviewer analysis, per-concern responses with pre-written text, experiment priority table, EchoJEPA-G gated release strategy, path to acceptance | **PRIMARY** |
| `01-paper-audit.md` | TIER 1-4 issue inventory with anticipated attacks, defense evidence, and response templates | Reference |
| `02-rebuttal-template.md` | Pre-review rebuttal text. Leads with controlled comparison, covers compute-matched concerns, baseline fairness | Superseded by 08 |
| `03-worst-case-scenarios.md` | Scenarios 1-7: broken VideoMAE, missing EchoCardMAE, model size, frozen probing, large gap, probe unfairness | Superseded by 08 |
| `04-competitive-positioning.md` | EchoJEPA vs EchoCardMAE / Echo-Vision-FM / EFNet: objective, protocol, scale, multi-view, clinical breadth | Reference |
| `05-probe-fairness.md` | Attentive probe inversion: root cause (normalization bugs + d=4 degeneration, NOT token starvation), d=1 verification results, Strategy E justification | Reference |
| `06-claim-validity.md` | Bulletproof vs confounded claims — which hills to die on, which to concede | Superseded by 08 |
| `07-camera-ready-actions.md` | Final assessment + 13 prioritized action items for camera-ready | Reference |
| `claude-rebuttal-master.md` | Original unstructured source document (preserved) | Archive |
| `review-simulation-prompt.md` | Self-contained prompt for simulating ICML review panel in Claude web app | Tool |

## goodfire/

Goodfire interpretability analysis of EchoJEPA representation geometry, attribution, and sparse autoencoder concept discovery. Internal technical report (v1.0, 2026-03-19). **Check permissions before citing figures directly.**

| File | Contents | Rebuttal use |
|------|----------|-------------|
| `goodfire_mar20.pdf` | 31-page analysis: representation geometry (activation distributions, intrinsic dimensionality, singular value spectra, trajectory straightness, frame shuffling, temporal Fourier power), attribution analysis (saliency, SmoothGrad, ForGrad, RISE), SAE concept extraction (BatchTopK, clinical correlations, reconstruction-concept tradeoff) | Frame shuffling (Fig 25-26) and temporal Fourier power (Fig 29) for ICML rebuttal. SAE concepts (rho=0.50 LVEF) and attribution atlases reserved for NatMed. |
