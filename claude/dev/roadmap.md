# Roadmap

Consolidated view of outstanding work for the Nature Medicine data pipeline. Updated 2026-03-26 18:00 UTC (**17 UHN tasks pred avg DONE** — 13 × all 5 + 4 new × 4 manuscript models. RV function in progress. Trajectory expansion: MR onset + LVEF 3-class DONE. 7/7 diseases PA DONE (4/4 models). MIMIC outcome chain running. CY sklearn = manuscript gold standard. MIMIC xfer: 4 diseases × 4 models DONE).

For code-level roadmap (extraction scripts, bug fixes), see `vjepa2/claude/dev/roadmap.md`.
For UHN label inventories and data quality notes, see `uhn-pipeline.md` (in this directory).
For individual task specs, see `../planning/tasks/`.
For prioritized run queue and full task inventory, see `vjepa2/experiments/nature_medicine/TASK_TRACKER.md`.
For comprehensive manuscript task inventory (every experiment in sn-article.tex with status), see `manuscript-tasks.md` (in this directory).

## Manuscript-Aligned Priority List (Updated 2026-03-19)

Organized by what `sn-article.tex` needs. ~30 `‡` markers (experiments not run) and ~70 `\tbd` placeholders remain.

**4 manuscript models**: EchoJEPA-G, EchoJEPA-L-K, EchoPrime, PanEcho. EchoJEPA-L is internal testing only.

### Tier 0: Current Status (2026-03-26 18:00 UTC / 2:00 PM ET)
- **GPUs 0-7**: RV function EP+Pan training (ETA ~3:30 PM ET). MIMIC outcome chain RUNNING (L-K/EP/Pan phase).
- **NEW UHN tasks DONE (4 × 4 manuscript models)**:
  - EDV: G **R²=0.774**/r=0.890, L-K 0.560/0.790, EP 0.425/0.719, Pan 0.554/0.815
  - ESV: G **R²=0.853**/r=0.931, L-K 0.721/0.882, EP 0.589/0.843, Pan 0.675/0.878
  - Diastolic function: G **AUROC=0.903**/bal_acc=74.7%, L-K 0.855/64.8%, EP 0.846/63.1%, Pan 0.830/61.2%
  - Cardiac output: G **R²=0.335**/r=0.615, L-K 0.185/0.499, EP 0.143/0.425, Pan 0.179/0.452
- **RV function in progress**: G val **0.845**, L-K val 0.746, EP+Pan training
- **Trajectory experiments:**
  - **trajectory_lvef_onset**: G **0.793** (flagship, done earlier)
  - **trajectory_mr_severity_onset**: G **0.733**, Pan 0.688, EP 0.666, L-K 0.605 — validates onset paradigm generalizes
  - **trajectory_lvef 3-class**: Pan 0.633, EP 0.628, G 0.613, L 0.536, L-K 0.532 — DONE (5 models)
  - Skipped: TAPSE (2K pairs too small), LV mass (56 positives), RV pressure (noisy)
- **MIMIC outcome chain**: G 10/11 done, L-K/EP/Pan chain running
- **sklearn reproduction**: CY results NOT reproduced (2-5pp gap). CY values = manuscript gold standard.
- **Disease pred avg — 7/7 DONE (all 4 manuscript models)**
- **MIMIC cross-institution disease xfer: 4 diseases × 4 models DONE**
- **Physiological coherence analysis now unblocked**: EDV+ESV+LVEF all done → can compute derived EF correlation
- **Next**: RV function PA (after EP+Pan). Monitor MIMIC chain. Bland-Altman analysis.

### P1: Finish UHN Probe Pipeline (Alif, compute-only — blocks §2.2-2.4 tables)

| # | Task | Effort | Status | Notes |
|---|------|--------|--------|-------|
| 1 | ~~**RV S' training + PA**~~ | **DONE** | All 5 PA: G 0.591, L-K 0.473, EP 0.353, L 0.350, Pan 0.301 | §2.3 RV mechanics |
| 2 | ~~**RV FAC training + PA**~~ | **DONE** | All 5 PA: G 0.539, L-K 0.444, L 0.325, Pan 0.301, EP 0.278 | §2.3 RV mechanics |
| 3 | ~~**Remaining pred avg**~~ | **DONE** | AR sev G 0.765, Pan 0.692. RV S' L-K/EP/Pan. All recovered. | UHN tables COMPLETE |
| 4 | **Bland-Altman analysis** | ~1 day code | TODO | All regression reporting. Wendy requirement. **Only remaining P1 item.** |

### P2: MIMIC Strategy E Probes (Alif — blocks §2.5 outcomes table, currently all `\tbd`)

| # | Task | Effort | Status | Notes |
|---|------|--------|--------|-------|
| 5 | **8 outcome tasks × 4 models** (mortality ×4, readmission, discharge, LOS, end-of-life) | ~4 days | **G 10/11 DONE** (PA complete). L-K/EP/Pan needs restart (chain killed). | Strategy E underperforms CY linear by 1-6pp. sklearn reproduction failed to close gap. |
| 6 | **Biomarkers** (troponin, NT-proBNP, creatinine, lactate) × 4 models | ~1 day | **G PA DONE**, L-K/EP/Pan partial | R² very low (0.008-0.076). All models struggle. |
| 7 | **Cross-institution** — more tasks beyond LVEF | ~hours/task | LVEF + 4 diseases DONE | §2.5 cross-site table. MR/TR severity feasible but 5→4 class mismatch needs post-hoc merge. |
| 7b | **sklearn CY reproduction** | DONE | **NOT REPRODUCED** (2-5pp gap) | Solver convergence + L1 + HP range. CY's numbers used in manuscript. |

### P3: Novel/JEPA-Unique Experiments (Alif — blocks §2.5 trajectory + §2.8 ablation ‡ markers)

| # | Task | Effort | Owner | Notes |
|---|------|--------|-------|-------|
| 8 | ~~**Forward prediction (Exp 6.1)**~~ | DONE | Alif | Implemented in `evals/forward_prediction/`. Tested on MIMIC takotsubo/STEMI/mortality. Forward pred AUROC ~0.52 (chance). See `evals/forward_prediction/RESULTS.md`. |
| 9 | ~~**Anomaly detection (Exp 6.4)**~~ | DONE | Alif | 4 approaches tested (pred error, repr distance mean/token, forward pred) on UHN + MIMIC. Best: takotsubo 0.711 zero-shot repr distance. Pred error uniformly at chance. See `evals/forward_prediction/RESULTS.md`. |
| 10 | **Trajectory expansion** | ~1 day CSV + 3 days training | Alif | **MR onset DONE** (G 0.733, Pan 0.688, EP 0.666, L-K 0.605). **LVEF 3-class PA DONE** (5 models). Remaining: MIMIC trajectory (no infrastructure). |

### P4: Supporting Experiments (blocks §2.6, §2.1 Extended Data)

| # | Task | Effort | Owner | Notes |
|---|------|--------|-------|-------|
| 11 | **Disease panel** (7 diseases × 4 models) | ~1-2 days | Alif/Reza | **7/7 training DONE** (takotsubo dropped). **Pred avg: 7/7 DONE (4/4 models)**. MIMIC xfer: 4 diseases × 4 models DONE. |
| 12 | ~~**Diastolic function, cardiac output**~~ + **RV function** | ~hours | Alif | **Diastolic fn DONE** (G 0.903). **Cardiac output DONE** (G R²=0.335). **RV function: G+L-K trained, EP+Pan training.** |

### P5: Delegated Work (blocked on other people)

| # | Task | Owner | Blocks | Notes |
|---|------|-------|--------|-------|
| 13 | **Combined echo+EHR model** | CY | §2.5 additive value claim | EHR baseline done; needs fusion |
| 14 | **Acuity conditioning (H_confound)** | CY | §2.5 confound control | Hierarchical regression, LRT |
| 15 | **Outcome baselines** (LVEF alone, LVEF+demo+Charlson, Elixhauser, EHR-only, echo ensemble) | CY | §2.5 baseline comparison | Manuscript ‡ markers on these |
| 16 | **Core lab reader recruitment** (3 min) | Ali/Wendy | Extended Data validation | Non-negotiable for IEC |
| 17 | **SAE training + frame shuffling** | Adib/Goodfire | §2.8 interpretability | Blocked on checkpoint delivery |
| 18 | **UMAP main figure** | Reza | Main Figure | t-SNE done, need UMAP + UHN |
| 19 | **Note phenotyping (rare disease label quality)** | CY | §2.6 | Strengthens disease panel |

### P6: Fairness (CY — blocks entire §2.7, currently all `\tbd`)

| # | Task | Status | Notes |
|---|------|--------|-------|
| 20 | **Email Joe for Chicago demographics** | BLOCKING | Age, sex, race/ethnicity for Table 1 + cross-site fairness |
| 21 | **LVEF MAE + mortality AUC by subgroup** | TODO | Sex, race, age, insurance |
| 22 | **ECE + equalized odds** | TODO | At 90% sensitivity threshold |
| 23 | **Fairness table (Tab 5)** | TODO | All 30 cells currently `\tbd` |

### P7: Methods/Admin (blocks Methods section `\tbd` placeholders)

| # | Task | Owner | Notes |
|---|------|-------|-------|
| 24 | **REB protocol number** | Wendy | §4.7 ethics |
| 25 | **Masking strategy details** | Alif | §4.2 — mask %, strategy, spatial/temporal extent |
| 26 | **Preprocessing params** | Alif | §4.3 — frame count, FPS, augmentation |
| 27 | **UHN demographics for Table 1** | Alif | Age, sex distribution |
| 28 | **SAE hyperparams** | Adib | §4.6 — feature count, sparsity penalty |
| 29 | **Supplementary counts** | Alif | §5 — "Tables 1-\tbd, Figures 1-\tbd" |
| 30 | **Funding statement** | PI | §5 acknowledgements |

### Tier Deferrable (revision round)
- Experiment 6.2 (autoregressive rollout) — single-step sufficient
- Experiment 6.3 (intervention trajectory divergence) — small N, underpowered
- Cross-view imputation — architecturally infeasible
- Extended fairness (intersectional, calibration equity H4.2)
- MIMIC d=1 attentive probes (23 × 5 = 115 models — large compute, may use sklearn)
- Cross-institution validation (UChicago)
- Fine-tuned ViT-L baseline (DeLong vs frozen G)
- Phase-specific masking (Exp 6.5)

### Critical Path

```
P1 DONE (all 13 UHN primary pred avg complete)
  → P4 disease panel DONE (7/7 PA complete)
    → P2 MIMIC Strategy E (G 10/11 done, L-K/EP/Pan chain needs restart)
      → sklearn gap INVESTIGATED (CY pipeline not reproduced, 2-5pp gap, solver/HP issues)
        → Decision: use CY's sklearn values for manuscript (marked †), attentive probe values are secondary
          → P3 (forward prediction + anomaly detection, ~3-4 days)
            → P5/CY (combined model + confounding, ~4 days parallel)
              → P6/CY (fairness, ~2 days, after Chicago demographics)
                → P7 (methods placeholders, ~1 day)
                  → Paper draft with all ‡ and \tbd resolved
```

---

## Recent Completed Work

### Done (March 25-26 — new UHN tasks + trajectory expansion)
- **EDV**: 4 manuscript models trained + pred avg DONE (G R²=0.774)
- **ESV**: 4 manuscript models trained + pred avg DONE (G R²=0.853)
- **Diastolic function** (4-class): 4 manuscript models trained + pred avg DONE (G AUROC=0.903)
- **Cardiac output**: 4 manuscript models trained + pred avg DONE (G R²=0.335)
- **RV function** (5-class): G+L-K trained (G val 0.845), EP+Pan training
- **Trajectory MR severity onset**: G 0.733, Pan 0.688, EP 0.666, L-K 0.605
- **Trajectory LVEF 3-class**: Pan 0.633, EP 0.628, G 0.613, L 0.536, L-K 0.532
- **Physiological coherence unblocked**: EDV+ESV+LVEF → can compute derived EF correlation

### Done (March 24 — sklearn reproduction + attentive feature extraction)
- **sklearn reproduction experiments DONE**: Three scripts created. Study-level mean-pool → sklearn: mort_1yr 0.821. Clip-level → PA: mort_1yr 0.808. Both underperform CY (0.846) by 2-5pp. Root cause: solver convergence (LBFGS max_iter=500), no L1, narrower HP grid.
- **Train-set attentive feature extraction DONE**: All 10 MIMIC tasks extracted (6.5h, 2 parallel pairs on split GPUs). NPZs saved at `evals/vitg-384/nature_medicine/mimic/video_classification_frozen/*-trainfeat-echojepa-g/clip_outputs.npz`.
- **Head mismatch bug found**: Multi-head attentive pooler means each HP head has its own cross-attention weights. Features from head N and head M are in different spaces. Cannot train sklearn on head N train features and test on head M test features (gives AUROC 0.21). Used `clip_probs_all_heads` instead.
- **Attentive probe direct PA results confirmed**: Val-selected best head gives same results as earlier pred avg runs. sklearn ensembling of 15 heads' outputs provides marginal regression improvement.
- **CY code review** (commit 549fb07): Documented full pipeline architecture.

### Done (March 22-23 — MIMIC outcome chain + CY baselines + disease completion)
- **MIMIC outcome chain launched** (11 tasks × 4 models): G phase complete (10/11 tasks trained + pred avg). in_hospital_mortality G failed (port collision), requeued. L-K phase in progress.
- **CY baseline results integrated into manuscript**: 40+ `\tbd` cells filled (outcomes table). XGBoost EHR, LVEF, Elixhauser, echo measurements for all 12 rows.
- **Port collision fix committed** (c3517d7): cleanup_orphans + wait_for_port_free + retry loop
- **Key finding**: Strategy E attentive probes underperform CY linear probes by 1-6pp on MIMIC mortality (0.791 vs 0.846 for 1yr). Checkpoint integrity verified (md5sum match, all 35 epochs completed).
- **Disease pred avg 7/7 DONE (4/4 models each)**
- **MIMIC cross-institution disease xfer: 4 diseases × 4 models DONE**: Amyloidosis G **0.947**, HCM G 0.847, DCM EP 0.721, STEMI G 0.657

### Done (March 21-22 earlier — disease probes + labels)
- **7/7 disease probes trained** (takotsubo dropped), all 4 manuscript models. Bicuspid AV Pan training completed.
- **MIMIC disease labels v4.2**: STEMI precision audit + global whitespace fix.
- **13 tasks pred avg ALL 5 models DONE** (completed Mar 21)

### Done (March 19 — manuscript update session)
- **Manuscript `sn-article.tex` major update**: Replaced stale numbers with pred avg results. Added vision-only + data-efficiency emphasis to abstract, intro, ablations, discussion. Added AV mean grad (0.579), E/e' (0.558*), RV S' (0.491*) results. Removed EchoJEPA-L from manuscript (internal only). Added ‡ markers on ~30 pending experiments. Replaced ~72 TBD placeholders with `\tbd` command (renders as red bold ??). Three-tier marker convention: unmarked = pred avg final, * = single-clip val, ‡ = experiment not yet run.
- **ICML scale comparison attribution fixed**: Ablation section now properly credits ICML for G-vs-L scale comparison, extends analysis to clinical tasks.
- **Priority restructure**: Documented all 30 manuscript-pending experiments organized by section (P1-P7).

### Done (March 18-20 — compute)
- **7 tasks pred avg ALL 5 DONE**: LVEF (G 0.778), TAPSE (G 0.633), RVSP (G 0.504), AV mean grad (G 0.579), AS severity (G 0.932), MR severity (G 0.882), TR severity (G 0.854), trajectory onset (G 0.793)
- **AV Vmax pred avg partial**: G 0.679 (+9.7pp), L 0.242. L-K running.
- **RV FAC G trained**: R²=0.472, Pearson=0.691. L training.
- **All 11 tasks × 5 models TRAINED**: LVEF, TAPSE, RVSP, TR sev, AV mean grad, AV Vmax, AR sev, AS sev, MR sev, E/e' medial, trajectory onset
- **MIMIC cross-institution LVEF ALL 5**: G 0.591, L-K 0.485, EP 0.452, L 0.375, Pan 0.269
- **Bugs 008-014 found and fixed**
- **RV S' training started**: G 0.491, L 0.234 done. L-K at ep 10/15.

### Done (March 14-17)
- **TEE/Stress filtering applied** to all 52 UHN probe CSV dirs (7,341 study UIDs excluded, 0-3.2% contamination)
- **MR severity complete** (5 models): G **0.860**, L-K 0.803, EchoPrime 0.770, L 0.771, PanEcho 0.724
- **AS severity complete** (5 models): G **0.908**, EchoPrime 0.827, L-K 0.821, L 0.786, PanEcho 0.762
- **AV Vmax complete** (5 models): G **R²=0.582**, EchoPrime 0.476, PanEcho 0.390, L-K 0.388, L 0.232
- **TR severity complete** (5 models): G **0.838**, L-K 0.787, EchoPrime 0.758, L 0.755, PanEcho 0.731
- **Trajectory LVEF onset complete** (5 models): G **0.793**, EchoPrime 0.776, PanEcho 0.759, L-K 0.677, L 0.514
- **Literature review completed**: B-mode hemodynamics prior art, foundation model competitive landscape, three-way supervision taxonomy
- **GLS demoted** to Extended Data (not cross-modal, PanEcho already does it, commercial speckle tracking exists)
- **Prediction averaging implemented** in eval.py (auto-enables when `val_only=True` + `study_sampling=True`)
- **Per-epoch checkpoint archiving** added to prevent future checkpoint loss
- **LVEF retrain chain** complete (5/5 models retrained, pred avg in progress)
- **AR severity launched** on GPUs 0-3 (completed all 5 models, see above)

### Done (March 11-13 — prior week)
- Strategy E verified (d=1 attentive probes help all 4 tested models)
- Z-score refactor, physiological range filtering, NCCL all_gather fix
- TAPSE + LVEF probe training (original, checkpoints partially lost)
- EchoMAE dropped, run scripts built, UMAP methodology decided
- CY: MIMIC outcome probes + EHR baselines (sklearn, NPZ pipeline)
- Adib: Attention map infra + SAE training started
- Reza: MIMIC t-SNE complete (23 tasks x 7 models)

---

## Blocking — Must complete before probe results

### Data Pipeline (Strategy E — d=1 attentive probes from video)

The evaluation runs from video — no NPZ extraction or mean pooling needed. What's required:

| Task | Status | Blocks |
|------|--------|--------|
| Build training CSVs (all clips/study, `s3_path label`) | **DONE** (47 tasks + 5 trajectory) | All probe training |
| Build view-filtered CSVs (`train_vf.csv`, `val_vf.csv`) | **DONE** (41 tasks) | View-specific probe training |
| Build test CSVs (all clips/study, `s3_path label`) | **DONE** (47 tasks + 5 trajectory) | Study-level prediction aggregation |
| Run scripts (Phase 1 orchestration) | **DONE** | Phase 1 probes |
| Implement study-level prediction aggregation in eval.py | **DONE** | Study-level metrics |
| Add B-mode-only view filters for remaining hemodynamic tasks | **DONE** (all hemodynamic tasks; diastolic_function + pa_pressure Tier 3 still TODO) | Pillar 1 (structure predicts flow) |
| Implement Bland-Altman analysis | **TODO** | All regression reporting |
| Model checkpoints available for all 5 models | **DONE** | Encoder forward pass |
| Patient splits (UHN + MIMIC) | **DONE** | Train/test separation |
| Label files (53 UHN + 23 MIMIC tasks) | **DONE** | Training labels |
| View-filtered CSV build script | **DONE** | View-specific tasks |

**View-filtered training (decided 2026-03-11):** View-specific tasks use pre-filtered CSVs containing only task-relevant views. DistributedStudySampler picks 1 clip/study/epoch from filtered set. Global tasks (mortality, biomarkers) use unfiltered CSVs. See `decisions/decision_log.md` (Decisions 03 and 04).

**Files:**
- Build script: `vjepa2/experiments/nature_medicine/uhn/build_viewfiltered_csvs.py`
- Filtered CSVs: `vjepa2/experiments/nature_medicine/uhn/probe_csvs/{task}/train_vf.csv`, `val_vf.csv`, `test_vf.csv`
- Run scripts: `vjepa2/scripts/run_uhn_probe.sh` (generic, auto-detects task type/views/sampling) + `vjepa2/scripts/run_phase1.sh` (Phase 1 orchestrator: 18 tasks by group)
- Trajectory CSVs: `vjepa2/experiments/nature_medicine/uhn/build_trajectory_csvs.py`

## DECIDED — Evaluation protocol verified (2026-03-11)

**Strategy E adopted: Uniform depth=1 attentive probes + prediction averaging for all models.**

Verification experiment (UHN 13-class view classification, 22K clips) confirmed d=1 attentive probes help ALL 4 tested models:
- EchoJEPA-G: 87.3% (+1.2pp over linear 86.1%)
- EchoJEPA-L: 84.3% (+17.3pp over linear 67.0%)
- EchoPrime: 54.6% (+9.3pp over linear 45.3%)
- PanEcho: 46.0% (+7.1pp over linear 38.9%)

EchoPrime d=4 confirmed degenerate (38.0% < linear). ICML inversion was artifact.

See `probe-results.md` for full epoch tables. See `../decisions/evaluation_protocol_decision.md` and `../decisions/probe_implementation_analysis.md` for the analysis.

## DECIDED — 4 manuscript models (2026-03-19)

**Manuscript models (4):** EchoJEPA-G (~1,012M), EchoJEPA-L-K (~304M), EchoPrime (~34M), PanEcho (~43M).

EchoJEPA-L is trained internally for ablation reference but NOT included in the manuscript — ICML already covers the G-vs-L scale comparison. EchoMAE dropped (undertrained checkpoint, lr=3.5e-6 ~170x too low). JEPA-vs-MAE comparison deferred to ICML preprint.

## Probe Training Status (2026-03-22)

### Strategy E Results (d=1 attentive probes from video)

All results: d=1 attentive probes, 15 epochs, 12-head HP grid. All values are prediction-averaged test-set metrics (study-level). Full per-model tables in `probe-results.md`.

**ALL 13 Primary Tasks — Pred Avg Complete (all 5 models)**

| Task | Type | G | L-K | EchoPrime | PanEcho | L (internal) |
|------|------|---|-----|-----------|---------|---|
| **LVEF** | R² | **0.778** | 0.702 | 0.681 | 0.665 | 0.577 |
| **TAPSE** | R² | **0.633** | 0.555 | 0.430 | 0.385 | 0.430 |
| **RVSP** | R² | **0.504** | 0.317 | 0.169 | 0.274 | 0.168 |
| **AV Vmax** | R² | **0.679** | 0.492 | 0.574 | 0.479 | 0.242 |
| **AV mean grad** | R² | **0.579** | 0.328 | 0.462 | 0.378 | 0.147 |
| **E/e' medial** | R² | **0.598** | 0.491 | 0.422 | 0.454 | 0.370 |
| **RV S'** | R² | **0.591** | 0.473 | 0.353 | 0.301 | 0.350 |
| **RV FAC** | R² | **0.539** | 0.444 | 0.278 | 0.301 | 0.325 |
| **AS severity** | AUROC | **0.932** | 0.868 | 0.868 | 0.813 | 0.846 |
| **MR severity** | AUROC | **0.882** | 0.837 | 0.818 | 0.789 | 0.808 |
| **TR severity** | AUROC | **0.854** | 0.817 | 0.780 | 0.778 | 0.787 |
| **AR severity** | AUROC | **0.765** | 0.680 | 0.701 | 0.692 | 0.670 |
| **Trajectory onset** | AUROC | **0.793** | 0.677 | 0.776 | 0.759 | 0.514 |

**Key findings:**
- G leads by +8-20pp on hemodynamics — structure-flow physics from scale
- L-K (4,579 patients) beats EP (109K) and Pan (24K) on LVEF, TAPSE, RVSP, TR severity despite 24-60× less data
- G-vs-EchoPrime gap narrows to +1.7pp on trajectory — text supervision efficient for prognostic features
- ALL models extract hemodynamic signal from B-mode — story is "scale advantage," not "unique capability"
- **Prediction averaging boosts all tasks**: LVEF +6.7pp, AV Vmax +9.7pp, TAPSE +7.3pp, RVSP +5.0pp, AS sev +2.4pp, MR sev +2.2pp (G values)
- **Cross-institution (UHN→MIMIC)**: G degrades gracefully (LVEF R² 0.778→0.591), Pan degrades most (0.665→0.269)
- **Vision-only emphasis**: EchoJEPA uses no labels/reports/measurements during pretraining, yet matches or exceeds supervised baselines
- **Bugs 008-014 all fixed**

## MVP — Required for submission

| Task | Spec | Owner | Status | Section |
|------|------|-------|--------|---------|
| Training/test CSVs (UHN + MIMIC) | — | Alif | **DONE** (47 UHN + 23 MIMIC + 5 trajectory) | All probe training |
| View-filtered CSVs (per-task) | — | Alif | **DONE** (41 UHN tasks) | View-specific probe training |
| Phase 1 run scripts | — | Alif | **DONE** | Probe orchestration |
| Study-level prediction aggregation | — | Alif | **DONE** (auto-enables at inference; Bugs 008-010 fixed) | Study-level metrics |
| Run scripts (training + pred avg, Bug 008-010 fixes) | — | Alif | **DONE** (`run_uhn_probe.sh` + `run_pred_avg.sh`) | Batch execution |
| Bland-Altman analysis | — | Alif | **TODO** | All regression reporting |
| B-mode-only view filters | — | Alif | **DONE** (all hemodynamic tasks filtered; diastolic_function + pa_pressure CSVs still need `bmode_only=True` build for Tier 3) | Pillar 1 hemodynamics |
| UHN label cleaning | `mvp_uhn_label_cleaning.md` | Alif | **DONE** | All UHN regression |
| UHN rare disease NPZs | `mvp_uhn_rare_disease_npzs.md` | Alif | **DONE** | Sec 2.2 |
| UHN missing labels | `mvp_uhn_missing_labels.md` | Alif | **DONE** | Sec 2.1 |
| Rare disease confound audit | `mvp_rare_disease_confound_audit.md` | Alif/CY | **DONE** | Sec 2.2 |
| UHN trajectory pairs | `mvp_uhn_trajectory_pairs.md` | Alif/CY | **DONE** | Sec 2.3 |
| Charlson/Elixhauser | `mvp_charlson_elixhauser.md` | CY | **DONE** | Sec 2.3 |
| EHR-only baseline | `mvp_ehr_only_baseline.md` | CY | **DONE** (XGBoost + TabPFN, 54 features) | Sec 2.3 |
| MIMIC outcome probes (sklearn, EchoJEPA-G) | — | CY | **DONE** — H3.1 passed. Ensemble AUC: mortality 0.846-0.912 | Sec 2.3 |
| Combined model (echo + EHR) | — | CY | **TODO** — key claim for additive prognostic value | Sec 2.3 |
| Acuity conditioning (H_confound) | — | CY | **TODO** — hierarchical regression, likelihood ratio test | Sec 2.3 |
| Fairness demographics | `mvp_fairness_demographics.md` | CY | **DONE** | Sec 2.4 |
| Acuity covariates | `mvp_acuity_covariates.md` | CY | **DONE** | Sec 2.3 |
| MIMIC 7-model extraction + splits | — | Alif | **DONE** | All MIMIC experiments |
| d=1 attentive probe training (MIMIC: 11 outcomes × 4) | — | Alif | **G 10/11 PA DONE**. Chain running L-K/EP/Pan. | Sec 2.3-2.4 |
| d=1 attentive probe training (UHN: 47×5) | — | Alif | **13 tasks × 5 models ALL PRED AVG DONE.** 7/7 diseases × 4 models PA DONE. 3 non-disease tasks TODO. | Sec 2.1, Extended Data |
| Core lab reader recruitment (3 minimum) | — | Ali/Wendy | TODO | Sec 2.1, Extended Data |
| Rare disease label verification | — | Ali/Wendy | TODO | Sec 2.2 |
| ~~**Forward prediction + anomaly detection (Exp 6.1/6.4)**~~ | `evals/forward_prediction/RESULTS.md` | Alif | **DONE** — 4 approaches × UHN + MIMIC. Best zero-shot: takotsubo 0.711 repr distance. Predictor-based scoring at chance. | Sec 2.3 |
| Chicago demographics from Joe | — | Alif | **TODO** (blocking fairness) | Sec 2.4 |

**MVP progress (updated 2026-03-24 01:00):** Infrastructure complete (CSVs, scripts, pred avg pipeline, Bugs 007-014 fixed). **13 UHN tasks × 5 models ALL PRED AVG DONE.** **7/7 diseases PA DONE (4/4 models).** **MIMIC outcomes: G 10/11 PA DONE**, L-K/EP/Pan chain needs restart. CY baselines integrated (40+ cells). **sklearn investigation DONE** — CY pipeline not reproduced (2-5pp gap, solver/HP), CY values are manuscript gold standard. **Remaining**: restart MIMIC chain for L-K/EP/Pan, in_hospital_mortality G requeue, readmission_30d G pred avg re-run, JEPA-unique experiments (P3), 3 non-disease UHN tasks, fairness (P6), methods placeholders (P7).

Spec files at `../planning/tasks/`. Each is self-contained with pipeline steps, output paths, and dependencies.

## Dependency Order (Strategy E pipeline)

```
Blocking:
  1. Build training CSVs (all clips/study per task)     — DONE (47 + 5 trajectory)
  2. Build view-filtered CSVs (per task)                — DONE (41 tasks)
  3. Implement study-level prediction aggregation       — DONE (auto-enables when val_only=True + study_sampling=True)
  4. Run d=1 attentive probes (UHN Phase 1: 18×5, Phase 2: 4×5 + MIMIC 14×7) — IN PROGRESS (8 tasks in various stages, ~40/335 runs)

Data prep (DONE):
  - Label files (53 UHN + 23 MIMIC)                    — DONE
  - Patient splits (UHN + MIMIC)                        — DONE
  - Fairness demographics, acuity, Elixhauser, EHR-only — DONE
  - Trajectory pairs (93K UHN patients)                 — DONE

Strong (after primary probes):
  - strong_troponin_48h, strong_note_phenotyping, strong_ef_baseline_matrix
  - strong_trajectory_pairs, strong_intervention_cohorts
```

## Strong Additions

Now integrated into the P1-P7 priority structure above. See priority tiers for current status and manuscript section mapping. Key remaining strong items:

| Task | Owner | Status | Priority |
|------|-------|--------|----------|
| Forward prediction (Exp 6.1) | Alif | **DONE** | P3 #8 |
| Anomaly detection (Exp 6.4) | Alif | **DONE** — 40 experiments (28 single-model + 12 multi-model). Best zero-shot: takotsubo 0.711. Multi-model comparison: VideoMAE-L leads (0.871). | P3 #9 |
| Disease panel (7 diseases × 4 models) | Alif/Reza | **7/7 PA DONE (4/4 models)** | P4 #11 |
| MIMIC Strategy E outcomes (11 × 4) | Alif | **G 10/11 PA DONE**, chain running L-K/EP/Pan | P2 #5 |
| Combined echo+EHR model | CY | TODO | P5 #13 |
| SAE Phase A | Adib/Goodfire | IN PROGRESS (GPU issues) | P5 #17 |
| Frame shuffling | Goodfire | TODO (blocked) | P5 #17 |
| UMAP main figure | Reza | t-SNE done | P5 #18 |
| Attention map figures | Adib | Infra DONE | P5 |
| Note phenotyping | CY | TODO | P5 #19 |
| EF baseline matrix | CY | TODO | P5 #15 |

## Deferrable

See `../planning/tasks/deferrable.md` for the full list (cross-institution validation, extended fairness analysis, SAE training on UHN, MIMIC trajectory).

## UHN Label Gap Analysis (2026-03-06)

Comparison of paper draft (sn-article.tex) vs built label NPZs:

### Built (53 NPZs)
- 20 regression: lvef, edv, esv, tapse, rvsp, lv_mass, la_vol, ao_root, mv_ea, mv_ee, mv_ee_medial, mv_dt, lvot_vti, aov_vmax, aov_mean_grad, aov_area, rv_fac, rv_sp, ivsd, cardiac_output, gls
- 17 classification: lv_cavity_size, lv_systolic_function, mr_severity, tr_severity, ar_severity, as_severity, pa_pressure, la_size, ra_size, rv_size, diastolic_function, pericardial_effusion, lv_hypertrophy, rv_function, pr_severity, cardiac_rhythm, rwma
- 9 rare disease: disease_hcm, disease_amyloidosis, disease_takotsubo, disease_stemi, disease_endocarditis, disease_dcm, disease_bicuspid_av, disease_myxomatous_mv, disease_rheumatic_mv (confound audit DONE)
- 6 trajectory: trajectory_multi, trajectory_lvef, trajectory_tapse, trajectory_lv_mass, trajectory_rv_sp, trajectory_mr_severity
- 1 RWMA binary: rwma (69,452 studies)

### Resolved Discrepancies
- E/E' ratio: paper says 53,786 (medial), old NPZ (mv_ee) had 19,898 (lateral). New mv_ee_medial has 37,778 studies using correct medial measurements. Gap to 53K is from echo.db studies without S3 mapping.
- RWMA: paper says 73,394, built 69,452 after S3 mapping. Sources: Syngo LV_fx_regional_wma_obs + WMA comments + HeartLab findings + LV_Fx_qualitative normal-inferred absent.
- Cardiac rhythm: paper says 97,332, built 44,672 (Syngo only). Additional ~50K from HeartLab not yet integrated.

## What's Done

### Data Infrastructure
- [x] UHN UID → StudyRef mapping (`aws_syngo.csv`, 320K studies)
- [x] UHN clip index (`uhn_clip_index.npz`, 18.1M clips)
- [x] UHN patient splits (138K patients, 70/10/20 temporal)
- [x] UHN label NPZs (53 tasks)
- [x] UHN label cleaning (physiological range filters)
- [x] UHN rare disease confound audit
- [x] UHN all-clip probe CSVs (47 tasks) — `build_probe_csvs.py`
- [x] UHN view-filtered probe CSVs (41 tasks) — `build_viewfiltered_csvs.py`
- [x] UHN trajectory probe CSVs (5 tasks) — `build_trajectory_csvs.py`
- [x] MIMIC 23 label CSVs + label NPZs
- [x] MIMIC probe CSVs (23 tasks, Z-scored regression) — `build_probe_csvs.py`
- [x] MIMIC shared infrastructure (clip_index, patient_split, labels/)
- [x] MIMIC covariate CSVs (demographics, acuity, Charlson/Elixhauser, EHR features)
- [x] MIMIC downstream pipeline (study-level + splits for all 7 models × 23 tasks)
- [x] Phase 1 run scripts — `scripts/run_uhn_probe.sh` + `scripts/run_phase1.sh`

### Probe Results
- [x] MIMIC pilot probes (5 tasks × 7 models, superseded by v2)
- [x] MIMIC v2 full probes (23 tasks × 7 models = 161/161) — see `probe-results.md` for full tables
- [x] MIMIC outcome probes (EchoJEPA-G, sklearn) — H3.1 passed, ensemble AUC 0.846-0.912 for mortality
- [x] MIMIC EHR-only baseline (XGBoost + TabPFN, 54 features) — mortality AUC 0.856-0.959
- [x] UHN view classification probes (2 models) — see `probe-results.md`

### Interpretability & Visualization
- [x] Attention map infrastructure (Adib) — hooks for ViT-G + EchoPrime, per-layer/per-head extraction
- [x] Head specialization finding — different heads attend to different anatomical regions
- [x] Temporal consistency finding — JEPA attention smooth across frames vs EchoPrime/VideoMAE jitter
- [x] MIMIC t-SNE visualizations (Reza) — 23 tasks x 7 models, validates embedding structure
- [ ] SAE training (Adib) — started on MIMIC study-level, GPU interrupted, restarting
- [ ] UMAP regeneration (Reza) — t-SNE done, need UMAP for publication + UHN main figure

### Extraction + Bug Fixes
- [x] UHN EchoJEPA-G extraction + shuffle fix
- [x] UHN EchoJEPA-L extraction (post-fix)
- [x] MIMIC 7 models extracted + shuffle fixed (PanEcho/EchoPrime/EchoFM re-extracted 2026-03-08)
- [x] All 10 code bugs identified and fixed (see `vjepa2/claude/dev/bugs/`):
  - Bugs 001-006: shuffle ordering, encoder normalization, EchoFM padding, video load tracking, drop_last forwarding, NCCL all_gather
  - Bug 007: checkpoint loss prevention (per-epoch archiving to local + S3)
  - Bug 008: inference probe loading (`resume_checkpoint: true` required in YAML)
  - Bug 009: /dev/shm exhaustion (orphan cleanup, reduced workers/batch size)
  - Bug 010: concurrent job safety (ppid=1 filtered orphan cleanup in all scripts)
- [x] UHN per-task split pipeline (`evals/regenerate_uhn_downstream.py`) — 53 tasks × 2 models (G, L)

## Output Destinations

All MIMIC CSVs at `data_exploration/mimic/csv/`. Embeddings at `experiments/nature_medicine/{uhn,mimic}/`. UHN labels at `experiments/nature_medicine/uhn/labels/`. Each task spec documents exact output paths.
