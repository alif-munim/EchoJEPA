# Nature Medicine Manuscript Reference

Detailed context for the Nature Medicine manuscript lives in:
- `uhn_echo/nature_medicine/CLAUDE.md` — manuscript structure, writing principles, ICML vs Nature Medicine scope, model comparison table, evaluation protocol, author list

## Core Thesis

Clinical echocardiography reduces a rich spatiotemporal recording to a handful of standardised measurements (primarily ejection fraction). Self-supervised representations from EchoJEPA encode substantially more clinical information than this measurement paradigm captures, enabling capabilities not previously demonstrated from frozen echocardiographic representations: rare disease detection, outcome prediction, biomarker inference, trajectory forecasting, and interpretable feature decomposition.

## Relationship to ICML Preprint

Two companion papers with distinct contributions:
- **ICML** — Establishes the *method*: JEPA training objective, multi-view attentive probing, controlled comparisons (JEPA vs MAE vs contrastive), robustness, sample efficiency, pediatric transfer. Uses attentive probes. Datasets: Toronto/Chicago (internal), EchoNet-Dynamic, EchoNet-Pediatric.
- **Nature Medicine** — Establishes *clinical capabilities beyond standard measurements*: RV mechanics, hemodynamics (structure predicts flow), trajectory forecasting, rare disease detection, outcome/biomarker prediction, SAE interpretability, fairness analysis. Uses frozen d=1 attentive probes from video with prediction averaging. Datasets: UHN (pretraining + benchmarks), MIMIC-IV-Echo linked to MIMIC-IV clinical data.

No primary result overlap between papers. Standard benchmarks are compressed to one paragraph in Nature Medicine and defer to ICML citation.

## Paper Results Sections (organized by Wendy's three pillars)

Core novelty (three pillars):
1. **RV mechanics** — learned functional organization (TAPSE, S', FAC, RV function grade, RV size)
2. **Hemodynamics** — structure predicts flow (MR/AS/TR severity from B-mode only, AV gradients)
3. **Trajectory forecasting** — predicting future measurements from current echo (93K pairs)

Supporting sections:
4. Standard benchmarks (brief, defers to ICML)
5. Disease detection (HCM, amyloidosis — supporting evidence)
6. Clinical outcome and biomarker prediction (MIMIC — CY sklearn ensemble results are manuscript gold standard; EHR baselines: LVEF, echo meas., Elixhauser, XGBoost. Headline: 90d mortality video 0.883 > XGBoost 0.868)
7. Fairness / equitable subgroup performance
8. SAE interpretability (Strong Addition — basic now, full in revision)

## Models

**UHN probes (5 models):** EchoJEPA-G (ViT-G, 18M UHN, JEPA), EchoJEPA-L (ViT-L, UHN, JEPA), EchoMAE (ViT-L, VideoMAE on echo), EchoPrime (MViT-v2-S, CLIP on echo+text), PanEcho (ConvNeXt-T, multi-task supervised).

**MIMIC probes (7 models):** Above 5 + EchoJEPA-L-K (Kinetics-pretrained JEPA), EchoFM (MAE+triplet on echo).

## Evaluation Protocol (Strategy E, adopted 2026-03-11)

All downstream tasks use **frozen d=1 attentive probes** trained from video through frozen encoders (no fine-tuning, no NPZ extraction). Training: `DistributedStudySampler` selects 1 random clip per study per epoch from view-filtered CSVs. Evaluation: probe predicts independently on all clips per study; predictions averaged for study-level result (prediction averaging). View-specific tasks (TAPSE, LVEF, hemodynamics) use pre-filtered CSVs containing only task-relevant views. The old NPZ-based mean-pool + sklearn pipeline is fully superseded.

**Key differences from ICML preprint protocol:** d=1 (vs d=4), 12-head HP grid (vs 6-head), 2-epoch warmup (vs none), BS2 (vs BS1), 15 epochs (vs 20), view-filtered training, study-level sampling, prediction averaging. Resolution (224px) and temporal sampling (16f/step2/2seg) unchanged. See `claude/architecture/evaluation-protocols.md` for the full comparison table and config archaeology.
