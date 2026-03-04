# Nature Medicine Manuscript Reference

Detailed context for the Nature Medicine manuscript lives in:
- `uhn_echo/nature_medicine/CLAUDE.md` — manuscript structure, writing principles, ICML vs Nature Medicine scope, model comparison table, evaluation protocol, author list

## Core Thesis

Clinical echocardiography reduces a rich spatiotemporal recording to a handful of standardised measurements (primarily ejection fraction). Self-supervised representations from EchoJEPA encode substantially more clinical information than this measurement paradigm captures, enabling capabilities not previously demonstrated from frozen echocardiographic representations: rare disease detection, outcome prediction, biomarker inference, trajectory forecasting, and interpretable feature decomposition.

## Relationship to ICML Preprint

Two companion papers with distinct contributions:
- **ICML** — Establishes the *method*: JEPA training objective, multi-view attentive probing, controlled comparisons (JEPA vs MAE vs contrastive), robustness, sample efficiency, pediatric transfer. Uses attentive probes. Datasets: Toronto/Chicago (internal), EchoNet-Dynamic, EchoNet-Pediatric.
- **Nature Medicine** — Establishes *clinical capabilities beyond standard measurements*: rare disease detection, outcome/biomarker prediction, latent forward prediction, SAE interpretability, fairness analysis. Uses frozen linear probes and mean-pooled study-level embeddings. Datasets: UHN (pretraining), MIMIC-IV-Echo linked to MIMIC-IV clinical data.

No primary result overlap between papers. Standard benchmarks are compressed to one paragraph in Nature Medicine and defer to ICML citation.

## Paper Results Sections (all novel except 2.1/2.7)

1. Physiological consistency (brief benchmarks, defers to Extended Data + ICML)
2. Rare disease detection
3. Clinical outcome and biomarker prediction (includes Elixhauser/triage comparisons)
4. Fairness / equitable subgroup performance
5. Latent forward prediction
6. SAE interpretability
7. Ablation summary (brief, citing ICML)

## Models (9 total)

EchoJEPA-G (ViT-G, 18M UHN, JEPA), EchoJEPA-L (ViT-L, MIMIC, JEPA), EchoMAE-L (ViT-L, MIMIC, MAE), EchoJEPA-L-K (Kinetics→MIMIC, JEPA), Echo-Vision-FM, PanEcho, EchoPrime, EchoFM, Random Init.

## Evaluation Protocol

All downstream tasks use **frozen linear probes** (no fine-tuning). Study-level predictions use mean-pooled per-clip embeddings across all views. View-specific tasks use clips from the relevant view only.
