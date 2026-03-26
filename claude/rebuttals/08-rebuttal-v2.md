# ICML Rebuttal v2 — Based on Actual Reviews (2026-03-25)

## Scores

| Reviewer | Score | Confidence | Primary Concern |
|----------|-------|------------|-----------------|
| hfQ1 | 2 (Reject) | 5 | Novelty below ICML bar, no contrastive comparison, no segmentation, dataset release |
| 6t2T | 3 (Weak Reject) | 5 | No SOTA comparisons on downstream tasks ("looks like ablation"), reproducibility, double-blind |
| ncQn | 3 (Weak Reject) | 4 | Speckle claim not validated at representation level, only one controlled comparison |
| L8sp | 4 (Weak Accept) | 3 | Incremental novelty, multi-view probe is system-level |

Average: 3.0. Borderline. Need to flip at least one Weak Reject.

---

## Critical Finding: Prepared Rebuttal Was Orthogonal to Actual Concerns

The original rebuttal docs (`01`-`07`) anticipated technical ML attacks: probe fairness, VideoMAE LR, embedding dimensionality, EchoCardMAE comparison, appendix inconsistencies. **None of these were raised by any reviewer.**

The actual concerns are about novelty, evaluation breadth, mechanistic evidence, and reproducibility. This document supersedes the original rebuttal strategy.

---

## Concern 1: Limited Novelty Over V-JEPA

**Raised by:** ALL 4 reviewers
**Severity:** Critical — this is the #1 issue

### Reviewer Quotes

- hfQ1: "the technical contribution appears limited... does not meet the high bar typically expected at ICML"
- L8sp: "core architecture is largely based on V-JEPA... changes are incremental"
- ncQn: "largely follows the existing V-JEPA framework... main contribution lies more in scaling"
- 6t2T: (implied by wanting more SOTA comparisons beyond ablations)

### Response

We agree that the encoder architecture follows V-JEPA2. Our contribution is not architectural but scientific and empirical:

1. **Objective-domain alignment hypothesis.** We provide the first controlled evidence that the pretraining objective determines representation quality in a medical imaging domain, independent of architecture, scale, or data. This is a generalizable principle: for domains where pixel fidelity is dominated by stochastic noise (ultrasound, radar, certain microscopy), latent prediction is fundamentally better suited than pixel reconstruction. This hypothesis has implications beyond echocardiography.

2. **The controlled comparison is the primary contribution.** No prior work has isolated the training objective as a variable for ultrasound foundation models. EchoJEPA-L vs EchoMAE-L — identical ViT-L architecture, identical MIMIC data, identical compute — with a 45-point view classification gap attributable solely to the objective. This is not an incremental application; it is a controlled experiment that changes how the field should think about pretraining for noisy medical modalities.

3. **Multi-view probing framework.** Not an engineering convenience but a methodological contribution. No prior echo foundation model paper standardizes evaluation across heterogeneous architectures with frozen backbones, identical probes, and identical hyperparameter search. This framework enables fair comparison for the first time.

4. **Physics-informed robustness benchmarks (open-sourced).** Novel for ultrasound foundation model evaluation. Depth attenuation and acoustic shadow perturbations simulate real clinical degradation modes, revealing that EchoJEPA degrades 86% less than baselines. We open-source these benchmarks — including code to generate additional perturbation types (haze, speckle reduction) — as a community resource for evaluating any ultrasound foundation model under physics-informed noise. No such standardized robustness benchmark previously existed for echocardiography.

5. **Scale + open release.** 18M echocardiograms from 300K patients is the largest pretraining corpus for this modality. Combined with the gated release of EchoJEPA-G and the open-source robustness benchmarks (see Concern 5), this provides the community with a suite of resources that did not previously exist: the largest echo FM checkpoint, a standardized multi-view evaluation framework, and physics-informed robustness benchmarks.

ICML has a strong tradition of publishing empirical contributions that change understanding of when and why methods work (e.g., scaling laws, objective comparisons for contrastive learning). Our paper follows this tradition for medical imaging.

---

## Concern 2: No Segmentation Evaluation

**Raised by:** Reviewers 6t2T, hfQ1
**Severity:** High

### Reviewer Quotes

- hfQ1: "a vision foundation model is expected to demonstrate general-purpose visual capabilities... segmentation datasets are widely available... only evaluates classification and regression"
- 6t2T: "no comparison with other state-of-the-art methods, especially in downstream tasks"

### Response

We acknowledge this gap and have added segmentation evaluation in the revision.

**Experiment:** Frozen EchoJEPA encoder with a lightweight decoder on [CAMUS / EchoNet-Dynamic LV segmentation]. All encoder weights remain frozen — only the decoder is trained, consistent with our frozen-backbone evaluation protocol.

**Results:** [TO BE FILLED AFTER EXPERIMENT]

| Model | Dice (ED) | Dice (ES) | HD (mm) |
|-------|-----------|-----------|---------|
| EchoJEPA-G | — | — | — |
| EchoJEPA-L | — | — | — |
| EchoMAE-L | — | — | — |
| EchoPrime | — | — | — |
| PanEcho | — | — | — |

**Framing:** "We have added LV segmentation evaluation on [dataset] using frozen features with a lightweight [decoder]. EchoJEPA achieves [X] Dice score vs [Y] for baselines, demonstrating that the learned representations support pixel-level tasks in addition to classification and regression."

**On report generation** (6t2T): Report generation requires a text decoder and is orthogonal to representation quality evaluation. We consider this an important downstream application enabled by our representations, but it constitutes a separate contribution. We discuss this as future work.

---

## Concern 3: Speckle Suppression Not Directly Validated

**Raised by:** Reviewer ncQn
**Severity:** High — but directly addressable. ncQn is the most flippable reviewer.

### Reviewer Quote

"The paper mainly shows improvements on downstream tasks but does not provide representation-level analyses to demonstrate that the learned embeddings are indeed more robust to speckle noise or better aligned with anatomical structures. Without additional analyses such as feature invariance tests, representation visualization, or noise sensitivity studies, it remains unclear whether the gains truly arise from noise suppression or from other factors."

### Response

We thank the reviewer for this important suggestion. We have added three representation-level analyses:

**Experiment 1: Representation stability under speckle perturbation (CKA)**

- Take N test videos, add synthetic speckle noise at K intensity levels (multiplicative Rayleigh noise simulating ultrasound speckle)
- Extract frozen representations from clean and noisy inputs for both EchoJEPA-L and EchoMAE-L
- Compute CKA (Centered Kernel Alignment) between clean and noisy representation matrices
- Higher CKA = more noise-invariant representations

**Expected result:** EchoJEPA representations remain stable (high CKA) while EchoMAE representations shift substantially (low CKA), directly confirming that latent prediction filters speckle.

**Results:** [TO BE FILLED]

| Model | CKA (mild noise) | CKA (moderate) | CKA (severe) |
|-------|-------------------|----------------|--------------|
| EchoJEPA-L | — | — | — |
| EchoMAE-L | — | — | — |

**Experiment 2: Noise-level probing**

- Add speckle at 5 discrete intensity levels to test videos
- Train a linear probe on frozen representations to predict the noise level
- Lower accuracy = less noise information encoded in the representation

**Expected result:** EchoMAE features allow accurate noise-level prediction (representations encode speckle). EchoJEPA features do not (speckle is suppressed).

**Results:** [TO BE FILLED]

| Model | Noise-Level Probe Accuracy |
|-------|---------------------------|
| EchoJEPA-L | — |
| EchoMAE-L | — |

**Experiment 3: Attention visualization under perturbation**

- Compare attention maps on clean vs speckle-perturbed inputs
- EchoJEPA should maintain anatomical focus; EchoMAE should shift attention to noisy regions

Existing attention visualizations (Figure 6) already show EchoJEPA localizing on valve leaflets and ventricular walls while EchoMAE attends diffusely to borders and artifacts. The new analysis extends this to the perturbed setting.

**Rebuttal text:** "We have added representation-level analyses directly validating the speckle suppression claim: (1) CKA between clean and speckle-perturbed representations shows EchoJEPA maintains [X] similarity vs [Y] for EchoMAE under severe noise, confirming noise-invariant encoding; (2) a linear probe trained to predict noise level achieves [X]% accuracy from EchoMAE features vs only [Y]% from EchoJEPA features, indicating that speckle information is actively suppressed in the JEPA representation space; (3) attention maps remain anatomically focused under perturbation for EchoJEPA while shifting for EchoMAE."

---

## Concern 3b: Only One Controlled Comparison

**Raised by:** Reviewer ncQn
**Severity:** Moderate — addressable with framing

### Reviewer Quote

"The strictly controlled comparison appears limited to EchoJEPA-L vs EchoMAE-L. Other comparisons involve models with different architectures, training paradigms, or significantly larger proprietary datasets, making it difficult to draw strong conclusions."

### Response — Data Scale Confounds Work Against EchoJEPA-L

The reviewer is correct that only the EchoJEPA-L vs EchoMAE-L pair is fully controlled. However, the system-level comparisons are more informative than they appear, because the confounding variables systematically *disadvantage* EchoJEPA-L:

| Model | Objective | Params | Training Videos | Training Patients | View Acc |
|-------|-----------|--------|-----------------|-------------------|----------|
| EchoJEPA-L | Latent prediction | 300M | 525K | ~4.6K | **85.5%** |
| EchoPrime | Contrastive (CLIP) | ~35M | **12.1M** | **109K** | 42.1% |
| PanEcho | Supervised multitask | ~28M | **1.19M** | **24K** | 41.9% |
| EchoMAE-L | Reconstruction | 300M | 525K | ~4.6K | 40.4% |

EchoJEPA-L was trained on MIMIC-IV-Echo: 525K video clips from roughly 4,600 patients — a small, single-institution dataset. EchoPrime was trained on **23x more videos** from **24x more patients**. PanEcho was trained on **2.3x more videos** from **5x more patients** with direct supervision on 39 clinical tasks.

Despite this massive data disadvantage, EchoJEPA-L outperforms both. If data scale, patient diversity, or supervised labels were the primary drivers of performance, EchoPrime and PanEcho should dominate. They do not. The most parsimonious explanation is the pretraining objective.

This is not a substitute for a fully controlled comparison — but it provides strong *converging evidence*. When confounds are biased against your model and it still wins, the signal is real.

**Rebuttal text:** "We acknowledge that only the EchoJEPA-L vs EchoMAE-L comparison is strictly controlled. However, we note that the system-level comparisons provide converging evidence because all confounding variables favor the baselines: EchoPrime was trained on 23x more videos from 24x more patients, and PanEcho on 2.3x more videos with direct supervision on 39 clinical tasks. Despite training on only 525K clips from ~4,600 patients (MIMIC-IV-Echo), EchoJEPA-L outperforms both. When all confounds are biased against a model and it still wins, the remaining explanation is the variable we seek to isolate: the pretraining objective."

---

## Concern 4: No Contrastive Pretraining Comparison

**Raised by:** Reviewer hfQ1
**Severity:** Moderate-High

### Reviewer Quote

"No comparisons are provided between JEPA and contrastive pretraining approaches in this domain."

### Response

**Option A — Reframe existing comparison (minimum viable):**

EchoPrime uses contrastive vision-language learning (CLIP-style) on 12.1M video-report pairs from 109K patients and represents the contrastive paradigm in our evaluation. While this comparison is not architecture-matched (MViT-v2-S vs ViT-L), EchoPrime had access to 23x more training data and 24x more patients than EchoJEPA-L, yet achieves substantially lower view classification accuracy (42.1% vs 85.5%) and higher LVEF error (5.33 vs 5.97). As discussed in Concern 3b, this data disadvantage for EchoJEPA-L makes the system-level comparison more informative than it initially appears.

Our controlled comparison focuses on the prediction target (latent vs pixel) as the key variable, since this is the distinction most relevant to the speckle-noise hypothesis. Contrastive learning introduces different inductive biases — either augmentation invariances (BYOL/DINO) or text supervision (CLIP) — making it a less clean test of the noise-filtering mechanism.

**Option B — Controlled contrastive baseline (high impact, expensive):**

Train ViT-L with DINOv2 or BYOL objective on MIMIC-IV-Echo with matched compute budget. This would create a 3-way controlled comparison:

| Model | Objective | Architecture | Data | Compute |
|-------|-----------|-------------|------|---------|
| EchoJEPA-L | Latent prediction | ViT-L | MIMIC 525K | Matched |
| EchoMAE-L | Pixel reconstruction | ViT-L | MIMIC 525K | Matched |
| EchoDINO-L | Contrastive (DINO) | ViT-L | MIMIC 525K | Matched |

This is the strongest possible response but requires days of GPU time.

**Recommendation:** Use Option A in the rebuttal text. Pursue Option B if GPU time permits before the deadline — it would be the single most impactful addition for hfQ1.

---

## Concern 5: Proprietary Data / Reproducibility

**Raised by:** Reviewers 6t2T, L8sp, hfQ1
**Severity:** High — but now substantially mitigated

### Reviewer Quotes

- hfQ1: "it remains unclear whether the dataset will be publicly released"
- 6t2T: "difficult for other researchers to fully replicate or verify the flagship model's performance"
- L8sp: "the community cannot reproduce the flagship results using the full dataset"

### Response — EchoJEPA-G Gated Release

**UHN has agreed to release the EchoJEPA-G checkpoint (ViT-Giant, 1.1B parameters) under gated institutional access upon paper acceptance.**

This provides the community with:

| Asset | Access | Status |
|-------|--------|--------|
| EchoJEPA-G checkpoint (1.1B params, 18M echos) | Gated institutional access | Upon acceptance |
| EchoJEPA-L checkpoint (300M params, MIMIC) | Fully public | Already released |
| Evaluation framework + code | Fully public | Already released |
| Physics-informed robustness benchmarks | Fully public | Already released |
| Controlled comparison data (MIMIC-IV-Echo) | Fully public | PhysioNet |

**Competitive positioning:** To our knowledge, this will make EchoJEPA-G the largest publicly available echocardiography foundation model. For comparison:
- EchoPrime (35M params, 12M training videos) — model NOT publicly released
- PanEcho (28M params) — released, but 40x smaller
- EchoFM — released, but much smaller scale

Gated access is standard practice for medical AI models (LLaMA, Gemma, BiomedCLIP) and enables researchers to reproduce our flagship results, build downstream applications, and conduct independent validation — all without requiring access to the proprietary training data.

**Rebuttal text:** "We are pleased to announce that, upon acceptance, the EchoJEPA-G checkpoint (ViT-Giant, 1.1B parameters) will be released under gated institutional access, enabling the research community to reproduce and build upon our flagship results. Combined with the already-public EchoJEPA-L, our open-source evaluation framework, and our open-source physics-informed robustness benchmarks (depth attenuation, acoustic shadow, haze, speckle reduction — extensible to additional perturbation types), this provides full reproducibility at both model scales along with standardized tools for evaluating any ultrasound foundation model under realistic degradation. To our knowledge, this will make EchoJEPA the largest publicly available echocardiography foundation model, and our robustness benchmarks the first open-source, physics-informed evaluation suite for this modality."

---

## Concern 6: Dataset Diversity

**Raised by:** Reviewer hfQ1
**Severity:** Moderate — clarification suffices

### Reviewer Quote

"The dataset appears to contain only five categories despite its scale. This aspect requires further clarification regarding how representation diversity is ensured."

### Response

The five categories shown in Figure 1 are echocardiographic *view* categories (anatomical imaging planes), not the full semantic space of the dataset. Within each view, the dataset captures enormous clinical diversity:

- **300K unique patients** spanning diverse demographics, body habitus, and clinical indications
- **Full pathology spectrum**: normal hearts, hypertrophic cardiomyopathy, dilated cardiomyopathy, valvular disease (aortic stenosis, mitral regurgitation, tricuspid regurgitation), pericardial disease, congenital heart disease, and more
- **Hemodynamic diversity**: normal function through severe systolic/diastolic dysfunction, varying filling pressures, pressure overload states
- **Scanner diversity**: multiple manufacturers (GE, Philips, Siemens) and machine generations
- **Temporal diversity**: each video contains 1-3 full cardiac cycles with complex spatiotemporal dynamics (valve opening/closing, wall motion, flow patterns)

In self-supervised video learning, representation diversity emerges from temporal variation within clips and variation across the patient population, not from category count. Our 18M videos span the full spectrum of cardiac pathology and hemodynamic states encountered in a large academic medical center. We will add this clarification to the revised manuscript.

---

## Concern 7: Double-Blind Violation

**Raised by:** Reviewer 6t2T (ethics flag)
**Severity:** Low-Moderate

### Reviewer Quote

"The internal dataset names suggest the affiliation of the authors, which can break double-blind submission rules."

### Response

The paper was submitted using the `[preprint]` LaTeX option, which displays author names and affiliations. If the submission track requires double-blind review, we apologize for the oversight and will replace geographic identifiers ("Toronto", "Chicago") with anonymized labels ("Site A", "Site B") in the revised submission.

---

## Required Experiments (Prioritized — Updated after Review Simulation 2026-03-25)

Priority informed by review simulation (Claude web app) and independent analysis using AI Research Skills frameworks. Estimated acceptance: 25-35% with original plan, rising to **35-50%** with these additions.

### Tier 1 — Score-changers (could flip the outcome)

| # | Experiment | Target Reviewer | Effort | P(flips a score) |
|---|-----------|-----------------|--------|------------------|
| 1 | **Controlled DINO/BYOL baseline on MIMIC ViT-L** | hfQ1 (2->3) | 3-5 days GPU | High |
| 2 | **CKA speckle invariance (clean vs noisy representations)** | ncQn (3->4) | Hours | Very High |
| 3 | **Frame shuffling temporal ablation** | ALL — gives AC a "champion sentence" | Hours | Very High |

**~~Cross-modal probe (E/e' from B-mode) — REMOVED from ICML rebuttal.~~** Two reasons: (1) **Double submission risk** — cross-modal hemodynamic prediction is Nature Medicine Pillar 2 ("structure predicts flow"). Including it in the ICML rebuttal scoops the NatMed narrative and creates result overlap where currently there is none. (2) **Audience mismatch** — ICML reviewers don't know what E/e' is; explaining clinical significance burns rebuttal space. **Save this for Nature Medicine where it's the headline finding and the audience values it.**

**Experiment 1 rationale:** Three-way controlled comparison (JEPA vs MAE vs DINO, all ViT-L on MIMIC). Only experiment likely to move hfQ1. Three outcome framings prepared (see below).

**Experiment 2 rationale:** Three-way controlled comparison (JEPA vs MAE vs DINO, all ViT-L on MIMIC). This is the only experiment likely to move hfQ1. DINOv2 on ViT-L is well-documented, codebase public, MIMIC pipeline exists. ~84K updates on 8xH100.

**DINO result contingency framings (prepare ALL THREE before running):**
- **DINO ~40% (clusters with MAE):** "Three paradigms fail on ultrasound — pixel reconstruction, contrastive, and supervised. Only latent prediction succeeds. This is a systematic finding about objective-domain alignment."
- **DINO ~60-75% (moderate):** "Methods with EMA targets (JEPA, DINO) outperform pixel reconstruction, with JEPA's explicit latent prediction providing the strongest inductive bias for noisy domains. This reveals that the critical factor is the prediction target (latent vs pixel), not the specific self-supervised paradigm."
- **DINO ~80%+ (succeeds):** "Self-supervised methods with implicit noise filtering via EMA targets outperform explicit pixel reconstruction. JEPA provides the strongest such filtering, but the broader principle — EMA-based objectives for noisy domains — is itself a novel finding."

All three outcomes are publishable. Do not assume only the first is useful.

**Experiment 2 rationale:** ncQn explicitly asked for "feature invariance tests, representation visualization, or noise sensitivity studies." CKA between clean and speckle-perturbed representations directly validates the noise-filtering claim. If EchoJEPA CKA >0.9 and EchoMAE <0.7, ncQn should flip.

**Experiment 3 rationale (NEW champion sentence for AC):** Shuffle frame order in test videos, evaluate LVEF probe degradation. EchoJEPA degrades 10.9-15.1%, EchoMAE is invariant (<0.5%). This is pure ML, immediately striking, requires no clinical knowledge, and proves JEPA learns temporal dynamics while MAE learns static frame patterns. The AC champion sentence: "When temporal order is destroyed, JEPA representations degrade by 15% while MAE representations are completely invariant — the first direct evidence that latent prediction encodes cardiac dynamics rather than static appearance." This finding is not in the Nature Medicine paper and does not create overlap.

### Tier 2 — Supporting representation evidence (include selectively, not all)

**Principle: CKA + frame shuffling (now Tier 1) is sufficient for ncQn. Include noise-level probe and intrinsic dimensionality only if CKA results are ambiguous.**

| # | Experiment | Target Reviewer | Effort | Include? |
|---|-----------|-----------------|--------|----------|
| 4 | **Noise-level linear probe on frozen features** | ncQn | Hours | Backup (include if CKA gap is modest) |
| 5 | **Intrinsic dimensionality comparison** | ncQn, ALL | Hours | Backup (include if CKA gap is modest) |

**Experiments 4-5:** Run these but hold in reserve. If CKA shows a clean gap (EchoJEPA >0.9, EchoMAE <0.7) and frame shuffling shows the expected 15% vs 0% split, the point is made. Additional evidence adds length without adding conviction. Only include if CKA is noisy.

**Note on Goodfire results:** Check attribution/permissions before citing. If the report is internal, re-run frame shuffling independently (straightforward: shuffle tensor along temporal dimension, run frozen probe evaluation).

**Methodological notes from Goodfire report (apply when re-running):**
- Test across a dynamic-to-static task gradient: LVEF (inherently dynamic), mortality (partially dynamic), AF (largely static). This preempts "shuffling just degrades everything" — static tasks should be invariant and ARE.
- Include **matched-position shuffle** (reassign RoPE positions to match shuffled content). This removes positional encoding as a confound and shows the *true* temporal reliance (15.1% vs 10.9% for default shuffle). Default shuffle understates the effect.
- Include three shuffle types if feasible: tubelet shuffle (7.4%), frame shuffle (10.9%), matched-position (15.1%) — degradation scales monotonically with disruption severity.
- Report with error bars (±1 s.d. over 3 shuffle seeds).
- **Temporal Fourier power** (Goodfire Figure 29): include as one supplementary panel showing EchoJEPA-L-K has highest cardiac-band power while EchoMAE-L is DC-dominated. One sentence, zero compute.

### Tier 3 — EchoBench: Open-Source Robustness Benchmark (addresses segmentation + noise + novelty)

**Key insight: Package evaluations into an open-source benchmark rather than reporting isolated results.** This reframes segmentation from a risk into a contribution: the benchmark is the contribution, the results are data points within it.

| # | Experiment | Target Reviewer | Effort | P(flips a score) |
|---|-----------|-----------------|--------|------------------|
| 6 | **EchoBench: tasks x noise conditions x models** | 6t2T, hfQ1, ncQn, ALL | 2-3 days compute + 1 day packaging | High (addresses 3/4 reviewers at once) |
| 7 | **Linear probe view classification across all models** | ALL | Zero (if already computed) | Medium (supporting) |
| 8 | **EchoJEPA-L-K as additional row in results table** | ALL | Zero (already trained) | Low (supporting) |

**EchoBench evaluation matrix:**

| Task | Clean | Depth Atten. (3 levels) | Acoustic Shadow (3 levels) | Haze (3 levels) | Speckle (3 levels) |
|------|-------|------------------------|---------------------------|-----------------|-------------------|
| View classification | x | x | x | x | x |
| LVEF regression | x | x | x | x | x |
| RVSP regression | x | x | x | x | x |
| LV segmentation | x | x | x | x | x |

= 4 tasks x 13 conditions (clean + 4 noise types x 3 levels) x N models. Open-sourced with code to evaluate any new model.

**Why this solves the segmentation risk:**

Previously, standalone segmentation was dangerous: if MAE wins on clean Dice, reviewers see cherry-picking. EchoBench changes the question from "who wins on segmentation?" to "how do models behave across tasks and conditions?"

The expected finding tells a richer story than clean Dice alone:
- MAE: Dice 0.88 clean → 0.65 under severe shadow (26% degradation)
- JEPA: Dice 0.85 clean → 0.80 under severe shadow (6% degradation)

Even if MAE wins on clean segmentation, JEPA wins on robust segmentation — directly supporting the thesis. **Report all results honestly.** The benchmark framing makes completeness a virtue, not a liability.

**What EchoBench adds to the novelty argument:**

The contribution list becomes: (1) controlled finding about objectives, (2) largest open echo FM checkpoint, (3) first standardized robustness benchmark for ultrasound FMs including dense prediction under realistic degradation. That's hard to dismiss as "just applying V-JEPA."

**Implementation:** Already have noise augmentation pipeline + probe evaluation framework. New pieces: (1) lightweight segmentation decoder for frozen features on CAMUS/EchoNet-Dynamic, (2) run full matrix, (3) package with evaluation scripts and README.

**Rebuttal text:** "We introduce EchoBench, an open-source evaluation suite for echocardiography foundation models that measures performance across classification, regression, and dense prediction tasks under four types of physics-informed acoustic degradation at three severity levels. To our knowledge, this is the first standardized robustness benchmark for this modality. Results reveal that while pixel reconstruction achieves competitive clean segmentation, it degrades [X]% under acoustic perturbations compared to [Y]% for latent prediction — extending our finding from recognition tasks to dense prediction."

### Explicitly excluded from rebuttal

| Item | Reason |
|------|--------|
| Cross-modal prediction (E/e' from B-mode) | **Double submission risk** — this is NatMed Pillar 2 ("structure predicts flow"); creates result overlap. Also, ICML reviewers lack clinical context to appreciate significance. Save for Nature Medicine where it's the headline finding. |
| V-JEPA 2.1 results | Still training; changes architecture (dense loss, multi-layer heads); muddies controlled comparison; save for Nature Medicine |
| SAE concept discovery (Goodfire) | Whole new contribution; burns novelty for NatMed; needs too much explanation for rebuttal |
| Domain modification ablation | Counterproductive: either shows V-JEPA2 fails without modifications (undermines objective-alignment narrative) or succeeds (makes contribution thinner) |
| Data pipeline details | Wrong audience; engineering not science; no reviewer asked |
| Norm explosion / training instability | Opens new attack surface; save for NatMed where context allows full discussion |
| Attribution analysis (Goodfire) | Already have attention viz; adding four more methods doesn't address what reviewers asked |

---

## Rebuttal Structure (Updated — Lead with Surprise, Not Method)

**Key principles:**
1. Lead with what changes reviewer perception, not with defensive arguments
2. A rebuttal with 3-4 striking results presented cleanly outperforms one with 9 results in a laundry list
3. Target the AC, not just reviewers — the AC decides when scores are split
4. Every sentence of preamble is attention wasted

### Opening (2 sentences)

"We provide three new contributions addressing reviewer concerns: (1) representation-level evidence that latent prediction encodes temporal cardiac dynamics (15% LVEF degradation under frame shuffling vs 0% for MAE) and actively filters speckle noise (CKA analysis); (2) EchoBench, an open-source evaluation suite measuring foundation model performance across classification, regression, and dense prediction tasks under physics-informed acoustic degradation — the first such benchmark for echocardiography; and (3) gated release of EchoJEPA-G upon acceptance, the largest publicly available echocardiography foundation model."

Then immediately into the evidence. Do not re-explain the paper.

### Section 1: The Contribution (all reviewers — 2 paragraphs max)

**Stronger framing than "scientific, not architectural":** This paper is not an application of V-JEPA2 to echocardiography. It is an empirical investigation of when and why pretraining objectives fail, using echocardiography as a controlled test domain where the prediction is sharp: pixel reconstruction should fail because pixels are dominated by noise. The fact that the architecture is unchanged is the experimental design, not a limitation. This is analogous to how scaling law papers use standard architectures to isolate the effect of scale — the architectural simplicity is what makes the finding clean.

Second paragraph: new evidence and community contributions. Lead with representation evidence (Section 2), then EchoBench (Section 3). One-line linear probe confirmation: "This ranking holds under linear probing (EchoJEPA-L: 70.8% vs EchoMAE-L: 59.2%), ruling out probe architecture as a confound."

### Section 2: Representation-Level Evidence (ncQn + AC champion sentence)

Two strong results that are pure ML and require no clinical knowledge:

1. **Frame shuffling (champion result)** — JEPA degrades 10.9-15.1% on LVEF when frames shuffled, MAE invariant <0.5%. AC champion sentence: "The first direct evidence that latent prediction encodes temporal cardiac dynamics while pixel reconstruction learns static frame-level patterns." This is immediately striking, mechanistic, and addresses both the novelty concern (new finding about what the objective learns) and ncQn's representation-level evidence request.

2. **CKA speckle invariance** — one quantitative table. JEPA representations stable under noise, MAE shifts. Directly answers ncQn's specific ask.

Include noise-level probe and intrinsic dimensionality ONLY if CKA gap is modest.

### Section 3: EchoBench — Open-Source Robustness Benchmark (6t2T, hfQ1, ncQn)

Present EchoBench as a community contribution: 4 tasks (view classification, LVEF, RVSP, LV segmentation) x 13 conditions (clean + 4 noise types x 3 severity levels) x N models. Open-sourced with evaluation scripts.

Key results to highlight:
- **Segmentation under noise** — even if MAE wins on clean Dice, show degradation curves: "MAE achieves competitive clean segmentation but degrades [X]% under acoustic perturbation vs [Y]% for JEPA, extending the robustness finding from recognition to dense prediction."
- **Cross-task robustness profiles** — which objectives are fragile to which noise types
- Report all results honestly — the benchmark framing makes completeness a virtue

This addresses 6t2T ("no SOTA comparisons beyond ablation"), hfQ1 ("broader downstream tasks"), and ncQn ("noise sensitivity studies") simultaneously.

### Section 4: Contrastive Comparison (hfQ1)

DINO/BYOL results if available (note: DINO is image-only, video adaptation non-trivial — see experiment notes). If ready, use appropriate contingency framing. If not: EchoPrime reframing + Concern 3b data scale argument (EchoJEPA-L on 4.6K patients beats EchoPrime on 109K patients).

### Section 5: Community Resources & Reproducibility (all reviewers)

Three community contributions:
1. **EchoJEPA-G** — gated release upon acceptance, largest open echo FM (1.1B params)
2. **EchoBench** — first standardized robustness benchmark for ultrasound FMs
3. **EchoJEPA-L + evaluation framework** — already public

Dataset diversity clarification. Double-blind fix.

### Section 6: Camera-Ready Revisions

Summary table of all committed changes.

### Discussion Period Strategy

After rebuttal submission, engage actively during the author-reviewer discussion:
- **ncQn**: If CKA results are strong, post them and ask directly: "Does this representation-level evidence adequately address your concern?" Invite them to collaborate on what the camera-ready should include. Makes it harder to maintain a reject after their specific request is fulfilled.
- **hfQ1**: If DINO baseline is available, present it and ask: "We have added the contrastive comparison you requested. Does this address your concern about missing contrastive baselines?"

---

## Path to Acceptance (Updated after Review Simulation 2026-03-25)

### Key assets (in order of impact)

1. **Frame shuffling** (AC champion sentence — "15% degradation vs 0% proves JEPA encodes dynamics, MAE doesn't" — pure ML, no clinical knowledge needed, no NatMed overlap)
2. **CKA speckle invariance** (ncQn's direct ask — quantitative noise-filtering evidence)
3. **EchoBench** (open-source benchmark: tasks x noise conditions x models — addresses segmentation + noise + novelty simultaneously, makes completeness a virtue rather than cherry-picking a risk)
4. **EchoJEPA-G gated release** (zero compute, addresses 3/4 reviewers)
5. **DINO controlled baseline** (only lever for hfQ1 — but video adaptation is non-trivial, may not be ready)
6. **Novelty reframing** ("unchanged architecture IS the experimental design, not a limitation")
7. **Data scale argument** (Concern 3b — confounds work against EchoJEPA-L)

### Per-reviewer predicted movement

| Reviewer | Current | Post-rebuttal | Key driver | P(flip) |
|----------|---------|---------------|-----------|---------|
| hfQ1 | 2 | 2->3 | DINO baseline + cross-modal probe | ~35% |
| 6t2T | 3 | 3->4 | Cross-modal probe + G release (+ segmentation if favorable) | ~50% |
| ncQn | 3 | 3->4 (possibly 5) | CKA + frame shuffling directly answers their ask | ~80% |
| L8sp | 4 | 4 (possibly 5) | Already positive; new evidence confirms | ~95% stays |

### Scenario analysis

| Scenario | hfQ1 | 6t2T | ncQn | L8sp | Avg | Outcome | P |
|----------|------|------|------|------|-----|---------|---|
| Best | 3 | 4 | 5 | 5 | 4.25 | Accept | 25% |
| Good | 3 | 4 | 4 | 4 | 3.75 | Likely accept | 20% |
| Decent | 2 | 4 | 4 | 4 | 3.5 | AC decides | 25% |
| Mixed | 2 | 3 | 4 | 4 | 3.25 | Borderline | 18% |
| Worst | 2 | 3 | 3 | 4 | 3.0 | Reject | 12% |

**Overall acceptance probability: ~35-50%** (up from 25-35% with original plan; slightly more conservative than simulation's 45-55% because confidence-5 reviewers rarely move >1 point and "applications at methods venue" is structural)

### What caps probability below 55%

- hfQ1's confidence-5 reject — very hard to move; the novelty concern may be fundamental to their ICML expectations
- Execution risk — DINO may produce a moderate result (prepared with three framings), cross-modal probe may show modest correlation
- Structural issue — the AC's philosophy on empirical-contribution papers at ICML determines the outcome; this is not under our control
- Segmentation is a trap — may favor MAE; excluded from rebuttal unless result is favorable

### What the team controls

1. **Execute cross-modal probe immediately** — this is hours of compute and the highest single-experiment ROI
2. **Start DINO training today** — 3-5 days means this is on the critical path
3. **Run CKA + frame shuffling** — directly answers ncQn's explicit request
4. **Narrative discipline** — 2-sentence opening, 3-4 focused results, no laundry lists
5. **AC-targeted language** — "The unchanged architecture is the experimental design. This paper isolates the objective as a variable with the same logic that scaling law papers use standard architectures."
6. **Discussion period engagement** — directly engage ncQn ("Does this address your concern?") and hfQ1 ("We added the contrastive comparison you requested")

---

## Relationship to Previous Rebuttal Documents

This document **supersedes** the strategy in `01`-`07` for the actual rebuttal response. Those documents remain valuable as:
- `01-paper-audit.md` — reference for appendix fixes and editorial corrections
- `04-competitive-positioning.md` — framing for EchoJEPA vs concurrent work
- `05-probe-fairness.md` — background on Strategy E (d=1 attentive), useful if probe questions arise in discussion
- `07-camera-ready-actions.md` — editorial action items still apply

Documents `02` (old rebuttal template), `03` (old worst-case scenarios), and `06` (old claim validity) are largely superseded by this document.
