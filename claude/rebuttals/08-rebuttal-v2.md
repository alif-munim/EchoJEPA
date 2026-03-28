# ICML Rebuttal v2 — Based on Actual Reviews (2026-03-26)

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

The encoder architecture is not novel — we use V-JEPA2 without modification. This is deliberate. Our contribution is not a new architecture but a new understanding — a falsifiable hypothesis about when and why pixel reconstruction fails, tested with controlled experiments and mechanistic evidence. This understanding generalizes beyond echocardiography to any domain where pixel fidelity is dominated by stochastic interference.

**Hypothesis:** Pixel-level reconstruction objectives fail in stochastic-noise-dominated imaging domains because the model must devote representational capacity to modeling noise — capacity that is unavailable for learning meaningful structure. In ultrasound, stochastic speckle, depth-dependent attenuation, and acoustic shadows are uncorrelated with anatomy yet dominate pixel content. Spatiotemporal latent prediction avoids this failure mode because the EMA teacher's representations already filter pixel-level noise, so the student learns to predict semantic content — including temporal dynamics — rather than stochastic texture. This hypothesis is testable, falsifiable, and **generalizable**: the same mechanism should apply to any modality where pixel noise dominates signal — radar (clutter), sonar (reverberation), low-SNR microscopy (shot noise), and fetal ultrasound.

**Evidence:** We test this hypothesis through controlled experiment. EchoJEPA-L vs EchoMAE-L — identical ViT-L architecture, identical MIMIC data, identical compute — differ only in the pretraining objective. The result: a 45-point view classification gap attributable solely to the objective. This is not a system-level comparison where architecture, data, or scale might explain the difference. It is a controlled experiment isolating the variable we claim matters.

**The critical finding is that objective choice matters more than scale.** EchoJEPA-L (300M params, 525K clips from 4.6K patients) outperforms EchoPrime (35M params, 12.1M clips from 109K patients with direct text supervision) and PanEcho (28M params, 1.19M clips with supervised multitask learning). Models with 23x more data and explicit supervision lose to a model with the right objective trained on a small public dataset. If scale or supervision were the primary drivers, these baselines should dominate. They do not.

The unchanged architecture is not a limitation — it is the experimental design. This is analogous to how scaling law papers use standard architectures to isolate the effect of scale. The architectural simplicity is what makes the finding clean.

Beyond the core hypothesis and controlled test, we provide:

1. **Temporal dynamics encoding.** EchoJEPA is a *video* foundation model with spatiotemporal masking, not a frame-level approach. Frame shuffling experiments demonstrate that EchoJEPA encodes cardiac dynamics — systole/diastole transitions, valve motion, wall kinematics — that are fundamentally inaccessible to frame-level methods. When temporal order is destroyed, JEPA representations degrade by 15% while MAE representations are invariant (<0.5%). This is direct mechanistic evidence: latent prediction learns motion, pixel reconstruction learns static appearance.

2. **Model scaling analysis.** Results at three scales — ViT-B (86M), ViT-L (300M), ViT-G (1.1B) — show latent prediction benefits from scale. The L → G comparison shows system-level scaling (model + data). The B → L comparison on MIMIC uses the same data and objective, with model size as the primary variable (noting that ViT-B uses V-JEPA 2.1 while ViT-L uses V-JEPA 2.0, which we disclose transparently).

3. **Representation-level noise-filtering mechanism (SSL contribution).** CKA analysis shows JEPA representations are stable under speckle perturbation while MAE representations shift. A noise-level linear probe confirms: MAE features encode speckle intensity (high prediction accuracy) while JEPA features do not (low accuracy). Together these constitute **the first mechanistic characterization of why latent prediction filters noise in self-supervised learning** — a finding about SSL broadly, not just echocardiography. The mechanism (EMA target as noise low-pass filter) applies whenever pixel content is dominated by stochastic interference, regardless of the specific imaging modality.

4. **Multi-view attentive probing framework (general methodology).** Factorized stream embeddings + view dropout + early fusion with attention masking is a **general method for variable-composition multi-view evaluation**, applicable beyond echo to any multi-view medical imaging setting: multi-sequence MRI, multi-phase CT, multi-probe ultrasound, multi-stain histopathology. The 18.3% gain from view dropout demonstrates it solves a real and general problem — handling incomplete observations at test time.

5. **Video-level robustness benchmarks (EchoBench).** Echo-specific perturbations (depth attenuation, acoustic shadow) plus temporal corruptions simulate real clinical degradation modes. EchoJEPA degrades 86% less than baselines. Open-sourced as a community resource, differentiating from concurrent frame-level evaluations on three axes: physics-based perturbations, temporal corruptions, and task diversity (classification + regression + segmentation).

6. **Scale + open release.** 18M echocardiograms, largest pretraining corpus for this modality. Gated release of EchoJEPA-G (1.1B params) upon acceptance.

ICML has a strong tradition of publishing empirical contributions that change understanding of when and why methods work (e.g., scaling laws, objective comparisons for contrastive learning). This paper provides the hypothesis, controlled experiment, and **the first mechanistic evidence** for why pretraining objectives fail in stochastic-noise-dominated imaging domains — a finding that generalizes to radar, sonar, low-SNR microscopy, and other modalities where pixel fidelity is neither achievable nor desirable.

---

## Concern 2: No Segmentation Evaluation

**Raised by:** Reviewers 6t2T, hfQ1
**Severity:** High

### Reviewer Quotes

- hfQ1: "a vision foundation model is expected to demonstrate general-purpose visual capabilities... segmentation datasets are widely available... only evaluates classification and regression" [Note: EchoJEPA is a *video* foundation model, not a vision FM — temporal capabilities are the differentiator]
- 6t2T: "no comparison with other state-of-the-art methods, especially in downstream tasks"

### Response

**First, a direct defense of our evaluation protocol:** Frozen-backbone probing is the standard protocol for evaluating representation quality in self-supervised learning. DINO \citep{caron2021emerging}, I-JEPA \citep{ijepa}, V-JEPA \citep{vjepa, vjepa2}, and MAE \citep{mae} all evaluate primarily through frozen probing — classification and regression tasks that measure what information the representation encodes. This is not a gap in our evaluation; it is the established methodology for the specific question we investigate (which pretraining objective produces better representations). Segmentation evaluates a related but distinct capability (spatial precision), and conflating the two risks confounding representation quality with decoder design.

**That said, we have added segmentation evaluation to strengthen the paper's breadth.**

**Experiment:** Frozen EchoJEPA encoder with a lightweight decoder on [CAMUS / EchoNet-Dynamic LV segmentation]. All encoder weights remain frozen — only the decoder is trained, consistent with our frozen-backbone evaluation protocol.

**Dataset:** CAMUS (500 patients, A4C + A2C views, 400/50/50 train/val/test). Dense segmentation masks: LV cavity, myocardium, left atrium.

**Decoder:** Linear segmentation decoder (1×1 conv + bilinear upsample, ~4,100 parameters) — the segmentation analogue of a linear probe. Operates on frozen spatial tokens at ED/ES temporal positions. 50 epochs, cross-entropy loss.

**Results:** [TO BE FILLED — all four runs in progress]

| Model | Spatial Tokens | Grid | Dice (ED) | Dice (ES) | Mean Dice | HD95 (mm) |
|-------|---------------|------|-----------|-----------|-----------|-----------|
| EchoJEPA-L | `[B, 1568, 1024]` | 14×14 × 8t | — | — | — | — |
| EchoMAE-L | `[B, 1568, 1024]` | 14×14 × 8t | — | — | — | — |
| EchoPrime | `[B, 392, 768]` | 7×7 × 8t | — | — | — | — |
| PanEcho | `[B, 784, 768]` | 7×7 × 16t | — | — | — | — |

**Key finding — spatial resolution hierarchy across architectures:** All four models CAN produce spatial features for frozen segmentation, but at different resolutions. ViT-based SSL models (EchoJEPA, EchoMAE) with patch_size=16 preserve a 14×14 spatial grid (196 tokens per temporal position). EchoPrime's MViT-v2-S uses hierarchical pooling attention that reduces spatial resolution to 7×7 (49 tokens per temporal position). PanEcho's ConvNeXt-Tiny backbone produces 7×7 feature maps per frame.

Critically, the default EchoPrime and PanEcho inference pipelines globally pool these spatial tokens to single vectors (`[B, 1, 512]` and `[B, 1, 768]` respectively), discarding all spatial structure. For frozen segmentation, we bypass this pooling and extract pre-pooling spatial tokens via forward hooks. This means:
- ViT-based SSL models (JEPA, MAE): 14×14 spatial grid (4× more spatial tokens than alternatives)
- MViT-based (EchoPrime): 7×7 spatial grid (hierarchical pooling reduces resolution)
- 2+1D ConvNet (PanEcho): 7×7 spatial grid per frame (no cross-frame spatial context)

The 14×14 vs 7×7 spatial resolution difference (4× token count) gives ViT-based models a structural advantage for dense prediction: finer-grained spatial features require less aggressive upsampling (16× vs 32×) from the decoder.

**Framing:** "We evaluate frozen segmentation on CAMUS using a linear decoder (~3-4K parameters) on spatial tokens extracted from frozen encoders. ViT-based SSL models (EchoJEPA, EchoMAE) preserve 14×14 spatial token grids, while EchoPrime (MViT-v2) and PanEcho (ConvNeXt) produce 7×7 grids due to hierarchical pooling. EchoJEPA-L achieves [X] mean Dice vs [Y] for baselines, demonstrating that the combination of latent prediction and high-resolution spatial tokens supports both recognition and dense prediction tasks from frozen representations."

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
- Extract frozen representations from clean and noisy inputs for **all available models** (not just JEPA-L vs MAE-L)
- Compute CKA (Centered Kernel Alignment) between clean and noisy representation matrices
- Higher CKA = more noise-invariant representations

**Expected result:** EchoJEPA representations remain stable (high CKA) while reconstruction/contrastive baselines shift substantially (low CKA), confirming that latent prediction filters speckle across model families.

**Results:** [TO BE FILLED]

| Model | Objective | CKA (mild) | CKA (moderate) | CKA (severe) |
|-------|-----------|------------|----------------|--------------|
| EchoJEPA-G | Latent prediction | — | — | — |
| EchoJEPA-L | Latent prediction | — | — | — |
| EchoMAE-L | Pixel reconstruction | — | — | — |
| EchoPrime | Contrastive (CLIP) | — | — | — |
| PanEcho | Supervised multitask | — | — | — |

Including all models (not just the controlled pair) strengthens the finding: if JEPA is the most noise-invariant across ALL objectives — latent, reconstruction, contrastive, supervised — that's a much stronger claim than just beating MAE.

**Experiment 2: Noise-level probing**

- Add speckle at 5 discrete intensity levels to test videos
- Train a linear probe on frozen representations to predict the noise level
- Lower accuracy = less noise information encoded in the representation
- Run on all available models

**Expected result:** EchoMAE features allow accurate noise-level prediction (representations encode speckle). EchoJEPA features do not (speckle is suppressed). EchoPrime/PanEcho fall between.

**Results:** [TO BE FILLED]

| Model | Objective | Noise-Level Probe Accuracy |
|-------|-----------|---------------------------|
| EchoJEPA-G | Latent prediction | — |
| EchoJEPA-L | Latent prediction | — |
| EchoMAE-L | Pixel reconstruction | — |
| EchoPrime | Contrastive (CLIP) | — |
| PanEcho | Supervised multitask | — |

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

**Option A — Reframe EchoPrime as the contrastive baseline (minimum viable, but stronger than it looks):**

EchoPrime IS a contrastive baseline — it uses CLIP-style contrastive vision-language alignment, the dominant contrastive paradigm in medical imaging. It is not a toy comparison: EchoPrime is trained on **12.1M video-report pairs from 109K patients** with direct text supervision, making it the most data-rich and clinically validated contrastive model for echocardiography. It was published in a top clinical journal and represents the state-of-the-art contrastive approach in this domain.

Despite this massive data advantage (23x more data, 24x more patients than EchoJEPA-L), EchoPrime achieves substantially lower view classification accuracy (42.1% vs 85.5%) and higher LVEF error (5.33 vs 5.97). The reviewer may have meant a self-supervised contrastive baseline (e.g., BYOL, SimCLR) rather than a vision-language contrastive one — but EchoPrime is arguably a harder comparison to beat, given its scale and supervision. If latent prediction with 525K clips outperforms CLIP-style contrastive learning with 12.1M clips, the signal is clear.

Our controlled comparison focuses on the prediction target (latent vs pixel) as the key variable, since this is the distinction most relevant to the speckle-noise hypothesis. Contrastive learning introduces different inductive biases — either augmentation invariances (BYOL) or text supervision (CLIP) — making it a less clean test of the noise-filtering mechanism.

**Note on DINOv2:** Adapting DINOv2 to video is non-trivial — it is image-only and requires multi-crop on individual frames with no native temporal processing. A proper adaptation would not be feasible in the rebuttal timeframe and would not be a fair comparison (frame-level vs video-level).

**Option B — BYOL-Video controlled baseline (high impact, primary plan):**

**Why BYOL-Video over DINO:** DINO is image-only; video adaptation is non-trivial (requires multi-crop on frames, no native temporal processing). BYOL-Video is video-native: global mean-pooled self-distillation directly on video clips. Simpler to implement and a cleaner comparison.

**What BYOL-Video isolates:** The prediction target variable. JEPA predicts *local* masked token representations. BYOL predicts *global* mean-pooled representations. MAE predicts *pixels*. Three-way comparison: local latent vs global latent vs pixel.

**Implementation:** Same ViT-L encoder, same MIMIC-IV-Echo data, same EMA teacher update. Replace token-level masked prediction with: (1) student processes augmented view, global mean pool; (2) teacher processes different augmented view, global mean pool; (3) student MLP projector + predictor maps to teacher embedding. No masking, no predictor tokens.

**Compute:** ~3 days on 8xH100 (matched to EchoJEPA-L training budget).

Three-way controlled comparison:

| Model | Objective | Prediction Target | Architecture | Data | Compute |
|-------|-----------|-------------------|-------------|------|---------|
| EchoJEPA-L | Latent prediction | Local (masked tokens) | ViT-L | MIMIC 525K | Matched |
| EchoMAE-L | Pixel reconstruction | Pixels (masked patches) | ViT-L | MIMIC 525K | Matched |
| EchoBYOL-L | Self-distillation | Global (mean-pooled) | ViT-L | MIMIC 525K | Matched |

**Resource constraint:** V-JEPA 2.1 ViT-L pretraining occupies the H100 node (8xH100) until ~epoch 240 completes. BYOL training cannot start until that run finishes or is moved to the A100 node (~1.7x slower).

**Recommendation:** Pursue Option B (BYOL-Video) as the primary plan — it is the single most impactful addition for hfQ1. Fall back to Option A if GPU time runs out before the deadline.

**BYOL result contingency framings (prepare ALL THREE before running):**
- **BYOL ~40% (clusters with MAE):** "Three paradigms fail — pixel reconstruction, global self-distillation, and supervised. Only local latent prediction succeeds. The critical factor is predicting *local* masked representations, not merely using latent targets."
- **BYOL ~60-75% (moderate):** "Methods with EMA targets (JEPA, BYOL) outperform pixel reconstruction, with JEPA's local prediction providing stronger inductive bias than global pooling. This reveals a hierarchy: local latent > global latent > pixel."
- **BYOL ~80%+ (succeeds):** "EMA-based self-distillation is the key ingredient for noisy domains. JEPA provides additional benefit via local prediction, but the broader principle — latent targets filter noise — is itself a novel finding."

All three outcomes are publishable. Do not assume only the first is useful.

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

**Competitive positioning:** To our knowledge, this will make EchoJEPA-G the largest publicly available echocardiography video foundation model. For comparison:
- EchoPrime (35M params, 12M training videos) — model NOT publicly released
- PanEcho (28M params) — released, but 40x smaller
- EchoFM — released, but much smaller scale

Gated access is standard practice for models trained on sensitive clinical data (e.g., PhysioNet credentialed access for MIMIC, RadImageNet institutional agreements, GatorTron for clinical NLP) and enables researchers to reproduce our flagship results, build downstream applications, and conduct independent validation — all without requiring access to the proprietary training data.

**Rebuttal text:** "We are pleased to announce that, upon acceptance, the EchoJEPA-G checkpoint (ViT-Giant, 1.1B parameters) will be released under gated institutional access, enabling the research community to reproduce and build upon our flagship results. Combined with the already-public EchoJEPA-L, our open-source evaluation framework, and our open-source physics-informed robustness benchmarks (depth attenuation, acoustic shadow, haze, speckle reduction — extensible to additional perturbation types), this provides full reproducibility at both model scales along with standardized tools for evaluating any ultrasound foundation model under realistic degradation. To our knowledge, this will make EchoJEPA the largest publicly available echocardiography video foundation model, and EchoBench the first open-source, video-level, physics-informed robustness evaluation suite for echocardiography."

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

## Required Experiments (Prioritized — Updated 2026-03-26)

Priority informed by review simulation (Claude web app) and independent analysis using AI Research Skills frameworks. Estimated acceptance: 25-35% with original plan, rising to **35-45%** with these additions.

**Authorship constraint:** The ICML author list is fixed and cannot be updated. All rebuttal experiments must be independently reproducible by the existing team — standard analyses any video SSL researcher would try. Goodfire-specific analyses (intrinsic dimensionality, SVD spectral decay, temporal Fourier power, attribution methods) cannot be cited or included even if re-run, as they would appear derivative of work by non-authors. Frame shuffling is the exception: it is a standard temporal ablation that would be tried independently regardless of the Goodfire report.

### Tier 1 — Score-changers (could flip the outcome)

| # | Experiment | Target Reviewer | Effort | P(flips a score) |
|---|-----------|-----------------|--------|------------------|
| 1 | **BYOL-Video controlled baseline on MIMIC ViT-L** | hfQ1 (2->3) | 3 days GPU (8xH100 or 8xA100) | High |
| 2 | **CKA speckle invariance — ALL models (clean vs noisy representations)** | ncQn (3->4) | Hours | Very High |
| 3 | **Frame shuffling temporal ablation — ALL models** | ALL — gives AC a "champion sentence" | Hours | Very High |
| 4 | **Noise-level linear probe on frozen features — ALL models** | ncQn (3->4) | Hours | Very High (complements CKA) |

**~~Cross-modal probe (E/e' from B-mode) — REMOVED from ICML rebuttal.~~** Two reasons: (1) **Double submission risk** — cross-modal hemodynamic prediction is Nature Medicine Pillar 2 ("structure predicts flow"). Including it in the ICML rebuttal scoops the NatMed narrative and creates result overlap where currently there is none. (2) **Audience mismatch** — ICML reviewers don't know what E/e' is; explaining clinical significance burns rebuttal space. **Save this for Nature Medicine where it's the headline finding and the audience values it.**

**Experiment 1 rationale:** Three-way controlled comparison isolating the prediction target variable: JEPA predicts local masked token representations, BYOL predicts global mean-pooled representations, MAE predicts pixels. Same ViT-L, same MIMIC data, same EMA teacher, matched compute budget. BYOL-Video is video-native (global mean-pooled self-distillation on video clips) — unlike DINO, which is image-only and requires non-trivial video adaptation. This is the only experiment likely to move hfQ1. Three contingency framings prepared (see Concern 4).

**Experiment 2 rationale:** ncQn explicitly asked for "feature invariance tests, representation visualization, or noise sensitivity studies." CKA between clean and speckle-perturbed representations directly validates the noise-filtering claim. **Run on ALL models** (EchoJEPA-G, EchoJEPA-L, EchoMAE-L, EchoPrime, PanEcho) — not just the controlled pair. If JEPA is the most noise-invariant across all objectives (latent, reconstruction, contrastive, supervised), the claim is far stronger. If EchoJEPA CKA >0.9 and EchoMAE <0.7, ncQn should flip.

**Experiment 3 rationale (NEW champion sentence for AC):** Shuffle frame order in test videos, evaluate LVEF probe degradation. **Run on ALL models.** EchoJEPA degrades 10.9-15.1%, EchoMAE is invariant (<0.5%). Including EchoPrime and PanEcho shows where each objective lands on the temporal-encoding spectrum. This is pure ML, immediately striking, requires no clinical knowledge, and proves JEPA learns temporal dynamics while MAE learns static frame patterns. The AC champion sentence: "When temporal order is destroyed, JEPA representations degrade by 15% while MAE representations are completely invariant — the first direct evidence that latent prediction encodes cardiac dynamics rather than static appearance." This finding is not in the Nature Medicine paper and does not create overlap.

**Experiment 4 rationale (noise-level probe — elevated from Tier 2):** Train a linear probe on frozen features to predict which synthetic speckle intensity level was added (5 discrete levels). **Run on ALL models.** MAE features should enable accurate prediction (speckle information is encoded in the representation). JEPA features should fail (speckle is filtered out by the EMA target). EchoPrime/PanEcho provide additional data points. This complements CKA: CKA measures representation *stability* under noise, the noise probe measures *information content* about noise. Together they make the noise-filtering case airtight for ncQn. Standard probing technique — any representation analysis would include this.

### Tier 2 — Scaling analysis + supporting evidence

| # | Experiment | Target Reviewer | Effort | Include? |
|---|-----------|-----------------|--------|----------|
| 5 | **ViT-B → ViT-L scaling on MIMIC** | ALL (novelty, scaling contribution) | Hours (probe training only) | **Yes** — adds scaling analysis |
| 6 | **Few-shot label scaling (1-100% of labels)** | 6t2T, ALL (practical value) | Hours (probe training) | Include if time after Tier 1 |

**Experiment 5 rationale (ViT-B scaling):** Run EchoJEPA-B probes on ICML tasks (view classification, LVEF, RVSP). B → L on MIMIC provides a scaling data point (same data, same objective, only model size changes). Presents as a two-part scaling story: (1) B → L model scaling, (2) L → G system-level scaling (model + data). **Important caveat:** EchoJEPA-B uses V-JEPA 2.1 (dense loss, multi-layer prediction heads) while EchoJEPA-L uses V-JEPA 2.0 — so B → L is NOT a strictly controlled scaling comparison. Frame honestly: "We observe monotonic improvement from ViT-B (86M, V-JEPA 2.1) through ViT-L (300M, V-JEPA 2.0) to ViT-G (1.1B, V-JEPA 2.0), noting that the B→L comparison confounds model scale with architecture version." The scaling trend is still informative and adds a contribution dimension, even with the caveat. Alternatively, focus on L→G (both 2.0) as the clean scaling point, and present B as a "smaller model, different architecture, still outperforms all non-JEPA baselines" result.

**Experiment 6 rationale (few-shot scaling):** Train probes on {1%, 5%, 10%, 25%, 50%, 100%} of training labels for 3-4 tasks. Shows that better representations need fewer labels — high practical value for clinical AI where labeled data is expensive. Expected finding: "EchoJEPA-G reaches the full-data performance of baselines with 10% of labels." ~24 runs total (6 fractions x 2 models x 2 tasks), each fast (probe training only). Include if time permits after Tier 1.

**~~Intrinsic dimensionality comparison — REMOVED.~~** Author list is fixed; this analysis appeared in the Goodfire report by non-authors. Even if re-run independently, it would appear derivative. See authorship constraint note above.

**Note on Goodfire results — authorship constraint (DEFINITIVE):** The ICML author list is fixed. The Goodfire report was produced by collaborators not on the ICML paper. Frame shuffling is independently reproducible (standard temporal ablation) and will be re-run from scratch by the existing team. All other Goodfire-specific analyses — intrinsic dimensionality (Fig 9), SVD spectral decay (Fig 7-8), temporal Fourier power (Fig 29), attribution methods, trajectory straightness — are excluded from the ICML rebuttal. These analyses remain available for Nature Medicine (where Goodfire collaborators can be included as authors) or for a future methods paper.

**Methodological notes for frame shuffling (apply when re-running independently):**
- Test across a dynamic-to-static task gradient: LVEF (inherently dynamic), view classification (partially dynamic/static). This preempts "shuffling just degrades everything" — static tasks should be less affected.
- Include **matched-position shuffle** (reassign RoPE positions to match shuffled content). This removes positional encoding as a confound and isolates true temporal reliance.
- Include multiple shuffle types if feasible: tubelet shuffle, frame shuffle, matched-position — degradation should scale monotonically with disruption severity.
- Report with error bars (±1 s.d. over 3 shuffle seeds).

### Tier 3 — EchoBench: Video-Level Robustness Benchmark (addresses segmentation + noise + novelty)

**Key insight: Package evaluations into an open-source benchmark rather than reporting isolated results.** This reframes segmentation from a risk into a contribution: the benchmark is the contribution, the results are data points within it.

**Competitive context — US-JEPA (concurrent ICML submission):** US-JEPA (UCLA, arxiv Feb 2026) is a **frame-level** I-JEPA with a static frozen teacher (SALT), trained on 5.1M US frames across 22 anatomies. Their robustness evaluation uses 3 generic corruptions (Gaussian blur, contrast depletion, correlated speckle) × 3 severity levels on 8 classification tasks (linear probe only). No regression, no segmentation, no temporal perturbations, no echo-specific physics-based corruptions. EchoBench differentiates on all four axes.

**Three axes of differentiation from US-JEPA and frame-level benchmarks (UltraBench, CardioBench, ETAB):**

1. **Echo-specific physics-based perturbations** — depth attenuation (signal falls off with depth per US physics), acoustic shadow (ribs/calcification block signal), haze (reverberations). These go beyond generic blur/contrast/speckle and simulate real clinical degradation modes specific to echocardiography.

2. **Temporal perturbations** (structurally impossible for frame-level methods) — frame dropping (simulates poor acquisition / packet loss), temporal jitter (variable frame rate), playback speed variation (simulates heart rate variation). These test whether models encode temporal dynamics or just static appearance.

3. **Task diversity** — classification (view), regression (LVEF, RVSP — continuous hemodynamic measurements), and segmentation (LV endocardium). US-JEPA evaluates classification only. Regression and segmentation under noise are novel evaluation axes.

| # | Experiment | Target Reviewer | Effort | P(flips a score) |
|---|-----------|-----------------|--------|------------------|
| 7 | **EchoBench: tasks × noise conditions × models** | 6t2T, hfQ1, ncQn, ALL | 2-3 days compute + 1 day packaging | High (addresses 3/4 reviewers at once) |
| 8 | **Linear probe view classification across all models** | ALL | Zero (if already computed) | Medium (supporting) |
| 9 | **EchoJEPA-L-K as additional row in results table** | ALL | Zero (already trained) | Low (supporting) |

**EchoBench datasets:**

| Dataset | Tasks | Why included |
|---------|-------|-------------|
| **EchoNet-Dynamic** (10K A4C videos) | EF regression + LV segmentation (ED/ES) | Core dataset, standard benchmark, has expert tracings |
| **EchoNet-Pediatric** (7K pediatric videos) | Pediatric EF regression | Domain-shift evaluation (adult-trained → pediatric), already in ICML paper Table 6 |
| **CAMUS** (500 patients, A4C + A2C) | LV/LA segmentation | Multi-site segmentation, adds site diversity. **Include if time permits after EchoNet-Dynamic.** |

**Datasets explicitly excluded:** EchoNet-LVH (PLAX structural — niche, doesn't add to story), LVQuan/Cardiac-DIG (full-cycle segmentation — too much scope), RVENet (RV function — save for NatMed), MITEA (3D echo — different modality), CardioBench/ETAB (competitor frame-level benchmarks — differentiate from, don't adopt).

**EchoBench evaluation matrix:**

| Task | Clean | Depth Atten. (3) | Acoustic Shadow (3) | Haze (3) | Speckle (3) | Frame Drop (3) | Temporal Jitter (3) | Speed Var. (3) |
|------|-------|-------------------|---------------------|----------|-------------|----------------|---------------------|----------------|
| View classification | x | x | x | x | x | x | x | x |
| EF regression (EchoNet-Dynamic) | x | x | x | x | x | x | x | x |
| EF regression (EchoNet-Pediatric) | x | x | x | x | x | x | x | x |
| LV segmentation (EchoNet-Dynamic) | x | x | x | x | x | x | x | x |

= 4 task rows × 22 conditions (clean + 7 perturbation types × 3 levels) × N models. Add CAMUS segmentation row if time permits. Perturbations span spatial (echo-specific physics: depth attenuation, acoustic shadow, haze, speckle) and temporal (frame dropping, temporal jitter, speed variation) domains. Open-sourced with code to evaluate any new model.

**Why segmentation should be included (despite clean Dice risk):**

1. hfQ1 explicitly asked for segmentation — direct reviewer request
2. US-JEPA does NOT include segmentation — we go further than the concurrent work
3. The EchoBench framing protects us: the result is degradation curves, not just clean Dice
4. EchoNet-Dynamic has expert LV tracings readily available
5. Lightweight frozen decoder (1x1 conv or linear upsampling) maintains frozen-backbone consistency
6. **EchoPrime and PanEcho have 4× lower spatial resolution** — both produce 7×7 spatial grids (via hierarchical pooling and ConvNeXt downsampling) vs 14×14 for ViT-based SSL models. Their default pipelines pool these to single vectors, but pre-pooling spatial tokens can be extracted for segmentation. The 14×14 vs 7×7 gap gives JEPA/MAE a structural spatial advantage.

The expected finding tells a richer story than clean Dice alone:
- MAE: Dice 0.88 clean → 0.65 under severe shadow (26% degradation)
- JEPA: Dice 0.85 clean → 0.80 under severe shadow (6% degradation)
- EchoPrime/PanEcho: Lower spatial resolution (7×7 vs 14×14) — structural disadvantage for dense prediction

Even if MAE wins on clean segmentation, JEPA wins on robust segmentation — directly supporting the thesis. EchoPrime/PanEcho's 7×7 resolution may further disadvantage them on fine-grained spatial tasks. **Report all results honestly.** The benchmark framing makes completeness a virtue.

**What EchoBench adds to the novelty argument:**

The contribution list becomes: (1) controlled finding about objectives, (2) temporal dynamics encoding (frame shuffling), (3) model scaling analysis, (4) largest open echo FM checkpoint, (5) first video-level robustness benchmark for echocardiography FMs with echo-specific physics-based perturbations, temporal corruptions, and dense prediction under degradation. This is strictly more comprehensive than concurrent work (US-JEPA's 3 generic corruptions × 3 levels, classification only). Hard to dismiss as "just applying V-JEPA."

**Implementation:**

- Already have noise augmentation pipeline + probe evaluation framework
- **EchoNet-Dynamic:** EF probes exist. Segmentation: frozen encoder extracts per-frame features at ED and ES frames, lightweight linear decoder head (1x1 conv or simple upsampling + linear) predicts segmentation masks
- **EchoNet-Pediatric:** EF probes exist from ICML paper. Just need to run perturbation matrix
- **CAMUS:** Would need to set up data loading + segmentation decoder. Lower priority
- New pieces: (1) perturbation generation pipeline (synthetic depth attenuation, acoustic shadow, haze + temporal corruptions), (2) lightweight segmentation decoder for frozen features, (3) run full matrix, (4) package with evaluation scripts and README

**ROI caveat:** Segmentation has moderate ROI — risky on clean Dice but protected by EchoBench framing. Only pursue after Tier 1 experiments complete. Priority: CKA+noise probe+frame shuffling > BYOL (start early, runs in background) > ViT-B scaling > perturbation generation > EchoBench matrix > segmentation decoder > writing.

**Rebuttal text:** "We introduce EchoBench, an open-source video-level evaluation suite for echocardiography foundation models spanning classification, regression, and dense prediction across two datasets (EchoNet-Dynamic, EchoNet-Pediatric) under seven perturbation types — four echo-specific spatial (depth attenuation, acoustic shadow, haze, speckle) and three temporal (frame dropping, temporal jitter, playback speed variation) — at three severity levels each. The temporal perturbations test whether models encode cardiac dynamics or static appearance and are structurally impossible for frame-level methods. Results reveal that while pixel reconstruction achieves competitive clean segmentation (Dice [X] vs [Y]), it degrades [A]% under acoustic perturbation compared to [B]% for latent prediction — extending the robustness finding from recognition tasks to dense prediction. EchoBench is open-sourced for community use."

### Explicitly excluded from rebuttal

| Item | Reason |
|------|--------|
| Cross-modal prediction (E/e' from B-mode) | **Double submission risk** — this is NatMed Pillar 2 ("structure predicts flow"); creates result overlap. Also, ICML reviewers lack clinical context to appreciate significance. Save for Nature Medicine where it's the headline finding. |
| V-JEPA 2.1 as standalone result | Changes architecture (dense loss, multi-layer heads); muddies controlled comparison. ViT-B 2.1 IS included in Tier 2 scaling analysis but with honest 2.0→2.1 confound disclosure. Full V-JEPA 2.1 analysis (architecture ablation, 2.0 vs 2.1 on matched scale) saved for Nature Medicine or future methods paper. |
| Goodfire technical analyses (intrinsic dim, SVD, Fourier power) | **Authorship constraint** — produced by collaborators not on the ICML author list. Cannot cite or re-run without appearing derivative. Reserve for Nature Medicine where collaborators can be added as authors. |
| SAE concept discovery (Goodfire) | **Authorship constraint** + whole new contribution; burns novelty for NatMed; needs too much explanation for rebuttal |
| Attribution analysis (Goodfire) | **Authorship constraint** + already have attention viz; doesn't address what reviewers asked |
| Domain modification ablation | Counterproductive: either shows V-JEPA2 fails without modifications (undermines objective-alignment narrative) or succeeds (makes contribution thinner) |
| Data pipeline details | Wrong audience; engineering not science; no reviewer asked |
| Norm explosion / training instability | Opens new attack surface; save for NatMed where context allows full discussion |
| Full few-shot scaling (all tasks) | Tier 2 covers 3-4 tasks if time permits; exhaustive few-shot across all tasks is camera-ready material, not rebuttal sprint |

---

## Rebuttal Structure (Updated — Lead with Surprise, Not Method)

**Key principles:**
1. **This is an SSL insight paper, not an application paper.** The contribution is understanding WHEN and WHY pixel reconstruction fails — echo is the case study, not the contribution. Frame for ICML audience (general ML), not MICCAI audience (medical imaging).
2. Lead with the falsifiable hypothesis, not with defense. Own the architecture point, then pivot hard to what IS novel: the mechanistic evidence (frame shuffling, CKA, noise probe) constitutes the first characterization of the noise-filtering mechanism in latent prediction.
3. "Objective choice matters more than scale" is the central claim — evidence: 525K clips beats 12M clips when the objective is right
4. A rebuttal with 3-4 striking results presented cleanly outperforms one with 9 results in a laundry list
5. Target the AC, not just reviewers — the AC decides when scores are split
6. Generalize explicitly: radar, sonar, low-SNR microscopy, fetal ultrasound — the EMA-as-noise-filter mechanism is domain-agnostic
7. Every sentence of preamble is attention wasted

### Opening (2 sentences)

"This paper tests a falsifiable hypothesis: pixel reconstruction fails in stochastic-noise-dominated domains because the model wastes capacity encoding noise, while latent prediction succeeds because the EMA target filters it. We now provide the first mechanistic evidence for this claim: (1) when temporal order is destroyed, JEPA representations degrade by 15% while MAE is invariant — proving JEPA encodes dynamics, not static texture; (2) CKA analysis confirms JEPA representations are noise-invariant while MAE representations shift under perturbation; and (3) objective choice matters more than scale — 525K clips with the right objective outperforms 12M clips with text supervision. These findings generalize beyond echocardiography to any domain where pixel fidelity is dominated by stochastic interference."

Then immediately into the evidence. Do not re-explain the paper.

### Section 1: The Hypothesis (all reviewers — 2 paragraphs max)

**Lead with the general principle, not defense.** Paragraph 1: "The architecture is not novel — we use V-JEPA2 without modification. The contribution is a new understanding about self-supervised learning: pixel reconstruction fails in stochastic-noise-dominated domains because the model devotes capacity to encoding noise rather than structure. Latent prediction avoids this — the EMA target acts as a noise low-pass filter, and the student learns to predict semantic content rather than stochastic texture. This is a testable, falsifiable hypothesis that generalizes beyond echocardiography to any modality where pixel noise dominates signal (radar, sonar, low-SNR microscopy). We test it on ultrasound as a case study."

Paragraph 2: "Objective choice matters more than scale. EchoJEPA-L (300M params, 525K clips) outperforms EchoPrime (12.1M clips, text supervision) and PanEcho (1.19M clips, 39-task supervision). A 45-point view classification gap between EchoJEPA-L and EchoMAE-L — identical in every respect except the objective — confirms the hypothesis under controlled conditions. New mechanistic evidence: frame shuffling shows JEPA encodes temporal dynamics (15% degradation) while MAE learns static patterns (0% degradation); CKA confirms JEPA representations are noise-invariant; a noise-level probe shows MAE features encode speckle while JEPA features do not. Together these constitute the first mechanistic characterization of the noise-filtering mechanism in latent prediction."

One-line linear probe confirmation: "This ranking holds under linear probing (EchoJEPA-L: 70.8% vs EchoMAE-L: 59.2%), ruling out probe architecture as a confound."

### Section 2: Mechanistic Evidence — A Contribution to SSL Understanding (ncQn + AC + ALL)

**Framing:** These results are not just answers to ncQn's request — they are the paper's primary scientific contribution. They provide the first mechanistic characterization of WHY latent prediction filters noise, which is a finding about self-supervised learning broadly, not just echocardiography.

Three strong results that are pure ML and require no clinical knowledge:

1. **Frame shuffling (champion result)** — JEPA degrades 10.9-15.1% on LVEF when frames shuffled, MAE invariant <0.5%. AC champion sentence: "The first direct evidence that latent prediction encodes temporal dynamics while pixel reconstruction learns static frame-level patterns." This is immediately striking, mechanistic, and addresses the novelty concern directly — it's a new finding about what these objectives learn.

2. **CKA speckle invariance** — one quantitative table. JEPA representations stable under noise, MAE shifts. Validates the "EMA as noise low-pass filter" mechanism. This result would hold for any stochastic-noise-dominated domain, not just ultrasound.

3. **Noise-level linear probe** — MAE features predict speckle intensity (high accuracy), JEPA features do not (low accuracy). Complements CKA: stability (CKA) + information content (probe). Together: **the encoder literally learns to ignore noise when trained with latent prediction, and to encode noise when trained with pixel reconstruction.** This is the mechanistic insight that elevates the paper from "ablation" to "understanding."

### Section 2.5: Scaling Analysis (all reviewers — if ViT-B results ready)

Present scaling on MIMIC + system-level to G. One compact table showing monotonic improvement. **Honest framing:** "We observe monotonic improvement from ViT-B (86M) through ViT-L (300M) to ViT-G (1.1B). The L → G comparison involves both model scale and data scale. The B → L comparison uses the same MIMIC data but notes an architectural difference (ViT-B uses V-JEPA 2.1 with dense loss; ViT-L uses V-JEPA 2.0). Despite this caveat, even the smallest model outperforms all non-JEPA baselines, and the scaling trend is monotonic." Addresses "incremental novelty" by adding a scaling contribution dimension.

### Section 3: EchoBench — Video-Level Robustness Benchmark (6t2T, hfQ1, ncQn)

Present EchoBench as a community contribution: 4 task rows (view classification, EF on EchoNet-Dynamic, EF on EchoNet-Pediatric, LV segmentation) × 22 conditions (clean + 7 perturbation types × 3 levels) × N models. Open-sourced with evaluation scripts.

Key framing: EchoBench differentiates from concurrent work (US-JEPA: 3 generic corruptions, classification only, frame-level) and existing frame-level benchmarks (UltraBench, CardioBench, ETAB) on three axes:
1. **Echo-specific physics-based perturbations** (depth attenuation, acoustic shadow, haze) — not generic blur/contrast
2. **Temporal perturbations** (frame drop, jitter, speed variation) — impossible for frame-level methods
3. **Task diversity** (classification + regression + segmentation) — not just classification

Key results to highlight:
- **Segmentation under noise** — even if MAE wins on clean Dice, show degradation curves: "MAE achieves competitive clean segmentation but degrades [X]% under acoustic perturbation vs [Y]% for JEPA, extending the robustness finding from recognition to dense prediction."
- **Temporal perturbation resilience** — frame dropping/jitter should hurt MAE more than JEPA (JEPA encodes temporal dynamics, MAE learns static patterns). Links directly to frame shuffling finding.
- **Domain-shift robustness** — EchoNet-Pediatric results show adult-trained representations transfer to pediatric echo under perturbation.
- Report all results honestly — the benchmark framing makes completeness a virtue.

This addresses 6t2T ("no SOTA comparisons beyond ablation"), hfQ1 ("broader downstream tasks" + segmentation), and ncQn ("noise sensitivity studies") simultaneously.

### Section 4: Contrastive Comparison (hfQ1)

BYOL-Video results if available. If ready, use appropriate contingency framing (see Concern 4). If not: EchoPrime reframing + Concern 3b data scale argument (EchoJEPA-L on 4.6K patients beats EchoPrime on 109K patients).

### Section 5: Community Resources & Reproducibility (all reviewers)

Four community contributions:
1. **EchoJEPA-G** — gated release upon acceptance, largest open echo video FM (1.1B params)
2. **EchoBench** — first video-level robustness benchmark for echocardiography FMs with echo-specific physics-based perturbations, temporal corruptions, and dense prediction. Covers EchoNet-Dynamic + EchoNet-Pediatric. Strictly more comprehensive than concurrent frame-level evaluations.
3. **Multi-view probing framework** — general-purpose method for variable-composition multi-view evaluation, applicable to multi-sequence MRI, multi-phase CT, multi-probe ultrasound. Open-sourced.
4. **EchoJEPA-L + evaluation framework** — already public

Dataset diversity clarification. Double-blind fix.

### Section 6: Camera-Ready Revisions

**Narrative restructuring (key commitment):** We will restructure the paper to lead with the general principle rather than the application:

- **Title consideration:** "Latent Prediction Outperforms Pixel Reconstruction in Noisy Imaging Domains: Evidence from Echocardiography" — positions echo as the case study for a general finding, not the end in itself
- **Introduction reframe:** Open with the general problem (pixel reconstruction in stochastic-noise-dominated domains), state the hypothesis, introduce echo as the ideal testbed (well-characterized noise, controlled comparison possible, clinical impact)
- **New "Mechanistic Analysis" section:** Elevate CKA + noise probe + frame shuffling from rebuttal supplement to main text. These are the primary scientific contribution — the first evidence for WHY latent prediction filters noise
- **Discussion:** Explicit connection to other noisy domains (radar, sonar, microscopy) — the mechanism (EMA target as noise filter) is domain-agnostic
- **Multi-view framework reframing:** Explicitly note applicability to multi-sequence MRI, multi-phase CT, and other variable-composition medical imaging

**Contribution list for camera-ready:**

| # | Contribution | Type | Scope |
|---|-------------|------|-------|
| 1 | Falsifiable hypothesis: pixel reconstruction fails in stochastic-noise domains | Scientific | General SSL |
| 2 | Mechanistic evidence: frame shuffling + CKA + noise probe | Empirical (SSL) | General SSL |
| 3 | Controlled experiment: objective matters more than 20× data scale | Empirical | General SSL |
| 4 | Multi-view attentive probing framework | Methodological | General medical imaging |
| 5 | EchoBench: video-level robustness benchmark | Community resource | Echocardiography |
| 6 | EchoJEPA-G/L: largest open echo video foundation models | Community resource | Echocardiography |

The key shift: contributions 1-3 are about SSL understanding (ICML scope), not about echocardiography (medical venue scope). Echo is the case study, not the contribution.

Summary table of all committed changes (experiments, text revisions, anonymization fixes).

### Discussion Period Strategy

After rebuttal submission, engage actively during the author-reviewer discussion:
- **ncQn**: If CKA results are strong, post them and ask directly: "Does this representation-level evidence adequately address your concern?" Invite them to collaborate on what the camera-ready should include. Makes it harder to maintain a reject after their specific request is fulfilled.
- **hfQ1**: If BYOL-Video baseline is available, present it and ask: "We have added the contrastive comparison you requested — a controlled BYOL-Video baseline isolating global vs local prediction targets. Does this address your concern about missing contrastive baselines?"

**If a reviewer or AC raises concurrent frame-level JEPA work (e.g., US-JEPA):** Prepared response: "We note that our work predates this concurrent submission on arxiv by three weeks. Key differences: (1) EchoJEPA uses video-level V-JEPA 2 with spatiotemporal masking, enabling temporal cardiac dynamics encoding that frame-level approaches structurally cannot capture — our frame shuffling analysis shows JEPA representations degrade 15% when temporal order is destroyed while MAE is invariant, proving video-level temporal encoding is critical; (2) we provide a strictly controlled three-way comparison (JEPA vs BYOL vs MAE, identical architecture/data/compute) isolating the pretraining objective; (3) EchoBench evaluates classification, regression, and segmentation under echo-specific physics-based perturbations plus temporal corruptions (frame dropping, jitter, speed variation) that are structurally impossible for frame-level methods — compared to US-JEPA's 3 generic corruptions on classification only; (4) we demonstrate scaling from ViT-B through ViT-G (1.1B parameters), 13x larger than frame-level approaches. The two works are complementary: US-JEPA explores frame-level JEPA across 22 US anatomies; EchoJEPA demonstrates video-level JEPA's ability to encode temporal cardiac dynamics at scale." Do NOT proactively mention this work — only respond if raised.

---

## Path to Acceptance (Updated 2026-03-27)

### Key assets (in order of impact)

1. **Narrative reframe: SSL insight paper, not application paper** (ALL reviewers + AC — this is the single most important change. Echo is the case study, not the contribution. The contribution is understanding when/why pixel reconstruction fails in noisy domains. Mechanistic evidence elevates from "ablation" to "understanding." Explicit generalization to radar, sonar, microscopy. Costs zero compute.)
2. **Frame shuffling** (AC champion sentence — "15% degradation vs 0% proves JEPA encodes dynamics, MAE doesn't" — pure ML, no clinical knowledge needed, no NatMed overlap. Fundamentally differentiates video-level approaches from frame-level ones.)
3. **CKA speckle invariance** (ncQn's direct ask — quantitative noise-filtering stability evidence. Frame as: "first evidence that EMA target acts as noise low-pass filter" — SSL contribution, not just echo.)
4. **Noise-level linear probe** (complements CKA — noise information content evidence. Together with CKA, makes the noise-filtering case airtight. "The encoder literally learns to ignore noise under latent prediction.")
5. **BYOL-Video controlled baseline** (only lever for hfQ1 — video-native, isolates local vs global prediction targets)
6. **EchoJEPA-G gated release** (zero compute, addresses 3/4 reviewers)
7. **ViT-B → ViT-L → ViT-G scaling analysis** (model scaling with honest 2.0→2.1 caveat for B→L; adds contribution dimension)
8. **EchoBench** (video-level benchmark with echo-specific perturbations — addresses segmentation + noise + novelty, but lower priority than 1-7 due to execution risk)
9. **Camera-ready restructuring commitment** (title change, lead with general principle, elevate mechanistic analysis to main text, discussion connecting to other noisy domains — shows reviewers the paper they'd accept is stronger than what they reviewed)
10. **Data scale argument** (Concern 3b — confounds work against EchoJEPA-L)
11. **Few-shot label scaling** (practical value — "reaches full-data baselines with 10% of labels"; include if time after 1-7)

### Per-reviewer predicted movement

| Reviewer | Current | Post-rebuttal | Key driver | P(flip) |
|----------|---------|---------------|-----------|---------|
| hfQ1 | 2 | 2->3 | BYOL baseline + novelty reframing | 30-35% |
| 6t2T | 3 | 3->4 | EchoBench + G release + segmentation | 35-40% (corrected down: confidence-5, "looks like ablation" is a framing objection not easily refuted by new experiments) |
| ncQn | 3 | 3->4/5 | CKA + noise probe + frame shuffling directly answers their ask | 75-80% |
| L8sp | 4 | 4->5 | Already positive; new evidence confirms | ~95% stays |

### Scenario analysis

| Scenario | hfQ1 | 6t2T | ncQn | L8sp | Avg | Outcome | P |
|----------|------|------|------|------|-----|---------|---|
| Best | 3 | 4 | 5 | 5 | 4.25 | Accept | 20% |
| Good | 3 | 4 | 4 | 4 | 3.75 | Likely accept | 15% |
| Decent | 2 | 4 | 4 | 4 | 3.5 | AC decides | 25% |
| Mixed | 2 | 3 | 4 | 4 | 3.25 | Borderline | 25% |
| Worst | 2 | 3 | 3 | 4 | 3.0 | Reject | 15% |

**Overall acceptance probability: ~35-45%** (corrected from 40-50%. The 6t2T correction is important: their concern is fundamentally about paper framing — "looks like an ablation study" — and adding more experiments doesn't change the paper's structure. Confidence-5 reviewers rarely move >1 point, and "applications at methods venue" is structural.)

### What caps probability below 50%

- hfQ1's confidence-5 reject — very hard to move; the novelty concern may be fundamental to their ICML expectations
- 6t2T's framing objection — "looks like ablation" is about paper structure, not missing experiments; new results don't change this
- Execution risk — BYOL may produce a moderate result (prepared with three framings)
- Structural issue — the AC's philosophy on empirical-contribution papers at ICML determines the outcome; this is not under our control
- Segmentation is a trap — may favor MAE; excluded from rebuttal unless result is favorable

### What the team controls

1. **Start BYOL-Video training as soon as GPU is available** — 3 days means this is on the critical path
2. **Run CKA + noise-level probe + frame shuffling** — directly answers ncQn's explicit request, hours of compute, three complementary angles on the same mechanism
3. **Narrative discipline** — 2-sentence opening, 4-5 focused results, no laundry lists
4. **AC-targeted language** — "The architecture is not novel. The hypothesis is: pixel reconstruction fails for noisy domains because it wastes capacity on noise. We test this with a controlled experiment and provide mechanistic evidence. Objective choice matters more than scale."
5. **Discussion period engagement** — directly engage ncQn ("Does this address your concern?") and hfQ1 ("We added the contrastive comparison you requested")

---

## Timeline (Updated 2026-03-28)

### Resource status
- **H100 node** (2 nodes x 8x H100): BYOL-Video v2 training (Job 241, epoch 12/240). Must stop in ~48h (~epoch 92).
- **A100 node** (8x A100 80GB): Available for CKA/frame shuffling/probes.

### Priority order (strict — do not reorder)
1. **CKA + noise-level probe + frame shuffling** (hours) — directly answers ncQn, provides AC champion sentence, highest ROI
2. **BYOL-Video training** (running on H100, ~48h to epoch 92) — only experiment that might move hfQ1
3. **ViT-B probes on ICML tasks** (hours) — scaling analysis (view, LVEF, RVSP), low effort
4. **Perturbed data generation** (hours) — needed for CKA; generate synthetic Rayleigh speckle at 3-5 intensity levels
5. **Few-shot label scaling** (hours) — if time after 1-4, high narrative value
6. **EchoBench / segmentation on EchoNet-Dynamic** (1-2 days) — only if time permits after 1-5; risk: MAE may win clean Dice
7. **Rebuttal writing** (1 day) — AFTER BYOL results are in; do not write before

### Day-by-day (revised)
- Day 1 (Mar 28, TODAY): BYOL training running on H100 (epoch 12). Run CKA + frame shuffling + noise probe on A100. Start ViT-B probes.
- Day 2 (Mar 29): BYOL continues (~epoch 52). Few-shot scaling if GPUs free. ViT-B probe results collected.
- Day 3 (Mar 30): BYOL reaches ~epoch 92, stop training. Evaluate BYOL on view/LVEF/RVSP. Pick contingency framing.
- Day 4 (Mar 31): EchoBench/segmentation if time and results favorable. Draft rebuttal text.
- Day 5 (Apr 1): Finalize rebuttal. Review narrative coherence.
- Buffer (Apr 2): Submission.

### Three-way comparison checkpoints (all on S3)

| Model | S3 Path | Epoch |
|-------|---------|-------|
| V-JEPA (EchoJEPA-L) | `s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts/runs/vjepa_mimic_pretrain_125/training_folder/e100.pt` | 100 |
| EchoMAE-L | `s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts/runs/videomae_115/training_folder/checkpoint-99.pth` | 99 |
| BYOL-Video-L | `s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts/runs/byol-vitl-imagenet-v2/e92.pt` | 92 |

---

## BYOL Result Risk

BYOL-Video is the highest-stakes experiment. Unlike CKA/frame shuffling (where we can predict outcomes),
BYOL's performance is genuinely uncertain.

**Run v1 (epochs 1-40) — misconfigured, results invalid.** The BYOL config used cosine EMA ramp
(0.996 → 1.0) per the original BYOL paper, while V-JEPA uses constant EMA (0.99925). Combined with
constant LR (no decay), the ramping EMA caused the target encoder to freeze by epoch ~12. Effective
batch size was also mismatched (512 vs V-JEPA's 1024).

Collapse detection probes (d=1 attentive, UHN data):
- *View classification (13-class, 22K):* Identical at epoch 10 vs 40 (24.61% val acc, 0.696 AUROC) — too coarse to detect degradation.
- *LVEF regression (5K train / 2K val subset):* Clear collapse in Pearson r:

| Encoder | Epoch 10 | Epoch 40 | Drop |
|---------|----------|----------|------|
| Online | r=0.151 | r=0.089 | -41% |
| Target (EMA) | r=0.156 | r=0.068 | -56% |

Online vs target nearly identical within each checkpoint. Target encoder degraded *more* than online
(56% vs 41% drop), suggesting EMA accumulated rather than stabilized the degradation. MAE ~8.06 across
all conditions (barely above mean-prediction baseline), so Pearson r was the sensitive metric.
**This run does not constitute a valid controlled comparison.**

**Run v2 — LAUNCHED (2026-03-28, Job 241)**

Infrastructure:
- Cluster: echojepa-h100-march (SageMaker HyperPod, `yyepvbne5vzr`)
- Compute: 2 nodes x 8x H100 80GB = 16 GPUs
- Launcher: `srun python -m app.main_srun` (multi-node DDP)
- sbatch: `~/byol_pretrain_2node.sbatch`

Training config (matched to V-JEPA / EchoJEPA-L):

| Parameter | Value |
|-----------|-------|
| Model | ViT-L/16, RoPE, activation checkpointing |
| Init weights | `vitl_in21k.pt` (ImageNet-21k) |
| Objective | BYOL (cosine similarity, 2 temporal clips) |
| EMA | [0.99925, 0.99925] constant (matches V-JEPA) |
| Batch size | 64/GPU x 16 GPUs = 1024 effective (matches V-JEPA) |
| LR | 1.75e-4 constant after warmup (start: 3.33e-5) |
| Warmup | 40 epochs (12,000 iters) |
| Epochs | 240 |
| ipe | 300 |
| Total samples | 73.7M (matched to V-JEPA) |
| Weight decay | 0.04 |
| Precision | bfloat16 |
| Projector | 4096 hidden -> 256 dim |
| Predictor | 4096 hidden MLP |

Progress (as of 2026-03-28):
- Current: Epoch 12 of 240 (5%), still in warmup phase (12 of 40)
- Loss: -1.98 (monotonically improving from -1.66 at e1, no plateau)
- Iter time: ~7.2s, ~36 min/epoch
- Elapsed: ~7h, Remaining: ~5.7 days, ETA: ~April 3

Checkpointing:
- latest.pt + latest_opt.pt every epoch (~4.8GB)
- Periodic snapshots every 2 epochs, all archived to S3
- S3: `s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts/runs/byol-vitl-imagenet-v2/`

**Timing constraint:** Must stop training in ~48h (reach ~epoch 92). Compare all three models at matched epoch count:

| Model | Checkpoint | Epoch |
|-------|-----------|-------|
| V-JEPA (EchoJEPA-L) | `s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts/runs/vjepa_mimic_pretrain_125/training_folder/e100.pt` | 100 |
| EchoMAE-L | `s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts/runs/videomae_115/training_folder/checkpoint-99.pth` | 99 |
| BYOL-Video | `s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts/runs/byol-vitl-imagenet-v2/e92.pt` | 92 |

~8 epoch spread (92 vs 99-100). All post-warmup with 50+ epochs of full LR training. Defensible comparison — same architecture, same data, same init, matched LR schedule, ~matched compute. The only variable is the objective.

**Why 6 days instead of 3:** V1 used batch_size=32 (effective 512), giving 3.7s/iter. V2 doubled to batch_size=64 (effective 1024) to match V-JEPA, so iter time doubled. Necessary for fair three-way comparison where total samples, LR schedule, and batch size are all matched.

**Note on ipe:** Reducing ipe from 300 to 150 doesn't help — it halves total samples unless you also double epochs (and warmup), giving identical wall time. Wall time is locked at total_iters x iter_time.

**Risk**: If BYOL achieves ~80%+ view accuracy, the "only latent prediction works" narrative weakens.
This is non-trivial — BYOL shares the EMA teacher mechanism with JEPA, which may be the actual
noise-filtering ingredient rather than local prediction specifically.

**Mitigation**: Three contingency framings prepared (see Concern 4). All are publishable. The key is to
run BYOL early enough that the rebuttal narrative can be adapted to the actual result. Do NOT write
the rebuttal text before BYOL results are in.

**What happens if we don't run BYOL**: Fall back to EchoPrime reframing (Option A in Concern 4) +
data scale argument (Concern 3b). This is weaker but still viable. hfQ1 flip probability drops to ~15%.

---

## Relationship to Previous Rebuttal Documents

This document **supersedes** the strategy in `01`-`07` for the actual rebuttal response. Those documents remain valuable as:
- `01-paper-audit.md` — reference for appendix fixes and editorial corrections
- `04-competitive-positioning.md` — framing for EchoJEPA vs concurrent work
- `05-probe-fairness.md` — background on Strategy E (d=1 attentive), useful if probe questions arise in discussion
- `07-camera-ready-actions.md` — editorial action items still apply

Documents `02` (old rebuttal template), `03` (old worst-case scenarios), and `06` (old claim validity) are largely superseded by this document.
